import os
import re
import requests
from collections import deque
from dotenv import load_dotenv
from SPARQLWrapper import SPARQLWrapper, JSON
from py2neo import Graph, Node, Relationship

# 安裝 google.generativeai: pip install google-generativeai
import google.generativeai as genai

###############################################################################
# 1. 環境初始化 & 連線設定
###############################################################################
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("請先設定 GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# 連接 Neo4j (請自行修改連線資訊)
neo4j_graph = Graph("bolt://localhost:7689", auth=("neo4j", "12345678"))

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# 全域快取
triple_cache = {}  # BFS 三元組快取
qid_to_label_cache = {}  # QID -> Label
qid_to_p31_cache = {}  # QID -> [instance of]


###############################################################################
# 2. 清理 Neo4j
###############################################################################
def clear_neo4j_data():
    """
    清空 Neo4j 資料庫所有節點與關係
    """
    neo4j_graph.delete_all()


###############################################################################
# 3. 取得屬性 / 實體標籤
###############################################################################
def get_property_label(pid: str) -> str:
    """
    從 Wikidata 取得指定 property (Pxx) 的英文 label。
    """
    if pid.startswith("http://www.wikidata.org/prop/direct/"):
        pid = pid.split("/")[-1]
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": pid,
        "props": "labels",
        "languages": "en",
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if "entities" in data and pid in data["entities"]:
            en_label = (
                data["entities"][pid].get("labels", {}).get("en", {}).get("value")
            )
            if en_label:
                return en_label
        return pid
    except Exception as e:
        print(f"[Error] get_property_label({pid}): {e}")
        return pid


def get_entity_label(qid: str) -> str:
    """
    從 Wikidata 取得實體 (Qxx) 的英文 label。
    """
    if qid in qid_to_label_cache:
        return qid_to_label_cache[qid]

    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qid}&format=json&props=labels&languages=en"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        if "entities" in data and qid in data["entities"]:
            en_label = (
                data["entities"][qid].get("labels", {}).get("en", {}).get("value")
            )
            if en_label:
                qid_to_label_cache[qid] = en_label
                return en_label
        return qid
    except Exception as e:
        print(f"[Error] get_entity_label({qid}): {e}")
        return qid


###############################################################################
# 4. 取得 instance of (P31)
###############################################################################
def get_instance_of(qid: str) -> list:
    """
    回傳該 QID 的所有 P31 (instance of) 項目 ID (如 [Q5(人), Q11424(電影)] ).
    有快取以減少重複查詢。
    """
    if qid in qid_to_p31_cache:
        return qid_to_p31_cache[qid]

    query = f"""
    SELECT ?type WHERE {{
      wd:{qid} wdt:P31 ?type .
    }}
    """
    s = SPARQLWrapper(WIKIDATA_SPARQL)
    s.setQuery(query)
    s.setReturnFormat(JSON)

    types_found = []
    try:
        ret = s.queryAndConvert()
        for row in ret["results"]["bindings"]:
            type_uri = row["type"]["value"]
            type_qid = type_uri.split("/")[-1]
            types_found.append(type_qid)
    except Exception as e:
        print(f"[Error] get_instance_of({qid}): {e}")

    qid_to_p31_cache[qid] = types_found
    return types_found


###############################################################################
# 5. 根據 subject/object 類型 & property label 動態生成敘述
###############################################################################
domain_range_rules = [
    # (domainQID, property_label, rangeQID, text_pattern)
    ("Q11424", "director", "Q5", "({obj}) is the director of ({subj})"),
    ("Q5", "successor", "Q5", "({obj}) is the successor of ({subj})"),
    # 你可以繼續加更多...
]


def describe_triple_with_domain_range(
    subj_qid: str, prop_qid: str, obj_qid: str
) -> str:
    """
    若符合 domain_range_rules, 則翻轉描述; 否則 fallback: (subj_qid)-[prop_label]->(obj_qid)
    """
    prop_label = get_property_label(prop_qid)  # 例如 "director", "successor"
    subj_types = get_instance_of(subj_qid)  # e.g. ["Q11424"]
    obj_types = get_instance_of(obj_qid)  # e.g. ["Q5"]

    for dom, p_label, rng, text_pattern in domain_range_rules:
        if (
            dom in subj_types
            and rng in obj_types
            and prop_label.lower() == p_label.lower()
        ):
            return text_pattern.format(subj=subj_qid, obj=obj_qid)

    return f"({subj_qid})-[{prop_label}]->({obj_qid})"


###############################################################################
# 6. LLM 幫忙抽取屬性 (第二階段 BFS)，針對「新實體」動態問 LLM
###############################################################################
def llm_extract_properties_for_entity(question: str, entity_label: str):
    """
    給 LLM 一個新實體 (如 'Democratic Party'), 問:
    "為了回答 {question}, 我還需要對 {entity_label} 哪些 property Pxx?"
    回傳 pids (list)
    """
    prompt = f"""
We have a new entity: '{entity_label}' relevant to the question: '{question}'.
Which Wikidata property IDs (Pxx) might be needed from this entity to help answer the question?
Output only the property IDs, separated by commas, with no explanation.
    """.strip()
    resp = model.generate_content(prompt)
    text = resp.text.replace("\n", " ").strip()
    pids = [x.strip() for x in text.split(",") if x.strip().startswith("P")]
    return pids


###############################################################################
# 7. BFS：先對初始實體抽取 property -> 取得三元組 -> 發現新實體 -> 對新實體再向 LLM 要 property -> 繼續
###############################################################################
def fetch_triples_dynamically(question: str, seed_qids: list, max_depth: int = 3):
    """
    queue: (qid, depth)
    visited: 紀錄走過的 QID
    """
    triple_list = []
    visited = set(seed_qids)
    queue = deque()
    for q in seed_qids:
        queue.append((q, 0))

    while queue:
        current_qid, depth = queue.popleft()
        if depth >= max_depth:
            continue

        # 1) 先拿 label
        e_label = get_entity_label(current_qid)
        # 2) 用 LLM 要 property
        pids_to_fetch = llm_extract_properties_for_entity(question, e_label)
        pids_to_fetch = list(set(pids_to_fetch))  # 去重

        if not pids_to_fetch:
            continue

        # 3) 用 wdt:Pxx 抓
        fetched = fetch_triples_for_qid(current_qid, pids_to_fetch)
        triple_list.extend(fetched)

        # BFS enqueue
        for s_iri, p_iri, o_iri in fetched:
            if o_iri.startswith("http://www.wikidata.org/entity/"):
                new_qid = o_iri.split("/")[-1]
                if new_qid not in visited:
                    visited.add(new_qid)
                    queue.append((new_qid, depth + 1))

    return triple_list


def fetch_triples_for_qid(qid: str, pids: list):
    """
    與舊程式相同: 用 wdt:Pxx
    """
    if not pids:
        return []
    results = []
    pid_uris = [f"wdt:{pid}" for pid in pids]
    pids_filter = ", ".join(pid_uris)
    query = f"""
    SELECT ?p ?o
    WHERE {{
      wd:{qid} ?p ?o .
      FILTER (?p IN ({pids_filter}))
    }}
    """
    s = SPARQLWrapper(WIKIDATA_SPARQL)
    s.setQuery(query)
    s.setReturnFormat(JSON)
    try:
        ret = s.queryAndConvert()
        subj_uri = f"http://www.wikidata.org/entity/{qid}"
        for row in ret["results"]["bindings"]:
            p_iri = row["p"]["value"]
            o_iri = row["o"]["value"]
            results.append((subj_uri, p_iri, o_iri))
            print(f"Fetched Triple: ({subj_uri}, {p_iri}, {o_iri})")
    except Exception as e:
        print(f"[Error] fetching triples for {qid}:", e)
    return results


###############################################################################
# 8. 上傳到 Neo4j
###############################################################################
def upload_triples_to_neo4j(triples):
    for s, p, o in triples:
        s_qid = s.split("/")[-1]
        p_qid = p.split("/")[-1]
        o_qid = o.split("/")[-1]

        s_label = get_entity_label(s_qid)
        p_label = get_property_label(p_qid)
        o_label = get_entity_label(o_qid)

        s_node = Node("Entity", uri=s, qid=s_qid, label=s_label)
        o_node = Node("Entity", uri=o, qid=o_qid, label=o_label)

        neo4j_graph.merge(s_node, "Entity", "uri")
        neo4j_graph.merge(o_node, "Entity", "uri")

        rel = Relationship(s_node, p_qid, o_node)
        rel["name"] = p_label
        neo4j_graph.create(rel)


###############################################################################
# 9. LLM Reasoning
###############################################################################
llm_reasoning_prompt = """
You are a helpful assistant that can answer questions based on a given knowledge subgraph.
Here is the knowledge subgraph:
{subgraph_desc}

Answer the following question based on the subgraph. **Do not use any external knowledge. Only base your answer on the provided subgraph.**
Question: {question}

Think step by step and provide the answer. **If the answer cannot be found in the subgraph, say you don't know.**
Answer:
"""


def llm_reasoning_on_subgraph(question: str, subgraph_desc: str):
    prompt = llm_reasoning_prompt.format(subgraph_desc=subgraph_desc, question=question)
    response = model.generate_content(prompt)
    return response.text.strip()


###############################################################################
# 10. 抽取實體名稱
###############################################################################
def extract_entity_names(question: str):
    prompt = f"""
From the question: '{question}', extract the entity names (one or more).
Consider any text enclosed in quotes as a potential entity.
Output only the entity names, separated by commas. 
Do not provide any explanation.
"""
    resp = model.generate_content(prompt)
    cleaned = resp.text.replace("\n", " ").strip()
    names = [x.strip().strip("'\"") for x in cleaned.split(",") if x.strip()]
    return names


def link_entities_to_wikidata(question: str):
    entity_names = extract_entity_names(question)
    # fallback: 'Titanic'
    if "titanic" in question.lower() and not any(
        "titanic" in e.lower() for e in entity_names
    ):
        entity_names.append("Titanic")

    ent_map = {}
    for name in entity_names:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": name,
            "limit": 1,
        }
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            if data.get("search"):
                top = data["search"][0]
                qid = top["id"]
                label = top["label"]
                ent_map[label] = qid
                qid_to_label_cache[qid] = label
        except:
            pass
    return ent_map


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    question = "When was the political party of the current president of the United States founded?"

    # 1) 先用 LLM 抽實體 -> wbsearch => QID
    linked_entities = link_entities_to_wikidata(question)
    print("Linked Entities:", linked_entities)

    # 2) 將QID放入 BFS
    seed_qids = list(linked_entities.values())
    if not seed_qids:
        print("No entity found from question, end.")
        exit()

    # 3) 多階段BFS (Iterative), LLM動態給property
    print("[BFS] Start multi-iteration BFS with LLM extracting property each time...")
    triple_list = fetch_triples_dynamically(question, seed_qids, max_depth=5)
    print(f"Total fetched: {len(triple_list)} triples")

    # 4) 清空Neo4j & 上傳
    print("Clearing Neo4j data...")
    clear_neo4j_data()
    if triple_list:
        upload_triples_to_neo4j(triple_list)
        print("Uploaded to Neo4j:", len(triple_list), "triples.")
    else:
        print("No triples to upload.")

    # 5) 給 LLM Reasoning
    lines = []
    for s, p, o in triple_list:
        s_qid = s.split("/")[-1]
        p_qid = p.split("/")[-1]
        o_qid = o.split("/")[-1]
        # optional: flip if domain/range => describe_triple_with_domain_range
        # 這裡就直接 " s_qid - p_label -> o_qid"
        # or use domain-range if you want
        p_label = get_property_label(p_qid)
        lines.append(f"({s_qid})-[{p_label}]->({o_qid})")

    subgraph_desc = "\n".join(lines)
    final_answer = llm_reasoning_on_subgraph(question, subgraph_desc)
    print("=== Final Answer ===")
    print(final_answer)
    print("=======================")
