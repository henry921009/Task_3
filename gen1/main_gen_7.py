import os
import re
import requests
import rdflib  # 用來處理 RDF
from rdflib import Graph as RDFGraph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from SPARQLWrapper import SPARQLWrapper, JSON
import google.generativeai as genai
import nltk
from dotenv import load_dotenv

nltk.download("punkt")

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("請先設定 GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


def clear_fuseki_data(fuseki_update_url: str):
    """
    清空 Fuseki 資料集中的所有三元組。
    """
    query = "CLEAR DEFAULT"
    params = {"update": query}
    try:
        response = requests.post(fuseki_update_url, data=params)
        response.raise_for_status()  # 檢查是否有 HTTP 錯誤
        print("Successfully cleared Fuseki data.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error clearing Fuseki data: {e}")
        return False


################################################################################
# 1. 從自然語問題中抽取 Entities & Relations
################################################################################

node_query_prompt = """
From the question: '{question}', extract the entity names (one or more).
Output only the entity names, separated by commas. Do not provide any explanation.
"""

relation_query_prompt = """
Given the question: '{question}' and the entities: '{node_queries}', 
list the possible relationships (Wikidata property IDs like Pxx) that might be relevant to answering this question.
Output only the property IDs, separated by commas, with no explanation.
"""


def generate_node_queries(question: str):
    prompt = node_query_prompt.format(question=question)
    response = model.generate_content(prompt)
    cleaned = response.text.replace("\n", " ").strip()
    return [x.strip() for x in cleaned.split(",") if x.strip()]


def generate_relation_queries(question: str, node_queries: list):
    prompt = relation_query_prompt.format(
        question=question, node_queries=", ".join(node_queries)
    )
    response = model.generate_content(prompt)
    cleaned = response.text.replace("\n", " ").strip()
    rels = [r.strip() for r in cleaned.split(",") if r.strip().startswith("P")]
    return rels


################################################################################
# 2. 從 Wikidata 官方 endpoint 抓取子圖 (只抓指定 QID + 指定 Properties)
################################################################################


def get_wikidata_qid(entity_name: str, question: str = None):
    """
    根據實體名稱從 Wikidata 取得 QID。

    Args:
        entity_name: 實體名稱。
        question: 原始問題 (可選)，用於提供實體描述資訊。

    Returns:
        如果找到匹配的 QID，則返回 QID；否則返回 None。
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": entity_name,
        "limit": 5,  # 限制返回的候選 QID 數量
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("search"):
            # 根據匹配度和實體類型進行篩選 (這裡只是一個簡單的示例)
            best_match = None
            for result in data["search"]:
                if best_match is None:
                    best_match = result
                elif result.get("match", {}).get("score", 0) > best_match.get(
                    "match", {}
                ).get("score", 0):
                    best_match = result

            if best_match:
                return best_match["id"]
        return None

    except requests.exceptions.RequestException as e:
        print(f"[Error] get_wikidata_qid for '{entity_name}': {e}")
        return None


def fetch_entity_triples_specific(qids: list, pids: list):
    """
    只抓 qid 與指定 pids 的三元組 (更精準、更輕量)。
    pids 格式： ["P569", "P31", ...]
    """
    all_triples = []
    for qid in qids:
        for pid in pids:
            # Correctly format the QID as a URI
            qid_uri = (
                f"<http://www.wikidata.org/entity/{qid}>"
                if not qid.startswith("http")
                else f"<{qid}>"
            )

            query = f"""
            SELECT ?o
            WHERE {{
              {qid_uri} wdt:{pid} ?o .
            }}
            """
            s = SPARQLWrapper(WIKIDATA_SPARQL)
            s.setQuery(query)
            s.setReturnFormat(JSON)

            try:
                ret = s.queryAndConvert()
                subj_uri = f"http://www.wikidata.org/entity/{qid}"
                pred_uri = f"http://www.wikidata.org/prop/direct/{pid}"
                for row in ret["results"]["bindings"]:
                    o_iri = row["o"]["value"]
                    all_triples.append((subj_uri, pred_uri, o_iri))
                    print(
                        f"Fetched (SPECIFIC) Triple for {qid}: ({subj_uri}, {pred_uri}, {o_iri})"
                    )
            except Exception as e:
                print(f"[Error] fetch_entity_triples_specific for {qid}, P{pid}:", e)
    return all_triples


def fetch_subgraph_multihop(initial_qids: list, initial_pids: list, max_depth: int):
    """
    使用 BFS 逐步擴展子圖，抓取多跳關係。
    """
    all_triples = []
    visited_qids = set(initial_qids)
    queue = [(qid, 0) for qid in initial_qids]

    while queue:
        current_qid, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        # 假設每次都使用初始的 PIDs
        triples = fetch_entity_triples_specific([current_qid], initial_pids)
        for triple in triples:
            subj, pred, obj = triple
            all_triples.append(triple)
            # 如果對象是另一個實體，且未被訪問，加入隊列
            if (
                obj.startswith("http://www.wikidata.org/entity/")
                and obj not in visited_qids
            ):
                visited_qids.add(obj)
                # Ensure obj is treated as a URI in the queue
                queue.append(
                    (obj.replace("http://www.wikidata.org/entity/", ""), depth + 1)
                )

    return all_triples


################################################################################
# 3. 把抓到的三元組轉成 RDF (Turtle)，然後上載到本地 Fuseki
################################################################################


def create_rdf_graph_from_triples(triples):
    """
    使用 rdflib 建立一個 RDF Graph，把 (s, p, o) 全部加進去。
    """
    g = RDFGraph()

    for s, p, o in triples:
        s_ref = URIRef(s)
        p_ref = URIRef(p)
        # 判斷 o 是否 URI 或 literal
        if o.startswith("http://") or o.startswith("https://"):
            o_ref = URIRef(o)
        else:
            # 可能是一個 literal，根據屬性ID指定數據類型
            if "P569" in p:  # 出生日期
                o_ref = Literal(o, datatype=XSD.dateTime)
            else:
                o_ref = Literal(o)
        g.add((s_ref, p_ref, o_ref))
        print(f"Added to RDF Graph: ({s_ref}, {p_ref}, {o_ref})")
    return g


def verify_fuseki_data(fuseki_query_endpoint, qid):
    """
    驗證 Fuseki 中是否能成功查到該 QID 的三元組。
    """
    test_query = f"""
    SELECT ?p ?o WHERE {{
        <http://www.wikidata.org/entity/{qid}> ?p ?o .
    }}
    """
    s = SPARQLWrapper(fuseki_query_endpoint)
    s.setQuery(test_query)
    s.setReturnFormat(JSON)
    try:
        ret = s.queryAndConvert()
        print(f"=== Fuseki Data for {qid} ===")
        for row in ret["results"]["bindings"]:
            p = row["p"]["value"]
            o = row["o"]["value"]
            print(f"<{p}> -> <{o}>")
    except Exception as e:
        print("Error verifying Fuseki data:", e)


def upload_graph_to_fuseki(rdf_graph: RDFGraph, fuseki_dataset_url: str, qid: str):
    """
    將 rdflib Graph 上傳到 Fuseki (HTTP POST)。
    """
    data = rdf_graph.serialize(format="turtle")  # 轉成turtle字串
    post_url = fuseki_dataset_url.rstrip("/") + "/data"
    headers = {"Content-Type": "text/turtle"}
    resp = requests.post(post_url, data=data, headers=headers)
    if resp.status_code not in [200, 201, 204]:
        print("Upload to Fuseki failed:", resp.status_code, resp.text)
        return False
    else:
        print("Upload to Fuseki success:", resp.status_code)

    # 除錯：檢查 Fuseki 中的數據
    verify_fuseki_data(fuseki_query_endpoint=f"{fuseki_dataset_url}/sparql", qid=qid)
    return True


################################################################################
# 4. LLM 產生 SPARQL (只針對我們的本地子圖)
################################################################################

kg_query_prompt = """
You are a SPARQL query generator. Below is the subgraph of Wikidata (now stored locally).
You can only use the following QIDs and PIDs in the subgraph. The user question is also given.

Available QIDs:
{available_qids}

Available PIDs:
{available_pids}

Subgraph Description:
{subgraph_desc}

Question:
{question}

Requirements:
1. Output only a valid SPARQL query as plain text.
2. Do not include any explanations, comments, or additional messages.
3. Use the provided URIs for QIDs and PIDs.
4. Do not use any other QIDs or PIDs not listed above.
5. Do not rely on external knowledge.
6. Use meaningful variable names that reflect the property they represent (e.g., ?dateOfBirth for P569, ?spouse for P26).
7. The query must connect the entities mentioned in the question, potentially using multiple hops.
8. **The query should reflect the relationships and constraints implied by the question.** For example, if the question asks for the spouse of the director of a movie, the query should include a link between the director and the movie.
9. **Example:** If the question is "Who is the author of 'The Old Man and the Sea'?" and the subgraph contains entities for 'The Old Man and the Sea' (Q26505) and 'Ernest Hemingway' (Q23434) and the property 'author' (P50), a valid query would be:
    ```sparql
    SELECT ?author WHERE {{
        <http://www.wikidata.org/entity/Q26505> <http://www.wikidata.org/prop/direct/P50> ?author .
    }}
    ```

Return only the SPARQL query without any surrounding text.
"""


def generate_kg_query(question, subgraph_desc, available_qids, available_pids):
    prompt = kg_query_prompt.format(
        subgraph_desc=subgraph_desc,
        available_qids=", ".join(available_qids),
        available_pids=", ".join(available_pids),
        question=question,
    )
    print("\n[Prompt to LLM]\n", prompt, "\n----------------\n")
    response = model.generate_content(prompt)
    raw_query = response.text.strip()
    # 移除可能的 Markdown 反引號
    cleaned = re.sub(r"```(\w+)?", "", raw_query)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()
    return cleaned


def execute_local_sparql_query(query_text, fuseki_query_endpoint):
    """
    對本地 Fuseki 執行 SPARQL Query。
    """
    s = SPARQLWrapper(fuseki_query_endpoint)
    s.setReturnFormat(JSON)
    s.setQuery(query_text)
    try:
        ret = s.queryAndConvert()
        return ret
    except Exception as e:
        print("execute_local_sparql_query Error:", e)
        return None


################################################################################
# 5. LLM 產生自然語言回答
################################################################################

answer_prompt = """
Given the query results: '{query_results}',
and the original question: '{question}',
please provide a concise English answer. Do not include any explanations or additional commentary.
"""


def get_labels_for_qids(qids: list):
    """
    根據 QIDs 從 Wikidata 取得對應的標籤（Labels）。
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": "|".join(qids),
        "props": "labels",
        "languages": "en",
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        labels = {}
        for entity_id, entity_data in data.get("entities", {}).items():
            label = entity_data.get("labels", {}).get("en", {}).get("value", entity_id)
            labels[entity_id] = label
        return labels
    except requests.exceptions.RequestException as e:
        print(f"Error fetching labels for QIDs {qids}: {e}")
        return {qid: qid for qid in qids}  # 返回 QID 本身作為回退


def generate_answer(query_results, question):
    if not query_results or "results" not in query_results:
        return "No results or invalid query."

    # 簡單將前幾筆結果列成文字
    qids = []
    for r in query_results["results"]["bindings"]:
        for var_name, val_dict in r.items():
            value = val_dict["value"]
            if value.startswith("http://www.wikidata.org/entity/"):
                qid = value.split("/")[-1]
                qids.append(qid)
            else:
                qids.append(value)  # 若不是 QID，直接加入

    # 去除重複 QIDs
    unique_qids = list(set(qids))

    # 取得 QIDs 的標籤
    labels = get_labels_for_qids(unique_qids)

    # 將 QIDs 轉換為標籤
    labeled_results = [labels.get(qid, qid) for qid in qids]

    # 組成回應
    if labeled_results:
        # 去除重複並以逗號分隔
        final_labels = ", ".join(list(set(labeled_results)))
        return final_labels
    else:
        return "No results found."


################################################################################
# MAIN DEMO
################################################################################

if __name__ == "__main__":
    # 設定 Fuseki 資料集和 Update URL
    FUSEKI_DATASET_URL = "http://localhost:3030/ds"  # 根據實際情況修改
    FUSEKI_QUERY_ENDPOINT = "http://localhost:3030/ds/sparql"
    FUSEKI_UPDATE_ENDPOINT = "http://localhost:3030/ds/update"

    # 使用者問題
    question = "Who is the president of the United States?"

    # 1. 從問題萃取 entities 和 properties
    node_queries = generate_node_queries(question)
    relation_queries = generate_relation_queries(question, node_queries)

    print("Node Queries:", node_queries)
    print("Relation Queries:", relation_queries)

    qids_to_fetch = []
    for entity_name in node_queries:
        qid = get_wikidata_qid(entity_name)
        if qid:
            qids_to_fetch.append(qid)
            print(f"Found QID for '{entity_name}': {qid}")
        else:
            print(f"Could not find QID for '{entity_name}'")

    # 2. 從 Wikidata 抓多跳的子圖
    if qids_to_fetch and relation_queries:
        triple_list = fetch_subgraph_multihop(
            qids_to_fetch, relation_queries, max_depth=30
        )
    else:
        print("No valid QIDs or no property IDs found, skipping subgraph fetching.")
        triple_list = []

    # 3. 建立本地 RDF Graph
    rdf_graph = create_rdf_graph_from_triples(triple_list)

    # 每次上傳前先清理 Fuseki 資料庫
    print("Clearing Fuseki data...")
    if clear_fuseki_data(FUSEKI_UPDATE_ENDPOINT):
        # 上載到 Fuseki
        if qids_to_fetch and len(triple_list) > 0:
            # 這裡的 QID 用於簡單驗證
            upload_success = upload_graph_to_fuseki(
                rdf_graph, FUSEKI_DATASET_URL, qids_to_fetch[0]
            )
            if not upload_success:
                print("上傳到 Fuseki 失敗，終止程式。")
                exit(1)
        else:
            print("No triples to upload.")
    else:
        print("清理 Fuseki 資料失敗，可能影響後續操作。")

    # 4. 准備可用的 QIDs 和 PIDs
    available_qids = list(
        set(
            [t[0] for t in triple_list]
            + [
                t[2]
                for t in triple_list
                if t[2].startswith("http://www.wikidata.org/entity/")
            ]
        )
    )
    available_pids = list(set([t[1] for t in triple_list]))

    # 5. 描述子圖 (簡易描述) - 給 LLM 產生 SPARQL 用
    desc_lines = []
    for s, p, o in triple_list:
        desc_lines.append(f"{s} -> {p} -> {o}")
    subgraph_desc = "\n".join(desc_lines)

    # 6. LLM 產生針對本地子圖的 SPARQL
    kg_query = generate_kg_query(
        question, subgraph_desc, available_qids, available_pids
    )
    print("[LLM generated SPARQL]\n", kg_query)

    # 7. 執行本地 SPARQL
    query_results = execute_local_sparql_query(kg_query, FUSEKI_QUERY_ENDPOINT)

    # 8. 最終回答
    final_answer = generate_answer(query_results, question)
    print("=== Final Answer ===")
    print(final_answer)
    print("=======================")
