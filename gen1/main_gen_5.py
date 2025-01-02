import os
import re
import requests
import rdflib  # 用來處理 RDF
from rdflib import Graph as RDFGraph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from SPARQLWrapper import SPARQLWrapper, JSON
import google.generativeai as genai
import nltk
from dotenv import load_dotenv  # 導入 dotenv

# 若需要NLTK
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
# 2. 從 Wikidata 官方 endpoint 抓取子圖 (只抓指定 QID + 鄰居)
################################################################################

def get_wikidata_qid(entity_name: str):
    """
    根據實體名稱從 Wikidata 取得 QID。
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "sites": "enwiki",  # 可以根據需要修改語言
        "props": "",
        "titles": entity_name,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print("Wikidata API Response:", data)
        if "entities" in data:
            for entity_id in data["entities"]:  # 直接遍歷 entities 的鍵
                return entity_id
        return None
    except requests.exceptions.RequestException as e:
        print(f"[Error] get_wikidata_qid for '{entity_name}': {e}")
        return None


def fetch_entity_triples(qids: list, limit: int = 1000):
    """
    從 Wikidata 抓取多個 wd:QID 的所有相關三元組。
    """
    all_triples = []
    for qid in qids:
        query = f"""
        SELECT ?p ?o
        WHERE {{
          wd:{qid} ?p ?o .
        }} LIMIT {limit}
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
                all_triples.append((subj_uri, p_iri, o_iri))
                print(
                    f"Fetched Triple for {qid}: ({subj_uri}, {p_iri}, {o_iri})"
                )  # 除錯輸出
        except Exception as e:
            print(f"[Error] fetch_entity_triples for {qid}:", e)
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
            if p.endswith("P569"):  # 出生日期
                o_ref = Literal(o, datatype=XSD.dateTime)
            else:
                o_ref = Literal(o)
        g.add((s_ref, p_ref, o_ref))
        print(f"Added to RDF Graph: ({s_ref}, {p_ref}, {o_ref})")  # 除錯輸出
    return g


def verify_fuseki_data(fuseki_query_endpoint, qid):
    """
    驗證 Fuseki 中是否有特定的 PIDs。
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
        return False  # 回傳失敗狀態
    else:
        print("Upload to Fuseki success:", resp.status_code)

    # 除錯：檢查 Fuseki 中的數據
    verify_fuseki_data(fuseki_query_endpoint=f"{fuseki_dataset_url}/sparql", qid=qid)
    return True  # 回傳成功狀態


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
    # 移除三個反引號
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
please provide a concise English answer. No extra commentary.
"""


def generate_answer(query_results, question):
    if not query_results or "results" not in query_results:
        return "No results or invalid query."
    # 簡單將前幾筆結果列成文字
    rows = []
    for r in query_results["results"]["bindings"]:
        row_repr = []
        for var_name, val_dict in r.items():
            row_repr.append(f"{var_name}={val_dict['value']}")
        rows.append(", ".join(row_repr))
    summarized = "\n".join(rows[:5])  # 最多顯示5筆

    # 再次丟給 LLM 產生自然語言
    prompt = answer_prompt.format(query_results=summarized, question=question)
    response = model.generate_content(prompt)
    return response.text.strip()


################################################################################
# MAIN DEMO
################################################################################

if __name__ == "__main__":
    # 設定 Fuseki 資料集和 Update URL
    FUSEKI_DATASET_URL = "http://localhost:4040/ds"  # 根據實際情況修改
    FUSEKI_QUERY_ENDPOINT = "http://localhost:4040/ds/sparql"
    FUSEKI_UPDATE_ENDPOINT = (
        "http://localhost:4040/ds/update"  # Fuseki 的 Update endpoint
    )

    # 使用者問題
    question = "When was Barack Obama born?"

    # 1. 從問題萃取 entities 和 properties
    node_queries = generate_node_queries(question)
    relation_queries = generate_relation_queries(question, node_queries)

    print("Node Queries:", node_queries)
    print("Relation Queries:", relation_queries)

    # 根據實體名稱取得 QID
    qids_to_fetch = []
    for entity_name in node_queries:
        qid = get_wikidata_qid(entity_name)
        if qid:
            qids_to_fetch.append(qid)
            print(f"Found QID for '{entity_name}': {qid}")
        else:
            print(f"Could not find QID for '{entity_name}'")

    # 2. 從 Wikidata 抓子圖
    if qids_to_fetch:
        triple_list = fetch_entity_triples(qids_to_fetch, limit=1000)
    else:
        print("No valid QIDs found, skipping subgraph fetching.")
        triple_list = []

    # 3. 建立本地 RDF Graph
    rdf_graph = create_rdf_graph_from_triples(triple_list)

    # **新增：在每次上傳前清理 Fuseki 資料庫**
    print("Clearing Fuseki data...")
    if clear_fuseki_data(FUSEKI_UPDATE_ENDPOINT):
        # 4. 上載到 Fuseki
        if qids_to_fetch:
            # 這裡的 QID 用於驗證，可以選擇一個代表性的 QID 或之後修改驗證邏輯
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

    # 5. 準備可用的 QIDs 和 PIDs
    available_qids = list(set([triple[0] for triple in triple_list]))
    available_pids = [
        f"http://www.wikidata.org/prop/direct/{pid}" for pid in relation_queries
    ]

    # 6. 描述子圖 (簡易描述)
    desc_lines = []
    for s, p, o in triple_list:
        desc_lines.append(f"{s} -> {p} -> {o}")
    subgraph_desc = "\n".join(desc_lines)

    # 7. LLM 產生只針對本地子圖的 SPARQL
    kg_query = generate_kg_query(
        question, subgraph_desc, available_qids, available_pids
    )
    print("[LLM generated SPARQL]\n", kg_query)

    # 8. 執行本地 SPARQL
    query_results = execute_local_sparql_query(kg_query, FUSEKI_QUERY_ENDPOINT)

    # 9. 最終回答
    final_answer = generate_answer(query_results, question)
    print("=== Final Answer ===")
    print(final_answer)
    print("=======================")