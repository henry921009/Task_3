import os
import re
import google.generativeai as genai
from SPARQLWrapper import SPARQLWrapper, JSON
from py2neo import Graph, Node, Relationship
import requests
import nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

nltk.download("punkt")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("請在 .env 檔案中設定 GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

graph = Graph("bolt://localhost:7689", auth=("neo4j", "12345678"))

node_query_prompt = """
From the question: '{question}', extract the entity names (one or more).
Output only the entity names, separated by commas. Do not provide any explanation.
"""

relation_query_prompt = """
Given the question: '{question}' and the entities: '{node_queries}', 
list the possible relationships (Wikidata property IDs like Pxx) that might be relevant to answering this question.
Output only the property IDs, separated by commas, with no explanation.
"""

kg_query_prompt = """
You are a SPARQL (or Cypher) query generator. Below is a subgraph of Wikidata (entities and relations) 
that has been extracted into a local database. You can only use the QIDs / PIDs that appear in this subgraph. 
The user question is also given.

Please generate ONE valid query (SPARQL or Cypher) that can be executed on this local subgraph 
to answer the question. Use the following guidelines:
1. Only rely on these QIDs and PIDs. 
2. If the needed QID/PID is not in the list, it means the subgraph doesn't have it. 
3. Return only the query text, with no extra explanation.

Subgraph Entities:
{relevant_nodes}

Subgraph Relations:
{relevant_relations}

Question:
{question}

Your output: The query alone, no explanations.
"""

answer_prompt = """
Given the query results: '{query_results}',
please return only the direct information extracted.
Do not provide any personal commentary or interpretation.
"""

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)


# ----------------------------------------------------------------------------
# 以下各種輔助函式(略)：retrieve_nodes, retrieve_neighbors, retrieve_relations, ...
# 你原程式的內容基本保持不變，重點是 generate_kg_query 改掉
# ----------------------------------------------------------------------------


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


def escape_sparql_string(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


# 下面省略 retrieve_nodes, retrieve_neighbors, retrieve_relations 等
# ... (你原程式一模一樣即可)

def retrieve_nodes(node_queries, limit=5) -> dict:
    """
    針對每個 entity 名稱，用 Wikidata 先找 QID。
    然後將它們存到 Neo4j。
    回傳結構: { "Barack Obama": [{"q_id":"Q76","label":"Barack Obama"}, ...], ... }
    """
    results = {}
    for query in node_queries:
        query_escaped = escape_sparql_string(query)
        sparql_query = f"""
            SELECT ?q_id ?q_idLabel
            WHERE {{
              ?q_id rdfs:label "{query_escaped}"@en .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }} LIMIT {limit}
        """
        sparql.setQuery(sparql_query)
        try:
            ret = sparql.queryAndConvert()
            entities = []
            for r in ret["results"]["bindings"]:
                qid_iri = r["q_id"][
                    "value"
                ]  # e.g. "http://www.wikidata.org/entity/Q76"
                qid = qid_iri.split("/")[-1]
                label = r["q_idLabel"]["value"]
                entities.append({"q_id": qid, "label": label})

                # 建立/合併到本地 Neo4j
                node = Node("Entity", q_id=qid, label=label)
                graph.merge(node, "Entity", "q_id")

            # 如果沒找到，就嘗試用 Wikidata Search API
            if not entities:
                search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&language=en&format=json&limit={limit}"
                sr = requests.get(search_url).json()
                for item in sr.get("search", []):
                    qid = item["id"]
                    label = item["label"]
                    entities.append({"q_id": qid, "label": label})

                    node = Node("Entity", q_id=qid, label=label)
                    graph.merge(node, "Entity", "q_id")

            results[query] = entities

        except Exception as e:
            print(f"[Error] retrieve_nodes - {e}")
            results[query] = []

    return results


def retrieve_neighbors(q_id, limit=5, depth=1):
    """
    如果想深度擴展子圖，可用這種函式。從 Wikidata 抓該 QID 的所有鄰居 (主語或賓語)。
    depth=1 表示只做一層。
    """
    if depth <= 0:
        return

    try:
        sparql_query = f"""
        SELECT ?property ?propertyLabel ?value ?valueLabel 
        WHERE {{
          wd:{q_id} ?p ?value .
          ?property wikibase:directClaim ?p .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT {limit}
        """
        sparql.setQuery(sparql_query)
        ret = sparql.queryAndConvert()
        for r in ret["results"]["bindings"]:
            pid_iri = r["property"][
                "value"
            ]  # e.g. "http://www.wikidata.org/entity/P31"
            pid = pid_iri.split("/")[-1]
            p_label = r["propertyLabel"]["value"]
            val_iri = r["value"]["value"]
            val_qid = val_iri.split("/")[-1] if "/entity/" in val_iri else None
            val_label = r.get("valueLabel", {}).get("value", val_qid)

            # 建到 Neo4j
            node_self = graph.nodes.match("Entity", q_id=q_id).first()
            if val_qid and node_self:
                node_val = Node("Entity", q_id=val_qid, label=val_label)
                graph.merge(node_val, "Entity", "q_id")
                rel = Relationship(node_self, p_label, node_val)
                graph.merge(rel, (type(node_self), type(node_val)), "q_id")

        # 如果要更深層，也可遞迴
        # for 新抓到的 val_qid in ...
        #   retrieve_neighbors(val_qid, limit, depth-1)

    except Exception as e:
        print(f"[Error] retrieve_neighbors - {e}")


def retrieve_relations(relation_queries, node_queries):
    # 先確定這些 nodes 都已經抓到
    relevant_nodes_dict = retrieve_nodes(node_queries)
    # 這裡可再對每個 node 執行 retrieve_neighbors(...) 拓展子圖

    # 再抓每個 PID 的 label，存到 Neo4j
    for pid in relation_queries:
        try:
            sparql_query = f"""
            SELECT ?pLabel WHERE {{
              wd:{pid} rdfs:label ?pLabel .
              FILTER(LANG(?pLabel)="en")
            }} LIMIT 1
            """
            sparql.setQuery(sparql_query)
            ret = sparql.queryAndConvert()
            for r in ret["results"]["bindings"]:
                label = r["pLabel"]["value"]
                # 如果想在 Neo4j 建 "Relation" 節點或怎樣，都可以：
                # relation_node = Node("Relation", p_id=pid, label=label)
                # graph.merge(relation_node, "Relation", "p_id")
        except Exception as e:
            print(f"[Error] retrieve_relations - {e}")


def get_subgraph_description(node_queries, depth=1):
    lines_entities = []
    lines_relations = []
    # 先從 retrieve_nodes(...) 取回該 query 的 QID, label
    for ent_name, ent_list in retrieve_nodes(node_queries).items():
        for e in ent_list:
            lines_entities.append(
                f"{ent_name} => QID: {e['q_id']}, Label: {e['label']}"
            )
            node_in_neo = graph.nodes.match("Entity", q_id=e["q_id"]).first()
            if node_in_neo:
                for rel in graph.relationships.match((node_in_neo,), r_type=None):
                    p_label = type(rel).__name__
                    target_qid = rel.end_node["q_id"]
                    target_label = rel.end_node["label"]
                    lines_relations.append(
                        f"{e['q_id']} -[{p_label}]-> {target_qid}({target_label})"
                    )

    return "\n".join(lines_entities), "\n".join(lines_relations)


###############################################
# ！！！在這裡：改寫 generate_kg_query ！！！
###############################################
def generate_kg_query(question, node_queries):
    # 先取得子圖描述
    subgraph_entities, subgraph_relations = get_subgraph_description(node_queries)

    prompt = kg_query_prompt.format(
        relevant_nodes=subgraph_entities,
        relevant_relations=subgraph_relations,
        question=question,
    )
    print("=== KG Query Prompt ===")
    print(prompt)
    print("=======================")

    try:
        # 呼叫 LLM 產生查詢
        response = model.generate_content(prompt)
        # 先看 LLM 原始輸出
        raw_query = response.text
        print("\n[LLM 原始輸出] ↓↓↓\n", raw_query, "\n^^^^^^^^^^^^^^^^\n")

        # 1) 用正則移除 ```sparql 或 ```python 等 code fence
        cleaned = re.sub(r"```(\w+)?", "", raw_query)
        # 2) 再把多餘的 ```
        cleaned = cleaned.replace("```", "")
        # 3) 去掉 'sparql' 字串
        cleaned = cleaned.replace("sparql", "")
        # 4) 最後 strip
        cleaned = cleaned.strip()

        print("[最終 Query] ↓↓↓\n", cleaned, "\n^^^^^^^^^^^^^^^^\n")
        return cleaned

    except Exception as e:
        print(f"[Error] generate_kg_query - {e}")
        return None


def execute_kg_query(sparql_query: str):
    # 針對 Wikidata 執行 SPARQL
    # 送給 wikidata endpoint
    local_sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    local_sparql.setReturnFormat(JSON)
    local_sparql.setQuery(sparql_query)

    try:
        ret = local_sparql.queryAndConvert()
        return ret
    except Exception as e:
        print(f"[Error] execute_kg_query - {e}")
        return None


def generate_answer(query_results, question):
    if not query_results or "results" not in query_results:
        return "No results or invalid query."

    row_texts = []
    for row in query_results["results"]["bindings"]:
        row_texts.append(", ".join(f"{k}={v['value']}" for k, v in row.items()))
    summarized_results = "\n".join(row_texts[:10])

    prompt = answer_prompt.format(query_results=summarized_results, question=question)
    response = model.generate_content(prompt)
    return response.text.strip()


if __name__ == "__main__":
    question = "Who is the spouse of Michelle Obama?"

    # 1. 從問題萃取 entities
    node_queries = generate_node_queries(question)
    print("Node Queries:", node_queries)

    # 2. 從問題萃取 properties
    rel_queries = generate_relation_queries(question, node_queries)
    print("Relation Queries:", rel_queries)

    # 3. 把 nodes & neighbors & relations 抓到本地 Neo4j，形成子圖
    #    (retrieve_nodes 已做了基本抓取)
    #    若需要擴展 depth:
    for nq in node_queries:
        # 先拿到 QIDs
        q_dict = retrieve_nodes([nq])
        for ent in q_dict.get(nq, []):
            retrieve_neighbors(ent["q_id"], limit=10, depth=1)
    # 4. 產生查詢（已經內建去除三個反引號）
    kg_query = generate_kg_query(question, node_queries)

    # 5. 執行查詢
    query_results = execute_kg_query(kg_query)
    # 6. LLM 最終回答
    final_answer = generate_answer(query_results, question)
    print("=== Final Answer ===")
    print(final_answer)
    print("=======================")
