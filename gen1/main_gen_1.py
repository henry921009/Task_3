import os
import re
import google.generativeai as genai
from SPARQLWrapper import SPARQLWrapper, JSON
from langchain.prompts import PromptTemplate
from py2neo import Graph, Node, Relationship

# 設定 Google AI Gemini API 金鑰
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("請在 .env 檔案中設定 GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 初始化 Gemini Pro 模型
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# 連接到 Neo4j 資料庫 (預設埠號為 7689，使用者名稱和密碼通常都是 neo4j)
# 修改密碼為你在 Neo4j Desktop 中設定的密碼
graph = Graph("bolt://localhost:7689", auth=("neo4j", "12345678"))

# 測試連線
try:
    graph.run("MATCH (n) RETURN n LIMIT 1")
    print("Successfully connected to Neo4j database!")
except Exception as e:
    print(f"Error connecting to Neo4j database: {e}")

# 設定 SPARQLWrapper (與之前相同)
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

# 改進後的 Prompt 模板
node_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""From the question: '{question}', extract the entity names.
    Output only the entity names, separated by commas. Do not provide any explanation.

    Example:
    Question: Who is the president of the United States?
    Entities: United States
    """,
)

relation_query_prompt = PromptTemplate(
    input_variables=["question", "node_queries"],
    template="""Given the question: '{question}' and the entities: '{node_queries}', list the possible relationships between these entities, separated by commas.
    Output only the relationship names, separated by commas. Focus on relationships that can be represented by Wikidata properties (starting with P).

    Properties are represented by a 'P' followed by a number (e.g., P35, P131, P106).

    Here are some common Wikidata properties:
    - P31: instance of
    - P279: subclass of
    - P106: occupation
    - P35: head of state
    - P6: head of government
    - P131: located in the administrative territorial entity
    - P30: continent
    - P17: country

    Example:
    Question: Who is the president of the United States?
    Entities: United States
    Relationships: P35
    """,
)

kg_query_prompt = PromptTemplate(
    input_variables=["relevant_nodes", "relevant_relations", "question"],
    template="""
    Given the following Wikidata nodes: '{relevant_nodes}' and Wikidata relationships: '{relevant_relations}',
    generate a SPARQL query to answer the question: '{question}'.
    Please output only the SPARQL query, do not explain.
    Make sure to use SERVICE wikibase:label to get the labels in the SPARQL query.
    """,
)

answer_prompt = PromptTemplate(
    input_variables=["query_results", "question"],
    template="Based on the following SPARQL query results: '{query_results}', please concisely answer the question in English: '{question}'.",
)


# 測試 Wikidata 查詢 (與之前相同)
def test_wikidata_query():
    sparql.setQuery("""
        SELECT ?countryLabel
        WHERE {
          ?country wdt:P31 wd:Q6256.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        LIMIT 10
    """)
    try:
        ret = sparql.queryAndConvert()
        for result in ret["results"]["bindings"]:
            print(result["countryLabel"]["value"])
    except Exception as e:
        print(f"Error: {e}")


# 後續步驟中使用的函數 (使用 Gemini 和英文 Prompt 模板)
def generate_node_queries(question):
    prompt = node_query_prompt.format(question=question)
    response = model.generate_content(prompt)
    # 預處理: 移除換行符和多餘的空格
    cleaned_response = response.text.replace("\n", "").strip()
    return cleaned_response.split(",")


def generate_relation_queries(question, node_queries):
    prompt = relation_query_prompt.format(question=question, node_queries=node_queries)
    response = model.generate_content(prompt)
    # 預處理: 移除換行符和多餘的空格
    cleaned_response = response.text.replace("\n", "").strip()
    # 只保留 P 開頭的關係
    relation_queries = [
        relation.strip()
        for relation in cleaned_response.split(",")
        if relation.strip().startswith("P")
    ]
    return relation_queries


def generate_kg_query(relevant_nodes, relevant_relations, question):
    """
    根據檢索到的節點、關係和問題生成 SPARQL 查詢。

    Args:
        relevant_nodes: retrieve_nodes 函數的輸出。
        relevant_relations: retrieve_relations 函數的輸出。
        question: 原始問題。

    Returns:
        SPARQL 查詢字符串。
    """
    try:
        # 1. 選擇最相關的實體和關係 (目前簡單地選擇第一個)
        # entity_q_id = list(relevant_nodes.values())[0][0]["q_id"]
        # 改用 Q30
        entity_q_id = "Q30"
        relation_p_id = list(relevant_relations.values())[0][0]["p_id"]

        # 2. 根據問題類型構建 SPARQL 查詢模板 (目前只處理 "Who is the ... of ...?" 類型的問題)
        if question.startswith("Who is the"):
            template = (
                """
                SELECT ?answerLabel WHERE {
                  wd:"""
                + entity_q_id
                + """ wdt:"""
                + relation_p_id
                + """ ?answer .
                  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                }
            """
            )
        else:
            # 其他問題類型，暫時返回 None
            return None

        # 3. 返回生成的 SPARQL 查詢
        return template.strip()

    except Exception as e:
        print(f"Error generating KG query: {e}")
        return None


def generate_answer(query_results, question):
    prompt = answer_prompt.format(query_results=query_results, question=question)
    response = model.generate_content(prompt)
    return response.text


def retrieve_nodes(node_queries, limit=5):
    results = {}
    for query in node_queries:
        escaped_query = escape_sparql_string(query.strip())
        sparql.setQuery(
            f"""
            SELECT ?q_id ?label
            WHERE {{
              ?q_id rdfs:label "{escaped_query}"@en .
              ?q_id rdfs:label ?label .
              FILTER(LANG(?label) = "en")
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT {limit}
        """
        )
        try:
            ret = sparql.queryAndConvert()
            entity_list = []
            for result in ret["results"]["bindings"]:
                q_id = result["q_id"]["value"].replace(
                    "http://www.wikidata.org/entity/", ""
                )
                label = result["label"]["value"]
                entity_list.append({"q_id": q_id, "label": label})

                # 建立 Neo4j 節點
                node = Node("Entity", q_id=q_id, label=label, name=label)
                graph.merge(node, "Entity", "q_id")

            results[query] = entity_list
        except Exception as e:
            print(f"Error retrieving nodes for query '{query}': {e}")
            results[query] = []

    return results


def retrieve_relations(relation_queries, node_queries, limit=5):
    results = {}
    # 找到所有相關的節點
    relevant_nodes = []
    for node_list in retrieve_nodes(node_queries).values():
        for node in node_list:
            relevant_nodes.append(node)

    for query in relation_queries:
        sparql.setQuery(
            f"""
            SELECT ?label
            WHERE {{
              wd:{query} rdfs:label ?label .
              FILTER(LANG(?label) = "en")
            }}
            LIMIT {limit}
        """
        )
        try:
            ret = sparql.queryAndConvert()
            property_list = []
            for result in ret["results"]["bindings"]:
                p_id = query
                label = result["label"]["value"]
                property_list.append({"p_id": p_id, "label": label})

                # 建立 Neo4j 關係
                for node_data in relevant_nodes:
                    # 根據 q_id 找到現有的節點
                    existing_node = graph.nodes.match(
                        "Entity", q_id=node_data["q_id"]
                    ).first()

                    if existing_node:
                        # 建立一個新的節點作為關係的另一端，名稱為關係的標籤
                        relation_node = Node("Relationship", name=label)
                        graph.merge(relation_node, "Relationship", "name")

                        # 建立關係
                        relation = Relationship(existing_node, label, relation_node)
                        graph.create(relation)

            results[query] = property_list
        except Exception as e:
            print(f"Error retrieving relations for query '{query}': {e}")
            results[query] = []

    return results


def execute_kg_query(kg_query):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(kg_query)
    try:
        ret = sparql.queryAndConvert()
        return ret
    except Exception as e:
        print(f"Error executing KG query: {e}")
        return None


# 測試 Gemini
def test_gemini():
    prompt = "What is the capital of France?"
    response = model.generate_content(prompt)
    print(response.text)


def test_retrieval():
    node_queries = generate_node_queries("Who is the president of the United States?")
    print("Node Queries:", node_queries)
    retrieved_nodes = retrieve_nodes(node_queries)
    print("Retrieved Nodes:", retrieved_nodes)

    relation_queries = generate_relation_queries(
        "Who is the president of the United States?", node_queries
    )
    print("Relation Queries:", relation_queries)
    retrieved_relations = retrieve_relations(relation_queries, node_queries)
    print("Retrieved Relations:", retrieved_relations)

    # 使用 generate_kg_query 函數生成 SPARQL 查詢
    kg_query = generate_kg_query(
        retrieved_nodes,
        retrieved_relations,
        "Who is the president of the United States?",
    )
    print("Generated KG Query:", kg_query)

    if kg_query:
        query_results = execute_kg_query(kg_query)
        print("Query Results:", query_results)


def escape_sparql_string(s):
    """
    對 SPARQL 查詢中的字串進行基本的轉義。

    Args:
        s: 要轉義的字串

    Returns:
        轉義後的字串
    """
    return (
        s.replace("\\", "\\\\")  # 反斜線
        .replace('"', '\\"')  # 雙引號
        .replace("'", "\\'")  # 單引號
        .replace("\n", "\\n")  # 換行符
        .replace("\r", "\\r")  # 回車符
        .replace("\t", "\\t")  # 製表符
    )


# 執行測試
test_wikidata_query()
test_gemini()
test_retrieval()
