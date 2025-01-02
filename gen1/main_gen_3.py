import os
import re
import google.generativeai as genai
from SPARQLWrapper import SPARQLWrapper, JSON
from langchain.prompts import PromptTemplate
from py2neo import Graph, Node, Relationship
import requests
import nltk

nltk.download("punkt_tab")

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

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
    You are a SPARQL query generator. You are given a subgraph represented by a list of Wikidata nodes and relationships, and a question. Generate a SPARQL query to answer the question based on the provided subgraph.

    Subgraph:
    {relevant_nodes}

    {relevant_relations}

    Question:
    {question}

    Please output only the SPARQL query, do not explain.
    Make sure to use SERVICE wikibase:label to get the labels in the SPARQL query.
    
    Note: Only use the information provided in the subgraph. Do not use any external knowledge.
    
    Examples:
    ---
    Subgraph:
    Entity: United States
      - QID: Q30, Label: United States of America
    
    Relation: P35
      - PID: P35, Label: head of state

    Question:
    Who is the president of the United States?

    SPARQL Query:
    SELECT ?answerLabel WHERE {
      wd:Q30 wdt:P35 ?answer .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ---
    Subgraph:
    Entity: Barack Obama
      - QID: Q76, Label: Barack Obama
    
    Relation: P569
      - PID: P569, Label: date of birth

    Question:
    When was Barack Obama born?

    SPARQL Query:
    SELECT ?answerLabel WHERE {
      wd:Q76 wdt:P569 ?answer .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ---
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
    # 預處理: 移除換行符、引號和多餘的空格
    cleaned_response = response.text.replace("\n", "").replace('"', "").strip()
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


# ... (其他函數保持不變) ...


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
        # 1. 選擇最相關的實體和關係
        entity_q_id = None
        relation_p_id = None

        # 使用 BM25 演算法選擇最佳實體
        corpus = [
            entity["label"].lower()
            for entity_name, entities in relevant_nodes.items()
            for entity in entities
        ]
        tokenized_corpus = [word_tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_question = word_tokenize(question.lower())
        doc_scores = bm25.get_scores(tokenized_question)

        best_match_q_id = None
        best_match_score = -1

        index = 0
        for entity_name, entities in relevant_nodes.items():
            for entity in entities:
                if doc_scores[index] > best_match_score:
                    best_match_score = doc_scores[index]
                    best_match_q_id = entity["q_id"]
                index += 1

        # 如果沒有找到匹配的實體，則返回 None
        if best_match_q_id is None:
            print(
                f"Error: Could not find a relevant entity for the question: {question}"
            )
            return None

        entity_q_id = best_match_q_id

        # 選擇最佳關係
        if relevant_relations:
            relation_p_id = list(relevant_relations.values())[0][0]["p_id"]

        # 2. 根據問題類型構建 SPARQL 查詢模板 (目前只處理 "Who is the ... of ...?" 類型的問題)
        if question.startswith("Who is the"):
            # Who is the ... of ...?
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
        elif question.startswith("When was") and "born" in question:
            # When was ... born?
            template = f"""
                SELECT ?answerLabel WHERE {{
                  wd:{entity_q_id} wdt:P569 ?date .
                  BIND(YEAR(?date) as ?answerLabel)
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                }}
                LIMIT 100
            """
        elif question.startswith("Where is"):
            # Where is ... located?
            template = (
                """
                SELECT ?answerLabel WHERE {
                  wd:"""
                + entity_q_id
                + """ wdt:P131 ?answer .
                  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                }
            """
            )
        else:
            # 其他問題類型，暫時返回 None
            print(f"Question type not supported yet: {question}")
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


def retrieve_nodes(node_queries, limit=5, depth=1):
    results = {}
    for query in node_queries:
        escaped_query = escape_sparql_string(query.strip())

        # 1. 使用 SPARQL 查詢 Wikidata
        sparql_query = f"""
            SELECT ?q_id ?label
            WHERE {{
              ?q_id rdfs:label "{escaped_query}"@en .
              ?q_id rdfs:label ?label .
              ?q_id wdt:P31 wd:Q6256 .
              FILTER(LANG(?label) = "en")
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT {limit}
        """
        sparql.setQuery(sparql_query)

        try:
            ret = sparql.queryAndConvert()
            entity_list = []
            for result in ret["results"]["bindings"]:
                q_id = result["q_id"]["value"].replace(
                    "http://www.wikidata.org/entity/", ""
                )
                label = result["label"]["value"]
                entity_list.append({"q_id": q_id, "label": label})

            # 2. 如果 SPARQL 查詢沒有找到結果，則使用 Wikidata 搜尋 API
            if not entity_list:
                print(
                    f"No results found with SPARQL for '{query}', using Wikidata Search API"
                )
                search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&language=en&format=json&limit={limit}"
                search_response = requests.get(search_url).json()

                for result in search_response.get("search", []):
                    q_id = result["id"]
                    label = result["label"]
                    entity_list.append({"q_id": q_id, "label": label})

            # 建立 Neo4j 節點
            for entity in entity_list:
                node = Node(
                    "Entity",
                    q_id=entity["q_id"],
                    label=entity["label"],
                    name=entity["label"],
                )
                graph.merge(node, "Entity", "q_id")

            results[query] = entity_list

            # 遞迴地檢索鄰居節點
            if depth > 0:
                for entity in entity_list:
                    retrieve_neighbors(entity["q_id"], limit, depth - 1)

        except Exception as e:
            print(f"Error retrieving nodes for query '{query}': {e}")
            results[query] = []

    return results


def retrieve_neighbors(q_id, limit=5, depth=0):
    """
    遞迴地檢索指定節點的鄰居節點，並將其儲存到 Neo4j 中。

    Args:
        q_id: 節點的 Q 編號。
        limit: 每個查詢的結果數量限制。
        depth: 遞迴深度。
    """
    try:
        sparql.setQuery(
            f"""
            SELECT ?neighbor ?neighborLabel
            WHERE {{
              wd:{q_id} ?p ?neighbor .
              ?neighbor rdfs:label ?neighborLabel .
              FILTER(LANG(?neighborLabel) = "en")
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT {limit}
        """
        )

        ret = sparql.queryAndConvert()
        for result in ret["results"]["bindings"]:
            neighbor_q_id = result["neighbor"]["value"].replace(
                "http://www.wikidata.org/entity/", ""
            )
            neighbor_label = result["neighborLabel"]["value"]

            # 建立 Neo4j 節點
            neighbor_node = Node(
                "Entity",
                q_id=neighbor_q_id,
                label=neighbor_label,
                name=neighbor_label,
            )
            graph.merge(neighbor_node, "Entity", "q_id")

            # 建立與鄰居節點的關係
            relation = Relationship(
                graph.nodes.match("Entity", q_id=q_id).first(),
                "RELATED_TO",
                neighbor_node,
            )
            graph.create(relation)

            # 遞迴地檢索鄰居節點的鄰居節點
            if depth > 0:
                retrieve_neighbors(neighbor_q_id, limit, depth - 1)

    except Exception as e:
        print(f"Error retrieving neighbors for node '{q_id}': {e}")


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
    node_queries = generate_node_queries("When was Barack Obama born?")
    print("Node Queries:", node_queries)
    retrieved_nodes = retrieve_nodes(node_queries)
    print("Retrieved Nodes:", retrieved_nodes)

    relation_queries = generate_relation_queries(
        "When was Barack Obama born?", node_queries
    )
    print("Relation Queries:", relation_queries)
    retrieved_relations = retrieve_relations(relation_queries, node_queries)
    print("Retrieved Relations:", retrieved_relations)

    # 使用 generate_kg_query 函數生成 SPARQL 查詢
    kg_query = generate_kg_query(
        retrieved_nodes,
        retrieved_relations,
        "When was Barack Obama born?",
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
