import requests
import rdflib
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import logging
from rdflib import URIRef
from collections import deque
from SPARQLWrapper import SPARQLWrapper, JSON
import ast
from typing import List, Tuple
import time
import csv

# 日誌設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 環境設定
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("請先設定 GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")


class Config:
    WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


def check_system_health():
    health_status = {"gemini_api": False, "wikidata_api": False}

    try:
        response = model.generate_content("Test")
        health_status["gemini_api"] = True
    except Exception as e:
        logger.error(f"Gemini API error: {e}")

    try:
        response = requests.get(Config.WIKIDATA_API)
        health_status["wikidata_api"] = response.status_code == 200
    except Exception as e:
        logger.error(f"Wikidata API error: {e}")

    return health_status


class KGInterface:
    def __init__(self):
        self.WIKIDATA_API = Config.WIKIDATA_API

    def get_entity_qid(self, entity_name: str) -> str:
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": entity_name,
            "type": "item",
        }
        try:
            response = requests.get(self.WIKIDATA_API, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("search"):
                return data["search"][0]["id"]
            return None
        except Exception as e:
            logger.error(f"Error getting QID: {e}")
            return None


class QuestionAnalyzer:
    def __init__(self, model):
        self.model = model

    def extract_entities_and_relations(self, question: str) -> list:
        prompt = f"""
        Analyze the question: "{question}"
        Identify all the entities and their relationships required to answer the question.
        For multi-hop queries, list the entities and relationships in the order they should be queried.
        Output format: Return a list of tuples in the format (entity, relationship). Do not include any explanation.

        Example:
        Q: "When did the CEO of Apple born?"
        A: [("Apple", "CEO"), ("Tim Cook", "date of birth")]

        Q: "When did the president of the United States born?"
        A: [("United States of America", "president"), ("Joe Biden", "date of birth")]
        """
        try:
            response = self.model.generate_content(prompt)
            entities_relations = ast.literal_eval(response.text.strip())
            logger.info(f"Extracted entities and relations: {entities_relations}")
            return entities_relations
        except Exception as e:
            logger.error(f"Error extracting entities and relations: {e}")
            return []


class RAGQueryProcessor:
    def __init__(self, model, kg_interface, question_analyzer):
        self.model = model
        self.kg_interface = kg_interface
        self.question_analyzer = question_analyzer
        self.sparql = SPARQLWrapper(Config.WIKIDATA_SPARQL)
        self.sparql.setReturnFormat(JSON)

    def construct_sparql_query(self, qid: str, relation_pid: str) -> str:
        query = f"""
        SELECT ?entity ?date
        WHERE {{
            wd:{qid} wdt:{relation_pid} ?entity .
            ?entity wdt:P569 ?date .
        }}
        """
        logger.info(f"Constructed SPARQL query: {query}")
        return query

    def execute_query(self, query: str) -> List[dict]:
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            return []

    def process_results(self, results: List[dict], question: str) -> str:
        if not results:
            return "No information found in the knowledge graph."

        processed_results = []
        for result in results:
            entity = result["entity"]["value"].split("/")[-1]
            date = result["date"]["value"]
            processed_results.append(f"{entity} - {date}")

        context = "\n".join(processed_results)
        prompt = f"""
        Based on these query results:
        {context}
        
        Please provide a natural language answer to the question: {question}
        Answer should be clear and concise, using the information from the results.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer from query results."


def process_question(question, entity_name, relation_pid):
    start_time = time.perf_counter()

    kg_interface = KGInterface()
    question_analyzer = QuestionAnalyzer(model)
    query_processor = RAGQueryProcessor(model, kg_interface, question_analyzer)

    # 問題分析
    entities_relations = question_analyzer.extract_entities_and_relations(question)
    qid = kg_interface.get_entity_qid(entity_name)

    # 查詢
    query = query_processor.construct_sparql_query(qid, relation_pid)
    results = query_processor.execute_query(query)

    # 計算時間與錯誤率
    total_time = time.perf_counter() - start_time
    answer = results[0]["date"]["value"] if results else "No information found."

    # 輸出結果
    logger.info(f"Results for '{question}': {results}")
    logger.info(f"Execution Time: {total_time:.4f}s")

    return total_time, answer


def main():
    logger.info("=== Starting RAG SPARQL KGQA System ===")
    health_status = check_system_health()
    if not all(health_status.values()):
        logger.error("System health check failed. Aborting...")
        return

    questions = [
        ("When did the CEO of Apple born?", "Apple", "P169"),
        (
            "When did the president of the United States born?",
            "United States of America",
            "P6",
        ),
    ]

    results = []

    for question, entity_name, relation_pid in questions:
        avg_time, answer = process_question(question, entity_name, relation_pid)
        results.append(["RAG", question, avg_time, answer])

    # 儲存到 CSV
    with open("wikidata_rag_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "Question", "Avg Time (s)", "Answer"])
        writer.writerows(results)

    logger.info("=== Process Completed Successfully ===")


if __name__ == "__main__":
    main()
