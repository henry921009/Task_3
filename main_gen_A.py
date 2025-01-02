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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
    """檢查所有必要服務是否正常運行"""
    health_status = {"gemini_api": False, "wikidata_api": False}

    try:
        # 測試 Gemini API
        response = model.generate_content("Test")
        health_status["gemini_api"] = True
    except Exception as e:
        logger.error(f"Gemini API error: {e}")

    try:
        # 測試 Wikidata API
        response = requests.get(Config.WIKIDATA_API)
        health_status["wikidata_api"] = response.status_code == 200
    except Exception as e:
        logger.error(f"Wikidata API error: {e}")

    return health_status

class QuestionAnalyzer:
    def __init__(
        self, model, kg_interface, ontology_mapping_path="ontology_mapping.json"
    ):
        self.model = model
        self.kg_interface = kg_interface
        self.ontology_mapping = self.load_ontology_mapping(ontology_mapping_path)

    def load_ontology_mapping(self, ontology_mapping_path: str) -> dict:
        """加載關係映射表"""
        try:
            with open(ontology_mapping_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading ontology mapping: {e}")
            return {}

    def extract_entities_and_relations(self, question: str) -> list:
        """
        提取問題中的實體和關係。
        返回一個列表，包含一系列 (實體, 關係) 的元組，按查詢順序排列。
        """
        prompt = f"""
        Analyze the question: "{question}"
        Identify all the entities and their relationships required to answer the question.
        For multi-hop queries, list the entities and relationships in the order they should be queried.
        Output format: Return a list of tuples in the format (entity, relationship). Do not include any explanation.

        Example:
        Q: "Who is the spouse of the author of 'The Old Man and the Sea'?"
        A: [("Ernest Hemingway", "spouse")]

        Q: "Who is the spouse of the director of 'Titanic'?"
        A: [("James Cameron", "spouse")]

        Q: "When did the president of the USA born?"
        A: [("United States of America", "president"), ("Joe Biden", "date of birth")]
        """
        try:
            response = self.model.generate_content(prompt)
            # 使用更安全的解析方法
            entities_relations = ast.literal_eval(response.text.strip())
            logger.info(f"Extracted entities and relations: {entities_relations}")
            return entities_relations
        except Exception as e:
            logger.error(f"Error extracting entities and relations: {e}")
            return []

    def map_relation_to_pid(self, relation_phrase: str) -> str:
        """將關係短語映射到PID"""
        # 先查本地映射表
        if relation_phrase in self.ontology_mapping:
            pid = self.ontology_mapping[relation_phrase]
            logger.info(
                f"Mapped relation phrase '{relation_phrase}' to PID '{pid}' via ontology mapping."
            )
            return pid

        # 如果本地映射表中沒有，使用LLM提取PID
        logger.info(
            f"No mapping found for relation phrase '{relation_phrase}'. Trying LLM extraction."
        )
        pid = self.extract_wikidata_pid_with_llm(relation_phrase)
        if pid:
            logger.info(f"Extracted PID '{pid}' via LLM.")
            return pid

        # 如果LLM提取失敗，搜尋Wikidata屬性
        logger.info(
            f"LLM extraction failed for '{relation_phrase}'. Searching Wikidata properties."
        )
        possible_pids = self.search_wikidata_properties(relation_phrase)
        if possible_pids:
            best_pid = self.select_best_pid(possible_pids, relation_phrase)
            logger.info(
                f"Using Wikidata PID: {best_pid} for relation: {relation_phrase}"
            )
            return best_pid

        logger.error(
            f"Could not find a suitable Wikidata PID for relation: {relation_phrase}"
        )
        return None

    def extract_wikidata_pid_with_llm(self, relation_phrase: str) -> str:
        """使用LLM從關係短語中提取PID"""
        prompt = f"""
        Given the relationship: "{relation_phrase}"
        Provide the EXACT corresponding Wikidata property ID (PID) that represents this relationship.

        Common examples:
        - "president" -> "P6" (head of government)
        - "spouse" -> "P26"
        - "date of birth" -> "P569"
        - "position held" -> "P39"
        
        Output only the PID, no explanation.
        For "{relation_phrase}", the PID is:
        """
        try:
            response = self.model.generate_content(prompt)
            pid = response.text.strip().upper()
            if pid.startswith("P") and pid[1:].isdigit():
                return pid
            else:
                logger.warning(f"LLM did not return a valid PID: {pid}")
                return None
        except Exception as e:
            logger.error(f"Error extracting Wikidata PID with LLM: {e}")
            return None

    def search_wikidata_properties(self, relation_phrase: str) -> list:
        """根據關係描述搜尋 Wikidata 屬性"""
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "type": "property",
            "search": relation_phrase,
        }
        try:
            response = requests.get(Config.WIKIDATA_API, params=params)
            response.raise_for_status()
            data = response.json()
            return [result["id"] for result in data.get("search", [])]
        except Exception as e:
            logger.error(f"Error searching Wikidata properties: {e}")
            return []

    def select_best_pid(self, possible_pids: list, relation_phrase: str) -> str:
        """選擇最佳的 Wikidata 屬性 ID (PID)"""
        # 這裡選擇第一個結果作為最佳PID
        return possible_pids[0] if possible_pids else None


class KGInterface:
    def __init__(self):
        self.WIKIDATA_API = "https://www.wikidata.org/w/api.php"

    def validate_entity_exists(self, qid: str) -> bool:
        """驗證實體是否存在"""
        params = {"action": "wbgetentities", "format": "json", "ids": qid}
        try:
            response = requests.get(self.WIKIDATA_API, params=params)
            data = response.json()
            return (
                "entities" in data
                and qid in data["entities"]
                and "missing" not in data["entities"][qid]
            )
        except Exception as e:
            logger.error(f"Error validating entity: {e}")
            return False

    def get_entity_qid(self, entity_name: str) -> str:
        """獲取實體QID"""
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

    def get_entity_label(self, qid: str, lang="en") -> str:
        """獲取實體的標籤"""
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": qid,
            "props": "labels",
            "languages": lang,
        }
        try:
            response = requests.get(self.WIKIDATA_API, params=params)
            data = response.json()
            if "entities" in data and qid in data["entities"]:
                return data["entities"][qid]["labels"][lang]["value"]
            return None
        except Exception as e:
            logger.error(f"Error getting label: {e}")
            return None

    def get_basic_triples(self, qid: str, pid: str) -> list:
        """獲取基本三元組"""
        query = f"""
        SELECT ?obj WHERE {{
            wd:{qid} wdt:{pid} ?obj .
        }}
        """
        try:
            sparql = SPARQLWrapper(Config.WIKIDATA_SPARQL)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            return [
                (qid, pid, item["obj"]["value"])
                for item in results["results"]["bindings"]
            ]
        except Exception as e:
            logger.error(f"Error getting triple: {e}")
            return []

class DirectQueryProcessor:
    def __init__(self, model, kg_interface, question_analyzer):
        self.model = model
        self.kg_interface = kg_interface
        self.question_analyzer = question_analyzer
        self.sparql = SPARQLWrapper(Config.WIKIDATA_SPARQL)
        self.sparql.setReturnFormat(JSON)

    def construct_sparql_query(self, entities_relations: List[Tuple[str, str]]) -> str:
        """構建 SPARQL 查詢"""
        select_vars = []
        where_clauses = []
        current_var = "a"
        
        for i, (entity, relation) in enumerate(entities_relations):
            qid = self.kg_interface.get_entity_qid(entity)
            pid = self.question_analyzer.map_relation_to_pid(relation)
            
            if qid is None or pid is None:
                raise ValueError(f"Could not resolve QID or PID for {entity} - {relation}")
            
            if i == 0:
                where_clauses.append(f"wd:{qid} wdt:{pid} ?{current_var} .")
                select_vars.append(f"?{current_var}")
                current_var = chr(ord(current_var) + 1)
            else:
                prev_var = chr(ord(current_var) - 1)
                where_clauses.append(f"?{prev_var} wdt:{pid} ?{current_var} .")
                select_vars.append(f"?{current_var}")
                current_var = chr(ord(current_var) + 1)

        query = f"""
        SELECT {' '.join(select_vars)}
        WHERE {{
            {' '.join(where_clauses)}
        }}
        """
        logger.info(f"Constructed SPARQL query: {query}")
        return query

    def execute_query(self, query: str) -> List[dict]:
        """執行 SPARQL 查詢"""
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            logger.info(f"Query returned {len(results.get('results', {}).get('bindings', []))} results")
            return results["results"]["bindings"]
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            raise

    def process_results(self, results: List[dict], question: str) -> str:
        """處理查詢結果"""
        if not results:
            return "No information found in the knowledge graph."

        # 獲取實體標籤
        processed_results = []
        for result in results:
            processed_result = {}
            for var, value in result.items():
                if value["type"] == "uri" and "entity" in value["value"]:
                    qid = value["value"].split("/")[-1]
                    label = self.kg_interface.get_entity_label(qid)
                    processed_result[var] = label if label else value["value"]
                else:
                    processed_result[var] = value["value"]
            processed_results.append(processed_result)

        # 使用 Gemini 生成自然語言回答
        context = json.dumps(processed_results, indent=2)
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

    def find_answer(self, question: str, entities_relations: List[Tuple[str, str]]) -> str:
        """查找答案的主要方法"""
        try:
            logger.info(f"Processing question: {question}")
            
            # 構建並執行查詢
            query = self.construct_sparql_query(entities_relations)
            results = self.execute_query(query)
            
            # 處理結果
            answer = self.process_results(results, question)
            return answer
            
        except Exception as e:
            logger.error(f"Error in direct SPARQL query: {e}")
            return f"Error processing query: {str(e)}"

def main():
    logger.info("=== Starting Direct SPARQL KGQA System ===")

    # 系統健康檢查
    health_status = check_system_health()
    if not all(health_status.values()):
        logger.error("System health check failed. Aborting...")
        return

    # 初始化組件
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        kg_interface = KGInterface()
        question_analyzer = QuestionAnalyzer(model, kg_interface, "ontology_mapping.json")
        query_processor = DirectQueryProcessor(model, kg_interface, question_analyzer)
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        return

    question = "When did the CEO of Apple born?"
    logger.info(f"Processing question: {question}")

    try:
        # 分析問題 - 提取實體和關係
        entities_relations = question_analyzer.extract_entities_and_relations(question)
        if not entities_relations:
            logger.error("Failed to extract entities and relations.")
            return

        # 使用直接 SPARQL 查詢
        answer = query_processor.find_answer(question, entities_relations)
        
        logger.info(f"Final answer: {answer}")
        logger.info("=== KGQA Process Completed Successfully ===")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        return

if __name__ == "__main__":
    main()