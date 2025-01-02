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
    FUSEKI_DATASET = "http://localhost:3030/ds"
    FUSEKI_QUERY = f"{FUSEKI_DATASET}/sparql"
    FUSEKI_UPDATE = f"{FUSEKI_DATASET}/update"
    WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


def check_system_health():
    """檢查所有必要服務是否正常運行"""
    health_status = {"gemini_api": False, "fuseki_server": False, "wikidata_api": False}

    try:
        # 測試 Gemini API
        response = model.generate_content("Test")
        health_status["gemini_api"] = True
    except Exception as e:
        logger.error(f"Gemini API error: {e}")

    try:
        # 測試 Fuseki
        response = requests.get(Config.FUSEKI_DATASET)
        health_status["fuseki_server"] = response.status_code == 200
    except Exception as e:
        logger.error(f"Fuseki server error: {e}")

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
        Provide the corresponding Wikidata property ID (PID).
        Output format: Return only the PID (e.g., P26), no explanation.

        Example:
        Relationship: "spouse"
        Output: "P26"

        Relationship: "position held"
        Output: "P39"
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

class SubgraphBuilder:
    def __init__(self, kg_interface: KGInterface):
        self.kg_interface = kg_interface
        self.graph = rdflib.Graph()

    def bfs_subgraph(self, start_qid: str, relations_pids: list, max_depth: int = 2):
        """
        使用BFS從起始實體出發，根據給定的PID抓取子圖。
        :param start_qid: 起始實體的QID
        :param relations_pids: 要追蹤的PID列表
        :param max_depth: BFS的最大深度
        """
        visited = set()
        queue = deque([(start_qid, 0)])

        while queue:
            current_qid, depth = queue.popleft()
            if depth > max_depth:
                continue
            if current_qid in visited:
                continue
            visited.add(current_qid)

            for pid in relations_pids:
                triples = self.kg_interface.get_basic_triples(current_qid, pid)
                for s, p, o in triples:
                    self.graph.add(
                        (
                            URIRef(f"http://www.wikidata.org/entity/{s}"),
                            URIRef(f"http://www.wikidata.org/prop/direct/{p}"),
                            URIRef(o),
                        )
                    )
                    # 如果o是一個實體且未訪問，加入隊列
                    if o.startswith("http://www.wikidata.org/entity/"):
                        o_qid = o.split("/")[-1]
                        if o_qid not in visited:
                            queue.append((o_qid, depth + 1))

class QueryProcessor:
    def __init__(self, model, subgraph: rdflib.Graph, kg_interface: KGInterface):
        self.model = model
        self.subgraph = subgraph
        self.kg_interface = kg_interface

    def find_answer(self, target_relation_pid: str) -> list:
        """
        在子圖中查找目標關係的對象。
        :param target_relation_pid: 目標關係的PID
        :return: 目標關係對象的QID列表或值列表
        """
        answers = set()
        predicate = URIRef(f"http://www.wikidata.org/prop/direct/{target_relation_pid}")
        for s, p, o in self.subgraph.triples((None, predicate, None)):
            if isinstance(o, URIRef) and o.startswith(
                "http://www.wikidata.org/entity/"
            ):
                answer_qid = o.split("/")[-1]
                answers.add(answer_qid)
            else:
                answers.add(o.toPython())
        return list(answers)

    def get_labels(self, qids: list) -> list:
        """
        獲取QID對應的標籤。
        :param qids: QID列表
        :return: 標籤列表
        """
        labels = []
        for qid in qids:
            if qid.startswith("Q") and qid[1:].isdigit():
                label = self.kg_interface.get_entity_label(qid)
                if label:
                    labels.append(label)
            else:
                labels.append(qid)  # 如果不是QID，直接使用值
        return labels
    
def clear_fuseki_data(fuseki_update_url: str):
    """清空Fuseki資料集"""
    query = "CLEAR DEFAULT"
    try:
        response = requests.post(fuseki_update_url, data={"update": query})
        return response.status_code in [200, 201, 204]
    except Exception as e:
        logger.error(f"Error clearing Fuseki: {e}")
        return False
    
def main():
    logger.info("=== Starting KGQA System ===")

    # 1. 系統健康檢查
    logger.info("Performing system health check...")
    health_status = check_system_health()
    for service, status in health_status.items():
        logger.info(f"{service}: {'OK' if status else 'FAILED'}")
    if not all(health_status.values()):
        logger.error("System health check failed. Aborting...")
        return

    # 2. 初始化組件
    logger.info("Initializing components...")
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        kg_interface = KGInterface()
        question_analyzer = QuestionAnalyzer(
            model, kg_interface, "ontology_mapping.json"
        )
        subgraph_builder = SubgraphBuilder(kg_interface)
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        return

    # 3. 準備處理問題
    question = "When did the president of the USA born?"  # 你可以修改這個問題
    logger.info(f"Processing question: {question}")

    try:
        # 4. 清理既有數據（可選）
        logger.info("Clearing existing Fuseki data...")
        if clear_fuseki_data(Config.FUSEKI_UPDATE):
            logger.info("Fuseki data cleared successfully")
        else:
            logger.error("Failed to clear Fuseki data")
            return

        # 5. 分析問題 - 提取實體和關係
        logger.info("Extracting entities and relations from question...")
        entities_relations = question_analyzer.extract_entities_and_relations(question)
        if not entities_relations:
            logger.error("Failed to extract entities and relations.")
            return

        # 6. 對每個實體和關係進行PID映射
        relations_pids = []
        for entity, relation in entities_relations:
            pid = question_analyzer.map_relation_to_pid(relation)
            if pid:
                relations_pids.append(pid)
            else:
                logger.error(f"Failed to map relation '{relation}' to PID.")
                return

        # 7. 獲取所有實體的QID
        logger.info("Getting QIDs for all entities...")
        qids = []
        for entity, _ in entities_relations:
            qid = kg_interface.get_entity_qid(entity)
            if qid:
                qids.append(qid)
                logger.info(f"Found QID for entity '{entity}': {qid}")
            else:
                logger.error(f"Failed to get QID for entity '{entity}'.")
                return

        # 8. 驗證所有實體
        logger.info("Validating all entities...")
        for qid in qids:
            if not kg_interface.validate_entity_exists(qid):
                logger.error(f"Entity validation failed for QID: {qid}")
                return
        logger.info("All entities validated successfully.")

        # 9. 構建子圖（多跳查詢）
        logger.info("Building subgraph using BFS...")
        for qid, pid in zip(qids, relations_pids):
            subgraph_builder.bfs_subgraph(qid, [pid], max_depth=2)
        logger.info("Subgraph built successfully.")

        # 10. 查找答案
        logger.info("Processing query on subgraph...")
        query_processor = QueryProcessor(model, subgraph_builder.graph, kg_interface)

        # 根據多跳查詢的順序，最後一個關係是最終目標
        target_relation_pid = relations_pids[-1]
        answer_qids = query_processor.find_answer(target_relation_pid)
        logger.info(f"Found answer QIDs or values: {answer_qids}")

        if not answer_qids:
            logger.info("No results found.")
            answer = "No results found."
        else:
            # 獲取實體標籤（如果是QID）
            answer_labels = query_processor.get_labels(answer_qids)
            answer = ", ".join(answer_labels) if answer_labels else "No results found."

        logger.info(f"Final answer: {answer}")

        # 11. 完成處理
        logger.info("=== KGQA Process Completed Successfully ===")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        return
    
if __name__ == "__main__":
    main()