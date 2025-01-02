import requests
import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import URIRef
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import logging

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

    def identify_basic_relation(self, question: str, entity: str) -> str:
        """識別關係，優先使用本地映射表，然後嘗試 LLM 提取，最後搜尋 Wikidata"""
        prompt = """
        Given:
        - Question: {question}
        - Main entity: {entity}

        Identify the main Wikidata property ID (PID) that represents the *start time* or *date of appointment* related to the question.
        Output format: Return only the Wikidata property ID (e.g., P580, P571), no explanation.

        Example:
        Question: "When did Tim Cook become the CEO of Apple?"
        Main entity: "Tim Cook"
        Output: P580
        """
        try:
            response = self.model.generate_content(
                prompt.format(question=question, entity=entity)
            )
            relation_phrase = response.text.strip().lower()

            # 1. 嘗試使用本地映射表
            if relation_phrase in self.ontology_mapping:
                return self.ontology_mapping[relation_phrase]

            # 2. 嘗試使用 LLM 提取 Wikidata 屬性 ID
            logger.info(
                f"No mapping found in local map for: {relation_phrase}. Trying LLM extraction."
            )
            wikidata_pid = self.extract_wikidata_pid_with_llm(question, entity)
            if wikidata_pid:
                return wikidata_pid

            # 3. 搜尋 Wikidata 屬性
            logger.info(
                f"LLM extraction failed. Searching Wikidata properties for: {relation_phrase}"
            )
            possible_pids = self.search_wikidata_properties(relation_phrase)
            if possible_pids:
                # 選擇最佳 PID 的策略（這裡可以根據需要進行調整）
                best_pid = self.select_best_pid(possible_pids, relation_phrase)
                logger.info(
                    f"Using Wikidata PID: {best_pid} for relation: {relation_phrase}"
                )
                return best_pid
            else:
                logger.error(
                    f"Could not find a suitable Wikidata PID for relation: {relation_phrase}"
                )
                return None

        except Exception as e:
            logger.error(f"Error in relation identification: {e}")
            return None

    def extract_wikidata_pid_with_llm(self, question: str, entity: str) -> str:
        """嘗試使用 LLM 直接從問題中提取 Wikidata 屬性 ID"""
        prompt = f"""
        Given:
        - Question: {question}
        - Main entity: {entity}

        Identify the main Wikidata property ID (PID) that represents the relationship being asked about.
        Output format: Return only the Wikidata property ID (e.g., P6, P39), no explanation.

        Example:
        Question: "Who is the president of the United States?"
        Main entity: "United States"
        Output: P35
        """
        try:
            response = self.model.generate_content(prompt)
            pid = response.text.strip()
            # 簡單驗證一下結果是否像 PID
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
            response = requests.get(self.kg_interface.WIKIDATA_API, params=params)
            response.raise_for_status()
            data = response.json()
            return [result["id"] for result in data.get("search", [])]
        except Exception as e:
            logger.error(f"Error searching Wikidata properties: {e}")
            return []

    def select_best_pid(self, possible_pids: list, relation_phrase: str) -> str:
        """選擇最佳的 Wikidata 屬性 ID (PID)"""
        # 策略 1: 選擇第一個結果
        # return possible_pids[0]

        # 策略 2: 根據標籤或描述的相似度選擇 (需要用到詞向量模型或其他相似度計算方法)
        # 這裡只是一個簡單的示例，你可以根據需要進行改進
        best_pid = None
        best_score = -1
        for pid in possible_pids:
            label = self.kg_interface.get_entity_label(pid)  # 獲取 PID 的標籤
            if label:
                # 簡單的相似度比較
                score = len(set(relation_phrase.split()) & set(label.lower().split()))
                if score > best_score:
                    best_score = score
                    best_pid = pid
        return (
            best_pid if best_pid else possible_pids[0]
        )  # 如果沒有找到更合適的，則返回第一個

    def extract_basic_entity(self, question: str) -> list:
        """最基本的實體提取"""
        prompt = """
        Analyze the question: '{question}'
        Extract the main entity that is being asked about.
        Consider:
        1. Look for proper nouns or named entities
        2. Identify what the question is specifically about
        3. Ignore qualifiers or descriptive words

        Output format: Return only the entity name, no explanation

        Examples:
        Q: "Who is the president of the United States?"
        A: United States

        Q: "What is the capital of France?"
        A: France

        Q: "When was Apple Inc. founded?"
        A: Apple Inc.
        """
        try:
            response = self.model.generate_content(prompt.format(question=question))
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return None


class KGInterface:
    def __init__(self):
        self.WIKIDATA_API = "https://www.wikidata.org/w/api.php"
        self.WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

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
            return data["entities"][qid]["labels"][lang]["value"]
        except Exception as e:
            logger.error(f"Error getting label: {e}")
            return None

    def get_basic_triple(self, qid: str, pid: str) -> list:
        """獲取基本三元組"""
        query = f"""
        SELECT ?obj WHERE {{
            wd:{qid} wdt:{pid} ?obj .
        }}
        """
        try:
            sparql = SPARQLWrapper(self.WIKIDATA_SPARQL)
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

    def add_basic_triples(self, qid: str, pid: str) -> bool:
        """添加基本三元組到子圖"""
        try:
            triples = self.kg_interface.get_basic_triple(qid, pid)
            for s, p, o in triples:
                self.graph.add(
                    (
                        URIRef(f"http://www.wikidata.org/entity/{s}"),
                        URIRef(f"http://www.wikidata.org/prop/direct/{p}"),
                        URIRef(o),
                    )
                )
            return True
        except Exception as e:
            logger.error(f"Error adding triples: {e}")
            return False

    def save_to_fuseki(self, fuseki_url: str) -> bool:
        """保存到Fuseki"""
        try:
            data = self.graph.serialize(format="turtle")
            headers = {"Content-Type": "text/turtle"}
            response = requests.post(f"{fuseki_url}/data", data=data, headers=headers)
            return response.status_code in [200, 201, 204]
        except Exception as e:
            logger.error(f"Error saving to Fuseki: {e}")
            return False


class QueryProcessor:
    def __init__(self, model):
        self.model = model
        # 添加標準前綴定義
        self.prefixes = """
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        """

    def generate_basic_sparql(
        self, question: str, main_entity: str, relation: str
    ) -> str:
        """生成基本SPARQL查詢"""
        prompt = """
        Generate a SPARQL query for:
        Question: {question}
        Main entity: {main_entity}
        Relation: {relation}
        
        Use these exact prefixes:
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        
        Output format:
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT ?answer WHERE {{
            {main_entity} {relation} ?answer .
        }}
        """
        try:
            response = self.model.generate_content(
                prompt.format(
                    question=question, main_entity=main_entity, relation=relation
                )
            )
            # 清理回應中的特殊標記
            query = response.text.strip()
            query = query.replace("```sparql", "").replace("```", "").strip()

            # 確保查詢包含前綴
            if not query.lower().startswith("prefix"):
                query = f"{self.prefixes}\n{query}"

            return query
        except Exception as e:
            logger.error(f"Error generating SPARQL: {e}")
            return None

    def process_results(self, results: dict) -> str:
        """處理查詢結果"""
        try:
            if not results or "results" not in results:
                return "No results found."

            answers = []
            for result in results["results"]["bindings"]:
                for var in result:
                    answers.append(result[var]["value"])

            return ", ".join(answers)
        except Exception as e:
            logger.error(f"Error processing results: {e}")
            return "Error processing results."

    def validate_sparql_query(self, query: str) -> bool:
        """驗證 SPARQL 查詢格式"""
        if not query:
            return False
        # 基本格式檢查
        required_elements = ["SELECT", "WHERE", "{", "}"]
        return all(element in query for element in required_elements)

    def execute_sparql_query(self, query: str, fuseki_endpoint: str) -> dict:
        """執行SPARQL查詢"""
        try:
            # 確保查詢包含必要的前綴
            if not query.lower().startswith("prefix"):
                query = f"{self.prefixes}\n{query}"

            logger.info(f"Executing query with prefixes:\n{query}")

            sparql = SPARQLWrapper(fuseki_endpoint)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            return sparql.query().convert()
        except Exception as e:
            logger.error(f"Error executing SPARQL: {e}")
            return None


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
        query_processor = QueryProcessor(model)
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        return

    # 3. 準備處理問題
    question = "When did Joe Biden born?"  # 你可以修改這個問題
    logger.info(f"Processing question: {question}")

    try:
        # 4. 清理既有數據
        logger.info("Clearing existing Fuseki data...")
        if clear_fuseki_data(Config.FUSEKI_UPDATE):
            logger.info("Fuseki data cleared successfully")
        else:
            logger.error("Failed to clear Fuseki data")
            return

        # 5. 分析問題
        logger.info("Extracting entity from question...")
        entity = question_analyzer.extract_basic_entity(question)
        logger.info(f"Identified entity: {entity}")
        if not entity:
            logger.error("Failed to identify entity")
            return

        logger.info("Identifying relation...")
        relation = question_analyzer.identify_basic_relation(question, entity)
        logger.info(f"Identified relation: {relation}")
        if not relation:
            logger.error("Failed to identify relation")
            return

        # 6. 獲取QID
        logger.info("Getting entity QID...")
        qid = kg_interface.get_entity_qid(entity)
        logger.info(f"Found QID: {qid}")
        if not qid:
            logger.error("Failed to get QID")
            return

        # 7. 驗證實體
        logger.info("Validating entity...")
        if not kg_interface.validate_entity_exists(qid):
            logger.error("Entity validation failed")
            return
        logger.info("Entity validated successfully")

        # 8. 構建子圖
        logger.info("Building subgraph...")
        if subgraph_builder.add_basic_triples(qid, relation):
            logger.info("Triples added to graph successfully")
        else:
            logger.error("Failed to add triples to graph")
            return

        logger.info("Saving to Fuseki...")
        if subgraph_builder.save_to_fuseki(Config.FUSEKI_DATASET):
            logger.info("Saved to Fuseki successfully")
        else:
            logger.error("Failed to save to Fuseki")
            return

        # 9. 生成查詢
        logger.info("Generating SPARQL query...")
        sparql = query_processor.generate_basic_sparql(
            question, f"wd:{qid}", f"wdt:{relation}"
        )
        if not sparql:
            logger.error("Failed to generate SPARQL query")
            return
        logger.info(f"Generated SPARQL query: {sparql}")

        # 10. 執行查詢
        logger.info("Executing SPARQL query...")
        results = query_processor.execute_sparql_query(sparql, Config.FUSEKI_QUERY)
        if not results:
            logger.error("Failed to execute SPARQL query")
            return
        logger.info("Query executed successfully")

        # 11. 處理結果
        logger.info("Processing results...")
        answer = query_processor.process_results(results)
        logger.info(f"Final answer: {answer}")

        # 12. 完成處理
        logger.info("=== KGQA Process Completed Successfully ===")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        return


if __name__ == "__main__":
    main()
