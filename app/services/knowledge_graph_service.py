# app/services/knowledge_graph_service.py
from neo4j import GraphDatabase
from typing import Dict, Any, List

class KnowledgeGraphService:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def get_disease_info(self, disease_name: str, language: str = 'en') -> Dict[str, Any]:
        """根据病害名称和语言，从图数据库中查询详细信息"""
        with self._driver.session() as session:
            result = session.read_transaction(self._find_disease_details, disease_name, language)
            return result

    @staticmethod
    def _find_disease_details(tx, disease_name, lang):
        # Cypher 查询语言: 查找病害节点及其所有相关信息
        query = """
        MATCH (d:Disease {name_en: $disease_name})
        OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
        OPTIONAL MATCH (d)<-[:TREATS]-(t:Treatment)
        OPTIONAL MATCH (d)-[:CAUSED_BY]->(p:Pathogen)
        WITH d, p,
             collect(DISTINCT s) AS symptoms,
             collect(DISTINCT t) AS treatments
        RETURN
            d['summary_' + $lang] AS summary,
            p['name_' + $lang] AS pathogen,
            [symptom IN symptoms | symptom['desc_' + $lang]] AS symptom_descriptions,
            [treatment IN treatments | treatment['desc_' + $lang]] AS treatment_descriptions
        """
        result = tx.run(query, disease_name=disease_name, lang=lang)
        record = result.single()
        if record:
            return record.data()
        return None

# --- 全局实例初始化 ---
# 这些信息应该来自环境变量，为方便起见我们暂时硬编码
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password" # 替换成你的密码

kg_service = KnowledgeGraphService(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)