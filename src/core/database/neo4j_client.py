"""Neo4j graph database client for medical knowledge graph"""
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable

from src.core.config import settings


class Neo4jClient:
    """Async Neo4j client for knowledge graph operations"""
    
    def __init__(self):
        self._driver: Optional[AsyncDriver] = None
    
    async def connect(self) -> None:
        """Establish connection to Neo4j"""
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
        )
        # Verify connectivity
        await self._driver.verify_connectivity()
    
    async def close(self) -> None:
        """Close Neo4j connection"""
        if self._driver:
            await self._driver.close()
            self._driver = None
    
    @asynccontextmanager
    async def session(self, database: str = "neo4j"):
        """Get a Neo4j session"""
        if not self._driver:
            await self.connect()
        
        session = self._driver.session(database=database)
        try:
            yield session
        finally:
            await session.close()
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: str = "neo4j",
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        async with self.session(database) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records
    
    async def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: str = "neo4j",
    ) -> Dict[str, Any]:
        """Execute a write query and return summary"""
        async with self.session(database) as session:
            result = await session.run(query, parameters or {})
            summary = await result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set,
            }
    
    # Knowledge Graph specific methods
    async def find_disease(self, disease_name: str) -> Optional[Dict[str, Any]]:
        """Find a disease by name"""
        query = """
        MATCH (d:Disease)
        WHERE toLower(d.name) CONTAINS toLower($name)
        RETURN d {.*} as disease
        LIMIT 1
        """
        results = await self.execute_query(query, {"name": disease_name})
        return results[0]["disease"] if results else None
    
    async def find_drug(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Find a drug by name"""
        query = """
        MATCH (d:Drug)
        WHERE toLower(d.name) CONTAINS toLower($name)
           OR toLower(d.generic_name) CONTAINS toLower($name)
           OR ANY(brand IN d.brand_names WHERE toLower(brand) CONTAINS toLower($name))
        RETURN d {.*} as drug
        LIMIT 1
        """
        results = await self.execute_query(query, {"name": drug_name})
        return results[0]["drug"] if results else None
    
    async def get_drug_interactions(self, drug_id: str) -> List[Dict[str, Any]]:
        """Get all interactions for a drug"""
        query = """
        MATCH (d1:Drug {id: $drug_id})-[i:INTERACTS_WITH]->(d2:Drug)
        RETURN d2.name as drug_name, d2.id as drug_id,
               i.severity as severity, i.description as description,
               i.mechanism as mechanism
        """
        return await self.execute_query(query, {"drug_id": drug_id})
    
    async def get_disease_symptoms(self, disease_id: str) -> List[Dict[str, Any]]:
        """Get symptoms associated with a disease"""
        query = """
        MATCH (d:Disease {id: $disease_id})-[r:HAS_SYMPTOM]->(s:Symptom)
        RETURN s.name as symptom, s.id as symptom_id,
               r.frequency as frequency, r.severity as typical_severity
        ORDER BY r.frequency DESC
        """
        return await self.execute_query(query, {"disease_id": disease_id})
    
    async def get_treatment_options(self, disease_id: str) -> List[Dict[str, Any]]:
        """Get treatment options for a disease"""
        query = """
        MATCH (d:Disease {id: $disease_id})-[r:TREATED_BY]->(t:Drug)
        OPTIONAL MATCH (t)-[:HAS_GUIDELINE]->(g:Guideline)
        RETURN t.name as drug_name, t.id as drug_id,
               r.line_of_therapy as line_of_therapy,
               r.evidence_level as evidence_level,
               collect(g.name) as guidelines
        ORDER BY r.line_of_therapy
        """
        return await self.execute_query(query, {"disease_id": disease_id})
    
    async def find_contraindications(
        self,
        drug_id: str,
        patient_conditions: List[str],
    ) -> List[Dict[str, Any]]:
        """Find contraindications for a drug given patient conditions"""
        query = """
        MATCH (d:Drug {id: $drug_id})-[c:CONTRAINDICATED_FOR]->(cond:Condition)
        WHERE cond.name IN $conditions OR cond.id IN $conditions
        RETURN cond.name as condition, c.severity as severity,
               c.reason as reason, c.alternative_drugs as alternatives
        """
        return await self.execute_query(
            query,
            {"drug_id": drug_id, "conditions": patient_conditions}
        )
    
    async def get_clinical_guidelines(
        self,
        disease_id: Optional[str] = None,
        drug_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get clinical guidelines for disease or drug"""
        if disease_id:
            query = """
            MATCH (d:Disease {id: $id})-[:HAS_GUIDELINE]->(g:Guideline)
            RETURN g {.*} as guideline
            ORDER BY g.publication_date DESC
            """
            return await self.execute_query(query, {"id": disease_id})
        elif drug_id:
            query = """
            MATCH (d:Drug {id: $id})-[:HAS_GUIDELINE]->(g:Guideline)
            RETURN g {.*} as guideline
            ORDER BY g.publication_date DESC
            """
            return await self.execute_query(query, {"id": drug_id})
        return []
    
    async def search_by_symptoms(
        self,
        symptoms: List[str],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find diseases matching a set of symptoms"""
        query = """
        UNWIND $symptoms as symptom_name
        MATCH (s:Symptom)
        WHERE toLower(s.name) CONTAINS toLower(symptom_name)
        WITH collect(DISTINCT s) as matched_symptoms
        MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s)
        WHERE s IN matched_symptoms
        WITH d, count(DISTINCT s) as match_count, 
             collect({symptom: s.name, frequency: r.frequency}) as matched
        RETURN d.name as disease, d.id as disease_id,
               match_count, matched,
               d.severity as typical_severity,
               d.urgency as urgency_level
        ORDER BY match_count DESC
        LIMIT $limit
        """
        return await self.execute_query(
            query,
            {"symptoms": symptoms, "limit": limit}
        )


# Singleton instance
_neo4j_client: Optional[Neo4jClient] = None


async def get_neo4j_client() -> Neo4jClient:
    """Get or create Neo4j client singleton"""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
        await _neo4j_client.connect()
    return _neo4j_client


async def close_neo4j_client() -> None:
    """Close Neo4j client"""
    global _neo4j_client
    if _neo4j_client:
        await _neo4j_client.close()
        _neo4j_client = None
