"""Knowledge Graph Service - Layer 1 Truth Layer"""
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.core.database.neo4j_client import Neo4jClient, get_neo4j_client
from src.core.database.redis_client import RedisClient, get_redis_client
from src.core.security.audit import AuditLogger, get_audit_logger, AuditAction

from .schema import (
    Disease, Drug, Symptom, Guideline, Condition,
    DrugInteraction, Contraindication, DiseaseSymptomRelation,
    TreatmentRelation, Severity, EvidenceLevel
)
from .queries import MedicalQueryBuilder


class KnowledgeGraphService:
    """
    Layer 1: Knowledge Graph Service
    
    The Truth Layer - provides verified medical knowledge from:
    - Clinical guidelines
    - Drug labels
    - Regulatory texts
    - Medical ontologies (ICD-10, SNOMED-CT, RxNorm)
    """
    
    CACHE_TTL = 3600  # 1 hour cache
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        redis_client: Optional[RedisClient] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.neo4j = neo4j_client
        self.redis = redis_client
        self.audit = audit_logger or get_audit_logger()
        self.query_builder = MedicalQueryBuilder()
    
    # Disease Operations
    async def get_disease(self, disease_id: str) -> Optional[Dict[str, Any]]:
        """Get disease by ID with caching"""
        cache_key = f"disease:{disease_id}"
        
        # Check cache
        if self.redis:
            cached = await self.redis.cache_get("kg", cache_key)
            if cached:
                return cached
        
        # Query graph
        query = """
        MATCH (d:Disease {id: $id})
        RETURN d {.*} as disease
        """
        results = await self.neo4j.execute_query(query, {"id": disease_id})
        
        if results:
            disease = results[0]["disease"]
            if self.redis:
                await self.redis.cache_set("kg", cache_key, disease, self.CACHE_TTL)
            return disease
        return None
    
    async def search_diseases(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search diseases by name or description"""
        cypher = """
        MATCH (d:Disease)
        WHERE toLower(d.name) CONTAINS toLower($query)
           OR toLower(d.description) CONTAINS toLower($query)
           OR d.icd10_code = $query
        RETURN d {.*} as disease
        ORDER BY 
            CASE WHEN toLower(d.name) STARTS WITH toLower($query) THEN 0 ELSE 1 END,
            d.name
        LIMIT $limit
        """
        return await self.neo4j.execute_query(cypher, {"query": query, "limit": limit})
    
    async def get_disease_symptoms(self, disease_id: str) -> List[Dict[str, Any]]:
        """Get all symptoms associated with a disease"""
        return await self.neo4j.get_disease_symptoms(disease_id)
    
    async def get_disease_treatments(self, disease_id: str) -> List[Dict[str, Any]]:
        """Get treatment options for a disease"""
        return await self.neo4j.get_treatment_options(disease_id)
    
    # Drug Operations
    async def get_drug(self, drug_id: str) -> Optional[Dict[str, Any]]:
        """Get drug by ID with full profile"""
        cache_key = f"drug:{drug_id}"
        
        if self.redis:
            cached = await self.redis.cache_get("kg", cache_key)
            if cached:
                return cached
        
        query = MedicalQueryBuilder.get_drug_complete_profile()
        results = await self.neo4j.execute_query(query, {"drug_id": drug_id})
        
        if results:
            drug_profile = results[0]
            if self.redis:
                await self.redis.cache_set("kg", cache_key, drug_profile, self.CACHE_TTL)
            return drug_profile
        return None
    
    async def search_drugs(
        self,
        query: str,
        drug_class: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search drugs by name, brand name, or class"""
        cypher = """
        MATCH (d:Drug)
        WHERE toLower(d.name) CONTAINS toLower($query)
           OR ANY(brand IN d.brand_names WHERE toLower(brand) CONTAINS toLower($query))
           OR d.rxnorm_cui = $query
        """
        if drug_class:
            cypher += " AND d.drug_class = $drug_class"
        
        cypher += """
        RETURN d {.*} as drug
        ORDER BY 
            CASE WHEN toLower(d.name) STARTS WITH toLower($query) THEN 0 ELSE 1 END,
            d.name
        LIMIT $limit
        """
        params = {"query": query, "limit": limit}
        if drug_class:
            params["drug_class"] = drug_class
        
        return await self.neo4j.execute_query(cypher, params)
    
    async def check_drug_interactions(
        self,
        drug_ids: List[str],
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Check for interactions between multiple drugs"""
        query = MedicalQueryBuilder.check_drug_interactions()
        interactions = await self.neo4j.execute_query(query, {"drug_ids": drug_ids})
        
        # Filter out null interactions
        interactions = [i for i in interactions if i.get("severity")]
        
        # Log the check
        self.audit.log(
            action=AuditAction.KNOWLEDGE_GRAPH_QUERY,
            description="Drug interaction check",
            user_id=user_id,
            details={
                "drug_ids": drug_ids,
                "interactions_found": len(interactions),
            },
        )
        
        return interactions
    
    async def check_contraindications(
        self,
        drug_id: str,
        patient_conditions: List[str],
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Check drug contraindications against patient conditions"""
        query = MedicalQueryBuilder.check_contraindications()
        contraindications = await self.neo4j.execute_query(
            query,
            {"drug_id": drug_id, "patient_conditions": patient_conditions}
        )
        
        self.audit.log(
            action=AuditAction.KNOWLEDGE_GRAPH_QUERY,
            description="Contraindication check",
            user_id=user_id,
            details={
                "drug_id": drug_id,
                "conditions_checked": len(patient_conditions),
                "contraindications_found": len(contraindications),
            },
        )
        
        return contraindications
    
    async def find_alternative_drugs(
        self,
        drug_id: str,
        patient_conditions: List[str],
    ) -> List[Dict[str, Any]]:
        """Find alternative drugs avoiding patient's contraindications"""
        query = MedicalQueryBuilder.find_alternative_drugs()
        return await self.neo4j.execute_query(
            query,
            {"drug_id": drug_id, "patient_conditions": patient_conditions}
        )
    
    # Symptom-based Diagnosis
    async def differential_diagnosis(
        self,
        symptoms: List[str],
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate differential diagnosis based on symptoms
        Returns ranked list of possible diseases
        """
        query = MedicalQueryBuilder.search_diseases_by_symptoms()
        results = await self.neo4j.execute_query(
            query,
            {"symptoms": symptoms, "limit": limit}
        )
        
        self.audit.log(
            action=AuditAction.KNOWLEDGE_GRAPH_QUERY,
            description="Differential diagnosis query",
            user_id=user_id,
            details={
                "symptoms": symptoms,
                "results_count": len(results),
            },
        )
        
        return results
    
    # Guideline Operations
    async def get_clinical_guidelines(
        self,
        disease_id: Optional[str] = None,
        drug_id: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get clinical guidelines for disease or drug"""
        if disease_id:
            query = """
            MATCH (d:Disease {id: $id})-[:HAS_GUIDELINE]->(g:Guideline)
            """
        elif drug_id:
            query = """
            MATCH (d:Drug {id: $id})-[:HAS_GUIDELINE]->(g:Guideline)
            """
        else:
            query = """
            MATCH (g:Guideline)
            """
        
        if organization:
            query += " WHERE g.organization = $org"
        
        query += """
        RETURN g {.*} as guideline
        ORDER BY g.publication_date DESC
        """
        
        params = {}
        if disease_id:
            params["id"] = disease_id
        elif drug_id:
            params["id"] = drug_id
        if organization:
            params["org"] = organization
        
        return await self.neo4j.execute_query(query, params)
    
    # Treatment Pathway
    async def get_treatment_pathway(
        self,
        disease_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get complete treatment pathway for a disease"""
        query = MedicalQueryBuilder.get_treatment_pathway()
        results = await self.neo4j.execute_query(query, {"disease_id": disease_id})
        
        if results:
            pathway = results[0]
            
            self.audit.log(
                action=AuditAction.KNOWLEDGE_GRAPH_QUERY,
                description="Treatment pathway query",
                user_id=user_id,
                details={"disease_id": disease_id},
            )
            
            return pathway
        return {}
    
    # Data Ingestion
    async def create_disease(self, disease: Disease) -> Dict[str, Any]:
        """Create or update a disease node"""
        query = MedicalQueryBuilder.create_disease(disease.model_dump())
        return await self.neo4j.execute_write(query, disease.model_dump())
    
    async def create_drug(self, drug: Drug) -> Dict[str, Any]:
        """Create or update a drug node"""
        query = MedicalQueryBuilder.create_drug(drug.model_dump())
        return await self.neo4j.execute_write(query, drug.model_dump())
    
    async def create_symptom(self, symptom: Symptom) -> Dict[str, Any]:
        """Create or update a symptom node"""
        query = MedicalQueryBuilder.create_symptom(symptom.model_dump())
        return await self.neo4j.execute_write(query, symptom.model_dump())
    
    async def create_drug_interaction(
        self,
        interaction: DrugInteraction,
    ) -> Dict[str, Any]:
        """Create drug interaction relationship"""
        query = MedicalQueryBuilder.create_drug_interaction()
        return await self.neo4j.execute_write(query, interaction.model_dump())
    
    async def create_disease_symptom_relation(
        self,
        relation: DiseaseSymptomRelation,
    ) -> Dict[str, Any]:
        """Create disease-symptom relationship"""
        query = MedicalQueryBuilder.create_disease_symptom_relation()
        return await self.neo4j.execute_write(query, relation.model_dump())
    
    async def create_treatment_relation(
        self,
        relation: TreatmentRelation,
    ) -> Dict[str, Any]:
        """Create disease-treatment relationship"""
        query = MedicalQueryBuilder.create_treatment_relation()
        return await self.neo4j.execute_write(query, relation.model_dump())
    
    # Graph Statistics
    async def get_graph_stats(self) -> Dict[str, int]:
        """Get knowledge graph statistics"""
        query = """
        MATCH (n)
        RETURN labels(n)[0] as label, count(n) as count
        """
        results = await self.neo4j.execute_query(query)
        return {r["label"]: r["count"] for r in results}
    
    # Cache Management
    async def invalidate_cache(self, entity_type: str, entity_id: str) -> None:
        """Invalidate cache for an entity"""
        if self.redis:
            await self.redis.cache_invalidate("kg", f"{entity_type}:{entity_id}")


# Singleton
_kg_service: Optional[KnowledgeGraphService] = None


async def get_knowledge_graph_service() -> KnowledgeGraphService:
    """Get or create knowledge graph service singleton"""
    global _kg_service
    if _kg_service is None:
        neo4j = await get_neo4j_client()
        redis = await get_redis_client()
        _kg_service = KnowledgeGraphService(neo4j, redis)
    return _kg_service
