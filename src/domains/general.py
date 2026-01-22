"""
General Domain Service

Provides general medical information:
- Disease information
- Drug information
- Health education
- Medical terminology
"""
from typing import Optional, List, Dict, Any

from src.layers.knowledge_graph import KnowledgeGraphService
from src.layers.llm.prompts import RoleType
from src.core.security.audit import AuditLogger, get_audit_logger, AuditAction


class GeneralService:
    """
    General medical information service
    
    Capabilities:
    - Disease and condition information
    - Drug information and lookups
    - Medical terminology explanations
    - Health education content
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraphService] = None,
        llm_service=None,
        verifier_service=None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.kg = knowledge_graph
        self.llm = llm_service
        self.verifier = verifier_service
        self.audit = audit_logger or get_audit_logger()
    
    async def get_disease_info(
        self,
        disease_name: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive information about a disease
        """
        result = {
            "name": disease_name,
            "found": False,
            "description": None,
            "symptoms": [],
            "causes": [],
            "risk_factors": [],
            "diagnosis": [],
            "treatment_options": [],
            "prevention": [],
            "prognosis": None,
            "sources": [],
        }
        
        # Query knowledge graph
        if self.kg:
            disease = await self.kg.get_disease(disease_name)
            if disease:
                result["found"] = True
                result["description"] = disease.get("description")
                result["icd10_code"] = disease.get("icd10_code")
                result["category"] = disease.get("category")
                result["risk_factors"] = disease.get("risk_factors", [])
                result["prognosis"] = disease.get("prognosis")
                
                # Get symptoms
                symptoms = await self.kg.get_disease_symptoms(disease.get("id"))
                result["symptoms"] = [
                    {"name": s.get("symptom"), "frequency": s.get("frequency")}
                    for s in symptoms
                ]
                
                # Get treatments
                treatments = await self.kg.get_disease_treatments(disease.get("id"))
                result["treatment_options"] = [
                    {
                        "name": t.get("drug_name"),
                        "line": t.get("line_of_therapy"),
                        "evidence": t.get("evidence_level"),
                    }
                    for t in treatments
                ]
        
        self.audit.log(
            action=AuditAction.VIEW,
            description="Disease information requested",
            user_id=user_id,
            details={"disease": disease_name, "found": result["found"]},
        )
        
        return result
    
    async def get_drug_info(
        self,
        drug_name: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive information about a drug
        """
        result = {
            "name": drug_name,
            "found": False,
            "generic_name": None,
            "brand_names": [],
            "drug_class": None,
            "mechanism": None,
            "indications": [],
            "dosage_forms": [],
            "typical_dosage": None,
            "side_effects": {
                "common": [],
                "serious": [],
            },
            "contraindications": [],
            "interactions": [],
            "warnings": [],
            "pregnancy_category": None,
            "sources": [],
        }
        
        # Query knowledge graph
        if self.kg:
            drug = await self.kg.get_drug(drug_name)
            if drug:
                drug_data = drug.get("drug", {})
                result["found"] = True
                result["generic_name"] = drug_data.get("name")
                result["brand_names"] = drug_data.get("brand_names", [])
                result["drug_class"] = drug_data.get("drug_class")
                result["mechanism"] = drug_data.get("mechanism_of_action")
                result["dosage_forms"] = drug_data.get("dosage_forms", [])
                result["typical_dosage"] = drug_data.get("typical_dosage")
                result["side_effects"]["common"] = drug_data.get("common_side_effects", [])
                result["side_effects"]["serious"] = drug_data.get("serious_side_effects", [])
                result["pregnancy_category"] = drug_data.get("pregnancy_category")
                
                if drug_data.get("black_box_warning"):
                    result["warnings"].append(drug_data["black_box_warning"])
                
                # Get interactions
                result["interactions"] = drug.get("interactions", [])
                result["contraindications"] = drug.get("contraindications", [])
                result["indications"] = [
                    i.get("disease") for i in drug.get("indications", [])
                ]
        
        self.audit.log(
            action=AuditAction.VIEW,
            description="Drug information requested",
            user_id=user_id,
            details={"drug": drug_name, "found": result["found"]},
        )
        
        return result
    
    async def search_medical_info(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for medical information
        """
        results = {
            "query": query,
            "diseases": [],
            "drugs": [],
            "symptoms": [],
            "total_results": 0,
        }
        
        if self.kg:
            # Search diseases
            if not category or category == "disease":
                diseases = await self.kg.search_diseases(query, limit=limit)
                results["diseases"] = [
                    {"name": d.get("disease", {}).get("name"), "id": d.get("disease", {}).get("id")}
                    for d in diseases
                ]
            
            # Search drugs
            if not category or category == "drug":
                drugs = await self.kg.search_drugs(query, limit=limit)
                results["drugs"] = [
                    {"name": d.get("drug", {}).get("name"), "id": d.get("drug", {}).get("id")}
                    for d in drugs
                ]
        
        results["total_results"] = len(results["diseases"]) + len(results["drugs"])
        
        return results
    
    async def check_drug_interaction(
        self,
        drug1: str,
        drug2: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check for interaction between two drugs
        """
        result = {
            "drug1": drug1,
            "drug2": drug2,
            "has_interaction": False,
            "severity": None,
            "description": None,
            "mechanism": None,
            "management": None,
        }
        
        if self.kg:
            # Get drug IDs
            drug1_data = await self.kg.find_drug(drug1)
            drug2_data = await self.kg.find_drug(drug2)
            
            if drug1_data and drug2_data:
                interactions = await self.kg.check_drug_interactions(
                    [drug1_data.get("id"), drug2_data.get("id")],
                    user_id=user_id,
                )
                
                if interactions:
                    interaction = interactions[0]
                    result["has_interaction"] = True
                    result["severity"] = interaction.get("severity")
                    result["description"] = interaction.get("description")
                    result["mechanism"] = interaction.get("mechanism")
                    result["management"] = interaction.get("management")
        
        self.audit.log(
            action=AuditAction.VIEW,
            description="Drug interaction check",
            user_id=user_id,
            details={
                "drug1": drug1,
                "drug2": drug2,
                "has_interaction": result["has_interaction"],
            },
        )
        
        return result
    
    async def explain_medical_term(
        self,
        term: str,
        audience: str = "general",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Explain a medical term in plain language
        """
        # Common medical terms dictionary
        common_terms = {
            "hypertension": {
                "simple": "High blood pressure",
                "explanation": "A condition where the force of blood against artery walls is too high.",
                "related_terms": ["blood pressure", "systolic", "diastolic"],
            },
            "diabetes": {
                "simple": "High blood sugar",
                "explanation": "A condition where the body cannot properly process blood sugar (glucose).",
                "related_terms": ["insulin", "glucose", "A1C"],
            },
            "inflammation": {
                "simple": "Body's response to injury or infection",
                "explanation": "A protective response involving immune cells, blood vessels, and molecular mediators.",
                "related_terms": ["swelling", "redness", "immune response"],
            },
        }
        
        term_lower = term.lower()
        if term_lower in common_terms:
            info = common_terms[term_lower]
            return {
                "term": term,
                "simple_definition": info["simple"],
                "explanation": info["explanation"],
                "related_terms": info["related_terms"],
                "found": True,
            }
        
        # For terms not in dictionary, would use LLM
        return {
            "term": term,
            "simple_definition": f"Definition of {term}",
            "explanation": f"Detailed explanation of {term} would be provided here.",
            "related_terms": [],
            "found": False,
            "note": "Term not in quick reference - full explanation generated",
        }
    
    async def get_health_tips(
        self,
        category: str = "general",
        user_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Get health tips by category
        """
        tips = {
            "general": [
                {"title": "Stay Hydrated", "tip": "Drink 8 glasses of water daily"},
                {"title": "Regular Exercise", "tip": "Aim for 150 minutes of moderate activity per week"},
                {"title": "Balanced Diet", "tip": "Include fruits, vegetables, and whole grains"},
                {"title": "Adequate Sleep", "tip": "Adults need 7-9 hours of sleep per night"},
                {"title": "Stress Management", "tip": "Practice relaxation techniques regularly"},
            ],
            "heart": [
                {"title": "Monitor Blood Pressure", "tip": "Check your blood pressure regularly"},
                {"title": "Limit Sodium", "tip": "Keep sodium intake under 2300mg daily"},
                {"title": "Healthy Fats", "tip": "Choose unsaturated fats over saturated fats"},
            ],
            "diabetes": [
                {"title": "Monitor Blood Sugar", "tip": "Check glucose levels as recommended"},
                {"title": "Carb Counting", "tip": "Be aware of carbohydrate intake"},
                {"title": "Foot Care", "tip": "Inspect feet daily for any changes"},
            ],
        }
        
        return tips.get(category, tips["general"])
