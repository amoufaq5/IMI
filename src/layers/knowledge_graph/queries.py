"""Medical Knowledge Graph Query Builder"""
from typing import Optional, List, Dict, Any
from enum import Enum


class QueryType(str, Enum):
    """Types of knowledge graph queries"""
    DISEASE_LOOKUP = "disease_lookup"
    DRUG_LOOKUP = "drug_lookup"
    SYMPTOM_SEARCH = "symptom_search"
    INTERACTION_CHECK = "interaction_check"
    CONTRAINDICATION_CHECK = "contraindication_check"
    TREATMENT_OPTIONS = "treatment_options"
    GUIDELINE_SEARCH = "guideline_search"
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"


class MedicalQueryBuilder:
    """Builder for constructing Cypher queries for medical knowledge graph"""
    
    @staticmethod
    def create_disease(disease: Dict[str, Any]) -> str:
        """Generate Cypher query to create a disease node"""
        return """
        MERGE (d:Disease {id: $id})
        SET d.name = $name,
            d.icd10_code = $icd10_code,
            d.icd11_code = $icd11_code,
            d.snomed_ct_code = $snomed_ct_code,
            d.description = $description,
            d.category = $category,
            d.subcategory = $subcategory,
            d.typical_severity = $typical_severity,
            d.urgency_level = $urgency_level,
            d.prevalence = $prevalence,
            d.risk_factors = $risk_factors,
            d.complications = $complications,
            d.prognosis = $prognosis,
            d.sources = $sources,
            d.updated_at = datetime()
        RETURN d
        """
    
    @staticmethod
    def create_drug(drug: Dict[str, Any]) -> str:
        """Generate Cypher query to create a drug node"""
        return """
        MERGE (d:Drug {id: $id})
        SET d.name = $name,
            d.brand_names = $brand_names,
            d.drug_class = $drug_class,
            d.therapeutic_class = $therapeutic_class,
            d.pharmacological_class = $pharmacological_class,
            d.mechanism_of_action = $mechanism_of_action,
            d.routes = $routes,
            d.typical_dosage = $typical_dosage,
            d.max_daily_dose = $max_daily_dose,
            d.dosage_forms = $dosage_forms,
            d.black_box_warning = $black_box_warning,
            d.pregnancy_category = $pregnancy_category,
            d.common_side_effects = $common_side_effects,
            d.serious_side_effects = $serious_side_effects,
            d.half_life = $half_life,
            d.rxnorm_cui = $rxnorm_cui,
            d.atc_code = $atc_code,
            d.sources = $sources,
            d.updated_at = datetime()
        RETURN d
        """
    
    @staticmethod
    def create_symptom(symptom: Dict[str, Any]) -> str:
        """Generate Cypher query to create a symptom node"""
        return """
        MERGE (s:Symptom {id: $id})
        SET s.name = $name,
            s.description = $description,
            s.body_system = $body_system,
            s.snomed_ct_code = $snomed_ct_code,
            s.is_red_flag = $is_red_flag,
            s.red_flag_conditions = $red_flag_conditions,
            s.assessment_questions = $assessment_questions,
            s.updated_at = datetime()
        RETURN s
        """
    
    @staticmethod
    def create_drug_interaction() -> str:
        """Generate Cypher query to create drug interaction relationship"""
        return """
        MATCH (d1:Drug {id: $drug1_id})
        MATCH (d2:Drug {id: $drug2_id})
        MERGE (d1)-[i:INTERACTS_WITH]->(d2)
        SET i.severity = $severity,
            i.description = $description,
            i.mechanism = $mechanism,
            i.clinical_significance = $clinical_significance,
            i.management = $management,
            i.evidence_level = $evidence_level,
            i.sources = $sources,
            i.updated_at = datetime()
        RETURN d1, i, d2
        """
    
    @staticmethod
    def create_disease_symptom_relation() -> str:
        """Generate Cypher query to link disease to symptom"""
        return """
        MATCH (d:Disease {id: $disease_id})
        MATCH (s:Symptom {id: $symptom_id})
        MERGE (d)-[r:HAS_SYMPTOM]->(s)
        SET r.frequency = $frequency,
            r.typical_severity = $typical_severity,
            r.onset_pattern = $onset_pattern,
            r.is_pathognomonic = $is_pathognomonic,
            r.updated_at = datetime()
        RETURN d, r, s
        """
    
    @staticmethod
    def create_treatment_relation() -> str:
        """Generate Cypher query to link disease to treatment"""
        return """
        MATCH (d:Disease {id: $disease_id})
        MATCH (t:Drug {id: $drug_id})
        MERGE (d)-[r:TREATED_BY]->(t)
        SET r.line_of_therapy = $line_of_therapy,
            r.evidence_level = $evidence_level,
            r.dosing_guidance = $dosing_guidance,
            r.duration = $duration,
            r.monitoring = $monitoring,
            r.guideline_source = $guideline_source,
            r.updated_at = datetime()
        RETURN d, r, t
        """
    
    @staticmethod
    def create_contraindication() -> str:
        """Generate Cypher query to create contraindication relationship"""
        return """
        MATCH (d:Drug {id: $drug_id})
        MATCH (c:Condition {id: $condition_id})
        MERGE (d)-[r:CONTRAINDICATED_FOR]->(c)
        SET r.severity = $severity,
            r.reason = $reason,
            r.is_absolute = $is_absolute,
            r.alternatives = $alternatives,
            r.monitoring_required = $monitoring_required,
            r.sources = $sources,
            r.updated_at = datetime()
        RETURN d, r, c
        """
    
    @staticmethod
    def search_diseases_by_symptoms() -> str:
        """Query to find diseases matching symptoms with scoring"""
        return """
        UNWIND $symptoms as symptom_name
        MATCH (s:Symptom)
        WHERE toLower(s.name) CONTAINS toLower(symptom_name)
           OR toLower(s.description) CONTAINS toLower(symptom_name)
        WITH collect(DISTINCT s) as matched_symptoms
        
        MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s)
        WHERE s IN matched_symptoms
        
        WITH d, 
             count(DISTINCT s) as symptom_match_count,
             collect({
                 symptom: s.name,
                 frequency: r.frequency,
                 is_pathognomonic: r.is_pathognomonic
             }) as matched_symptoms,
             sum(CASE 
                 WHEN r.is_pathognomonic THEN 10
                 WHEN r.frequency = 'common' THEN 3
                 WHEN r.frequency = 'occasional' THEN 2
                 ELSE 1
             END) as relevance_score
        
        RETURN d.id as disease_id,
               d.name as disease_name,
               d.description as description,
               d.typical_severity as severity,
               d.urgency_level as urgency,
               symptom_match_count,
               matched_symptoms,
               relevance_score
        ORDER BY relevance_score DESC, symptom_match_count DESC
        LIMIT $limit
        """
    
    @staticmethod
    def check_drug_interactions() -> str:
        """Query to check interactions between multiple drugs"""
        return """
        UNWIND $drug_ids as drug_id
        MATCH (d:Drug {id: drug_id})
        WITH collect(d) as drugs
        
        UNWIND drugs as d1
        UNWIND drugs as d2
        WITH d1, d2
        WHERE d1.id < d2.id
        
        OPTIONAL MATCH (d1)-[i:INTERACTS_WITH]-(d2)
        WHERE i IS NOT NULL
        
        RETURN d1.name as drug1,
               d1.id as drug1_id,
               d2.name as drug2,
               d2.id as drug2_id,
               i.severity as severity,
               i.description as description,
               i.mechanism as mechanism,
               i.management as management
        ORDER BY 
            CASE i.severity
                WHEN 'life_threatening' THEN 1
                WHEN 'critical' THEN 2
                WHEN 'severe' THEN 3
                WHEN 'moderate' THEN 4
                WHEN 'mild' THEN 5
                ELSE 6
            END
        """
    
    @staticmethod
    def check_contraindications() -> str:
        """Query to check drug contraindications against patient conditions"""
        return """
        MATCH (d:Drug {id: $drug_id})-[c:CONTRAINDICATED_FOR]->(cond:Condition)
        WHERE cond.name IN $patient_conditions 
           OR cond.id IN $patient_conditions
        
        RETURN cond.name as condition,
               cond.id as condition_id,
               c.severity as severity,
               c.reason as reason,
               c.is_absolute as is_absolute,
               c.alternatives as alternatives,
               c.monitoring_required as monitoring
        ORDER BY 
            CASE c.severity
                WHEN 'life_threatening' THEN 1
                WHEN 'critical' THEN 2
                WHEN 'severe' THEN 3
                WHEN 'moderate' THEN 4
                ELSE 5
            END
        """
    
    @staticmethod
    def get_treatment_pathway() -> str:
        """Query to get complete treatment pathway for a disease"""
        return """
        MATCH (d:Disease {id: $disease_id})
        
        OPTIONAL MATCH (d)-[t:TREATED_BY]->(drug:Drug)
        OPTIONAL MATCH (d)-[:HAS_GUIDELINE]->(g:Guideline)
        OPTIONAL MATCH (d)-[:REQUIRES_TEST]->(lab:LabTest)
        OPTIONAL MATCH (d)-[:MAY_REQUIRE]->(proc:Procedure)
        
        WITH d, 
             collect(DISTINCT {
                 drug: drug.name,
                 drug_id: drug.id,
                 line: t.line_of_therapy,
                 evidence: t.evidence_level,
                 dosing: t.dosing_guidance
             }) as treatments,
             collect(DISTINCT {
                 name: g.name,
                 org: g.organization,
                 summary: g.summary
             }) as guidelines,
             collect(DISTINCT {
                 name: lab.name,
                 loinc: lab.loinc_code
             }) as labs,
             collect(DISTINCT {
                 name: proc.name,
                 cpt: proc.cpt_code
             }) as procedures
        
        RETURN d.name as disease,
               d.description as description,
               d.urgency_level as urgency,
               treatments,
               guidelines,
               labs,
               procedures
        """
    
    @staticmethod
    def get_drug_complete_profile() -> str:
        """Query to get complete drug profile with all relationships"""
        return """
        MATCH (d:Drug {id: $drug_id})
        
        OPTIONAL MATCH (d)-[i:INTERACTS_WITH]-(other:Drug)
        OPTIONAL MATCH (d)-[c:CONTRAINDICATED_FOR]->(cond:Condition)
        OPTIONAL MATCH (disease:Disease)-[t:TREATED_BY]->(d)
        OPTIONAL MATCH (d)-[:HAS_GUIDELINE]->(g:Guideline)
        
        WITH d,
             collect(DISTINCT {
                 drug: other.name,
                 severity: i.severity,
                 description: i.description
             }) as interactions,
             collect(DISTINCT {
                 condition: cond.name,
                 severity: c.severity,
                 reason: c.reason
             }) as contraindications,
             collect(DISTINCT {
                 disease: disease.name,
                 line: t.line_of_therapy
             }) as indications,
             collect(DISTINCT g.name) as guidelines
        
        RETURN d {.*} as drug,
               interactions,
               contraindications,
               indications,
               guidelines
        """
    
    @staticmethod
    def find_alternative_drugs() -> str:
        """Query to find alternative drugs for a condition"""
        return """
        MATCH (original:Drug {id: $drug_id})
        MATCH (disease:Disease)-[:TREATED_BY]->(original)
        MATCH (disease)-[t:TREATED_BY]->(alt:Drug)
        WHERE alt.id <> original.id
        
        // Check that alternative doesn't have contraindications
        OPTIONAL MATCH (alt)-[c:CONTRAINDICATED_FOR]->(cond:Condition)
        WHERE cond.name IN $patient_conditions
        
        WITH alt, t, collect(cond.name) as contraindicated_conditions
        WHERE size(contraindicated_conditions) = 0
        
        RETURN alt.name as drug_name,
               alt.id as drug_id,
               alt.drug_class as drug_class,
               t.line_of_therapy as line_of_therapy,
               t.evidence_level as evidence_level,
               alt.mechanism_of_action as mechanism
        ORDER BY t.line_of_therapy
        """
    
    @staticmethod
    def semantic_search_setup() -> str:
        """Query to create vector index for semantic search"""
        return """
        CREATE VECTOR INDEX medical_embeddings IF NOT EXISTS
        FOR (n:MedicalEntity)
        ON (n.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 768,
            `vector.similarity_function`: 'cosine'
        }}
        """
    
    @staticmethod
    def semantic_search() -> str:
        """Query for semantic similarity search"""
        return """
        CALL db.index.vector.queryNodes('medical_embeddings', $top_k, $query_embedding)
        YIELD node, score
        RETURN node.id as id,
               node.name as name,
               labels(node)[0] as entity_type,
               node.description as description,
               score
        ORDER BY score DESC
        """
