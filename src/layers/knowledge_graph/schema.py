"""Medical Knowledge Graph Schema - Node and Relationship definitions"""
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Severity levels"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    LIFE_THREATENING = "life_threatening"


class EvidenceLevel(str, Enum):
    """Evidence levels for medical recommendations"""
    LEVEL_1A = "1a"  # Systematic review of RCTs
    LEVEL_1B = "1b"  # Individual RCT
    LEVEL_2A = "2a"  # Systematic review of cohort studies
    LEVEL_2B = "2b"  # Individual cohort study
    LEVEL_3A = "3a"  # Systematic review of case-control studies
    LEVEL_3B = "3b"  # Individual case-control study
    LEVEL_4 = "4"    # Case series
    LEVEL_5 = "5"    # Expert opinion


class DrugClass(str, Enum):
    """Drug classification"""
    OTC = "otc"
    PRESCRIPTION = "prescription"
    CONTROLLED = "controlled"
    INVESTIGATIONAL = "investigational"


class RouteOfAdministration(str, Enum):
    """Drug administration routes"""
    ORAL = "oral"
    INTRAVENOUS = "intravenous"
    INTRAMUSCULAR = "intramuscular"
    SUBCUTANEOUS = "subcutaneous"
    TOPICAL = "topical"
    INHALATION = "inhalation"
    SUBLINGUAL = "sublingual"
    RECTAL = "rectal"
    TRANSDERMAL = "transdermal"
    OPHTHALMIC = "ophthalmic"
    OTIC = "otic"
    NASAL = "nasal"


# Node Models
class Disease(BaseModel):
    """Disease node in knowledge graph"""
    id: str
    name: str
    icd10_code: Optional[str] = None
    icd11_code: Optional[str] = None
    snomed_ct_code: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    typical_severity: Optional[Severity] = None
    urgency_level: Optional[str] = None  # emergency, urgent, routine
    prevalence: Optional[str] = None
    risk_factors: List[str] = Field(default_factory=list)
    complications: List[str] = Field(default_factory=list)
    prognosis: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    sources: List[str] = Field(default_factory=list)


class Drug(BaseModel):
    """Drug node in knowledge graph"""
    id: str
    name: str  # Generic name
    brand_names: List[str] = Field(default_factory=list)
    drug_class: DrugClass = DrugClass.PRESCRIPTION
    therapeutic_class: Optional[str] = None
    pharmacological_class: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    
    # Dosing
    routes: List[RouteOfAdministration] = Field(default_factory=list)
    typical_dosage: Optional[str] = None
    max_daily_dose: Optional[str] = None
    dosage_forms: List[str] = Field(default_factory=list)
    
    # Safety
    black_box_warning: Optional[str] = None
    pregnancy_category: Optional[str] = None
    lactation_risk: Optional[str] = None
    
    # Regulatory
    fda_approval_date: Optional[date] = None
    ndc_codes: List[str] = Field(default_factory=list)
    rxnorm_cui: Optional[str] = None
    atc_code: Optional[str] = None
    
    # Side effects
    common_side_effects: List[str] = Field(default_factory=list)
    serious_side_effects: List[str] = Field(default_factory=list)
    
    # Pharmacokinetics
    half_life: Optional[str] = None
    onset_of_action: Optional[str] = None
    duration_of_action: Optional[str] = None
    metabolism: Optional[str] = None
    excretion: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    sources: List[str] = Field(default_factory=list)


class Symptom(BaseModel):
    """Symptom node in knowledge graph"""
    id: str
    name: str
    description: Optional[str] = None
    body_system: Optional[str] = None
    snomed_ct_code: Optional[str] = None
    
    # Red flags
    is_red_flag: bool = False
    red_flag_conditions: List[str] = Field(default_factory=list)
    
    # Assessment
    assessment_questions: List[str] = Field(default_factory=list)
    severity_indicators: Dict[str, str] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Guideline(BaseModel):
    """Clinical guideline node"""
    id: str
    name: str
    organization: str  # e.g., AHA, ACC, NICE
    version: Optional[str] = None
    publication_date: Optional[date] = None
    last_reviewed: Optional[date] = None
    
    summary: Optional[str] = None
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_level: Optional[EvidenceLevel] = None
    
    # Scope
    target_population: Optional[str] = None
    conditions_covered: List[str] = Field(default_factory=list)
    
    # Source
    url: Optional[str] = None
    doi: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Condition(BaseModel):
    """Medical condition node (for contraindications)"""
    id: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None  # pregnancy, renal_impairment, etc.
    snomed_ct_code: Optional[str] = None


class LabTest(BaseModel):
    """Laboratory test node"""
    id: str
    name: str
    loinc_code: Optional[str] = None
    description: Optional[str] = None
    specimen_type: Optional[str] = None
    reference_range: Optional[str] = None
    units: Optional[str] = None
    critical_values: Optional[Dict[str, Any]] = None


class Procedure(BaseModel):
    """Medical procedure node"""
    id: str
    name: str
    cpt_code: Optional[str] = None
    icd10_pcs_code: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    risks: List[str] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)


# Relationship Models
class DrugInteraction(BaseModel):
    """Drug-Drug interaction relationship"""
    drug1_id: str
    drug2_id: str
    severity: Severity
    description: str
    mechanism: Optional[str] = None
    clinical_significance: Optional[str] = None
    management: Optional[str] = None
    evidence_level: Optional[EvidenceLevel] = None
    sources: List[str] = Field(default_factory=list)


class Contraindication(BaseModel):
    """Drug-Condition contraindication relationship"""
    drug_id: str
    condition_id: str
    severity: Severity
    reason: str
    is_absolute: bool = True  # absolute vs relative contraindication
    alternatives: List[str] = Field(default_factory=list)
    monitoring_required: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DiseaseSymptomRelation(BaseModel):
    """Disease-Symptom relationship"""
    disease_id: str
    symptom_id: str
    frequency: str  # common, occasional, rare
    typical_severity: Optional[Severity] = None
    onset_pattern: Optional[str] = None  # acute, gradual, episodic
    is_pathognomonic: bool = False  # uniquely characteristic


class TreatmentRelation(BaseModel):
    """Disease-Drug treatment relationship"""
    disease_id: str
    drug_id: str
    line_of_therapy: int = 1  # 1st line, 2nd line, etc.
    evidence_level: Optional[EvidenceLevel] = None
    dosing_guidance: Optional[str] = None
    duration: Optional[str] = None
    monitoring: Optional[str] = None
    guideline_source: Optional[str] = None


class ClinicalGuideline(BaseModel):
    """Structured clinical guideline for decision support"""
    id: str
    name: str
    condition: str
    
    # Decision criteria
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    
    # Recommendations
    first_line_treatment: List[str] = Field(default_factory=list)
    second_line_treatment: List[str] = Field(default_factory=list)
    
    # Monitoring
    monitoring_parameters: List[str] = Field(default_factory=list)
    follow_up_schedule: Optional[str] = None
    
    # Red flags
    referral_criteria: List[str] = Field(default_factory=list)
    emergency_criteria: List[str] = Field(default_factory=list)
    
    evidence_level: Optional[EvidenceLevel] = None
    source: Optional[str] = None
    last_updated: Optional[date] = None


class Interaction(BaseModel):
    """Generic interaction model"""
    id: str
    type: str  # drug-drug, drug-food, drug-disease
    entity1_id: str
    entity1_type: str
    entity2_id: str
    entity2_type: str
    severity: Severity
    description: str
    mechanism: Optional[str] = None
    management: Optional[str] = None
