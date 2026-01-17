"""
UMI Lab Value Interpretation Service
Provides lab result analysis, normal range checking, and clinical interpretation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path


class ResultStatus(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"
    PANIC = "panic"


class Urgency(str, Enum):
    ROUTINE = "routine"
    ABNORMAL = "abnormal"
    URGENT = "urgent"
    CRITICAL = "critical"


@dataclass
class LabResult:
    """A single lab test result."""
    test_name: str
    value: float
    unit: str
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    loinc_code: Optional[str] = None


@dataclass
class LabInterpretation:
    """Interpretation of a lab result."""
    test_name: str
    value: float
    unit: str
    status: ResultStatus
    urgency: Urgency
    reference_range: str
    interpretation: str
    possible_causes: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    related_tests: List[str] = field(default_factory=list)


# Comprehensive lab reference database
LAB_REFERENCE_DB: Dict[str, Dict[str, Any]] = {
    # Basic Metabolic Panel (BMP)
    "glucose": {
        "name": "Glucose (Fasting)",
        "loinc": "2345-7",
        "unit": "mg/dL",
        "normal_low": 70,
        "normal_high": 100,
        "critical_low": 50,
        "critical_high": 400,
        "panic_low": 40,
        "panic_high": 500,
        "low_causes": ["Hypoglycemia", "Insulin overdose", "Adrenal insufficiency", "Liver disease", "Malnutrition"],
        "high_causes": ["Diabetes mellitus", "Stress response", "Cushing's syndrome", "Pancreatitis", "Medications (steroids)"],
        "low_actions": ["Check for symptoms (sweating, confusion)", "Administer glucose if symptomatic", "Investigate cause"],
        "high_actions": ["Confirm fasting status", "Check HbA1c", "Consider diabetes workup", "Review medications"],
        "related": ["HbA1c", "Insulin", "C-peptide"],
    },
    "creatinine": {
        "name": "Creatinine",
        "loinc": "2160-0",
        "unit": "mg/dL",
        "normal_low": 0.7,
        "normal_high": 1.3,
        "normal_high_female": 1.1,
        "critical_high": 10.0,
        "low_causes": ["Low muscle mass", "Malnutrition", "Liver disease"],
        "high_causes": ["Acute kidney injury", "Chronic kidney disease", "Dehydration", "Rhabdomyolysis", "Nephrotoxic drugs"],
        "high_actions": ["Calculate eGFR", "Review medications", "Check urine output", "Assess hydration", "Consider nephrology consult"],
        "related": ["BUN", "eGFR", "Cystatin C", "Urinalysis"],
    },
    "bun": {
        "name": "Blood Urea Nitrogen",
        "loinc": "3094-0",
        "unit": "mg/dL",
        "normal_low": 7,
        "normal_high": 20,
        "critical_high": 100,
        "low_causes": ["Low protein diet", "Liver disease", "Overhydration", "Malnutrition"],
        "high_causes": ["Kidney disease", "Dehydration", "GI bleeding", "High protein diet", "Heart failure", "Shock"],
        "high_actions": ["Check creatinine", "Calculate BUN/Cr ratio", "Assess hydration", "Check for GI bleeding"],
        "related": ["Creatinine", "eGFR"],
    },
    "sodium": {
        "name": "Sodium",
        "loinc": "2951-2",
        "unit": "mEq/L",
        "normal_low": 136,
        "normal_high": 145,
        "critical_low": 120,
        "critical_high": 160,
        "panic_low": 115,
        "panic_high": 165,
        "low_causes": ["SIADH", "Diuretics", "Heart failure", "Cirrhosis", "Vomiting/diarrhea", "Adrenal insufficiency"],
        "high_causes": ["Dehydration", "Diabetes insipidus", "Excessive salt intake", "Hyperaldosteronism"],
        "low_actions": ["Assess volume status", "Check urine sodium", "Review medications", "Correct slowly (avoid osmotic demyelination)"],
        "high_actions": ["Assess hydration", "Provide free water", "Check urine osmolality", "Correct gradually"],
        "related": ["Potassium", "Chloride", "Osmolality"],
    },
    "potassium": {
        "name": "Potassium",
        "loinc": "2823-3",
        "unit": "mEq/L",
        "normal_low": 3.5,
        "normal_high": 5.0,
        "critical_low": 2.5,
        "critical_high": 6.5,
        "panic_low": 2.0,
        "panic_high": 7.0,
        "low_causes": ["Diuretics", "Vomiting/diarrhea", "Alkalosis", "Insulin therapy", "Hypomagnesemia"],
        "high_causes": ["Kidney disease", "ACE inhibitors/ARBs", "Potassium supplements", "Acidosis", "Hemolysis (artifact)", "Tissue damage"],
        "low_actions": ["ECG monitoring", "Oral/IV potassium replacement", "Check magnesium", "Review medications"],
        "high_actions": ["Repeat to rule out hemolysis", "ECG immediately", "Calcium gluconate if ECG changes", "Insulin/glucose", "Review medications"],
        "related": ["Sodium", "Magnesium", "Bicarbonate"],
    },
    "chloride": {
        "name": "Chloride",
        "loinc": "2075-0",
        "unit": "mEq/L",
        "normal_low": 98,
        "normal_high": 106,
        "critical_low": 80,
        "critical_high": 120,
        "low_causes": ["Vomiting", "Metabolic alkalosis", "SIADH", "Diuretics"],
        "high_causes": ["Dehydration", "Metabolic acidosis", "Renal tubular acidosis", "Excessive saline"],
        "related": ["Sodium", "Bicarbonate", "Anion gap"],
    },
    "bicarbonate": {
        "name": "Bicarbonate (CO2)",
        "loinc": "1963-8",
        "unit": "mEq/L",
        "normal_low": 22,
        "normal_high": 29,
        "critical_low": 10,
        "critical_high": 40,
        "low_causes": ["Metabolic acidosis", "DKA", "Lactic acidosis", "Renal failure", "Diarrhea"],
        "high_causes": ["Metabolic alkalosis", "Vomiting", "Diuretics", "Respiratory compensation"],
        "low_actions": ["Calculate anion gap", "Check lactate", "Check ketones", "Arterial blood gas"],
        "related": ["Anion gap", "Lactate", "pH"],
    },
    "calcium": {
        "name": "Calcium (Total)",
        "loinc": "17861-6",
        "unit": "mg/dL",
        "normal_low": 8.5,
        "normal_high": 10.5,
        "critical_low": 6.0,
        "critical_high": 14.0,
        "low_causes": ["Hypoparathyroidism", "Vitamin D deficiency", "Chronic kidney disease", "Hypomagnesemia", "Pancreatitis"],
        "high_causes": ["Hyperparathyroidism", "Malignancy", "Vitamin D toxicity", "Thiazides", "Immobilization"],
        "low_actions": ["Check ionized calcium", "Check PTH", "Check vitamin D", "Check magnesium", "ECG"],
        "high_actions": ["IV fluids", "Check PTH", "Check PTHrP", "Consider bisphosphonates", "ECG"],
        "related": ["Ionized calcium", "PTH", "Vitamin D", "Phosphorus", "Albumin"],
    },
    
    # Complete Blood Count (CBC)
    "hemoglobin": {
        "name": "Hemoglobin",
        "loinc": "718-7",
        "unit": "g/dL",
        "normal_low_male": 13.5,
        "normal_high_male": 17.5,
        "normal_low_female": 12.0,
        "normal_high_female": 16.0,
        "critical_low": 7.0,
        "critical_high": 20.0,
        "low_causes": ["Iron deficiency", "B12/folate deficiency", "Chronic disease", "Blood loss", "Hemolysis", "Bone marrow failure"],
        "high_causes": ["Polycythemia vera", "Chronic hypoxia", "Dehydration", "EPO-producing tumors"],
        "low_actions": ["Check MCV for type", "Iron studies", "Reticulocyte count", "Consider transfusion if symptomatic"],
        "high_actions": ["Check oxygen saturation", "Consider phlebotomy", "Hematology referral"],
        "related": ["Hematocrit", "MCV", "RBC", "Iron studies", "Reticulocytes"],
    },
    "hematocrit": {
        "name": "Hematocrit",
        "loinc": "4544-3",
        "unit": "%",
        "normal_low_male": 38.8,
        "normal_high_male": 50.0,
        "normal_low_female": 34.9,
        "normal_high_female": 44.5,
        "critical_low": 20,
        "critical_high": 60,
        "low_causes": ["Anemia", "Blood loss", "Overhydration"],
        "high_causes": ["Dehydration", "Polycythemia", "Chronic hypoxia"],
        "related": ["Hemoglobin", "RBC"],
    },
    "wbc": {
        "name": "White Blood Cell Count",
        "loinc": "6690-2",
        "unit": "K/uL",
        "normal_low": 4.5,
        "normal_high": 11.0,
        "critical_low": 2.0,
        "critical_high": 30.0,
        "panic_low": 1.0,
        "panic_high": 50.0,
        "low_causes": ["Bone marrow suppression", "Chemotherapy", "Viral infections", "Autoimmune", "Medications"],
        "high_causes": ["Infection", "Inflammation", "Leukemia", "Stress", "Steroids", "Smoking"],
        "low_actions": ["Check differential", "Review medications", "Consider infection precautions", "Hematology consult if severe"],
        "high_actions": ["Check differential", "Look for infection source", "Blood cultures if febrile", "Consider peripheral smear"],
        "related": ["Differential", "ANC", "Bands"],
    },
    "platelets": {
        "name": "Platelet Count",
        "loinc": "777-3",
        "unit": "K/uL",
        "normal_low": 150,
        "normal_high": 400,
        "critical_low": 50,
        "critical_high": 1000,
        "panic_low": 20,
        "low_causes": ["ITP", "TTP/HUS", "DIC", "Bone marrow failure", "Medications", "Hypersplenism", "Viral infections"],
        "high_causes": ["Reactive (infection, inflammation)", "Iron deficiency", "Myeloproliferative disorders", "Post-splenectomy"],
        "low_actions": ["Review medications", "Check peripheral smear", "Avoid IM injections", "Hematology consult", "Consider transfusion if bleeding"],
        "high_actions": ["Look for underlying cause", "Consider aspirin if very high", "Hematology referral if persistent"],
        "related": ["MPV", "Peripheral smear"],
    },
    
    # Liver Function Tests (LFTs)
    "alt": {
        "name": "Alanine Aminotransferase (ALT)",
        "loinc": "1742-6",
        "unit": "U/L",
        "normal_low": 7,
        "normal_high": 56,
        "critical_high": 1000,
        "high_causes": ["Hepatitis (viral, alcoholic, autoimmune)", "NAFLD", "Medications", "Ischemia", "Muscle injury"],
        "high_actions": ["Check hepatitis serologies", "Review medications", "Check AST/ALT ratio", "Liver ultrasound"],
        "related": ["AST", "ALP", "Bilirubin", "GGT", "Albumin"],
    },
    "ast": {
        "name": "Aspartate Aminotransferase (AST)",
        "loinc": "1920-8",
        "unit": "U/L",
        "normal_low": 10,
        "normal_high": 40,
        "critical_high": 1000,
        "high_causes": ["Liver disease", "Myocardial infarction", "Muscle injury", "Hemolysis"],
        "high_actions": ["Check ALT", "Calculate AST/ALT ratio", "Check CK if muscle injury suspected"],
        "related": ["ALT", "CK", "LDH"],
    },
    "bilirubin_total": {
        "name": "Bilirubin (Total)",
        "loinc": "1975-2",
        "unit": "mg/dL",
        "normal_low": 0.1,
        "normal_high": 1.2,
        "critical_high": 15.0,
        "high_causes": ["Hemolysis", "Liver disease", "Biliary obstruction", "Gilbert's syndrome"],
        "high_actions": ["Check direct bilirubin", "Check hemolysis labs", "Liver ultrasound", "Review medications"],
        "related": ["Direct bilirubin", "ALT", "AST", "ALP"],
    },
    "albumin": {
        "name": "Albumin",
        "loinc": "1751-7",
        "unit": "g/dL",
        "normal_low": 3.5,
        "normal_high": 5.0,
        "critical_low": 1.5,
        "low_causes": ["Liver disease", "Malnutrition", "Nephrotic syndrome", "Inflammation", "Burns"],
        "low_actions": ["Check liver function", "Check urine protein", "Assess nutritional status"],
        "related": ["Total protein", "Prealbumin", "Liver enzymes"],
    },
    "alp": {
        "name": "Alkaline Phosphatase (ALP)",
        "loinc": "6768-6",
        "unit": "U/L",
        "normal_low": 44,
        "normal_high": 147,
        "high_causes": ["Biliary obstruction", "Bone disease", "Pregnancy", "Growing children", "Medications"],
        "high_actions": ["Check GGT to differentiate liver vs bone", "Liver ultrasound if hepatic", "Bone workup if skeletal"],
        "related": ["GGT", "Bilirubin", "Calcium", "Vitamin D"],
    },
    
    # Lipid Panel
    "cholesterol_total": {
        "name": "Total Cholesterol",
        "loinc": "2093-3",
        "unit": "mg/dL",
        "normal_high": 200,
        "borderline_high": 239,
        "high_causes": ["Diet", "Familial hypercholesterolemia", "Hypothyroidism", "Nephrotic syndrome", "Medications"],
        "high_actions": ["Full lipid panel", "Calculate cardiovascular risk", "Lifestyle modifications", "Consider statin therapy"],
        "related": ["LDL", "HDL", "Triglycerides"],
    },
    "ldl": {
        "name": "LDL Cholesterol",
        "loinc": "13457-7",
        "unit": "mg/dL",
        "optimal": 100,
        "near_optimal": 129,
        "borderline_high": 159,
        "high": 189,
        "very_high": 190,
        "high_causes": ["Diet high in saturated fat", "Genetics", "Obesity", "Diabetes", "Hypothyroidism"],
        "high_actions": ["Lifestyle modifications", "Consider statin based on CV risk", "Repeat in 4-6 weeks after intervention"],
        "related": ["Total cholesterol", "HDL", "Triglycerides", "ApoB"],
    },
    "hdl": {
        "name": "HDL Cholesterol",
        "loinc": "2085-9",
        "unit": "mg/dL",
        "normal_low_male": 40,
        "normal_low_female": 50,
        "optimal": 60,
        "low_causes": ["Sedentary lifestyle", "Smoking", "Obesity", "Diabetes", "Genetics"],
        "low_actions": ["Exercise", "Smoking cessation", "Weight loss", "Consider niacin"],
        "related": ["Total cholesterol", "LDL", "Triglycerides"],
    },
    "triglycerides": {
        "name": "Triglycerides",
        "loinc": "2571-8",
        "unit": "mg/dL",
        "normal_high": 150,
        "borderline_high": 199,
        "high": 499,
        "very_high": 500,
        "high_causes": ["Diet", "Obesity", "Diabetes", "Alcohol", "Medications", "Hypothyroidism"],
        "high_actions": ["Dietary changes", "Limit alcohol", "Treat underlying conditions", "Consider fibrates if very high"],
        "related": ["Total cholesterol", "LDL", "HDL", "Glucose"],
    },
    
    # Thyroid Function
    "tsh": {
        "name": "Thyroid Stimulating Hormone",
        "loinc": "3016-3",
        "unit": "mIU/L",
        "normal_low": 0.4,
        "normal_high": 4.0,
        "critical_low": 0.01,
        "critical_high": 50,
        "low_causes": ["Hyperthyroidism", "Graves' disease", "Thyroiditis", "Excessive thyroid hormone", "Pituitary disease"],
        "high_causes": ["Hypothyroidism", "Hashimoto's thyroiditis", "Iodine deficiency", "Thyroid hormone resistance"],
        "low_actions": ["Check Free T4 and T3", "Consider thyroid antibodies", "Thyroid uptake scan"],
        "high_actions": ["Check Free T4", "Check thyroid antibodies", "Consider levothyroxine if confirmed hypothyroid"],
        "related": ["Free T4", "Free T3", "Thyroid antibodies"],
    },
    "free_t4": {
        "name": "Free T4 (Thyroxine)",
        "loinc": "14749-6",
        "unit": "ng/dL",
        "normal_low": 0.8,
        "normal_high": 1.8,
        "low_causes": ["Hypothyroidism", "Pituitary disease", "Severe illness"],
        "high_causes": ["Hyperthyroidism", "Thyroiditis", "Excessive thyroid hormone replacement"],
        "related": ["TSH", "Free T3"],
    },
    
    # Diabetes Monitoring
    "hba1c": {
        "name": "Hemoglobin A1c",
        "loinc": "4548-4",
        "unit": "%",
        "normal_high": 5.6,
        "prediabetes_low": 5.7,
        "prediabetes_high": 6.4,
        "diabetes_threshold": 6.5,
        "target_diabetic": 7.0,
        "high_causes": ["Diabetes mellitus", "Prediabetes", "Poor glycemic control"],
        "high_actions": ["Confirm diagnosis if new", "Intensify diabetes management", "Review medications", "Dietary counseling"],
        "related": ["Fasting glucose", "Fructosamine"],
    },
    
    # Cardiac Markers
    "troponin": {
        "name": "Troponin I/T",
        "loinc": "10839-9",
        "unit": "ng/mL",
        "normal_high": 0.04,
        "critical_high": 0.1,
        "high_causes": ["Myocardial infarction", "Myocarditis", "Heart failure", "PE", "Renal failure", "Sepsis"],
        "high_actions": ["Serial troponins", "ECG", "Cardiology consult", "Consider cath if STEMI/NSTEMI"],
        "related": ["CK-MB", "BNP", "ECG"],
    },
    "bnp": {
        "name": "B-type Natriuretic Peptide",
        "loinc": "30934-4",
        "unit": "pg/mL",
        "normal_high": 100,
        "heart_failure_likely": 400,
        "high_causes": ["Heart failure", "Renal failure", "Pulmonary embolism", "Atrial fibrillation"],
        "high_actions": ["Echocardiogram", "Assess volume status", "Optimize heart failure therapy"],
        "related": ["Troponin", "Echocardiogram"],
    },
    
    # Inflammatory Markers
    "crp": {
        "name": "C-Reactive Protein",
        "loinc": "1988-5",
        "unit": "mg/L",
        "normal_high": 3.0,
        "high_causes": ["Infection", "Inflammation", "Autoimmune disease", "Malignancy", "Tissue injury"],
        "high_actions": ["Look for infection source", "Consider autoimmune workup", "Serial monitoring"],
        "related": ["ESR", "Procalcitonin", "WBC"],
    },
    "esr": {
        "name": "Erythrocyte Sedimentation Rate",
        "loinc": "4537-7",
        "unit": "mm/hr",
        "normal_high_male": 15,
        "normal_high_female": 20,
        "high_causes": ["Infection", "Inflammation", "Autoimmune disease", "Malignancy", "Anemia"],
        "high_actions": ["Correlate with clinical picture", "Consider CRP", "Investigate underlying cause"],
        "related": ["CRP", "WBC"],
    },
    
    # Coagulation
    "pt_inr": {
        "name": "PT/INR",
        "loinc": "5902-2",
        "unit": "INR",
        "normal_low": 0.9,
        "normal_high": 1.1,
        "therapeutic_warfarin_low": 2.0,
        "therapeutic_warfarin_high": 3.0,
        "critical_high": 5.0,
        "high_causes": ["Warfarin therapy", "Liver disease", "Vitamin K deficiency", "DIC"],
        "high_actions": ["Review anticoagulation", "Check for bleeding", "Consider vitamin K", "FFP if bleeding"],
        "related": ["PTT", "Fibrinogen", "D-dimer"],
    },
    "ptt": {
        "name": "Partial Thromboplastin Time",
        "loinc": "3173-2",
        "unit": "seconds",
        "normal_low": 25,
        "normal_high": 35,
        "critical_high": 100,
        "high_causes": ["Heparin therapy", "Factor deficiency", "Lupus anticoagulant", "DIC"],
        "high_actions": ["Review heparin dosing", "Mixing study if unexplained", "Check for bleeding"],
        "related": ["PT/INR", "Anti-Xa", "Fibrinogen"],
    },
}


class LabInterpreter:
    """
    Interprets laboratory test results and provides clinical guidance.
    """
    
    def __init__(self):
        self.reference_db = LAB_REFERENCE_DB
    
    def normalize_test_name(self, test_name: str) -> Optional[str]:
        """Normalize test name to database key."""
        test_lower = test_name.lower().strip()
        
        # Direct match
        if test_lower in self.reference_db:
            return test_lower
        
        # Common aliases
        aliases = {
            "glu": "glucose",
            "fasting glucose": "glucose",
            "blood glucose": "glucose",
            "cr": "creatinine",
            "scr": "creatinine",
            "na": "sodium",
            "k": "potassium",
            "cl": "chloride",
            "co2": "bicarbonate",
            "hco3": "bicarbonate",
            "ca": "calcium",
            "hgb": "hemoglobin",
            "hb": "hemoglobin",
            "hct": "hematocrit",
            "white blood cells": "wbc",
            "white count": "wbc",
            "leukocytes": "wbc",
            "plt": "platelets",
            "platelet count": "platelets",
            "sgpt": "alt",
            "sgot": "ast",
            "tbili": "bilirubin_total",
            "total bilirubin": "bilirubin_total",
            "alb": "albumin",
            "alk phos": "alp",
            "alkaline phosphatase": "alp",
            "tc": "cholesterol_total",
            "total cholesterol": "cholesterol_total",
            "ldl-c": "ldl",
            "ldl cholesterol": "ldl",
            "hdl-c": "hdl",
            "hdl cholesterol": "hdl",
            "tg": "triglycerides",
            "trigs": "triglycerides",
            "a1c": "hba1c",
            "glycated hemoglobin": "hba1c",
            "hemoglobin a1c": "hba1c",
            "ft4": "free_t4",
            "free thyroxine": "free_t4",
            "trop": "troponin",
            "troponin i": "troponin",
            "troponin t": "troponin",
            "hs-crp": "crp",
            "c-reactive protein": "crp",
            "sed rate": "esr",
            "inr": "pt_inr",
            "protime": "pt_inr",
            "prothrombin time": "pt_inr",
            "aptt": "ptt",
        }
        
        if test_lower in aliases:
            return aliases[test_lower]
        
        return None
    
    def get_reference_range(
        self,
        test_key: str,
        sex: str = "unknown",
        age_years: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get reference range for a test, adjusted for sex/age if applicable."""
        if test_key not in self.reference_db:
            return None, None
        
        ref = self.reference_db[test_key]
        
        # Check for sex-specific ranges
        if sex.lower() == "male":
            low = ref.get("normal_low_male", ref.get("normal_low"))
            high = ref.get("normal_high_male", ref.get("normal_high"))
        elif sex.lower() == "female":
            low = ref.get("normal_low_female", ref.get("normal_low"))
            high = ref.get("normal_high_female", ref.get("normal_high"))
        else:
            low = ref.get("normal_low")
            high = ref.get("normal_high")
        
        return low, high
    
    def determine_status(
        self,
        test_key: str,
        value: float,
        sex: str = "unknown"
    ) -> Tuple[ResultStatus, Urgency]:
        """Determine the status and urgency of a lab result."""
        if test_key not in self.reference_db:
            return ResultStatus.NORMAL, Urgency.ROUTINE
        
        ref = self.reference_db[test_key]
        low, high = self.get_reference_range(test_key, sex)
        
        # Check panic values first
        panic_low = ref.get("panic_low")
        panic_high = ref.get("panic_high")
        if panic_low and value <= panic_low:
            return ResultStatus.PANIC, Urgency.CRITICAL
        if panic_high and value >= panic_high:
            return ResultStatus.PANIC, Urgency.CRITICAL
        
        # Check critical values
        critical_low = ref.get("critical_low")
        critical_high = ref.get("critical_high")
        if critical_low and value <= critical_low:
            return ResultStatus.CRITICAL_LOW, Urgency.CRITICAL
        if critical_high and value >= critical_high:
            return ResultStatus.CRITICAL_HIGH, Urgency.CRITICAL
        
        # Check normal range
        if low is not None and value < low:
            return ResultStatus.LOW, Urgency.ABNORMAL
        if high is not None and value > high:
            return ResultStatus.HIGH, Urgency.ABNORMAL
        
        return ResultStatus.NORMAL, Urgency.ROUTINE
    
    def interpret(
        self,
        test_name: str,
        value: float,
        unit: Optional[str] = None,
        sex: str = "unknown",
        age_years: Optional[float] = None
    ) -> LabInterpretation:
        """
        Interpret a single lab result.
        """
        test_key = self.normalize_test_name(test_name)
        
        if not test_key or test_key not in self.reference_db:
            return LabInterpretation(
                test_name=test_name,
                value=value,
                unit=unit or "",
                status=ResultStatus.NORMAL,
                urgency=Urgency.ROUTINE,
                reference_range="Not available",
                interpretation="Test not found in database. Please consult laboratory reference.",
                possible_causes=[],
                recommended_actions=["Consult with laboratory or specialist"],
                related_tests=[]
            )
        
        ref = self.reference_db[test_key]
        status, urgency = self.determine_status(test_key, value, sex)
        low, high = self.get_reference_range(test_key, sex)
        
        # Build reference range string
        if low is not None and high is not None:
            ref_range = f"{low} - {high} {ref['unit']}"
        elif low is not None:
            ref_range = f">= {low} {ref['unit']}"
        elif high is not None:
            ref_range = f"<= {high} {ref['unit']}"
        else:
            ref_range = "See laboratory reference"
        
        # Build interpretation
        if status == ResultStatus.NORMAL:
            interpretation = f"{ref['name']} is within normal limits."
        elif status == ResultStatus.LOW:
            interpretation = f"{ref['name']} is below the normal range."
        elif status == ResultStatus.HIGH:
            interpretation = f"{ref['name']} is above the normal range."
        elif status == ResultStatus.CRITICAL_LOW:
            interpretation = f"CRITICAL: {ref['name']} is critically low. Immediate attention required."
        elif status == ResultStatus.CRITICAL_HIGH:
            interpretation = f"CRITICAL: {ref['name']} is critically high. Immediate attention required."
        elif status == ResultStatus.PANIC:
            interpretation = f"PANIC VALUE: {ref['name']} is at a life-threatening level. Immediate intervention required."
        
        # Get causes and actions based on status
        if status in [ResultStatus.LOW, ResultStatus.CRITICAL_LOW]:
            causes = ref.get("low_causes", [])
            actions = ref.get("low_actions", [])
        elif status in [ResultStatus.HIGH, ResultStatus.CRITICAL_HIGH, ResultStatus.PANIC]:
            causes = ref.get("high_causes", [])
            actions = ref.get("high_actions", [])
        else:
            causes = []
            actions = []
        
        return LabInterpretation(
            test_name=ref["name"],
            value=value,
            unit=ref["unit"],
            status=status,
            urgency=urgency,
            reference_range=ref_range,
            interpretation=interpretation,
            possible_causes=causes,
            recommended_actions=actions,
            related_tests=ref.get("related", [])
        )
    
    def interpret_panel(
        self,
        results: List[LabResult],
        sex: str = "unknown",
        age_years: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Interpret a panel of lab results and provide summary.
        """
        interpretations = []
        critical_findings = []
        abnormal_findings = []
        
        for result in results:
            interp = self.interpret(
                result.test_name,
                result.value,
                result.unit,
                sex,
                age_years
            )
            interpretations.append(interp)
            
            if interp.urgency == Urgency.CRITICAL:
                critical_findings.append(interp)
            elif interp.urgency == Urgency.ABNORMAL:
                abnormal_findings.append(interp)
        
        # Determine overall urgency
        if critical_findings:
            overall_urgency = Urgency.CRITICAL
        elif abnormal_findings:
            overall_urgency = Urgency.ABNORMAL
        else:
            overall_urgency = Urgency.ROUTINE
        
        return {
            "interpretations": interpretations,
            "critical_findings": critical_findings,
            "abnormal_findings": abnormal_findings,
            "overall_urgency": overall_urgency,
            "summary": self._generate_summary(interpretations, critical_findings, abnormal_findings)
        }
    
    def _generate_summary(
        self,
        interpretations: List[LabInterpretation],
        critical: List[LabInterpretation],
        abnormal: List[LabInterpretation]
    ) -> str:
        """Generate a summary of lab panel results."""
        total = len(interpretations)
        normal_count = total - len(critical) - len(abnormal)
        
        summary_parts = [f"Reviewed {total} lab values."]
        
        if critical:
            critical_names = [i.test_name for i in critical]
            summary_parts.append(f"CRITICAL: {', '.join(critical_names)} require immediate attention.")
        
        if abnormal:
            abnormal_names = [i.test_name for i in abnormal]
            summary_parts.append(f"Abnormal: {', '.join(abnormal_names)}.")
        
        if normal_count == total:
            summary_parts.append("All values within normal limits.")
        
        return " ".join(summary_parts)
    
    def get_available_tests(self) -> List[str]:
        """Return list of tests in the database."""
        return [ref["name"] for ref in self.reference_db.values()]
    
    def calculate_anion_gap(
        self,
        sodium: float,
        chloride: float,
        bicarbonate: float
    ) -> Dict[str, Any]:
        """Calculate anion gap from electrolytes."""
        anion_gap = sodium - (chloride + bicarbonate)
        
        status = "normal"
        interpretation = "Anion gap is within normal limits (8-12 mEq/L)."
        causes = []
        
        if anion_gap > 12:
            status = "elevated"
            interpretation = "Elevated anion gap suggests metabolic acidosis."
            causes = [
                "Lactic acidosis",
                "Diabetic ketoacidosis (DKA)",
                "Uremia (renal failure)",
                "Toxic ingestion (methanol, ethylene glycol, salicylates)",
                "Starvation ketosis"
            ]
        elif anion_gap < 8:
            status = "low"
            interpretation = "Low anion gap may indicate hypoalbuminemia or laboratory error."
            causes = [
                "Hypoalbuminemia",
                "Multiple myeloma",
                "Laboratory error"
            ]
        
        return {
            "anion_gap": round(anion_gap, 1),
            "unit": "mEq/L",
            "status": status,
            "interpretation": interpretation,
            "possible_causes": causes
        }
    
    def calculate_corrected_calcium(
        self,
        total_calcium: float,
        albumin: float
    ) -> Dict[str, Any]:
        """Calculate albumin-corrected calcium."""
        corrected = total_calcium + 0.8 * (4.0 - albumin)
        
        status = "normal"
        if corrected < 8.5:
            status = "low"
        elif corrected > 10.5:
            status = "high"
        
        return {
            "corrected_calcium": round(corrected, 1),
            "unit": "mg/dL",
            "status": status,
            "interpretation": f"Corrected calcium: {round(corrected, 1)} mg/dL (adjusted for albumin {albumin} g/dL)"
        }


# Singleton instance
_interpreter = None


def get_lab_interpreter() -> LabInterpreter:
    """Get singleton instance of LabInterpreter."""
    global _interpreter
    if _interpreter is None:
        _interpreter = LabInterpreter()
    return _interpreter
