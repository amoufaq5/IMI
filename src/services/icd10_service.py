"""
UMI ICD-10 Coding Service
Provides diagnosis coding, code lookup, and clinical classification
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import re


class ICD10Category(str, Enum):
    INFECTIOUS = "A00-B99"
    NEOPLASMS = "C00-D49"
    BLOOD = "D50-D89"
    ENDOCRINE = "E00-E89"
    MENTAL = "F01-F99"
    NERVOUS = "G00-G99"
    EYE = "H00-H59"
    EAR = "H60-H95"
    CIRCULATORY = "I00-I99"
    RESPIRATORY = "J00-J99"
    DIGESTIVE = "K00-K95"
    SKIN = "L00-L99"
    MUSCULOSKELETAL = "M00-M99"
    GENITOURINARY = "N00-N99"
    PREGNANCY = "O00-O9A"
    PERINATAL = "P00-P96"
    CONGENITAL = "Q00-Q99"
    SYMPTOMS = "R00-R99"
    INJURY = "S00-T88"
    EXTERNAL = "V00-Y99"
    FACTORS = "Z00-Z99"


@dataclass
class ICD10Code:
    """An ICD-10 diagnosis code."""
    code: str
    description: str
    category: str
    chapter: str
    is_billable: bool = True
    includes: List[str] = field(default_factory=list)
    excludes: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class CodingSuggestion:
    """A suggested ICD-10 code for a diagnosis."""
    code: str
    description: str
    confidence: float  # 0.0 to 1.0
    rationale: str
    alternatives: List[Tuple[str, str]] = field(default_factory=list)


# Common ICD-10 codes database (subset for quick wins)
ICD10_DATABASE: Dict[str, Dict[str, Any]] = {
    # Infectious Diseases
    "A09": {"desc": "Infectious gastroenteritis and colitis, unspecified", "cat": "Infectious", "chapter": "I"},
    "A41.9": {"desc": "Sepsis, unspecified organism", "cat": "Infectious", "chapter": "I"},
    "A49.9": {"desc": "Bacterial infection, unspecified", "cat": "Infectious", "chapter": "I"},
    "B34.9": {"desc": "Viral infection, unspecified", "cat": "Infectious", "chapter": "I"},
    "B97.29": {"desc": "Other coronavirus as the cause of diseases classified elsewhere", "cat": "Infectious", "chapter": "I"},
    
    # Neoplasms
    "C34.90": {"desc": "Malignant neoplasm of unspecified part of bronchus or lung", "cat": "Neoplasms", "chapter": "II"},
    "C50.919": {"desc": "Malignant neoplasm of unspecified site of unspecified female breast", "cat": "Neoplasms", "chapter": "II"},
    "C61": {"desc": "Malignant neoplasm of prostate", "cat": "Neoplasms", "chapter": "II"},
    "D64.9": {"desc": "Anemia, unspecified", "cat": "Blood", "chapter": "III"},
    
    # Endocrine
    "E03.9": {"desc": "Hypothyroidism, unspecified", "cat": "Endocrine", "chapter": "IV"},
    "E05.90": {"desc": "Thyrotoxicosis, unspecified without thyrotoxic crisis", "cat": "Endocrine", "chapter": "IV"},
    "E11.9": {"desc": "Type 2 diabetes mellitus without complications", "cat": "Endocrine", "chapter": "IV"},
    "E11.65": {"desc": "Type 2 diabetes mellitus with hyperglycemia", "cat": "Endocrine", "chapter": "IV"},
    "E11.21": {"desc": "Type 2 diabetes mellitus with diabetic nephropathy", "cat": "Endocrine", "chapter": "IV"},
    "E11.40": {"desc": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified", "cat": "Endocrine", "chapter": "IV"},
    "E11.319": {"desc": "Type 2 diabetes mellitus with unspecified diabetic retinopathy without macular edema", "cat": "Endocrine", "chapter": "IV"},
    "E10.9": {"desc": "Type 1 diabetes mellitus without complications", "cat": "Endocrine", "chapter": "IV"},
    "E78.5": {"desc": "Hyperlipidemia, unspecified", "cat": "Endocrine", "chapter": "IV"},
    "E78.00": {"desc": "Pure hypercholesterolemia, unspecified", "cat": "Endocrine", "chapter": "IV"},
    "E66.9": {"desc": "Obesity, unspecified", "cat": "Endocrine", "chapter": "IV"},
    "E87.6": {"desc": "Hypokalemia", "cat": "Endocrine", "chapter": "IV"},
    "E87.1": {"desc": "Hypo-osmolality and hyponatremia", "cat": "Endocrine", "chapter": "IV"},
    
    # Mental and Behavioral
    "F32.9": {"desc": "Major depressive disorder, single episode, unspecified", "cat": "Mental", "chapter": "V"},
    "F33.0": {"desc": "Major depressive disorder, recurrent, mild", "cat": "Mental", "chapter": "V"},
    "F41.1": {"desc": "Generalized anxiety disorder", "cat": "Mental", "chapter": "V"},
    "F41.9": {"desc": "Anxiety disorder, unspecified", "cat": "Mental", "chapter": "V"},
    "F17.210": {"desc": "Nicotine dependence, cigarettes, uncomplicated", "cat": "Mental", "chapter": "V"},
    "F10.20": {"desc": "Alcohol dependence, uncomplicated", "cat": "Mental", "chapter": "V"},
    
    # Nervous System
    "G43.909": {"desc": "Migraine, unspecified, not intractable, without status migrainosus", "cat": "Nervous", "chapter": "VI"},
    "G44.1": {"desc": "Vascular headache, not elsewhere classified", "cat": "Nervous", "chapter": "VI"},
    "G47.00": {"desc": "Insomnia, unspecified", "cat": "Nervous", "chapter": "VI"},
    "G89.29": {"desc": "Other chronic pain", "cat": "Nervous", "chapter": "VI"},
    "G40.909": {"desc": "Epilepsy, unspecified, not intractable, without status epilepticus", "cat": "Nervous", "chapter": "VI"},
    
    # Circulatory System
    "I10": {"desc": "Essential (primary) hypertension", "cat": "Circulatory", "chapter": "IX"},
    "I11.9": {"desc": "Hypertensive heart disease without heart failure", "cat": "Circulatory", "chapter": "IX"},
    "I25.10": {"desc": "Atherosclerotic heart disease of native coronary artery without angina pectoris", "cat": "Circulatory", "chapter": "IX"},
    "I21.9": {"desc": "Acute myocardial infarction, unspecified", "cat": "Circulatory", "chapter": "IX"},
    "I21.3": {"desc": "ST elevation (STEMI) myocardial infarction of unspecified site", "cat": "Circulatory", "chapter": "IX"},
    "I48.91": {"desc": "Unspecified atrial fibrillation", "cat": "Circulatory", "chapter": "IX"},
    "I48.0": {"desc": "Paroxysmal atrial fibrillation", "cat": "Circulatory", "chapter": "IX"},
    "I50.9": {"desc": "Heart failure, unspecified", "cat": "Circulatory", "chapter": "IX"},
    "I50.22": {"desc": "Chronic systolic (congestive) heart failure", "cat": "Circulatory", "chapter": "IX"},
    "I63.9": {"desc": "Cerebral infarction, unspecified", "cat": "Circulatory", "chapter": "IX"},
    "I73.9": {"desc": "Peripheral vascular disease, unspecified", "cat": "Circulatory", "chapter": "IX"},
    "I87.2": {"desc": "Venous insufficiency (chronic) (peripheral)", "cat": "Circulatory", "chapter": "IX"},
    
    # Respiratory System
    "J00": {"desc": "Acute nasopharyngitis [common cold]", "cat": "Respiratory", "chapter": "X"},
    "J02.9": {"desc": "Acute pharyngitis, unspecified", "cat": "Respiratory", "chapter": "X"},
    "J06.9": {"desc": "Acute upper respiratory infection, unspecified", "cat": "Respiratory", "chapter": "X"},
    "J18.9": {"desc": "Pneumonia, unspecified organism", "cat": "Respiratory", "chapter": "X"},
    "J20.9": {"desc": "Acute bronchitis, unspecified", "cat": "Respiratory", "chapter": "X"},
    "J44.1": {"desc": "Chronic obstructive pulmonary disease with acute exacerbation", "cat": "Respiratory", "chapter": "X"},
    "J44.9": {"desc": "Chronic obstructive pulmonary disease, unspecified", "cat": "Respiratory", "chapter": "X"},
    "J45.20": {"desc": "Mild intermittent asthma, uncomplicated", "cat": "Respiratory", "chapter": "X"},
    "J45.909": {"desc": "Unspecified asthma, uncomplicated", "cat": "Respiratory", "chapter": "X"},
    
    # Digestive System
    "K21.0": {"desc": "Gastro-esophageal reflux disease with esophagitis", "cat": "Digestive", "chapter": "XI"},
    "K21.9": {"desc": "Gastro-esophageal reflux disease without esophagitis", "cat": "Digestive", "chapter": "XI"},
    "K29.70": {"desc": "Gastritis, unspecified, without bleeding", "cat": "Digestive", "chapter": "XI"},
    "K30": {"desc": "Functional dyspepsia", "cat": "Digestive", "chapter": "XI"},
    "K35.80": {"desc": "Unspecified acute appendicitis", "cat": "Digestive", "chapter": "XI"},
    "K57.30": {"desc": "Diverticulosis of large intestine without perforation or abscess without bleeding", "cat": "Digestive", "chapter": "XI"},
    "K58.9": {"desc": "Irritable bowel syndrome without diarrhea", "cat": "Digestive", "chapter": "XI"},
    "K76.0": {"desc": "Fatty (change of) liver, not elsewhere classified", "cat": "Digestive", "chapter": "XI"},
    "K80.20": {"desc": "Calculus of gallbladder without cholecystitis without obstruction", "cat": "Digestive", "chapter": "XI"},
    
    # Skin
    "L30.9": {"desc": "Dermatitis, unspecified", "cat": "Skin", "chapter": "XII"},
    "L50.9": {"desc": "Urticaria, unspecified", "cat": "Skin", "chapter": "XII"},
    "L70.0": {"desc": "Acne vulgaris", "cat": "Skin", "chapter": "XII"},
    "L03.90": {"desc": "Cellulitis, unspecified", "cat": "Skin", "chapter": "XII"},
    
    # Musculoskeletal
    "M54.5": {"desc": "Low back pain", "cat": "Musculoskeletal", "chapter": "XIII"},
    "M54.2": {"desc": "Cervicalgia", "cat": "Musculoskeletal", "chapter": "XIII"},
    "M25.50": {"desc": "Pain in unspecified joint", "cat": "Musculoskeletal", "chapter": "XIII"},
    "M79.3": {"desc": "Panniculitis, unspecified", "cat": "Musculoskeletal", "chapter": "XIII"},
    "M17.9": {"desc": "Osteoarthritis of knee, unspecified", "cat": "Musculoskeletal", "chapter": "XIII"},
    "M19.90": {"desc": "Unspecified osteoarthritis, unspecified site", "cat": "Musculoskeletal", "chapter": "XIII"},
    "M81.0": {"desc": "Age-related osteoporosis without current pathological fracture", "cat": "Musculoskeletal", "chapter": "XIII"},
    "M79.1": {"desc": "Myalgia", "cat": "Musculoskeletal", "chapter": "XIII"},
    "M62.830": {"desc": "Muscle spasm of back", "cat": "Musculoskeletal", "chapter": "XIII"},
    
    # Genitourinary
    "N18.3": {"desc": "Chronic kidney disease, stage 3 (moderate)", "cat": "Genitourinary", "chapter": "XIV"},
    "N18.4": {"desc": "Chronic kidney disease, stage 4 (severe)", "cat": "Genitourinary", "chapter": "XIV"},
    "N18.9": {"desc": "Chronic kidney disease, unspecified", "cat": "Genitourinary", "chapter": "XIV"},
    "N39.0": {"desc": "Urinary tract infection, site not specified", "cat": "Genitourinary", "chapter": "XIV"},
    "N40.0": {"desc": "Benign prostatic hyperplasia without lower urinary tract symptoms", "cat": "Genitourinary", "chapter": "XIV"},
    
    # Symptoms and Signs
    "R00.0": {"desc": "Tachycardia, unspecified", "cat": "Symptoms", "chapter": "XVIII"},
    "R05": {"desc": "Cough", "cat": "Symptoms", "chapter": "XVIII"},
    "R06.02": {"desc": "Shortness of breath", "cat": "Symptoms", "chapter": "XVIII"},
    "R07.9": {"desc": "Chest pain, unspecified", "cat": "Symptoms", "chapter": "XVIII"},
    "R10.9": {"desc": "Unspecified abdominal pain", "cat": "Symptoms", "chapter": "XVIII"},
    "R10.84": {"desc": "Generalized abdominal pain", "cat": "Symptoms", "chapter": "XVIII"},
    "R11.2": {"desc": "Nausea with vomiting, unspecified", "cat": "Symptoms", "chapter": "XVIII"},
    "R19.7": {"desc": "Diarrhea, unspecified", "cat": "Symptoms", "chapter": "XVIII"},
    "R42": {"desc": "Dizziness and giddiness", "cat": "Symptoms", "chapter": "XVIII"},
    "R50.9": {"desc": "Fever, unspecified", "cat": "Symptoms", "chapter": "XVIII"},
    "R51": {"desc": "Headache", "cat": "Symptoms", "chapter": "XVIII"},
    "R53.83": {"desc": "Other fatigue", "cat": "Symptoms", "chapter": "XVIII"},
    "R63.4": {"desc": "Abnormal weight loss", "cat": "Symptoms", "chapter": "XVIII"},
    "R73.03": {"desc": "Prediabetes", "cat": "Symptoms", "chapter": "XVIII"},
    
    # Injury
    "S06.0X0A": {"desc": "Concussion without loss of consciousness, initial encounter", "cat": "Injury", "chapter": "XIX"},
    "S52.509A": {"desc": "Unspecified fracture of the lower end of unspecified radius, initial encounter", "cat": "Injury", "chapter": "XIX"},
    "S82.009A": {"desc": "Unspecified fracture of unspecified patella, initial encounter", "cat": "Injury", "chapter": "XIX"},
    "T78.40XA": {"desc": "Allergy, unspecified, initial encounter", "cat": "Injury", "chapter": "XIX"},
    
    # Factors Influencing Health Status
    "Z00.00": {"desc": "Encounter for general adult medical examination without abnormal findings", "cat": "Factors", "chapter": "XXI"},
    "Z23": {"desc": "Encounter for immunization", "cat": "Factors", "chapter": "XXI"},
    "Z79.4": {"desc": "Long term (current) use of insulin", "cat": "Factors", "chapter": "XXI"},
    "Z79.82": {"desc": "Long term (current) use of aspirin", "cat": "Factors", "chapter": "XXI"},
    "Z79.01": {"desc": "Long term (current) use of anticoagulants", "cat": "Factors", "chapter": "XXI"},
    "Z79.899": {"desc": "Other long term (current) drug therapy", "cat": "Factors", "chapter": "XXI"},
    "Z87.891": {"desc": "Personal history of nicotine dependence", "cat": "Factors", "chapter": "XXI"},
    "Z96.1": {"desc": "Presence of intraocular lens", "cat": "Factors", "chapter": "XXI"},
}

# Symptom to ICD-10 mapping for suggestions
SYMPTOM_CODE_MAPPING: Dict[str, List[Tuple[str, str, float]]] = {
    "headache": [
        ("R51", "Headache", 0.9),
        ("G43.909", "Migraine, unspecified", 0.7),
        ("G44.1", "Vascular headache", 0.5),
    ],
    "chest pain": [
        ("R07.9", "Chest pain, unspecified", 0.9),
        ("I21.9", "Acute myocardial infarction, unspecified", 0.3),
        ("I25.10", "Atherosclerotic heart disease", 0.4),
    ],
    "abdominal pain": [
        ("R10.9", "Unspecified abdominal pain", 0.9),
        ("R10.84", "Generalized abdominal pain", 0.8),
        ("K30", "Functional dyspepsia", 0.5),
    ],
    "cough": [
        ("R05", "Cough", 0.9),
        ("J06.9", "Acute upper respiratory infection", 0.7),
        ("J20.9", "Acute bronchitis", 0.5),
    ],
    "shortness of breath": [
        ("R06.02", "Shortness of breath", 0.9),
        ("J45.909", "Asthma, uncomplicated", 0.5),
        ("I50.9", "Heart failure, unspecified", 0.4),
    ],
    "fever": [
        ("R50.9", "Fever, unspecified", 0.9),
        ("A49.9", "Bacterial infection, unspecified", 0.5),
        ("B34.9", "Viral infection, unspecified", 0.6),
    ],
    "fatigue": [
        ("R53.83", "Other fatigue", 0.9),
        ("D64.9", "Anemia, unspecified", 0.4),
        ("E03.9", "Hypothyroidism, unspecified", 0.4),
    ],
    "dizziness": [
        ("R42", "Dizziness and giddiness", 0.9),
        ("I10", "Essential hypertension", 0.3),
    ],
    "nausea": [
        ("R11.2", "Nausea with vomiting", 0.9),
        ("K21.9", "GERD without esophagitis", 0.5),
        ("K29.70", "Gastritis, unspecified", 0.5),
    ],
    "back pain": [
        ("M54.5", "Low back pain", 0.9),
        ("M54.2", "Cervicalgia", 0.7),
        ("M62.830", "Muscle spasm of back", 0.6),
    ],
    "joint pain": [
        ("M25.50", "Pain in unspecified joint", 0.9),
        ("M19.90", "Osteoarthritis, unspecified", 0.6),
        ("M17.9", "Osteoarthritis of knee", 0.5),
    ],
    "sore throat": [
        ("J02.9", "Acute pharyngitis, unspecified", 0.9),
        ("J06.9", "Acute upper respiratory infection", 0.7),
    ],
    "rash": [
        ("L30.9", "Dermatitis, unspecified", 0.8),
        ("L50.9", "Urticaria, unspecified", 0.6),
        ("T78.40XA", "Allergy, unspecified", 0.5),
    ],
    "diabetes": [
        ("E11.9", "Type 2 diabetes without complications", 0.9),
        ("E11.65", "Type 2 diabetes with hyperglycemia", 0.7),
        ("E10.9", "Type 1 diabetes without complications", 0.5),
    ],
    "hypertension": [
        ("I10", "Essential hypertension", 0.95),
        ("I11.9", "Hypertensive heart disease", 0.5),
    ],
    "depression": [
        ("F32.9", "Major depressive disorder, single episode", 0.9),
        ("F33.0", "Major depressive disorder, recurrent", 0.7),
    ],
    "anxiety": [
        ("F41.9", "Anxiety disorder, unspecified", 0.9),
        ("F41.1", "Generalized anxiety disorder", 0.8),
    ],
    "urinary tract infection": [
        ("N39.0", "Urinary tract infection, site not specified", 0.95),
    ],
    "pneumonia": [
        ("J18.9", "Pneumonia, unspecified organism", 0.9),
    ],
    "asthma": [
        ("J45.909", "Asthma, uncomplicated", 0.9),
        ("J45.20", "Mild intermittent asthma", 0.7),
    ],
    "copd": [
        ("J44.9", "COPD, unspecified", 0.9),
        ("J44.1", "COPD with acute exacerbation", 0.7),
    ],
    "heart failure": [
        ("I50.9", "Heart failure, unspecified", 0.9),
        ("I50.22", "Chronic systolic heart failure", 0.7),
    ],
    "atrial fibrillation": [
        ("I48.91", "Unspecified atrial fibrillation", 0.9),
        ("I48.0", "Paroxysmal atrial fibrillation", 0.7),
    ],
}


class ICD10Service:
    """
    Provides ICD-10 diagnosis coding and lookup services.
    """
    
    def __init__(self):
        self.code_db = ICD10_DATABASE
        self.symptom_mapping = SYMPTOM_CODE_MAPPING
    
    def lookup_code(self, code: str) -> Optional[ICD10Code]:
        """Look up an ICD-10 code and return its details."""
        code_upper = code.upper().strip()
        
        if code_upper in self.code_db:
            data = self.code_db[code_upper]
            return ICD10Code(
                code=code_upper,
                description=data["desc"],
                category=data["cat"],
                chapter=data["chapter"],
                is_billable=True
            )
        
        return None
    
    def search_codes(
        self,
        query: str,
        limit: int = 10
    ) -> List[ICD10Code]:
        """Search for ICD-10 codes by description or code."""
        query_lower = query.lower().strip()
        results = []
        
        for code, data in self.code_db.items():
            # Match by code
            if query_lower in code.lower():
                results.append(ICD10Code(
                    code=code,
                    description=data["desc"],
                    category=data["cat"],
                    chapter=data["chapter"]
                ))
                continue
            
            # Match by description
            if query_lower in data["desc"].lower():
                results.append(ICD10Code(
                    code=code,
                    description=data["desc"],
                    category=data["cat"],
                    chapter=data["chapter"]
                ))
        
        return results[:limit]
    
    def suggest_codes(
        self,
        symptoms: str,
        diagnoses: Optional[List[str]] = None
    ) -> List[CodingSuggestion]:
        """
        Suggest ICD-10 codes based on symptoms and/or diagnoses.
        """
        suggestions = []
        symptoms_lower = symptoms.lower()
        
        # Check symptom mapping
        for symptom, codes in self.symptom_mapping.items():
            if symptom in symptoms_lower:
                for code, desc, confidence in codes:
                    # Avoid duplicates
                    if not any(s.code == code for s in suggestions):
                        alternatives = [(c, d) for c, d, _ in codes if c != code][:3]
                        suggestions.append(CodingSuggestion(
                            code=code,
                            description=desc,
                            confidence=confidence,
                            rationale=f"Based on symptom: {symptom}",
                            alternatives=alternatives
                        ))
        
        # Check diagnoses if provided
        if diagnoses:
            for diagnosis in diagnoses:
                diag_lower = diagnosis.lower()
                for symptom, codes in self.symptom_mapping.items():
                    if symptom in diag_lower:
                        for code, desc, confidence in codes:
                            if not any(s.code == code for s in suggestions):
                                suggestions.append(CodingSuggestion(
                                    code=code,
                                    description=desc,
                                    confidence=confidence * 1.1,  # Boost for explicit diagnosis
                                    rationale=f"Based on diagnosis: {diagnosis}",
                                    alternatives=[]
                                ))
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return suggestions[:10]
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate an ICD-10 code format and existence."""
        code_upper = code.upper().strip()
        
        # Check format
        pattern = r'^[A-Z]\d{2}(\.\d{1,4})?[A-Z]?$'
        if not re.match(pattern, code_upper):
            return {
                "valid": False,
                "exists": False,
                "reason": "Invalid ICD-10 code format"
            }
        
        # Check existence
        if code_upper in self.code_db:
            return {
                "valid": True,
                "exists": True,
                "code": code_upper,
                "description": self.code_db[code_upper]["desc"]
            }
        
        return {
            "valid": True,
            "exists": False,
            "reason": "Code format is valid but not found in database"
        }
    
    def get_category_codes(self, category: str) -> List[ICD10Code]:
        """Get all codes in a specific category."""
        results = []
        category_lower = category.lower()
        
        for code, data in self.code_db.items():
            if data["cat"].lower() == category_lower:
                results.append(ICD10Code(
                    code=code,
                    description=data["desc"],
                    category=data["cat"],
                    chapter=data["chapter"]
                ))
        
        return results
    
    def get_related_codes(self, code: str) -> List[ICD10Code]:
        """Get codes related to a given code (same category/chapter)."""
        code_upper = code.upper().strip()
        
        if code_upper not in self.code_db:
            return []
        
        base_data = self.code_db[code_upper]
        results = []
        
        # Get codes with same first 3 characters (same subcategory)
        base_prefix = code_upper[:3]
        for c, data in self.code_db.items():
            if c.startswith(base_prefix) and c != code_upper:
                results.append(ICD10Code(
                    code=c,
                    description=data["desc"],
                    category=data["cat"],
                    chapter=data["chapter"]
                ))
        
        return results[:10]
    
    def encode_encounter(
        self,
        chief_complaint: str,
        diagnoses: List[str],
        procedures: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate ICD-10 codes for a clinical encounter.
        """
        primary_suggestions = self.suggest_codes(chief_complaint, diagnoses)
        
        # Determine primary diagnosis
        primary_code = None
        secondary_codes = []
        
        if primary_suggestions:
            primary_code = {
                "code": primary_suggestions[0].code,
                "description": primary_suggestions[0].description,
                "confidence": primary_suggestions[0].confidence
            }
            
            for suggestion in primary_suggestions[1:5]:
                secondary_codes.append({
                    "code": suggestion.code,
                    "description": suggestion.description,
                    "confidence": suggestion.confidence
                })
        
        return {
            "primary_diagnosis": primary_code,
            "secondary_diagnoses": secondary_codes,
            "all_suggestions": [
                {
                    "code": s.code,
                    "description": s.description,
                    "confidence": s.confidence,
                    "rationale": s.rationale
                }
                for s in primary_suggestions
            ],
            "requires_review": any(s.confidence < 0.7 for s in primary_suggestions[:3]) if primary_suggestions else True,
            "notes": "Codes suggested by AI. Please verify accuracy before billing."
        }
    
    def get_available_categories(self) -> List[str]:
        """Return list of available ICD-10 categories."""
        categories = set()
        for data in self.code_db.values():
            categories.add(data["cat"])
        return sorted(list(categories))


# Singleton instance
_service = None


def get_icd10_service() -> ICD10Service:
    """Get singleton instance of ICD10Service."""
    global _service
    if _service is None:
        _service = ICD10Service()
    return _service
