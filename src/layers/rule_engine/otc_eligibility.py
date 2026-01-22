"""
OTC Eligibility Engine - Determines if patient can safely use OTC medications

Based on pharmacy practice guidelines for responsible OTC recommendations.
"""
from typing import Optional, List, Dict, Any, Set
from enum import Enum
from pydantic import BaseModel, Field


class OTCDecisionType(str, Enum):
    """OTC decision types"""
    ELIGIBLE = "eligible"
    ELIGIBLE_WITH_CAUTION = "eligible_with_caution"
    REFER_TO_PHARMACIST = "refer_to_pharmacist"
    REFER_TO_DOCTOR = "refer_to_doctor"
    NOT_ELIGIBLE = "not_eligible"


class OTCDecision(BaseModel):
    """OTC eligibility decision"""
    decision: OTCDecisionType
    eligible_products: List[str] = Field(default_factory=list)
    contraindicated_products: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    precautions: List[str] = Field(default_factory=list)
    dosing_guidance: Optional[str] = None
    max_duration_days: Optional[int] = None
    referral_reason: Optional[str] = None
    
    # Explainability
    rules_applied: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)


class OTCProduct(BaseModel):
    """OTC product definition"""
    id: str
    name: str
    active_ingredients: List[str]
    therapeutic_class: str
    indications: List[str]
    contraindications: List[str]
    drug_interactions: List[str]
    age_restrictions: Dict[str, int] = Field(default_factory=dict)  # min_age, max_age
    pregnancy_safe: bool = False
    breastfeeding_safe: bool = False
    max_daily_dose: Optional[str] = None
    max_duration_days: int = 7
    warnings: List[str] = Field(default_factory=list)


class OTCEligibilityEngine:
    """
    Deterministic OTC eligibility assessment
    
    Evaluates whether a patient can safely use OTC medications
    based on their symptoms, conditions, and current medications.
    """
    
    # Conditions that always require doctor referral
    REFERRAL_CONDITIONS: Set[str] = {
        "pregnancy",
        "liver_disease", "hepatic_impairment", "cirrhosis",
        "kidney_disease", "renal_impairment", "dialysis",
        "heart_failure", "cardiac_arrhythmia",
        "uncontrolled_hypertension",
        "bleeding_disorder", "on_anticoagulants",
        "immunocompromised", "hiv", "chemotherapy",
        "organ_transplant",
    }
    
    # Age restrictions for common OTC categories
    AGE_RESTRICTIONS = {
        "cough_suppressant": {"min_age": 4},
        "decongestant": {"min_age": 6},
        "antihistamine": {"min_age": 2},
        "nsaid": {"min_age": 12},
        "aspirin": {"min_age": 18},  # Reye's syndrome risk
        "antidiarrheal": {"min_age": 6},
        "laxative": {"min_age": 6},
        "sleep_aid": {"min_age": 18},
        "acid_reducer": {"min_age": 12},
    }
    
    # Common drug interactions
    DRUG_INTERACTIONS = {
        "nsaid": ["warfarin", "aspirin", "ssri", "lithium", "methotrexate", "ace_inhibitor"],
        "acetaminophen": ["warfarin", "alcohol"],
        "antihistamine": ["sedatives", "alcohol", "maoi"],
        "decongestant": ["maoi", "beta_blocker", "antihypertensive"],
        "antacid": ["tetracycline", "fluoroquinolone", "iron"],
    }
    
    # Symptom to OTC category mapping
    SYMPTOM_OTC_MAP = {
        "headache": ["acetaminophen", "ibuprofen", "aspirin"],
        "fever": ["acetaminophen", "ibuprofen"],
        "cold": ["decongestant", "antihistamine", "cough_suppressant"],
        "cough": ["cough_suppressant", "expectorant"],
        "allergies": ["antihistamine", "nasal_corticosteroid"],
        "heartburn": ["antacid", "h2_blocker", "ppi"],
        "diarrhea": ["loperamide", "bismuth_subsalicylate"],
        "constipation": ["fiber", "osmotic_laxative", "stimulant_laxative"],
        "muscle_pain": ["acetaminophen", "ibuprofen", "topical_analgesic"],
        "insomnia": ["diphenhydramine", "melatonin"],
        "nausea": ["bismuth_subsalicylate", "ginger"],
    }
    
    def __init__(self):
        self.products: Dict[str, OTCProduct] = self._load_default_products()
    
    def _load_default_products(self) -> Dict[str, OTCProduct]:
        """Load default OTC product database"""
        return {
            "acetaminophen": OTCProduct(
                id="acetaminophen",
                name="Acetaminophen (Tylenol)",
                active_ingredients=["acetaminophen"],
                therapeutic_class="analgesic",
                indications=["headache", "fever", "pain"],
                contraindications=["liver_disease", "alcohol_use_disorder"],
                drug_interactions=["warfarin"],
                age_restrictions={"min_age": 0},
                pregnancy_safe=True,
                breastfeeding_safe=True,
                max_daily_dose="3000mg (2000mg if liver concerns)",
                max_duration_days=10,
                warnings=["Do not exceed recommended dose", "Avoid alcohol"],
            ),
            "ibuprofen": OTCProduct(
                id="ibuprofen",
                name="Ibuprofen (Advil, Motrin)",
                active_ingredients=["ibuprofen"],
                therapeutic_class="nsaid",
                indications=["headache", "fever", "pain", "inflammation"],
                contraindications=[
                    "kidney_disease", "heart_failure", "gi_bleeding",
                    "aspirin_allergy", "third_trimester_pregnancy"
                ],
                drug_interactions=["warfarin", "aspirin", "ace_inhibitor", "lithium"],
                age_restrictions={"min_age": 6},
                pregnancy_safe=False,
                breastfeeding_safe=True,
                max_daily_dose="1200mg OTC",
                max_duration_days=10,
                warnings=["Take with food", "May increase cardiovascular risk"],
            ),
            "diphenhydramine": OTCProduct(
                id="diphenhydramine",
                name="Diphenhydramine (Benadryl)",
                active_ingredients=["diphenhydramine"],
                therapeutic_class="antihistamine",
                indications=["allergies", "insomnia", "cold"],
                contraindications=["glaucoma", "urinary_retention", "dementia"],
                drug_interactions=["sedatives", "alcohol", "maoi"],
                age_restrictions={"min_age": 2, "max_age": 65},
                pregnancy_safe=True,
                breastfeeding_safe=False,
                max_daily_dose="300mg",
                max_duration_days=7,
                warnings=["Causes drowsiness", "Avoid in elderly"],
            ),
            "pseudoephedrine": OTCProduct(
                id="pseudoephedrine",
                name="Pseudoephedrine (Sudafed)",
                active_ingredients=["pseudoephedrine"],
                therapeutic_class="decongestant",
                indications=["nasal_congestion", "sinus_congestion"],
                contraindications=[
                    "hypertension", "heart_disease", "hyperthyroidism",
                    "diabetes", "glaucoma", "prostate_enlargement"
                ],
                drug_interactions=["maoi", "beta_blocker"],
                age_restrictions={"min_age": 6},
                pregnancy_safe=False,
                breastfeeding_safe=False,
                max_daily_dose="240mg",
                max_duration_days=7,
                warnings=["May increase blood pressure", "Behind pharmacy counter"],
            ),
            "omeprazole": OTCProduct(
                id="omeprazole",
                name="Omeprazole (Prilosec OTC)",
                active_ingredients=["omeprazole"],
                therapeutic_class="ppi",
                indications=["heartburn", "gerd"],
                contraindications=[],
                drug_interactions=["clopidogrel", "methotrexate"],
                age_restrictions={"min_age": 18},
                pregnancy_safe=True,
                breastfeeding_safe=True,
                max_daily_dose="20mg",
                max_duration_days=14,
                warnings=["Not for immediate relief", "14-day treatment course"],
            ),
            "loperamide": OTCProduct(
                id="loperamide",
                name="Loperamide (Imodium)",
                active_ingredients=["loperamide"],
                therapeutic_class="antidiarrheal",
                indications=["diarrhea"],
                contraindications=["bloody_diarrhea", "fever_with_diarrhea", "c_diff"],
                drug_interactions=[],
                age_restrictions={"min_age": 6},
                pregnancy_safe=True,
                breastfeeding_safe=True,
                max_daily_dose="8mg OTC",
                max_duration_days=2,
                warnings=["Do not use if fever or bloody stool"],
            ),
        }
    
    def assess_eligibility(
        self,
        symptoms: List[str],
        age: int,
        conditions: List[str],
        current_medications: List[str],
        is_pregnant: bool = False,
        is_breastfeeding: bool = False,
        allergies: List[str] = None,
    ) -> OTCDecision:
        """
        Assess OTC eligibility for a patient
        
        Returns deterministic, explainable decision
        """
        allergies = allergies or []
        decision = OTCDecision(decision=OTCDecisionType.ELIGIBLE)
        
        # Normalize inputs
        conditions_lower = set(c.lower().replace(" ", "_") for c in conditions)
        meds_lower = set(m.lower().replace(" ", "_") for m in current_medications)
        symptoms_lower = set(s.lower().replace(" ", "_") for s in symptoms)
        allergies_lower = set(a.lower() for a in allergies)
        
        # Check for referral conditions
        referral_conditions = conditions_lower.intersection(self.REFERRAL_CONDITIONS)
        if referral_conditions:
            decision.decision = OTCDecisionType.REFER_TO_DOCTOR
            decision.referral_reason = f"Medical conditions require doctor consultation: {', '.join(referral_conditions)}"
            decision.rules_applied.append("referral_condition_check")
            decision.reasoning.append(f"Patient has conditions that require medical supervision: {referral_conditions}")
            return decision
        
        # Check pregnancy
        if is_pregnant:
            decision.decision = OTCDecisionType.REFER_TO_DOCTOR
            decision.referral_reason = "Pregnancy requires doctor consultation before OTC use"
            decision.rules_applied.append("pregnancy_check")
            decision.reasoning.append("Pregnant patients should consult healthcare provider")
            return decision
        
        # Find eligible products for symptoms
        eligible_products = []
        for symptom in symptoms_lower:
            if symptom in self.SYMPTOM_OTC_MAP:
                for product_class in self.SYMPTOM_OTC_MAP[symptom]:
                    for product_id, product in self.products.items():
                        if product.therapeutic_class == product_class or product_id == product_class:
                            if product_id not in [p for p in eligible_products]:
                                eligible_products.append(product_id)
        
        # Filter products based on patient factors
        final_eligible = []
        for product_id in eligible_products:
            product = self.products.get(product_id)
            if not product:
                continue
            
            is_eligible = True
            product_warnings = []
            
            # Check age restrictions
            if "min_age" in product.age_restrictions:
                if age < product.age_restrictions["min_age"]:
                    is_eligible = False
                    decision.contraindicated_products.append(
                        f"{product.name}: Age restriction (min {product.age_restrictions['min_age']} years)"
                    )
                    decision.rules_applied.append(f"age_restriction_{product_id}")
            
            if "max_age" in product.age_restrictions:
                if age > product.age_restrictions["max_age"]:
                    product_warnings.append(f"Use with caution in patients over {product.age_restrictions['max_age']}")
            
            # Check contraindications
            product_contras = set(c.lower().replace(" ", "_") for c in product.contraindications)
            matching_contras = conditions_lower.intersection(product_contras)
            if matching_contras:
                is_eligible = False
                decision.contraindicated_products.append(
                    f"{product.name}: Contraindicated with {', '.join(matching_contras)}"
                )
                decision.rules_applied.append(f"contraindication_{product_id}")
            
            # Check drug interactions
            product_interactions = set(i.lower().replace(" ", "_") for i in product.drug_interactions)
            matching_interactions = meds_lower.intersection(product_interactions)
            if matching_interactions:
                product_warnings.append(
                    f"Potential interaction with: {', '.join(matching_interactions)}"
                )
                decision.rules_applied.append(f"interaction_check_{product_id}")
            
            # Check allergies
            for ingredient in product.active_ingredients:
                if ingredient.lower() in allergies_lower:
                    is_eligible = False
                    decision.contraindicated_products.append(
                        f"{product.name}: Allergy to {ingredient}"
                    )
            
            # Check breastfeeding
            if is_breastfeeding and not product.breastfeeding_safe:
                product_warnings.append("Use with caution while breastfeeding")
            
            if is_eligible:
                final_eligible.append(product.name)
                decision.warnings.extend(product_warnings)
                decision.warnings.extend(product.warnings)
        
        decision.eligible_products = final_eligible
        
        # Determine final decision
        if not final_eligible:
            decision.decision = OTCDecisionType.REFER_TO_PHARMACIST
            decision.referral_reason = "No suitable OTC products identified"
            decision.reasoning.append("Could not find appropriate OTC options for this patient")
        elif decision.warnings:
            decision.decision = OTCDecisionType.ELIGIBLE_WITH_CAUTION
            decision.reasoning.append("OTC products available but use with caution")
        else:
            decision.decision = OTCDecisionType.ELIGIBLE
            decision.reasoning.append("Patient eligible for OTC treatment")
        
        # Set max duration
        if final_eligible:
            min_duration = min(
                self.products[p.split(" ")[0].lower()].max_duration_days
                for p in final_eligible
                if p.split(" ")[0].lower() in self.products
            ) if any(p.split(" ")[0].lower() in self.products for p in final_eligible) else 7
            decision.max_duration_days = min_duration
            decision.precautions.append(
                f"Do not use for more than {min_duration} days without consulting a doctor"
            )
        
        return decision
    
    def get_product_info(self, product_id: str) -> Optional[OTCProduct]:
        """Get detailed product information"""
        return self.products.get(product_id.lower())
    
    def add_product(self, product: OTCProduct) -> None:
        """Add a new OTC product to the database"""
        self.products[product.id.lower()] = product
    
    def get_dosing_guidance(
        self,
        product_id: str,
        age: int,
        weight_kg: Optional[float] = None,
    ) -> Optional[str]:
        """Get age/weight-appropriate dosing guidance"""
        product = self.products.get(product_id.lower())
        if not product:
            return None
        
        # Basic dosing - in production, this would be more sophisticated
        if product_id == "acetaminophen":
            if age < 12:
                if weight_kg:
                    dose_mg = min(weight_kg * 15, 1000)
                    return f"{dose_mg:.0f}mg every 4-6 hours, max 5 doses/day"
                return "Use pediatric formulation, dose by weight"
            return "650-1000mg every 4-6 hours, max 3000mg/day"
        
        elif product_id == "ibuprofen":
            if age < 12:
                if weight_kg:
                    dose_mg = min(weight_kg * 10, 400)
                    return f"{dose_mg:.0f}mg every 6-8 hours with food"
                return "Use pediatric formulation, dose by weight"
            return "200-400mg every 4-6 hours with food, max 1200mg/day"
        
        return product.max_daily_dose
