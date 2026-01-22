"""
Contraindication Checker - Drug-Condition and Drug-Drug safety checks

Implements deterministic safety logic for medication safety.
"""
from typing import Optional, List, Dict, Any, Set, Tuple
from enum import Enum
from pydantic import BaseModel, Field


class ContraindicationSeverity(str, Enum):
    """Severity levels for contraindications"""
    ABSOLUTE = "absolute"        # Never use
    RELATIVE = "relative"        # Use with extreme caution
    CAUTION = "caution"          # Monitor closely
    INTERACTION = "interaction"  # Drug-drug interaction


class ContraindicationResult(BaseModel):
    """Result of contraindication check"""
    is_safe: bool
    severity: Optional[ContraindicationSeverity] = None
    contraindications: List[Dict[str, Any]] = Field(default_factory=list)
    interactions: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    alternatives: List[str] = Field(default_factory=list)
    monitoring_required: List[str] = Field(default_factory=list)
    
    # Explainability
    rules_triggered: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)


class DrugConditionRule(BaseModel):
    """Rule for drug-condition contraindication"""
    drug: str
    drug_class: Optional[str] = None
    condition: str
    severity: ContraindicationSeverity
    reason: str
    alternatives: List[str] = Field(default_factory=list)
    monitoring: Optional[str] = None


class DrugDrugRule(BaseModel):
    """Rule for drug-drug interaction"""
    drug1: str
    drug2: str
    severity: ContraindicationSeverity
    mechanism: str
    clinical_effect: str
    management: str


class ContraindicationChecker:
    """
    Deterministic contraindication and interaction checker
    
    Provides safety checks for:
    - Drug-condition contraindications
    - Drug-drug interactions
    - Drug-allergy checks
    - Age/pregnancy/lactation restrictions
    """
    
    def __init__(self):
        self.drug_condition_rules = self._load_drug_condition_rules()
        self.drug_drug_rules = self._load_drug_drug_rules()
        self.drug_classes = self._load_drug_classes()
    
    def _load_drug_classes(self) -> Dict[str, List[str]]:
        """Map drugs to their therapeutic classes"""
        return {
            "nsaid": ["ibuprofen", "naproxen", "aspirin", "celecoxib", "meloxicam", "diclofenac", "ketorolac"],
            "ace_inhibitor": ["lisinopril", "enalapril", "ramipril", "captopril", "benazepril"],
            "arb": ["losartan", "valsartan", "irbesartan", "olmesartan", "candesartan"],
            "beta_blocker": ["metoprolol", "atenolol", "propranolol", "carvedilol", "bisoprolol"],
            "calcium_channel_blocker": ["amlodipine", "diltiazem", "verapamil", "nifedipine"],
            "statin": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin"],
            "anticoagulant": ["warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban", "heparin", "enoxaparin"],
            "antiplatelet": ["aspirin", "clopidogrel", "prasugrel", "ticagrelor"],
            "ssri": ["sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram"],
            "snri": ["venlafaxine", "duloxetine", "desvenlafaxine"],
            "benzodiazepine": ["alprazolam", "lorazepam", "diazepam", "clonazepam"],
            "opioid": ["morphine", "oxycodone", "hydrocodone", "fentanyl", "tramadol", "codeine"],
            "fluoroquinolone": ["ciprofloxacin", "levofloxacin", "moxifloxacin"],
            "macrolide": ["azithromycin", "clarithromycin", "erythromycin"],
            "sulfonylurea": ["glipizide", "glyburide", "glimepiride"],
            "thiazolidinedione": ["pioglitazone", "rosiglitazone"],
            "ppi": ["omeprazole", "pantoprazole", "esomeprazole", "lansoprazole"],
            "h2_blocker": ["famotidine", "ranitidine", "cimetidine"],
            "diuretic_loop": ["furosemide", "bumetanide", "torsemide"],
            "diuretic_thiazide": ["hydrochlorothiazide", "chlorthalidone", "metolazone"],
            "diuretic_potassium_sparing": ["spironolactone", "eplerenone", "triamterene"],
        }
    
    def _load_drug_condition_rules(self) -> List[DrugConditionRule]:
        """Load drug-condition contraindication rules"""
        return [
            # NSAIDs
            DrugConditionRule(
                drug_class="nsaid",
                drug="",
                condition="kidney_disease",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="NSAIDs can cause acute kidney injury and worsen chronic kidney disease",
                alternatives=["acetaminophen"],
                monitoring="Renal function if must use",
            ),
            DrugConditionRule(
                drug_class="nsaid",
                drug="",
                condition="heart_failure",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="NSAIDs cause fluid retention and can precipitate heart failure exacerbation",
                alternatives=["acetaminophen"],
            ),
            DrugConditionRule(
                drug_class="nsaid",
                drug="",
                condition="gi_bleeding",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="NSAIDs increase risk of GI bleeding",
                alternatives=["acetaminophen"],
            ),
            DrugConditionRule(
                drug_class="nsaid",
                drug="",
                condition="pregnancy_third_trimester",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="NSAIDs can cause premature closure of ductus arteriosus",
                alternatives=["acetaminophen"],
            ),
            
            # ACE Inhibitors
            DrugConditionRule(
                drug_class="ace_inhibitor",
                drug="",
                condition="pregnancy",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="ACE inhibitors are teratogenic - can cause fetal renal dysgenesis",
                alternatives=["labetalol", "nifedipine", "methyldopa"],
            ),
            DrugConditionRule(
                drug_class="ace_inhibitor",
                drug="",
                condition="angioedema_history",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Risk of life-threatening angioedema",
                alternatives=["arb with caution", "calcium_channel_blocker"],
            ),
            DrugConditionRule(
                drug_class="ace_inhibitor",
                drug="",
                condition="bilateral_renal_artery_stenosis",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Can cause acute renal failure",
                alternatives=["calcium_channel_blocker"],
            ),
            
            # Beta Blockers
            DrugConditionRule(
                drug_class="beta_blocker",
                drug="",
                condition="asthma",
                severity=ContraindicationSeverity.RELATIVE,
                reason="Can cause bronchospasm - use cardioselective agents with caution",
                alternatives=["calcium_channel_blocker"],
                monitoring="Respiratory status",
            ),
            DrugConditionRule(
                drug_class="beta_blocker",
                drug="",
                condition="bradycardia",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Can worsen bradycardia",
                alternatives=["calcium_channel_blocker (dihydropyridine)"],
            ),
            DrugConditionRule(
                drug_class="beta_blocker",
                drug="",
                condition="heart_block",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Can worsen AV block",
                alternatives=["dihydropyridine calcium_channel_blocker"],
            ),
            
            # Metformin
            DrugConditionRule(
                drug="metformin",
                condition="kidney_disease_severe",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Risk of lactic acidosis with eGFR <30",
                alternatives=["insulin", "sulfonylurea"],
            ),
            DrugConditionRule(
                drug="metformin",
                condition="liver_disease_severe",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Risk of lactic acidosis",
                alternatives=["insulin"],
            ),
            
            # Statins
            DrugConditionRule(
                drug_class="statin",
                drug="",
                condition="active_liver_disease",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Can worsen liver function",
                alternatives=["ezetimibe", "pcsk9_inhibitor"],
            ),
            DrugConditionRule(
                drug_class="statin",
                drug="",
                condition="pregnancy",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Teratogenic - cholesterol needed for fetal development",
                alternatives=["discontinue during pregnancy"],
            ),
            
            # Anticoagulants
            DrugConditionRule(
                drug_class="anticoagulant",
                drug="",
                condition="active_bleeding",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Will worsen bleeding",
                alternatives=["hold anticoagulation"],
            ),
            DrugConditionRule(
                drug="warfarin",
                condition="pregnancy",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Teratogenic - warfarin embryopathy",
                alternatives=["heparin", "enoxaparin"],
            ),
            
            # Opioids
            DrugConditionRule(
                drug_class="opioid",
                drug="",
                condition="respiratory_depression",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Can worsen respiratory depression",
                alternatives=["non-opioid analgesics"],
            ),
            DrugConditionRule(
                drug_class="opioid",
                drug="",
                condition="paralytic_ileus",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Can worsen ileus",
                alternatives=["non-opioid analgesics"],
            ),
            
            # Fluoroquinolones
            DrugConditionRule(
                drug_class="fluoroquinolone",
                drug="",
                condition="myasthenia_gravis",
                severity=ContraindicationSeverity.ABSOLUTE,
                reason="Can exacerbate muscle weakness",
                alternatives=["other antibiotic classes"],
            ),
            DrugConditionRule(
                drug_class="fluoroquinolone",
                drug="",
                condition="tendon_disorder",
                severity=ContraindicationSeverity.RELATIVE,
                reason="Increased risk of tendon rupture",
                alternatives=["other antibiotic classes"],
            ),
        ]
    
    def _load_drug_drug_rules(self) -> List[DrugDrugRule]:
        """Load drug-drug interaction rules"""
        return [
            # Warfarin interactions
            DrugDrugRule(
                drug1="warfarin",
                drug2="aspirin",
                severity=ContraindicationSeverity.CAUTION,
                mechanism="Additive anticoagulant/antiplatelet effect",
                clinical_effect="Increased bleeding risk",
                management="Monitor INR closely, consider GI protection",
            ),
            DrugDrugRule(
                drug1="warfarin",
                drug2="nsaid",
                severity=ContraindicationSeverity.RELATIVE,
                mechanism="NSAIDs inhibit platelet function and can cause GI bleeding",
                clinical_effect="Significantly increased bleeding risk",
                management="Avoid if possible, use acetaminophen instead",
            ),
            
            # ACE-I + K-sparing diuretic
            DrugDrugRule(
                drug1="ace_inhibitor",
                drug2="diuretic_potassium_sparing",
                severity=ContraindicationSeverity.CAUTION,
                mechanism="Both cause potassium retention",
                clinical_effect="Hyperkalemia risk",
                management="Monitor potassium levels closely",
            ),
            
            # Serotonin syndrome combinations
            DrugDrugRule(
                drug1="ssri",
                drug2="tramadol",
                severity=ContraindicationSeverity.RELATIVE,
                mechanism="Both increase serotonin",
                clinical_effect="Serotonin syndrome risk",
                management="Use with caution, monitor for symptoms",
            ),
            DrugDrugRule(
                drug1="ssri",
                drug2="maoi",
                severity=ContraindicationSeverity.ABSOLUTE,
                mechanism="Massive serotonin increase",
                clinical_effect="Life-threatening serotonin syndrome",
                management="Contraindicated - 14 day washout required",
            ),
            
            # QT prolongation
            DrugDrugRule(
                drug1="fluoroquinolone",
                drug2="macrolide",
                severity=ContraindicationSeverity.RELATIVE,
                mechanism="Both prolong QT interval",
                clinical_effect="Risk of Torsades de Pointes",
                management="Avoid combination, monitor ECG if necessary",
            ),
            
            # Statin + CYP3A4 inhibitors
            DrugDrugRule(
                drug1="simvastatin",
                drug2="clarithromycin",
                severity=ContraindicationSeverity.ABSOLUTE,
                mechanism="CYP3A4 inhibition increases statin levels",
                clinical_effect="Rhabdomyolysis risk",
                management="Use pravastatin or rosuvastatin instead",
            ),
            
            # Opioid + Benzodiazepine
            DrugDrugRule(
                drug1="opioid",
                drug2="benzodiazepine",
                severity=ContraindicationSeverity.RELATIVE,
                mechanism="Additive CNS depression",
                clinical_effect="Respiratory depression, overdose death",
                management="Avoid if possible, use lowest effective doses",
            ),
            
            # Methotrexate + NSAIDs
            DrugDrugRule(
                drug1="methotrexate",
                drug2="nsaid",
                severity=ContraindicationSeverity.RELATIVE,
                mechanism="NSAIDs reduce methotrexate clearance",
                clinical_effect="Methotrexate toxicity",
                management="Avoid NSAIDs or use with extreme caution",
            ),
            
            # Lithium + NSAIDs
            DrugDrugRule(
                drug1="lithium",
                drug2="nsaid",
                severity=ContraindicationSeverity.CAUTION,
                mechanism="NSAIDs reduce lithium clearance",
                clinical_effect="Lithium toxicity",
                management="Monitor lithium levels, consider dose reduction",
            ),
            
            # Digoxin interactions
            DrugDrugRule(
                drug1="digoxin",
                drug2="amiodarone",
                severity=ContraindicationSeverity.CAUTION,
                mechanism="Amiodarone inhibits P-glycoprotein",
                clinical_effect="Increased digoxin levels, toxicity",
                management="Reduce digoxin dose by 50%, monitor levels",
            ),
        ]
    
    def _get_drug_class(self, drug: str) -> Optional[str]:
        """Get the therapeutic class of a drug"""
        drug_lower = drug.lower()
        for drug_class, drugs in self.drug_classes.items():
            if drug_lower in drugs:
                return drug_class
        return None
    
    def _normalize_drug(self, drug: str) -> str:
        """Normalize drug name"""
        return drug.lower().strip().replace("-", "").replace(" ", "")
    
    def _normalize_condition(self, condition: str) -> str:
        """Normalize condition name"""
        return condition.lower().strip().replace("-", "_").replace(" ", "_")
    
    def check_drug_condition(
        self,
        drug: str,
        conditions: List[str],
    ) -> ContraindicationResult:
        """Check drug against patient conditions"""
        result = ContraindicationResult(is_safe=True)
        
        drug_normalized = self._normalize_drug(drug)
        drug_class = self._get_drug_class(drug_normalized)
        conditions_normalized = [self._normalize_condition(c) for c in conditions]
        
        for rule in self.drug_condition_rules:
            # Check if rule applies to this drug
            rule_applies = False
            if rule.drug and self._normalize_drug(rule.drug) == drug_normalized:
                rule_applies = True
            elif rule.drug_class and rule.drug_class == drug_class:
                rule_applies = True
            
            if not rule_applies:
                continue
            
            # Check if patient has the condition
            rule_condition = self._normalize_condition(rule.condition)
            for patient_condition in conditions_normalized:
                if rule_condition in patient_condition or patient_condition in rule_condition:
                    result.contraindications.append({
                        "drug": drug,
                        "condition": rule.condition,
                        "severity": rule.severity.value,
                        "reason": rule.reason,
                    })
                    result.alternatives.extend(rule.alternatives)
                    if rule.monitoring:
                        result.monitoring_required.append(rule.monitoring)
                    result.rules_triggered.append(f"drug_condition_{drug}_{rule.condition}")
                    result.reasoning.append(f"{drug} contraindicated with {rule.condition}: {rule.reason}")
                    
                    if rule.severity == ContraindicationSeverity.ABSOLUTE:
                        result.is_safe = False
                        result.severity = ContraindicationSeverity.ABSOLUTE
                    elif rule.severity == ContraindicationSeverity.RELATIVE and result.is_safe:
                        result.severity = ContraindicationSeverity.RELATIVE
                        result.warnings.append(f"Use {drug} with caution due to {rule.condition}")
        
        return result
    
    def check_drug_drug(
        self,
        drugs: List[str],
    ) -> ContraindicationResult:
        """Check for drug-drug interactions"""
        result = ContraindicationResult(is_safe=True)
        
        drugs_normalized = [self._normalize_drug(d) for d in drugs]
        drug_classes = {d: self._get_drug_class(d) for d in drugs_normalized}
        
        # Check each pair
        for i, drug1 in enumerate(drugs_normalized):
            for drug2 in drugs_normalized[i+1:]:
                for rule in self.drug_drug_rules:
                    rule_drug1 = self._normalize_drug(rule.drug1)
                    rule_drug2 = self._normalize_drug(rule.drug2)
                    
                    # Check direct drug match or class match
                    match = False
                    if (drug1 == rule_drug1 and drug2 == rule_drug2) or \
                       (drug1 == rule_drug2 and drug2 == rule_drug1):
                        match = True
                    elif (drug_classes.get(drug1) == rule_drug1 and drug2 == rule_drug2) or \
                         (drug_classes.get(drug1) == rule_drug2 and drug2 == rule_drug1):
                        match = True
                    elif (drug1 == rule_drug1 and drug_classes.get(drug2) == rule_drug2) or \
                         (drug1 == rule_drug2 and drug_classes.get(drug2) == rule_drug1):
                        match = True
                    elif (drug_classes.get(drug1) == rule_drug1 and drug_classes.get(drug2) == rule_drug2) or \
                         (drug_classes.get(drug1) == rule_drug2 and drug_classes.get(drug2) == rule_drug1):
                        match = True
                    
                    if match:
                        result.interactions.append({
                            "drug1": drug1,
                            "drug2": drug2,
                            "severity": rule.severity.value,
                            "mechanism": rule.mechanism,
                            "clinical_effect": rule.clinical_effect,
                            "management": rule.management,
                        })
                        result.rules_triggered.append(f"drug_drug_{drug1}_{drug2}")
                        result.reasoning.append(
                            f"Interaction between {drug1} and {drug2}: {rule.clinical_effect}"
                        )
                        
                        if rule.severity == ContraindicationSeverity.ABSOLUTE:
                            result.is_safe = False
                            result.severity = ContraindicationSeverity.ABSOLUTE
                        elif rule.severity == ContraindicationSeverity.RELATIVE:
                            if result.severity != ContraindicationSeverity.ABSOLUTE:
                                result.severity = ContraindicationSeverity.RELATIVE
                            result.warnings.append(rule.management)
        
        return result
    
    def check_all(
        self,
        drug: str,
        conditions: List[str],
        current_medications: List[str],
        allergies: List[str] = None,
        age: Optional[int] = None,
        is_pregnant: bool = False,
        is_breastfeeding: bool = False,
    ) -> ContraindicationResult:
        """Comprehensive safety check for a drug"""
        allergies = allergies or []
        
        # Start with drug-condition check
        result = self.check_drug_condition(drug, conditions)
        
        # Add pregnancy/breastfeeding as conditions
        if is_pregnant:
            pregnancy_result = self.check_drug_condition(drug, ["pregnancy"])
            result.contraindications.extend(pregnancy_result.contraindications)
            result.warnings.extend(pregnancy_result.warnings)
            result.alternatives.extend(pregnancy_result.alternatives)
            if not pregnancy_result.is_safe:
                result.is_safe = False
        
        # Check drug-drug interactions
        all_drugs = current_medications + [drug]
        interaction_result = self.check_drug_drug(all_drugs)
        result.interactions = interaction_result.interactions
        result.warnings.extend(interaction_result.warnings)
        if not interaction_result.is_safe:
            result.is_safe = False
        
        # Check allergies
        drug_normalized = self._normalize_drug(drug)
        for allergy in allergies:
            if self._normalize_drug(allergy) == drug_normalized:
                result.is_safe = False
                result.severity = ContraindicationSeverity.ABSOLUTE
                result.contraindications.append({
                    "drug": drug,
                    "condition": f"allergy to {allergy}",
                    "severity": "absolute",
                    "reason": "Patient has documented allergy",
                })
                result.rules_triggered.append(f"allergy_{drug}")
        
        # Remove duplicates
        result.alternatives = list(set(result.alternatives))
        result.warnings = list(set(result.warnings))
        
        return result
