"""
Red Flag Detector - Identifies dangerous symptoms requiring immediate attention

Implements evidence-based red flag detection for patient safety.
"""
from typing import Optional, List, Dict, Any, Set
from enum import Enum
from pydantic import BaseModel, Field


class RedFlagSeverity(str, Enum):
    """Red flag severity levels"""
    CRITICAL = "critical"    # Immediate emergency
    HIGH = "high"            # Urgent - same day
    MODERATE = "moderate"    # Soon - within 48 hours


class RedFlag(BaseModel):
    """A detected red flag"""
    symptom: str
    severity: RedFlagSeverity
    possible_conditions: List[str]
    action_required: str
    time_sensitivity: str
    additional_questions: List[str] = Field(default_factory=list)


class RedFlagResult(BaseModel):
    """Result of red flag detection"""
    has_red_flags: bool = False
    red_flags: List[RedFlag] = Field(default_factory=list)
    highest_severity: Optional[RedFlagSeverity] = None
    immediate_action: Optional[str] = None
    referral_recommendation: Optional[str] = None
    
    # Explainability
    rules_triggered: List[str] = Field(default_factory=list)


class RedFlagRule(BaseModel):
    """Red flag detection rule"""
    id: str
    keywords: List[str]  # Symptoms/phrases to match
    severity: RedFlagSeverity
    possible_conditions: List[str]
    action: str
    time_sensitivity: str
    body_system: Optional[str] = None
    age_specific: Optional[Dict[str, Any]] = None
    additional_questions: List[str] = Field(default_factory=list)


class RedFlagDetector:
    """
    Deterministic red flag detection engine
    
    Identifies symptoms and presentations that require
    immediate medical attention or urgent referral.
    """
    
    def __init__(self):
        self.rules = self._load_rules()
    
    def _load_rules(self) -> List[RedFlagRule]:
        """Load red flag detection rules"""
        return [
            # Cardiovascular Red Flags
            RedFlagRule(
                id="chest_pain_cardiac",
                keywords=["chest pain", "chest pressure", "chest tightness", "crushing chest"],
                severity=RedFlagSeverity.CRITICAL,
                possible_conditions=["Myocardial infarction", "Unstable angina", "Aortic dissection"],
                action="Call 911 immediately",
                time_sensitivity="Minutes matter - every minute of delay increases mortality",
                body_system="cardiovascular",
                additional_questions=[
                    "Does the pain radiate to your arm, jaw, or back?",
                    "Are you sweating or feeling nauseous?",
                    "Do you have shortness of breath?",
                ],
            ),
            RedFlagRule(
                id="sudden_severe_headache",
                keywords=["worst headache", "thunderclap headache", "sudden severe headache"],
                severity=RedFlagSeverity.CRITICAL,
                possible_conditions=["Subarachnoid hemorrhage", "Cerebral aneurysm rupture"],
                action="Call 911 immediately",
                time_sensitivity="Immediate - potential brain bleed",
                body_system="neurological",
                additional_questions=[
                    "Did the headache start suddenly or gradually?",
                    "Do you have neck stiffness?",
                    "Have you had any loss of consciousness?",
                ],
            ),
            RedFlagRule(
                id="stroke_symptoms",
                keywords=[
                    "facial drooping", "arm weakness", "speech difficulty", "slurred speech",
                    "sudden numbness", "sudden confusion", "sudden vision loss",
                    "sudden trouble walking", "sudden dizziness"
                ],
                severity=RedFlagSeverity.CRITICAL,
                possible_conditions=["Stroke", "TIA"],
                action="Call 911 immediately - time is brain",
                time_sensitivity="Treatment window is 3-4.5 hours for tPA",
                body_system="neurological",
                additional_questions=[
                    "When did symptoms start?",
                    "Can you smile and show your teeth?",
                    "Can you raise both arms?",
                ],
            ),
            RedFlagRule(
                id="breathing_emergency",
                keywords=[
                    "cannot breathe", "severe shortness of breath", "gasping for air",
                    "blue lips", "cyanosis", "choking"
                ],
                severity=RedFlagSeverity.CRITICAL,
                possible_conditions=["Respiratory failure", "Pulmonary embolism", "Anaphylaxis", "Airway obstruction"],
                action="Call 911 immediately",
                time_sensitivity="Immediate - life-threatening",
                body_system="respiratory",
            ),
            RedFlagRule(
                id="anaphylaxis",
                keywords=[
                    "throat swelling", "tongue swelling", "difficulty swallowing",
                    "hives with breathing difficulty", "allergic reaction severe"
                ],
                severity=RedFlagSeverity.CRITICAL,
                possible_conditions=["Anaphylaxis"],
                action="Use epinephrine if available, call 911",
                time_sensitivity="Immediate - can be fatal within minutes",
                body_system="immune",
            ),
            RedFlagRule(
                id="severe_bleeding",
                keywords=["uncontrolled bleeding", "severe bleeding", "hemorrhage", "blood won't stop"],
                severity=RedFlagSeverity.CRITICAL,
                possible_conditions=["Hemorrhage", "Trauma", "Coagulopathy"],
                action="Apply pressure, call 911",
                time_sensitivity="Immediate",
                body_system="hematologic",
            ),
            RedFlagRule(
                id="loss_of_consciousness",
                keywords=["passed out", "fainted", "unconscious", "unresponsive", "syncope"],
                severity=RedFlagSeverity.CRITICAL,
                possible_conditions=["Cardiac arrhythmia", "Seizure", "Hypoglycemia", "Stroke"],
                action="Call 911 if not immediately recovered",
                time_sensitivity="Immediate evaluation needed",
                body_system="neurological",
            ),
            RedFlagRule(
                id="seizure",
                keywords=["seizure", "convulsion", "fitting", "shaking uncontrollably"],
                severity=RedFlagSeverity.CRITICAL,
                possible_conditions=["Epilepsy", "Brain lesion", "Metabolic disorder", "Drug toxicity"],
                action="Protect from injury, call 911 if first seizure or lasting >5 min",
                time_sensitivity="Immediate if prolonged or first occurrence",
                body_system="neurological",
            ),
            
            # High Severity Red Flags
            RedFlagRule(
                id="meningitis_signs",
                keywords=["stiff neck with fever", "neck stiffness fever", "photophobia fever", "rash with fever"],
                severity=RedFlagSeverity.HIGH,
                possible_conditions=["Meningitis", "Encephalitis"],
                action="Seek emergency care immediately",
                time_sensitivity="Hours - requires urgent antibiotics",
                body_system="infectious",
                additional_questions=[
                    "Do you have a rash that doesn't fade when pressed?",
                    "Is light bothering your eyes?",
                    "Have you been around anyone with meningitis?",
                ],
            ),
            RedFlagRule(
                id="appendicitis_signs",
                keywords=[
                    "right lower abdominal pain", "pain started around navel moved to right",
                    "rebound tenderness", "pain worse with movement"
                ],
                severity=RedFlagSeverity.HIGH,
                possible_conditions=["Appendicitis"],
                action="Seek emergency evaluation",
                time_sensitivity="Hours - risk of rupture",
                body_system="gastrointestinal",
                additional_questions=[
                    "Did the pain start around your belly button?",
                    "Does it hurt more when you release pressure on your abdomen?",
                    "Do you have fever or loss of appetite?",
                ],
            ),
            RedFlagRule(
                id="testicular_torsion",
                keywords=["sudden testicular pain", "testicle pain severe", "twisted testicle"],
                severity=RedFlagSeverity.HIGH,
                possible_conditions=["Testicular torsion"],
                action="Seek emergency care immediately",
                time_sensitivity="6 hours to save testicle",
                body_system="genitourinary",
            ),
            RedFlagRule(
                id="ectopic_pregnancy",
                keywords=[
                    "pregnancy bleeding", "pregnant with abdominal pain",
                    "missed period with pain", "positive pregnancy test with bleeding"
                ],
                severity=RedFlagSeverity.HIGH,
                possible_conditions=["Ectopic pregnancy", "Miscarriage"],
                action="Seek emergency evaluation",
                time_sensitivity="Hours - risk of rupture and hemorrhage",
                body_system="reproductive",
            ),
            RedFlagRule(
                id="diabetic_emergency",
                keywords=[
                    "diabetic confusion", "fruity breath", "diabetic vomiting",
                    "very high blood sugar", "blood sugar over 400"
                ],
                severity=RedFlagSeverity.HIGH,
                possible_conditions=["Diabetic ketoacidosis", "Hyperosmolar state"],
                action="Seek emergency care",
                time_sensitivity="Hours",
                body_system="endocrine",
            ),
            RedFlagRule(
                id="deep_vein_thrombosis",
                keywords=[
                    "leg swelling one side", "calf pain swelling", "leg red hot swollen",
                    "recent surgery leg pain", "long flight leg swelling"
                ],
                severity=RedFlagSeverity.HIGH,
                possible_conditions=["Deep vein thrombosis", "Pulmonary embolism risk"],
                action="Seek same-day medical evaluation",
                time_sensitivity="Same day - risk of PE",
                body_system="vascular",
                additional_questions=[
                    "Have you had recent surgery or been immobile?",
                    "Is one leg more swollen than the other?",
                    "Do you have any shortness of breath?",
                ],
            ),
            
            # Moderate Severity Red Flags
            RedFlagRule(
                id="blood_in_stool",
                keywords=["blood in stool", "black stool", "tarry stool", "melena", "rectal bleeding"],
                severity=RedFlagSeverity.MODERATE,
                possible_conditions=["GI bleeding", "Hemorrhoids", "Colorectal cancer", "Ulcer"],
                action="Seek medical evaluation within 24-48 hours",
                time_sensitivity="24-48 hours unless heavy bleeding",
                body_system="gastrointestinal",
                additional_questions=[
                    "How much blood have you noticed?",
                    "Is the blood bright red or dark/black?",
                    "Have you had any abdominal pain or weight loss?",
                ],
            ),
            RedFlagRule(
                id="blood_in_urine",
                keywords=["blood in urine", "hematuria", "pink urine", "red urine"],
                severity=RedFlagSeverity.MODERATE,
                possible_conditions=["UTI", "Kidney stones", "Bladder cancer", "Kidney disease"],
                action="Seek medical evaluation within 24-48 hours",
                time_sensitivity="24-48 hours",
                body_system="genitourinary",
            ),
            RedFlagRule(
                id="unexplained_weight_loss",
                keywords=["unexplained weight loss", "losing weight without trying", "unintentional weight loss"],
                severity=RedFlagSeverity.MODERATE,
                possible_conditions=["Cancer", "Hyperthyroidism", "Diabetes", "Depression"],
                action="Schedule medical evaluation",
                time_sensitivity="Within 1-2 weeks",
                body_system="systemic",
                additional_questions=[
                    "How much weight have you lost?",
                    "Over what time period?",
                    "Have you had changes in appetite?",
                ],
            ),
            RedFlagRule(
                id="persistent_fever",
                keywords=["fever over a week", "persistent fever", "fever not going away", "prolonged fever"],
                severity=RedFlagSeverity.MODERATE,
                possible_conditions=["Infection", "Autoimmune disease", "Malignancy"],
                action="Seek medical evaluation",
                time_sensitivity="Within 24-48 hours",
                body_system="infectious",
            ),
            RedFlagRule(
                id="new_neurological_symptoms",
                keywords=[
                    "new weakness", "new numbness", "balance problems new",
                    "coordination problems", "tremor new"
                ],
                severity=RedFlagSeverity.MODERATE,
                possible_conditions=["Stroke", "MS", "Brain tumor", "Neuropathy"],
                action="Seek medical evaluation",
                time_sensitivity="Within 24-48 hours",
                body_system="neurological",
            ),
            
            # Mental Health Red Flags
            RedFlagRule(
                id="suicidal_ideation",
                keywords=[
                    "suicidal", "want to die", "kill myself", "end my life",
                    "no reason to live", "better off dead"
                ],
                severity=RedFlagSeverity.CRITICAL,
                possible_conditions=["Suicidal ideation", "Major depression", "Crisis"],
                action="Call 988 (Suicide Hotline) or go to nearest ER",
                time_sensitivity="Immediate - life-threatening",
                body_system="psychiatric",
                additional_questions=[
                    "Do you have a plan?",
                    "Do you have access to means?",
                    "Is there someone who can stay with you?",
                ],
            ),
            RedFlagRule(
                id="homicidal_ideation",
                keywords=["want to hurt someone", "going to kill", "homicidal"],
                severity=RedFlagSeverity.CRITICAL,
                possible_conditions=["Psychiatric emergency"],
                action="Seek immediate psychiatric evaluation",
                time_sensitivity="Immediate",
                body_system="psychiatric",
            ),
            RedFlagRule(
                id="psychosis",
                keywords=[
                    "hearing voices", "seeing things not there", "paranoid",
                    "people following me", "being watched", "delusions"
                ],
                severity=RedFlagSeverity.HIGH,
                possible_conditions=["Psychosis", "Schizophrenia", "Drug-induced psychosis"],
                action="Seek psychiatric evaluation",
                time_sensitivity="Same day",
                body_system="psychiatric",
            ),
        ]
    
    def detect(
        self,
        symptoms: List[str],
        chief_complaint: Optional[str] = None,
        age: Optional[int] = None,
        additional_context: Optional[str] = None,
    ) -> RedFlagResult:
        """
        Detect red flags in patient presentation
        
        Returns deterministic, explainable results
        """
        result = RedFlagResult()
        
        # Combine all text to search
        search_text = " ".join(symptoms).lower()
        if chief_complaint:
            search_text += " " + chief_complaint.lower()
        if additional_context:
            search_text += " " + additional_context.lower()
        
        detected_flags: List[RedFlag] = []
        
        for rule in self.rules:
            # Check if any keywords match
            for keyword in rule.keywords:
                if keyword.lower() in search_text:
                    # Check age-specific rules
                    if rule.age_specific and age:
                        if "min_age" in rule.age_specific and age < rule.age_specific["min_age"]:
                            continue
                        if "max_age" in rule.age_specific and age > rule.age_specific["max_age"]:
                            continue
                    
                    flag = RedFlag(
                        symptom=keyword,
                        severity=rule.severity,
                        possible_conditions=rule.possible_conditions,
                        action_required=rule.action,
                        time_sensitivity=rule.time_sensitivity,
                        additional_questions=rule.additional_questions,
                    )
                    
                    # Avoid duplicates
                    if not any(f.symptom == flag.symptom for f in detected_flags):
                        detected_flags.append(flag)
                        result.rules_triggered.append(rule.id)
                    break  # Only match once per rule
        
        if detected_flags:
            result.has_red_flags = True
            result.red_flags = detected_flags
            
            # Determine highest severity
            severities = [f.severity for f in detected_flags]
            if RedFlagSeverity.CRITICAL in severities:
                result.highest_severity = RedFlagSeverity.CRITICAL
                result.immediate_action = "Call 911 or go to emergency room immediately"
                result.referral_recommendation = "Emergency Room"
            elif RedFlagSeverity.HIGH in severities:
                result.highest_severity = RedFlagSeverity.HIGH
                result.immediate_action = "Seek medical attention today"
                result.referral_recommendation = "Urgent Care or Emergency Room"
            else:
                result.highest_severity = RedFlagSeverity.MODERATE
                result.immediate_action = "Schedule medical appointment within 24-48 hours"
                result.referral_recommendation = "Primary Care Physician"
        
        return result
    
    def get_follow_up_questions(self, red_flags: List[RedFlag]) -> List[str]:
        """Get follow-up questions for detected red flags"""
        questions = []
        for flag in red_flags:
            questions.extend(flag.additional_questions)
        return list(set(questions))  # Remove duplicates
    
    def add_rule(self, rule: RedFlagRule) -> None:
        """Add a custom red flag rule"""
        self.rules.append(rule)
