"""
ASMETHOD-style Triage Engine for Patient Assessment

ASMETHOD Framework:
- Age: Patient age considerations
- Self or someone else: Who is the patient
- Medication: Current medications
- Extra medicines: OTC/supplements
- Time/Duration: How long symptoms present
- History: Medical history
- Other symptoms: Associated symptoms
- Danger signals: Red flags
"""
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class TriageUrgency(str, Enum):
    """Triage urgency levels"""
    EMERGENCY = "emergency"      # Call 911 / Go to ER immediately
    URGENT = "urgent"            # See doctor within 24 hours
    SEMI_URGENT = "semi_urgent"  # See doctor within 48-72 hours
    ROUTINE = "routine"          # Schedule appointment
    SELF_CARE = "self_care"      # OTC treatment appropriate


class TriageResult(BaseModel):
    """Result of triage assessment"""
    urgency: TriageUrgency
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: List[str] = Field(default_factory=list)
    red_flags_detected: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    referral_type: Optional[str] = None  # ER, specialist, PCP
    otc_eligible: bool = False
    requires_prescription: bool = False
    follow_up_timeframe: Optional[str] = None
    
    # Explainability
    rules_triggered: List[str] = Field(default_factory=list)
    assessment_timestamp: datetime = Field(default_factory=datetime.utcnow)


class PatientAssessment(BaseModel):
    """Patient assessment data for triage"""
    # Demographics
    age: int
    sex: Optional[str] = None
    is_pregnant: bool = False
    is_breastfeeding: bool = False
    
    # Chief complaint
    chief_complaint: str
    symptoms: List[str] = Field(default_factory=list)
    symptom_duration_hours: Optional[float] = None
    symptom_severity: Optional[int] = Field(default=None, ge=1, le=10)
    
    # Medical history
    medical_conditions: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    otc_medications: List[str] = Field(default_factory=list)
    
    # Vital signs (if available)
    temperature_f: Optional[float] = None
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None
    
    # Additional context
    recent_travel: bool = False
    recent_surgery: bool = False
    immunocompromised: bool = False


class TriageRule(BaseModel):
    """A single triage rule"""
    id: str
    name: str
    description: str
    urgency: TriageUrgency
    conditions: Dict[str, Any]  # Conditions to match
    priority: int = 0  # Higher priority rules evaluated first


class TriageEngine:
    """
    Deterministic triage engine using ASMETHOD framework
    
    This is the Safety Layer - it uses if-then medical reasoning
    to ensure patient safety through deterministic rules.
    """
    
    # Emergency red flags - immediate ER referral
    EMERGENCY_SYMPTOMS = {
        "chest_pain", "chest_pressure", "crushing_chest_pain",
        "difficulty_breathing", "severe_shortness_of_breath", "cannot_breathe",
        "stroke_symptoms", "facial_drooping", "arm_weakness", "speech_difficulty",
        "severe_bleeding", "uncontrolled_bleeding",
        "loss_of_consciousness", "fainting", "syncope",
        "severe_allergic_reaction", "anaphylaxis", "throat_swelling",
        "severe_head_injury", "head_trauma",
        "suicidal_thoughts", "self_harm", "homicidal_thoughts",
        "seizure", "convulsions",
        "severe_abdominal_pain", "rigid_abdomen",
        "signs_of_shock", "cold_clammy_skin", "rapid_weak_pulse",
        "severe_burns", "chemical_burns_eyes",
        "poisoning", "overdose",
        "pregnancy_bleeding", "severe_pregnancy_pain",
    }
    
    # Urgent symptoms - see doctor within 24 hours
    URGENT_SYMPTOMS = {
        "high_fever",  # >103°F
        "persistent_vomiting",
        "blood_in_stool", "blood_in_urine",
        "severe_dehydration",
        "worsening_symptoms",
        "new_confusion", "altered_mental_status",
        "severe_pain",
        "eye_injury", "sudden_vision_changes",
        "severe_headache", "worst_headache_of_life",
        "stiff_neck_with_fever",
        "rash_with_fever",
        "difficulty_swallowing",
        "testicular_pain",
    }
    
    # Age-specific considerations
    HIGH_RISK_AGE_INFANT = 3  # months
    HIGH_RISK_AGE_CHILD = 2   # years
    HIGH_RISK_AGE_ELDERLY = 65  # years
    
    def __init__(self):
        self.rules: List[TriageRule] = self._load_default_rules()
    
    def _load_default_rules(self) -> List[TriageRule]:
        """Load default triage rules"""
        return [
            TriageRule(
                id="emergency_chest_pain",
                name="Chest Pain Emergency",
                description="Chest pain with cardiac risk factors",
                urgency=TriageUrgency.EMERGENCY,
                conditions={
                    "symptoms_contain": ["chest_pain", "chest_pressure"],
                    "or_conditions": [
                        {"age_gte": 40},
                        {"has_condition": ["diabetes", "hypertension", "heart_disease"]},
                        {"symptoms_contain": ["shortness_of_breath", "arm_pain", "jaw_pain"]},
                    ]
                },
                priority=100,
            ),
            TriageRule(
                id="emergency_stroke",
                name="Stroke Symptoms",
                description="Signs of stroke - FAST protocol",
                urgency=TriageUrgency.EMERGENCY,
                conditions={
                    "symptoms_contain_any": [
                        "facial_drooping", "arm_weakness", "speech_difficulty",
                        "sudden_confusion", "sudden_numbness", "sudden_vision_loss",
                        "sudden_severe_headache",
                    ]
                },
                priority=100,
            ),
            TriageRule(
                id="emergency_breathing",
                name="Severe Breathing Difficulty",
                description="Respiratory emergency",
                urgency=TriageUrgency.EMERGENCY,
                conditions={
                    "symptoms_contain_any": [
                        "severe_shortness_of_breath", "cannot_breathe",
                        "blue_lips", "cyanosis",
                    ],
                    "or_conditions": [
                        {"oxygen_saturation_lt": 92},
                        {"respiratory_rate_gt": 30},
                    ]
                },
                priority=100,
            ),
            TriageRule(
                id="urgent_high_fever",
                name="High Fever",
                description="Fever requiring urgent evaluation",
                urgency=TriageUrgency.URGENT,
                conditions={
                    "temperature_gte": 103.0,
                    "or_conditions": [
                        {"age_lte": 2},
                        {"age_gte": 65},
                        {"immunocompromised": True},
                    ]
                },
                priority=80,
            ),
            TriageRule(
                id="infant_fever",
                name="Infant Fever",
                description="Any fever in infant under 3 months",
                urgency=TriageUrgency.EMERGENCY,
                conditions={
                    "age_months_lte": 3,
                    "temperature_gte": 100.4,
                },
                priority=100,
            ),
        ]
    
    def assess(self, patient: PatientAssessment) -> TriageResult:
        """
        Perform triage assessment on patient
        
        Returns deterministic, explainable triage result
        """
        result = TriageResult(
            urgency=TriageUrgency.SELF_CARE,
            confidence=1.0,
            otc_eligible=True,
        )
        
        # Check emergency symptoms first
        emergency_flags = self._check_emergency_symptoms(patient)
        if emergency_flags:
            result.urgency = TriageUrgency.EMERGENCY
            result.red_flags_detected = emergency_flags
            result.recommendations = ["Call 911 or go to nearest emergency room immediately"]
            result.referral_type = "ER"
            result.otc_eligible = False
            result.requires_prescription = True
            result.rules_triggered.append("emergency_symptom_detection")
            result.reasoning.append(f"Emergency symptoms detected: {', '.join(emergency_flags)}")
            return result
        
        # Check urgent symptoms
        urgent_flags = self._check_urgent_symptoms(patient)
        if urgent_flags:
            result.urgency = TriageUrgency.URGENT
            result.red_flags_detected = urgent_flags
            result.recommendations = ["See a doctor within 24 hours"]
            result.referral_type = "PCP"
            result.otc_eligible = False
            result.requires_prescription = True
            result.rules_triggered.append("urgent_symptom_detection")
            result.reasoning.append(f"Urgent symptoms detected: {', '.join(urgent_flags)}")
        
        # Check vital signs
        vital_urgency = self._assess_vital_signs(patient)
        if vital_urgency and vital_urgency.value < result.urgency.value:
            result.urgency = vital_urgency
            result.rules_triggered.append("vital_signs_assessment")
        
        # Check age-specific risks
        age_risk = self._assess_age_risk(patient)
        if age_risk:
            result.reasoning.append(age_risk)
            result.rules_triggered.append("age_risk_assessment")
            # Escalate urgency for high-risk ages
            if patient.age < self.HIGH_RISK_AGE_CHILD or patient.age >= self.HIGH_RISK_AGE_ELDERLY:
                if result.urgency == TriageUrgency.SELF_CARE:
                    result.urgency = TriageUrgency.ROUTINE
                    result.otc_eligible = False
        
        # Check pregnancy
        if patient.is_pregnant:
            result.reasoning.append("Patient is pregnant - additional caution required")
            result.rules_triggered.append("pregnancy_check")
            result.otc_eligible = False
            if result.urgency == TriageUrgency.SELF_CARE:
                result.urgency = TriageUrgency.ROUTINE
                result.recommendations.append("Consult with OB/GYN before taking any medication")
        
        # Check symptom duration
        duration_assessment = self._assess_duration(patient)
        if duration_assessment:
            result.reasoning.append(duration_assessment)
            result.rules_triggered.append("duration_assessment")
        
        # Apply custom rules
        for rule in sorted(self.rules, key=lambda r: -r.priority):
            if self._evaluate_rule(rule, patient):
                if rule.urgency.value < result.urgency.value:
                    result.urgency = rule.urgency
                result.rules_triggered.append(rule.id)
                result.reasoning.append(f"Rule triggered: {rule.name} - {rule.description}")
        
        # Set follow-up timeframe
        result.follow_up_timeframe = self._get_follow_up_timeframe(result.urgency)
        
        # Final OTC eligibility check
        if result.urgency in [TriageUrgency.EMERGENCY, TriageUrgency.URGENT]:
            result.otc_eligible = False
            result.requires_prescription = True
        
        return result
    
    def _check_emergency_symptoms(self, patient: PatientAssessment) -> List[str]:
        """Check for emergency symptoms"""
        detected = []
        patient_symptoms = set(s.lower().replace(" ", "_") for s in patient.symptoms)
        patient_symptoms.add(patient.chief_complaint.lower().replace(" ", "_"))
        
        for symptom in self.EMERGENCY_SYMPTOMS:
            if symptom in patient_symptoms:
                detected.append(symptom)
            # Also check partial matches
            for ps in patient_symptoms:
                if symptom in ps or ps in symptom:
                    if symptom not in detected:
                        detected.append(symptom)
        
        return detected
    
    def _check_urgent_symptoms(self, patient: PatientAssessment) -> List[str]:
        """Check for urgent symptoms"""
        detected = []
        patient_symptoms = set(s.lower().replace(" ", "_") for s in patient.symptoms)
        
        for symptom in self.URGENT_SYMPTOMS:
            if symptom in patient_symptoms:
                detected.append(symptom)
        
        # Check temperature
        if patient.temperature_f and patient.temperature_f >= 103.0:
            detected.append("high_fever")
        
        return detected
    
    def _assess_vital_signs(self, patient: PatientAssessment) -> Optional[TriageUrgency]:
        """Assess vital signs for abnormalities"""
        # Critical vital signs -> Emergency
        if patient.oxygen_saturation and patient.oxygen_saturation < 90:
            return TriageUrgency.EMERGENCY
        if patient.heart_rate and (patient.heart_rate < 40 or patient.heart_rate > 150):
            return TriageUrgency.EMERGENCY
        if patient.blood_pressure_systolic:
            if patient.blood_pressure_systolic < 80 or patient.blood_pressure_systolic > 200:
                return TriageUrgency.EMERGENCY
        if patient.respiratory_rate and patient.respiratory_rate > 30:
            return TriageUrgency.EMERGENCY
        
        # Concerning vital signs -> Urgent
        if patient.oxygen_saturation and patient.oxygen_saturation < 94:
            return TriageUrgency.URGENT
        if patient.heart_rate and (patient.heart_rate < 50 or patient.heart_rate > 120):
            return TriageUrgency.URGENT
        if patient.temperature_f and patient.temperature_f >= 103.0:
            return TriageUrgency.URGENT
        
        return None
    
    def _assess_age_risk(self, patient: PatientAssessment) -> Optional[str]:
        """Assess age-related risks"""
        if patient.age < self.HIGH_RISK_AGE_CHILD:
            return f"Patient is {patient.age} years old - pediatric considerations apply"
        if patient.age >= self.HIGH_RISK_AGE_ELDERLY:
            return f"Patient is {patient.age} years old - geriatric considerations apply"
        return None
    
    def _assess_duration(self, patient: PatientAssessment) -> Optional[str]:
        """Assess symptom duration"""
        if patient.symptom_duration_hours:
            if patient.symptom_duration_hours > 168:  # > 1 week
                return "Symptoms present for over 1 week - medical evaluation recommended"
            elif patient.symptom_duration_hours > 72:  # > 3 days
                return "Symptoms present for over 3 days - consider medical evaluation"
        return None
    
    def _evaluate_rule(self, rule: TriageRule, patient: PatientAssessment) -> bool:
        """Evaluate a single triage rule against patient data"""
        conditions = rule.conditions
        
        # Check symptom conditions
        if "symptoms_contain" in conditions:
            required = set(conditions["symptoms_contain"])
            patient_symptoms = set(s.lower().replace(" ", "_") for s in patient.symptoms)
            if not required.issubset(patient_symptoms):
                return False
        
        if "symptoms_contain_any" in conditions:
            required = set(conditions["symptoms_contain_any"])
            patient_symptoms = set(s.lower().replace(" ", "_") for s in patient.symptoms)
            if not required.intersection(patient_symptoms):
                return False
        
        # Check age conditions
        if "age_gte" in conditions and patient.age < conditions["age_gte"]:
            return False
        if "age_lte" in conditions and patient.age > conditions["age_lte"]:
            return False
        
        # Check temperature
        if "temperature_gte" in conditions:
            if not patient.temperature_f or patient.temperature_f < conditions["temperature_gte"]:
                return False
        
        # Check conditions
        if "has_condition" in conditions:
            required = set(c.lower() for c in conditions["has_condition"])
            patient_conditions = set(c.lower() for c in patient.medical_conditions)
            if not required.intersection(patient_conditions):
                return False
        
        return True
    
    def _get_follow_up_timeframe(self, urgency: TriageUrgency) -> str:
        """Get recommended follow-up timeframe"""
        timeframes = {
            TriageUrgency.EMERGENCY: "Immediate",
            TriageUrgency.URGENT: "Within 24 hours",
            TriageUrgency.SEMI_URGENT: "Within 48-72 hours",
            TriageUrgency.ROUTINE: "Within 1-2 weeks",
            TriageUrgency.SELF_CARE: "If symptoms worsen or persist beyond 7 days",
        }
        return timeframes.get(urgency, "As needed")
    
    def add_rule(self, rule: TriageRule) -> None:
        """Add a custom triage rule"""
        self.rules.append(rule)
    
    def get_assessment_questions(self, chief_complaint: str) -> List[str]:
        """Get ASMETHOD assessment questions for a complaint"""
        base_questions = [
            "How old is the patient?",
            "Is this for yourself or someone else?",
            "What medications are you currently taking?",
            "Are you taking any over-the-counter medicines or supplements?",
            "How long have you had these symptoms?",
            "Do you have any medical conditions or allergies?",
            "Are you experiencing any other symptoms?",
            "Have you noticed any warning signs like fever, difficulty breathing, or severe pain?",
        ]
        
        # Add complaint-specific questions
        complaint_lower = chief_complaint.lower()
        
        if "headache" in complaint_lower:
            base_questions.extend([
                "Is this the worst headache of your life?",
                "Do you have neck stiffness or sensitivity to light?",
                "Have you had any recent head injury?",
            ])
        elif "chest" in complaint_lower or "heart" in complaint_lower:
            base_questions.extend([
                "Does the pain radiate to your arm, jaw, or back?",
                "Are you short of breath?",
                "Are you sweating or feeling nauseous?",
            ])
        elif "abdominal" in complaint_lower or "stomach" in complaint_lower:
            base_questions.extend([
                "Where exactly is the pain located?",
                "Is the pain constant or does it come and go?",
                "Have you had any vomiting, diarrhea, or blood in stool?",
            ])
        
        return base_questions
