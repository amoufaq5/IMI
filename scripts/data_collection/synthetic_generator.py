"""
Synthetic Medical Case Generator

Generates synthetic training data for medical LLM fine-tuning:
- Patient symptom cases with triage decisions
- Drug interaction scenarios
- Clinical decision support cases
- USMLE-style questions
- Regulatory compliance scenarios
"""
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import itertools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


# Medical knowledge bases for generation
SYMPTOMS = {
    "cardiovascular": [
        "chest pain", "palpitations", "shortness of breath", "leg swelling",
        "dizziness", "fainting", "irregular heartbeat", "fatigue",
    ],
    "respiratory": [
        "cough", "wheezing", "shortness of breath", "chest tightness",
        "sputum production", "hemoptysis", "stridor", "nasal congestion",
    ],
    "gastrointestinal": [
        "abdominal pain", "nausea", "vomiting", "diarrhea", "constipation",
        "bloating", "heartburn", "blood in stool", "difficulty swallowing",
    ],
    "neurological": [
        "headache", "dizziness", "numbness", "tingling", "weakness",
        "vision changes", "confusion", "memory problems", "seizures",
    ],
    "musculoskeletal": [
        "joint pain", "muscle pain", "back pain", "stiffness", "swelling",
        "limited range of motion", "muscle weakness", "cramps",
    ],
    "general": [
        "fever", "fatigue", "weight loss", "night sweats", "chills",
        "loss of appetite", "malaise", "sleep problems",
    ],
}

CONDITIONS = [
    {"name": "Hypertension", "category": "cardiovascular", "chronic": True},
    {"name": "Type 2 Diabetes", "category": "endocrine", "chronic": True},
    {"name": "Asthma", "category": "respiratory", "chronic": True},
    {"name": "COPD", "category": "respiratory", "chronic": True},
    {"name": "Heart Failure", "category": "cardiovascular", "chronic": True},
    {"name": "Atrial Fibrillation", "category": "cardiovascular", "chronic": True},
    {"name": "Chronic Kidney Disease", "category": "renal", "chronic": True},
    {"name": "Osteoarthritis", "category": "musculoskeletal", "chronic": True},
    {"name": "Depression", "category": "psychiatric", "chronic": True},
    {"name": "Anxiety", "category": "psychiatric", "chronic": True},
    {"name": "GERD", "category": "gastrointestinal", "chronic": True},
    {"name": "Hypothyroidism", "category": "endocrine", "chronic": True},
]

MEDICATIONS = {
    "cardiovascular": [
        {"name": "Lisinopril", "class": "ACE Inhibitor", "dose": "10-40mg daily"},
        {"name": "Metoprolol", "class": "Beta Blocker", "dose": "25-200mg daily"},
        {"name": "Amlodipine", "class": "Calcium Channel Blocker", "dose": "5-10mg daily"},
        {"name": "Atorvastatin", "class": "Statin", "dose": "10-80mg daily"},
        {"name": "Aspirin", "class": "Antiplatelet", "dose": "81-325mg daily"},
        {"name": "Warfarin", "class": "Anticoagulant", "dose": "varies by INR"},
    ],
    "diabetes": [
        {"name": "Metformin", "class": "Biguanide", "dose": "500-2000mg daily"},
        {"name": "Glipizide", "class": "Sulfonylurea", "dose": "5-20mg daily"},
        {"name": "Sitagliptin", "class": "DPP-4 Inhibitor", "dose": "100mg daily"},
        {"name": "Empagliflozin", "class": "SGLT2 Inhibitor", "dose": "10-25mg daily"},
    ],
    "respiratory": [
        {"name": "Albuterol", "class": "Beta-2 Agonist", "dose": "2 puffs PRN"},
        {"name": "Fluticasone", "class": "Inhaled Corticosteroid", "dose": "1-2 puffs BID"},
        {"name": "Montelukast", "class": "Leukotriene Inhibitor", "dose": "10mg daily"},
    ],
    "pain": [
        {"name": "Ibuprofen", "class": "NSAID", "dose": "200-800mg TID"},
        {"name": "Acetaminophen", "class": "Analgesic", "dose": "325-1000mg Q6H"},
        {"name": "Naproxen", "class": "NSAID", "dose": "250-500mg BID"},
    ],
    "gi": [
        {"name": "Omeprazole", "class": "PPI", "dose": "20-40mg daily"},
        {"name": "Famotidine", "class": "H2 Blocker", "dose": "20-40mg BID"},
    ],
}

DRUG_INTERACTIONS = [
    {
        "drug1": "Warfarin", "drug2": "Aspirin",
        "severity": "major", "effect": "Increased bleeding risk",
        "mechanism": "Both affect hemostasis through different mechanisms",
        "management": "Avoid combination unless specifically indicated; monitor for bleeding",
    },
    {
        "drug1": "Lisinopril", "drug2": "Potassium supplements",
        "severity": "major", "effect": "Hyperkalemia",
        "mechanism": "ACE inhibitors reduce potassium excretion",
        "management": "Monitor potassium levels; avoid supplements unless hypokalemic",
    },
    {
        "drug1": "Metformin", "drug2": "Contrast dye",
        "severity": "major", "effect": "Lactic acidosis risk",
        "mechanism": "Contrast can cause AKI, impairing metformin clearance",
        "management": "Hold metformin 48h before and after contrast; check renal function",
    },
    {
        "drug1": "Ibuprofen", "drug2": "Lisinopril",
        "severity": "moderate", "effect": "Reduced antihypertensive effect, increased renal risk",
        "mechanism": "NSAIDs inhibit prostaglandins needed for ACE inhibitor efficacy",
        "management": "Use lowest NSAID dose for shortest duration; monitor BP and renal function",
    },
    {
        "drug1": "Warfarin", "drug2": "Omeprazole",
        "severity": "moderate", "effect": "Increased INR",
        "mechanism": "CYP2C19 inhibition reduces warfarin metabolism",
        "management": "Monitor INR more frequently; consider dose adjustment",
    },
    {
        "drug1": "Metoprolol", "drug2": "Albuterol",
        "severity": "moderate", "effect": "Reduced bronchodilator effect",
        "mechanism": "Beta-blockers antagonize beta-2 agonist effects",
        "management": "Use cardioselective beta-blocker; monitor respiratory status",
    },
]

TRIAGE_LEVELS = {
    "emergency": {
        "description": "Life-threatening, requires immediate attention",
        "symptoms": ["severe chest pain", "difficulty breathing", "stroke symptoms", 
                    "severe bleeding", "loss of consciousness", "severe allergic reaction"],
        "recommendation": "Call 911 or go to emergency room immediately",
    },
    "urgent": {
        "description": "Serious condition requiring prompt medical attention",
        "symptoms": ["moderate chest pain", "high fever with rash", "severe abdominal pain",
                    "signs of infection", "moderate breathing difficulty"],
        "recommendation": "See a doctor within 24 hours or visit urgent care",
    },
    "semi_urgent": {
        "description": "Condition requiring medical attention within days",
        "symptoms": ["persistent symptoms", "worsening condition", "moderate pain",
                    "concerning but stable symptoms"],
        "recommendation": "Schedule appointment within 2-3 days",
    },
    "routine": {
        "description": "Non-urgent condition for routine care",
        "symptoms": ["mild symptoms", "chronic condition follow-up", "medication refill",
                    "preventive care"],
        "recommendation": "Schedule routine appointment",
    },
    "self_care": {
        "description": "Minor condition suitable for self-care",
        "symptoms": ["common cold", "minor headache", "mild muscle ache",
                    "minor cuts/bruises"],
        "recommendation": "Self-care with OTC medications; see doctor if symptoms persist",
    },
}


class SyntheticCaseGenerator:
    """Generates synthetic medical cases for training"""
    
    def __init__(self, output_dir: Path = DATA_DIR / "synthetic"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_patient_demographics(self) -> Dict[str, Any]:
        """Generate random patient demographics"""
        age = random.choices(
            [random.randint(18, 30), random.randint(31, 50), 
             random.randint(51, 70), random.randint(71, 90)],
            weights=[0.2, 0.3, 0.3, 0.2]
        )[0]
        
        gender = random.choice(["male", "female"])
        
        # More conditions with age
        num_conditions = random.choices([0, 1, 2, 3], 
            weights=[0.3 if age < 50 else 0.1, 
                    0.4 if age < 50 else 0.2,
                    0.2 if age < 50 else 0.4,
                    0.1 if age < 50 else 0.3])[0]
        
        conditions = random.sample(CONDITIONS, min(num_conditions, len(CONDITIONS)))
        
        # Medications based on conditions
        medications = []
        for cond in conditions:
            if cond["name"] == "Hypertension":
                medications.extend(random.sample(MEDICATIONS["cardiovascular"][:4], 
                                                random.randint(1, 2)))
            elif cond["name"] == "Type 2 Diabetes":
                medications.extend(random.sample(MEDICATIONS["diabetes"], 
                                                random.randint(1, 2)))
            elif cond["name"] in ["Asthma", "COPD"]:
                medications.extend(random.sample(MEDICATIONS["respiratory"], 
                                                random.randint(1, 2)))
        
        return {
            "age": age,
            "gender": gender,
            "conditions": [c["name"] for c in conditions],
            "medications": [m["name"] for m in medications],
            "allergies": random.sample(["Penicillin", "Sulfa", "NSAIDs", "None"], 1),
        }
    
    def generate_symptom_case(self) -> Dict[str, Any]:
        """Generate a symptom assessment case"""
        patient = self.generate_patient_demographics()
        
        # Select symptom category and symptoms
        category = random.choice(list(SYMPTOMS.keys()))
        num_symptoms = random.randint(1, 4)
        symptoms = random.sample(SYMPTOMS[category], min(num_symptoms, len(SYMPTOMS[category])))
        
        # Add some cross-category symptoms
        if random.random() > 0.5:
            other_cat = random.choice([c for c in SYMPTOMS.keys() if c != category])
            symptoms.append(random.choice(SYMPTOMS[other_cat]))
        
        # Determine severity and duration
        severity = random.randint(1, 10)
        duration_hours = random.choice([2, 6, 12, 24, 48, 72, 168])
        
        # Determine triage level based on symptoms and severity
        if severity >= 8 or any(s in TRIAGE_LEVELS["emergency"]["symptoms"] for s in symptoms):
            triage = "emergency"
        elif severity >= 6 or duration_hours <= 6:
            triage = "urgent"
        elif severity >= 4:
            triage = "semi_urgent"
        elif severity >= 2:
            triage = "routine"
        else:
            triage = "self_care"
        
        chief_complaint = symptoms[0] if symptoms else "general discomfort"
        
        return {
            "patient": patient,
            "symptoms": symptoms,
            "chief_complaint": chief_complaint,
            "severity": severity,
            "duration_hours": duration_hours,
            "triage_level": triage,
            "triage_info": TRIAGE_LEVELS[triage],
        }
    
    def generate_triage_training_example(self) -> Dict[str, str]:
        """Generate a triage training example in instruction format"""
        case = self.generate_symptom_case()
        patient = case["patient"]
        
        instruction = f"""You are a medical triage assistant. Assess the following patient and provide a triage recommendation.

Patient: {patient['age']} year old {patient['gender']}
Medical History: {', '.join(patient['conditions']) if patient['conditions'] else 'None'}
Current Medications: {', '.join(patient['medications']) if patient['medications'] else 'None'}
Allergies: {', '.join(patient['allergies'])}

Chief Complaint: {case['chief_complaint']}
Symptoms: {', '.join(case['symptoms'])}
Severity (1-10): {case['severity']}
Duration: {case['duration_hours']} hours

Provide triage level and recommendation."""

        output = f"""**Triage Level: {case['triage_level'].upper()}**

{case['triage_info']['description']}

**Recommendation:** {case['triage_info']['recommendation']}

**Reasoning:**
- Chief complaint: {case['chief_complaint']}
- Severity score: {case['severity']}/10
- Duration: {case['duration_hours']} hours
- Patient factors: {patient['age']} years old with {len(patient['conditions'])} chronic conditions

**Next Steps:**
1. {case['triage_info']['recommendation']}
2. Monitor for worsening symptoms
3. {'Seek immediate care if symptoms worsen' if case['triage_level'] != 'emergency' else 'Do not delay treatment'}

*This assessment is for guidance only. Always consult a healthcare professional for medical decisions.*"""

        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "source": "synthetic_triage",
            "adapter": "patient_triage",
            "metadata": {
                "triage_level": case["triage_level"],
                "severity": case["severity"],
            }
        }
    
    def generate_drug_interaction_example(self) -> Dict[str, str]:
        """Generate a drug interaction training example"""
        interaction = random.choice(DRUG_INTERACTIONS)
        
        instruction = f"""Check for drug interactions between {interaction['drug1']} and {interaction['drug2']}. 
Explain the interaction, its severity, mechanism, and management recommendations."""

        output = f"""**Drug Interaction Alert**

**Drugs:** {interaction['drug1']} + {interaction['drug2']}
**Severity:** {interaction['severity'].upper()}

**Effect:** {interaction['effect']}

**Mechanism:** {interaction['mechanism']}

**Clinical Management:**
{interaction['management']}

**Recommendations:**
1. {'Avoid this combination if possible' if interaction['severity'] == 'major' else 'Use with caution'}
2. Monitor patient closely for signs of {interaction['effect'].lower()}
3. Document interaction in patient record
4. Consider alternative medications if available

*Always verify interactions with current drug references and clinical judgment.*"""

        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "source": "synthetic_interaction",
            "adapter": "clinical_pharmacist",
            "metadata": {
                "severity": interaction["severity"],
                "drugs": [interaction["drug1"], interaction["drug2"]],
            }
        }
    
    def generate_clinical_case(self) -> Dict[str, str]:
        """Generate a clinical decision support case for doctors"""
        patient = self.generate_patient_demographics()
        category = random.choice(["cardiovascular", "respiratory", "gastrointestinal"])
        symptoms = random.sample(SYMPTOMS[category], random.randint(2, 4))
        
        # Generate vitals
        vitals = {
            "HR": random.randint(60, 110),
            "BP": f"{random.randint(100, 160)}/{random.randint(60, 100)}",
            "RR": random.randint(12, 24),
            "Temp": round(random.uniform(97.5, 101.5), 1),
            "SpO2": random.randint(92, 100),
        }
        
        instruction = f"""You are a clinical decision support system. Analyze this case and provide differential diagnosis.

**Patient:** {patient['age']}yo {patient['gender']}
**Chief Complaint:** {symptoms[0]}
**HPI:** Patient presents with {', '.join(symptoms)} for the past {random.choice([1,2,3,5,7])} days.
**PMH:** {', '.join(patient['conditions']) if patient['conditions'] else 'None significant'}
**Medications:** {', '.join(patient['medications']) if patient['medications'] else 'None'}
**Allergies:** {', '.join(patient['allergies'])}

**Vitals:**
- HR: {vitals['HR']} bpm
- BP: {vitals['BP']} mmHg
- RR: {vitals['RR']} /min
- Temp: {vitals['Temp']}°F
- SpO2: {vitals['SpO2']}%

Provide differential diagnosis and recommended workup."""

        # Generate differential based on symptoms
        differentials = self._generate_differentials(category, symptoms, patient)
        
        output = f"""**Clinical Assessment**

**Differential Diagnosis (in order of likelihood):**
{self._format_differentials(differentials)}

**Recommended Workup:**
1. Labs: CBC, BMP, {self._get_relevant_labs(category)}
2. Imaging: {self._get_relevant_imaging(category)}
3. Additional: {self._get_additional_workup(category)}

**Immediate Considerations:**
- {'Vital signs concerning for' + self._assess_vitals(vitals) if self._assess_vitals(vitals) else 'Vital signs stable'}
- Consider patient's comorbidities: {', '.join(patient['conditions']) if patient['conditions'] else 'None'}
- Review current medications for interactions

**Disposition:**
{self._get_disposition(vitals, symptoms)}

*Clinical judgment should guide final decisions.*"""

        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "source": "synthetic_clinical",
            "adapter": "clinical_decision",
            "metadata": {
                "category": category,
                "patient_age": patient["age"],
            }
        }
    
    def _generate_differentials(self, category: str, symptoms: List[str], 
                                patient: Dict) -> List[Dict]:
        """Generate differential diagnoses"""
        differentials_db = {
            "cardiovascular": [
                {"name": "Acute Coronary Syndrome", "probability": "high"},
                {"name": "Heart Failure Exacerbation", "probability": "moderate"},
                {"name": "Arrhythmia", "probability": "moderate"},
                {"name": "Pericarditis", "probability": "low"},
            ],
            "respiratory": [
                {"name": "Pneumonia", "probability": "high"},
                {"name": "COPD Exacerbation", "probability": "moderate"},
                {"name": "Asthma Exacerbation", "probability": "moderate"},
                {"name": "Pulmonary Embolism", "probability": "low"},
            ],
            "gastrointestinal": [
                {"name": "Gastritis/PUD", "probability": "high"},
                {"name": "Cholecystitis", "probability": "moderate"},
                {"name": "Pancreatitis", "probability": "moderate"},
                {"name": "Appendicitis", "probability": "low"},
            ],
        }
        return differentials_db.get(category, [])[:4]
    
    def _format_differentials(self, differentials: List[Dict]) -> str:
        """Format differentials for output"""
        lines = []
        for i, d in enumerate(differentials, 1):
            lines.append(f"{i}. **{d['name']}** - {d['probability']} probability")
        return '\n'.join(lines)
    
    def _get_relevant_labs(self, category: str) -> str:
        labs = {
            "cardiovascular": "Troponin, BNP, Lipid panel",
            "respiratory": "ABG, Procalcitonin, D-dimer",
            "gastrointestinal": "LFTs, Lipase, Amylase",
        }
        return labs.get(category, "As clinically indicated")
    
    def _get_relevant_imaging(self, category: str) -> str:
        imaging = {
            "cardiovascular": "ECG, CXR, Echo if indicated",
            "respiratory": "CXR, CT chest if PE suspected",
            "gastrointestinal": "Abdominal X-ray, RUQ ultrasound or CT abdomen",
        }
        return imaging.get(category, "As clinically indicated")
    
    def _get_additional_workup(self, category: str) -> str:
        workup = {
            "cardiovascular": "Continuous telemetry monitoring",
            "respiratory": "Sputum culture if productive cough",
            "gastrointestinal": "H. pylori testing if indicated",
        }
        return workup.get(category, "Based on clinical findings")
    
    def _assess_vitals(self, vitals: Dict) -> str:
        concerns = []
        if vitals["HR"] > 100:
            concerns.append("tachycardia")
        if vitals["SpO2"] < 94:
            concerns.append("hypoxia")
        if vitals["Temp"] > 100.4:
            concerns.append("fever")
        return ", ".join(concerns) if concerns else ""
    
    def _get_disposition(self, vitals: Dict, symptoms: List[str]) -> str:
        if vitals["SpO2"] < 92 or vitals["HR"] > 120:
            return "Consider admission for monitoring and treatment"
        elif vitals["Temp"] > 101:
            return "Observation with reassessment; admission if no improvement"
        else:
            return "Outpatient management possible if stable; close follow-up recommended"
    
    def generate_usmle_question(self) -> Dict[str, str]:
        """Generate USMLE-style question"""
        patient = self.generate_patient_demographics()
        category = random.choice(["cardiovascular", "respiratory", "gastrointestinal", "neurological"])
        
        # Question templates
        templates = [
            {
                "stem": f"A {patient['age']}-year-old {patient['gender']} presents with {{symptoms}}. "
                       f"Physical examination reveals {{findings}}. Which of the following is the most likely diagnosis?",
                "type": "diagnosis",
            },
            {
                "stem": f"A {patient['age']}-year-old {patient['gender']} with a history of {{conditions}} "
                       f"presents with {{symptoms}}. What is the most appropriate next step in management?",
                "type": "management",
            },
            {
                "stem": f"A {patient['age']}-year-old {patient['gender']} is started on {{medication}}. "
                       f"Which of the following adverse effects should be monitored?",
                "type": "pharmacology",
            },
        ]
        
        template = random.choice(templates)
        symptoms = random.sample(SYMPTOMS[category], 2)
        
        question_stem = template["stem"].format(
            symptoms=", ".join(symptoms),
            findings="relevant clinical findings",
            conditions=patient["conditions"][0] if patient["conditions"] else "no significant medical history",
            medication=random.choice(MEDICATIONS.get(category, MEDICATIONS["pain"]))["name"],
        )
        
        # Generate options
        options = ["Option A - Correct answer", "Option B - Distractor 1", 
                  "Option C - Distractor 2", "Option D - Distractor 3", "Option E - Distractor 4"]
        
        instruction = f"""Answer this USMLE-style question:

{question_stem}

A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
E. {options[4]}"""

        output = f"""**Correct Answer: A**

**Explanation:**
{options[0]} is correct because it directly addresses the clinical presentation described.

**Why other options are incorrect:**
- B: {options[1]} - Does not fit the clinical picture
- C: {options[2]} - Would present differently
- D: {options[3]} - Less likely given the history
- E: {options[4]} - Not supported by the findings

**High-Yield Points:**
1. Key clinical features to recognize
2. Important differential considerations
3. First-line management approach

**Related Topics:** {category.title()}, Clinical Medicine"""

        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "source": "synthetic_usmle",
            "adapter": "education",
            "metadata": {
                "question_type": template["type"],
                "category": category,
            }
        }
    
    def generate_regulatory_scenario(self) -> Dict[str, str]:
        """Generate regulatory compliance scenario"""
        scenarios = [
            {
                "topic": "Cleaning Validation",
                "question": "What are the key requirements for cleaning validation in pharmaceutical manufacturing?",
                "answer": """**Cleaning Validation Requirements**

**Regulatory Basis:** FDA 21 CFR 211, EU GMP Annex 15

**Key Requirements:**

1. **Written Procedures**
   - Detailed cleaning SOPs for each equipment
   - Defined acceptance criteria
   - Sampling procedures (swab and rinse)

2. **Acceptance Criteria**
   - Maximum Allowable Carryover (MAC) calculation
   - Visual cleanliness standards
   - Microbial limits where applicable

3. **Validation Protocol**
   - Minimum 3 consecutive successful runs
   - Worst-case product selection
   - Documented sampling locations

4. **Documentation**
   - Validation protocol and report
   - Training records
   - Change control for modifications

5. **Ongoing Monitoring**
   - Periodic revalidation
   - Monitoring of cleaning effectiveness
   - Deviation handling procedures

**Common Deficiencies:**
- Inadequate acceptance criteria justification
- Insufficient sampling locations
- Missing worst-case assessment""",
            },
            {
                "topic": "Deviation Management",
                "question": "How should deviations be handled according to GMP requirements?",
                "answer": """**Deviation Management Process**

**Regulatory Basis:** FDA 21 CFR 211.192, EU GMP Chapter 1

**Process Steps:**

1. **Detection & Documentation**
   - Immediate documentation of deviation
   - Initial impact assessment
   - Containment actions if needed

2. **Classification**
   - Critical: Direct product quality impact
   - Major: Potential quality impact
   - Minor: No direct quality impact

3. **Investigation**
   - Root cause analysis (5 Whys, Fishbone)
   - Impact assessment on product/batch
   - Extension to other batches/products

4. **CAPA Implementation**
   - Corrective actions for immediate issue
   - Preventive actions to prevent recurrence
   - Effectiveness verification

5. **Closure & Trending**
   - QA review and approval
   - Trend analysis
   - Management review

**Timeline Requirements:**
- Initial documentation: Within 24 hours
- Investigation completion: 30 days (typical)
- CAPA implementation: Per risk assessment""",
            },
            {
                "topic": "Data Integrity",
                "question": "What are the ALCOA+ principles for data integrity?",
                "answer": """**ALCOA+ Data Integrity Principles**

**Regulatory Basis:** FDA Data Integrity Guidance, WHO TRS 996

**ALCOA+ Explained:**

**A - Attributable**
- Data traceable to person who generated it
- Electronic signatures where applicable
- Audit trails enabled

**L - Legible**
- Data readable throughout retention period
- Permanent recording (no pencil)
- Clear and unambiguous

**C - Contemporaneous**
- Recorded at time of activity
- No backdating or predating
- Real-time documentation

**O - Original**
- First capture of data (or true copy)
- Original records preserved
- Certified copies when needed

**A - Accurate**
- Free from errors
- Reflects actual observation
- Corrections properly documented

**+ Additional Principles:**
- **Complete**: All data included
- **Consistent**: Logical sequence
- **Enduring**: Maintained for retention period
- **Available**: Accessible when needed

**Common Violations:**
- Shared login credentials
- Disabled audit trails
- Predating/backdating entries
- Unauthorized data deletion""",
            },
        ]
        
        scenario = random.choice(scenarios)
        
        return {
            "instruction": scenario["question"],
            "input": "",
            "output": scenario["answer"],
            "source": "synthetic_regulatory",
            "adapter": "regulatory_qa",
            "metadata": {
                "topic": scenario["topic"],
            }
        }
    
    def generate_dataset(self, num_examples: int = 1000) -> Dict[str, List[Dict]]:
        """Generate complete synthetic dataset"""
        logger.info(f"Generating {num_examples} synthetic examples...")
        
        datasets = {
            "patient_triage": [],
            "clinical_pharmacist": [],
            "clinical_decision": [],
            "education": [],
            "regulatory_qa": [],
        }
        
        # Distribution of examples
        distribution = {
            "triage": 0.25,
            "interaction": 0.15,
            "clinical": 0.20,
            "usmle": 0.25,
            "regulatory": 0.15,
        }
        
        for i in range(num_examples):
            r = random.random()
            
            if r < distribution["triage"]:
                example = self.generate_triage_training_example()
                datasets["patient_triage"].append(example)
            elif r < distribution["triage"] + distribution["interaction"]:
                example = self.generate_drug_interaction_example()
                datasets["clinical_pharmacist"].append(example)
            elif r < distribution["triage"] + distribution["interaction"] + distribution["clinical"]:
                example = self.generate_clinical_case()
                datasets["clinical_decision"].append(example)
            elif r < distribution["triage"] + distribution["interaction"] + distribution["clinical"] + distribution["usmle"]:
                example = self.generate_usmle_question()
                datasets["education"].append(example)
            else:
                example = self.generate_regulatory_scenario()
                datasets["regulatory_qa"].append(example)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Generated {i + 1}/{num_examples} examples")
        
        return datasets
    
    def save_datasets(self, datasets: Dict[str, List[Dict]]):
        """Save generated datasets"""
        for adapter, examples in datasets.items():
            if examples:
                output_path = self.output_dir / f"{adapter}_synthetic.json"
                with open(output_path, 'w') as f:
                    json.dump(examples, f, indent=2)
                logger.info(f"Saved {len(examples)} examples to {output_path}")
    
    def generate_and_save(self, num_examples: int = 1000):
        """Generate and save complete dataset"""
        logger.info("=" * 60)
        logger.info("Synthetic Medical Case Generator")
        logger.info("=" * 60)
        
        datasets = self.generate_dataset(num_examples)
        self.save_datasets(datasets)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Generation Summary")
        logger.info("=" * 60)
        total = 0
        for adapter, examples in datasets.items():
            logger.info(f"  {adapter}: {len(examples)} examples")
            total += len(examples)
        logger.info(f"  Total: {total} examples")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic medical training data")
    parser.add_argument("--num-examples", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "synthetic", help="Output directory")
    args = parser.parse_args()
    
    generator = SyntheticCaseGenerator(args.output_dir)
    generator.generate_and_save(args.num_examples)


if __name__ == "__main__":
    main()
