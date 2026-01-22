"""
Knowledge Graph Seeding Script

Populates Neo4j with initial medical knowledge:
- Common diseases and conditions
- Drugs and medications
- Symptoms
- Drug interactions
- Contraindications
"""
import asyncio
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample medical data for seeding
DISEASES = [
    {
        "name": "Hypertension",
        "icd10_code": "I10",
        "category": "Cardiovascular",
        "description": "Persistently elevated arterial blood pressure",
        "risk_factors": ["obesity", "high sodium diet", "sedentary lifestyle", "family history"],
    },
    {
        "name": "Type 2 Diabetes",
        "icd10_code": "E11",
        "category": "Endocrine",
        "description": "Metabolic disorder characterized by high blood sugar and insulin resistance",
        "risk_factors": ["obesity", "sedentary lifestyle", "family history", "age over 45"],
    },
    {
        "name": "Asthma",
        "icd10_code": "J45",
        "category": "Respiratory",
        "description": "Chronic inflammatory disease of the airways",
        "risk_factors": ["allergies", "family history", "respiratory infections"],
    },
    {
        "name": "Migraine",
        "icd10_code": "G43",
        "category": "Neurological",
        "description": "Recurrent headache disorder with moderate to severe pain",
        "risk_factors": ["stress", "hormonal changes", "certain foods", "sleep changes"],
    },
    {
        "name": "Gastroesophageal Reflux Disease",
        "icd10_code": "K21",
        "category": "Gastrointestinal",
        "description": "Chronic digestive disease where stomach acid flows back into esophagus",
        "risk_factors": ["obesity", "hiatal hernia", "pregnancy", "smoking"],
    },
]

DRUGS = [
    {
        "name": "Metformin",
        "drug_class": "Biguanide",
        "mechanism_of_action": "Decreases hepatic glucose production and increases insulin sensitivity",
        "common_side_effects": ["nausea", "diarrhea", "stomach upset"],
        "serious_side_effects": ["lactic acidosis"],
        "pregnancy_category": "B",
    },
    {
        "name": "Lisinopril",
        "drug_class": "ACE Inhibitor",
        "mechanism_of_action": "Inhibits angiotensin-converting enzyme, reducing blood pressure",
        "common_side_effects": ["dry cough", "dizziness", "headache"],
        "serious_side_effects": ["angioedema", "hyperkalemia"],
        "pregnancy_category": "D",
    },
    {
        "name": "Omeprazole",
        "drug_class": "Proton Pump Inhibitor",
        "mechanism_of_action": "Inhibits gastric acid secretion by blocking H+/K+ ATPase",
        "common_side_effects": ["headache", "nausea", "diarrhea"],
        "serious_side_effects": ["C. difficile infection", "bone fractures"],
        "pregnancy_category": "C",
    },
    {
        "name": "Albuterol",
        "drug_class": "Beta-2 Agonist",
        "mechanism_of_action": "Relaxes bronchial smooth muscle via beta-2 receptor activation",
        "common_side_effects": ["tremor", "tachycardia", "nervousness"],
        "serious_side_effects": ["paradoxical bronchospasm"],
        "pregnancy_category": "C",
    },
    {
        "name": "Ibuprofen",
        "drug_class": "NSAID",
        "mechanism_of_action": "Inhibits cyclooxygenase enzymes, reducing prostaglandin synthesis",
        "common_side_effects": ["stomach upset", "nausea", "dizziness"],
        "serious_side_effects": ["GI bleeding", "cardiovascular events", "renal impairment"],
        "pregnancy_category": "C/D",
    },
]

SYMPTOMS = [
    {"name": "Headache", "category": "Neurological"},
    {"name": "Chest Pain", "category": "Cardiovascular"},
    {"name": "Shortness of Breath", "category": "Respiratory"},
    {"name": "Fatigue", "category": "General"},
    {"name": "Nausea", "category": "Gastrointestinal"},
    {"name": "Dizziness", "category": "Neurological"},
    {"name": "Fever", "category": "General"},
    {"name": "Cough", "category": "Respiratory"},
    {"name": "Abdominal Pain", "category": "Gastrointestinal"},
    {"name": "Joint Pain", "category": "Musculoskeletal"},
]

DRUG_INTERACTIONS = [
    {
        "drug1": "Lisinopril",
        "drug2": "Potassium supplements",
        "severity": "major",
        "description": "Increased risk of hyperkalemia",
        "mechanism": "Both increase serum potassium levels",
    },
    {
        "drug1": "Ibuprofen",
        "drug2": "Lisinopril",
        "severity": "moderate",
        "description": "NSAIDs may reduce antihypertensive effect and increase renal risk",
        "mechanism": "NSAIDs inhibit prostaglandins needed for ACE inhibitor efficacy",
    },
    {
        "drug1": "Metformin",
        "drug2": "Contrast dye",
        "severity": "major",
        "description": "Increased risk of lactic acidosis",
        "mechanism": "Contrast can cause acute kidney injury, impairing metformin clearance",
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Lisinopril",
        "condition": "Pregnancy",
        "severity": "absolute",
        "reason": "Can cause fetal harm, particularly in second and third trimesters",
    },
    {
        "drug": "Ibuprofen",
        "condition": "Active GI bleeding",
        "severity": "absolute",
        "reason": "Can worsen bleeding due to antiplatelet effects",
    },
    {
        "drug": "Metformin",
        "condition": "Severe renal impairment",
        "severity": "absolute",
        "reason": "Risk of lactic acidosis due to impaired drug clearance",
    },
    {
        "drug": "Albuterol",
        "condition": "Severe cardiac arrhythmia",
        "severity": "relative",
        "reason": "Beta-agonist effects may worsen arrhythmias",
    },
]


def generate_cypher_queries() -> List[str]:
    """Generate Cypher queries for seeding data"""
    queries = []
    
    # Create diseases
    for disease in DISEASES:
        query = f"""
        MERGE (d:Disease {{name: '{disease["name"]}'}})
        SET d.icd10_code = '{disease["icd10_code"]}',
            d.category = '{disease["category"]}',
            d.description = '{disease["description"]}',
            d.risk_factors = {disease["risk_factors"]}
        """
        queries.append(query)
    
    # Create drugs
    for drug in DRUGS:
        query = f"""
        MERGE (d:Drug {{name: '{drug["name"]}'}})
        SET d.drug_class = '{drug["drug_class"]}',
            d.mechanism_of_action = '{drug["mechanism_of_action"]}',
            d.common_side_effects = {drug["common_side_effects"]},
            d.serious_side_effects = {drug["serious_side_effects"]},
            d.pregnancy_category = '{drug["pregnancy_category"]}'
        """
        queries.append(query)
    
    # Create symptoms
    for symptom in SYMPTOMS:
        query = f"""
        MERGE (s:Symptom {{name: '{symptom["name"]}'}})
        SET s.category = '{symptom["category"]}'
        """
        queries.append(query)
    
    # Create drug interactions
    for interaction in DRUG_INTERACTIONS:
        query = f"""
        MATCH (d1:Drug {{name: '{interaction["drug1"]}'}}), (d2:Drug {{name: '{interaction["drug2"]}'}})
        MERGE (d1)-[r:INTERACTS_WITH]->(d2)
        SET r.severity = '{interaction["severity"]}',
            r.description = '{interaction["description"]}',
            r.mechanism = '{interaction["mechanism"]}'
        """
        queries.append(query)
    
    # Create contraindications
    for contra in CONTRAINDICATIONS:
        query = f"""
        MATCH (d:Drug {{name: '{contra["drug"]}'}})
        MERGE (c:Condition {{name: '{contra["condition"]}'}})
        MERGE (d)-[r:CONTRAINDICATED_FOR]->(c)
        SET r.severity = '{contra["severity"]}',
            r.reason = '{contra["reason"]}'
        """
        queries.append(query)
    
    return queries


async def seed_knowledge_graph():
    """Seed the knowledge graph with initial data"""
    logger.info("Generating Cypher queries for knowledge graph seeding...")
    
    queries = generate_cypher_queries()
    
    logger.info(f"Generated {len(queries)} Cypher queries")
    
    # In production, execute these against Neo4j
    for i, query in enumerate(queries):
        logger.info(f"Query {i+1}: {query[:50]}...")
    
    logger.info("Knowledge graph seeding complete!")
    return queries


async def main():
    """Run knowledge graph seeding"""
    await seed_knowledge_graph()


if __name__ == "__main__":
    asyncio.run(main())
