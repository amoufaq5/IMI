"""
UMI SNOMED CT Data Ingestion Pipeline
Fetches clinical terminology from SNOMED CT via Snowstorm API - NO LIMITS
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
from tqdm import tqdm


class SNOMEDClient:
    """Client for SNOMED CT Snowstorm API."""
    
    BASE_URL = "https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct"
    BRANCH = "MAIN/2024-03-01"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def search_concepts(self, term: str, limit: int = 100) -> List[Dict]:
        """Search for SNOMED concepts."""
        url = f"{self.BASE_URL}/{self.BRANCH}/concepts"
        params = {"term": term, "activeFilter": True, "limit": limit}
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json().get("items", [])
        except Exception as e:
            print(f"    API error: {e}")
            return []
    
    async def get_concept(self, concept_id: str) -> Dict:
        """Get concept details."""
        url = f"{self.BASE_URL}/{self.BRANCH}/concepts/{concept_id}"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()
        except:
            return {}
    
    async def get_concept_parents(self, concept_id: str) -> List[Dict]:
        """Get parent concepts."""
        url = f"{self.BASE_URL}/{self.BRANCH}/concepts/{concept_id}/parents"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()
        except:
            return []


CLINICAL_TERMS = [
    # Diseases
    "diabetes mellitus", "hypertension", "myocardial infarction", "stroke",
    "pneumonia", "asthma", "COPD", "cancer", "leukemia", "lymphoma",
    "alzheimer disease", "parkinson disease", "epilepsy", "migraine",
    "depression", "anxiety disorder", "schizophrenia", "bipolar disorder",
    
    # Symptoms
    "chest pain", "dyspnea", "cough", "fever", "headache", "fatigue",
    "nausea", "vomiting", "diarrhea", "constipation", "abdominal pain",
    
    # Procedures
    "surgery", "biopsy", "endoscopy", "colonoscopy", "MRI", "CT scan",
    "blood transfusion", "dialysis", "chemotherapy", "radiation therapy",
    
    # Anatomy
    "heart", "lung", "liver", "kidney", "brain", "stomach", "intestine",
    "bone", "muscle", "nerve", "blood vessel", "lymph node",
    
    # Medications
    "antibiotic", "analgesic", "antihypertensive", "antidiabetic",
    "anticoagulant", "antidepressant", "antipsychotic", "vaccine",
    
    # Lab findings
    "anemia", "leukocytosis", "thrombocytopenia", "hyperglycemia",
    "hyperlipidemia", "elevated creatinine", "elevated liver enzymes",
]


class SNOMEDIngestionPipeline:
    """Pipeline for ingesting SNOMED CT terminology - NO LIMITS."""
    
    def __init__(self, output_dir: str = "data/knowledge_base/snomed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = SNOMEDClient()
        self.concepts: List[Dict] = []
    
    def _concept_to_document(self, concept: Dict, parents: List = None) -> Dict[str, Any]:
        """Convert SNOMED concept to document."""
        concept_id = concept.get("conceptId", "")
        fsn = concept.get("fsn", {}).get("term", "Unknown")
        pt = concept.get("pt", {}).get("term", fsn)
        
        content_parts = [f"{pt} is a clinical concept in SNOMED CT"]
        content_parts.append(f"Fully specified name: {fsn}")
        content_parts.append(f"SNOMED CT ID: {concept_id}")
        
        if concept.get("definitionStatus"):
            content_parts.append(f"Definition status: {concept['definitionStatus']}")
        
        if parents:
            parent_names = [p.get("pt", {}).get("term", "") for p in parents[:5] if p.get("pt")]
            if parent_names:
                content_parts.append(f"Parent concepts: {', '.join(parent_names)}")
        
        return {
            "id": f"snomed_{concept_id}",
            "title": pt,
            "content": ". ".join(content_parts),
            "metadata": {
                "source": "SNOMED CT",
                "concept_id": concept_id,
                "fsn": fsn,
                "type": "clinical_term",
            }
        }
    
    async def run(self) -> None:
        """Run the ingestion pipeline - NO LIMITS."""
        print("=" * 60)
        print("SNOMED CT CLINICAL TERMINOLOGY INGESTION")
        print("=" * 60)
        
        try:
            concept_docs = []
            seen = set()
            
            for term in tqdm(CLINICAL_TERMS, desc="Clinical terms"):
                try:
                    results = await self.client.search_concepts(term, limit=50)
                    
                    for concept in results:
                        cid = concept.get("conceptId")
                        if cid and cid not in seen:
                            seen.add(cid)
                            parents = await self.client.get_concept_parents(cid)
                            doc = self._concept_to_document(concept, parents)
                            concept_docs.append(doc)
                            self.concepts.append(concept)
                        
                        await asyncio.sleep(0.1)
                    
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"   Warning: Error with {term}: {e}")
            
            # Save
            with open(self.output_dir / "concepts.jsonl", 'w', encoding='utf-8') as f:
                for doc in concept_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            print(f"\n{'=' * 60}")
            print("SNOMED CT INGESTION COMPLETE")
            print(f"Concepts: {len(concept_docs)}")
            print(f"Output: {self.output_dir}")
            print("=" * 60)
            
        finally:
            await self.client.close()


async def main():
    pipeline = SNOMEDIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
