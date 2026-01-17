"""
UMI UMLS Data Ingestion Pipeline
Fetches medical terminology from UMLS Metathesaurus - NO LIMITS
Requires UMLS API key from https://uts.nlm.nih.gov/uts/
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import httpx
from tqdm import tqdm


class UMLSClient:
    """Client for UMLS REST API."""
    
    AUTH_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("UMLS_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=60.0)
        self._tgt = None
    
    async def close(self):
        await self.client.aclose()
    
    async def _get_tgt(self) -> str:
        """Get Ticket Granting Ticket."""
        if not self.api_key:
            return ""
        
        try:
            response = await self.client.post(
                self.AUTH_URL,
                data={"apikey": self.api_key},
            )
            if response.status_code == 201:
                # Extract TGT from Location header
                location = response.headers.get("location", "")
                return location
        except:
            pass
        return ""
    
    async def _get_service_ticket(self) -> str:
        """Get service ticket for API call."""
        if not self._tgt:
            self._tgt = await self._get_tgt()
        
        if not self._tgt:
            return ""
        
        try:
            response = await self.client.post(
                self._tgt,
                data={"service": "http://umlsks.nlm.nih.gov"},
            )
            if response.status_code == 200:
                return response.text
        except:
            pass
        return ""
    
    async def search_concepts(self, query: str, page_size: int = 50) -> List[Dict]:
        """Search for UMLS concepts."""
        ticket = await self._get_service_ticket()
        if not ticket:
            return []
        
        url = f"{self.BASE_URL}/search/current"
        params = {
            "string": query,
            "ticket": ticket,
            "pageSize": page_size,
        }
        
        try:
            response = await self.client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return data.get("result", {}).get("results", [])
        except Exception as e:
            print(f"    API error: {e}")
        
        return []
    
    async def get_concept(self, cui: str) -> Dict:
        """Get concept details by CUI."""
        ticket = await self._get_service_ticket()
        if not ticket:
            return {}
        
        url = f"{self.BASE_URL}/content/current/CUI/{cui}"
        params = {"ticket": ticket}
        
        try:
            response = await self.client.get(url, params=params)
            if response.status_code == 200:
                return response.json().get("result", {})
        except:
            pass
        
        return {}
    
    async def get_concept_definitions(self, cui: str) -> List[Dict]:
        """Get definitions for a concept."""
        ticket = await self._get_service_ticket()
        if not ticket:
            return []
        
        url = f"{self.BASE_URL}/content/current/CUI/{cui}/definitions"
        params = {"ticket": ticket}
        
        try:
            response = await self.client.get(url, params=params)
            if response.status_code == 200:
                return response.json().get("result", [])
        except:
            pass
        
        return []


MEDICAL_TERMS = [
    # Diseases
    "diabetes mellitus", "hypertension", "myocardial infarction", "stroke",
    "pneumonia", "asthma", "chronic obstructive pulmonary disease", "cancer",
    "alzheimer disease", "parkinson disease", "epilepsy", "depression",
    "anxiety", "schizophrenia", "arthritis", "osteoporosis",
    
    # Symptoms
    "chest pain", "dyspnea", "cough", "fever", "headache", "fatigue",
    "nausea", "vomiting", "diarrhea", "abdominal pain", "back pain",
    
    # Procedures
    "surgery", "biopsy", "endoscopy", "colonoscopy", "MRI scan",
    "computed tomography", "blood transfusion", "dialysis", "chemotherapy",
    
    # Medications
    "aspirin", "metformin", "atorvastatin", "lisinopril", "amlodipine",
    "omeprazole", "sertraline", "gabapentin", "insulin", "warfarin",
    
    # Anatomy
    "heart", "lung", "liver", "kidney", "brain", "stomach",
    "intestine", "bone", "muscle", "nerve", "blood vessel",
    
    # Lab tests
    "complete blood count", "basic metabolic panel", "lipid panel",
    "hemoglobin A1c", "thyroid function test", "liver function test",
]


class UMLSIngestionPipeline:
    """Pipeline for ingesting UMLS terminology - NO LIMITS."""
    
    def __init__(self, output_dir: str = "data/knowledge_base/umls", api_key: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = UMLSClient(api_key)
        self.concepts: List[Dict] = []
    
    def _concept_to_document(self, concept: Dict, definitions: List = None) -> Dict[str, Any]:
        """Convert UMLS concept to document."""
        cui = concept.get("ui", "")
        name = concept.get("name", "Unknown")
        
        content_parts = [f"{name} is a medical concept in UMLS"]
        content_parts.append(f"Concept Unique Identifier (CUI): {cui}")
        
        semantic_types = concept.get("semanticTypes", [])
        if semantic_types:
            types = [st.get("name", "") for st in semantic_types if st.get("name")]
            if types:
                content_parts.append(f"Semantic types: {', '.join(types)}")
        
        if definitions:
            for defn in definitions[:3]:
                value = defn.get("value", "")
                source = defn.get("rootSource", "")
                if value:
                    content_parts.append(f"Definition ({source}): {value[:500]}")
        
        atoms = concept.get("atomCount", 0)
        if atoms:
            content_parts.append(f"Number of atoms (synonyms): {atoms}")
        
        return {
            "id": f"umls_{cui}",
            "title": name,
            "content": ". ".join(content_parts),
            "metadata": {
                "source": "UMLS",
                "cui": cui,
                "semantic_types": [st.get("name") for st in semantic_types] if semantic_types else [],
                "type": "medical_concept",
            }
        }
    
    async def run(self) -> None:
        """Run the ingestion pipeline - NO LIMITS."""
        print("=" * 60)
        print("UMLS MEDICAL TERMINOLOGY INGESTION")
        print("=" * 60)
        
        if not self.client.api_key:
            print("WARNING: No UMLS API key provided.")
            print("Set UMLS_API_KEY environment variable or get key from https://uts.nlm.nih.gov/uts/")
            print("Skipping UMLS ingestion.")
            return
        
        try:
            concept_docs = []
            seen = set()
            
            for term in tqdm(MEDICAL_TERMS, desc="Medical terms"):
                try:
                    results = await self.client.search_concepts(term, page_size=25)
                    
                    for result in results:
                        cui = result.get("ui")
                        if cui and cui not in seen:
                            seen.add(cui)
                            
                            # Get full concept details
                            concept = await self.client.get_concept(cui)
                            definitions = await self.client.get_concept_definitions(cui)
                            
                            if concept:
                                doc = self._concept_to_document(concept, definitions)
                                concept_docs.append(doc)
                                self.concepts.append(concept)
                            
                            await asyncio.sleep(0.2)
                    
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"   Warning: Error with {term}: {e}")
            
            # Save
            with open(self.output_dir / "concepts.jsonl", 'w', encoding='utf-8') as f:
                for doc in concept_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            print(f"\n{'=' * 60}")
            print("UMLS INGESTION COMPLETE")
            print(f"Concepts: {len(concept_docs)}")
            print(f"Output: {self.output_dir}")
            print("=" * 60)
            
        finally:
            await self.client.close()


async def main():
    pipeline = UMLSIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
