"""
<<<<<<< HEAD
UMI UMLS Data Ingestion Pipeline
Fetches medical terminology from UMLS Metathesaurus - NO LIMITS
Requires UMLS API key from https://uts.nlm.nih.gov/uts/
=======
UMI UMLS (Unified Medical Language System) Data Ingestion Pipeline
Fetches medical terminology and concepts from NLM UMLS API
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
"""

import asyncio
import json
<<<<<<< HEAD
import os
from pathlib import Path
from typing import Any, Dict, List
=======
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)

import httpx
from tqdm import tqdm


<<<<<<< HEAD
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
=======
@dataclass
class UMLSConcept:
    """Represents a UMLS concept."""
    cui: str
    name: str
    semantic_types: List[str]
    definitions: List[str]
    atoms: List[Dict[str, str]]
    relations: List[Dict[str, str]]
    source_vocabularies: List[str]


class UMLSClient:
    """
    Client for UMLS REST API.
    https://documentation.uts.nlm.nih.gov/rest/home.html
    
    Requires UMLS API key from https://uts.nlm.nih.gov/uts/
    """
    
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    AUTH_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.environ.get("UMLS_API_KEY")
        self.client = httpx.AsyncClient(timeout=30.0)
        self.tgt = None  # Ticket Granting Ticket
    
    async def _get_service_ticket(self) -> Optional[str]:
        """Get a service ticket for API access."""
        if not self.api_key:
            print("WARNING: UMLS API key not set. Set UMLS_API_KEY environment variable.")
            return None
        
        try:
            # Get TGT if not exists
            if not self.tgt:
                response = await self.client.post(
                    self.AUTH_URL,
                    data={"apikey": self.api_key},
                )
                response.raise_for_status()
                self.tgt = response.headers.get("location")
            
            # Get service ticket
            if self.tgt:
                response = await self.client.post(
                    self.tgt,
                    data={"service": "http://umlsks.nlm.nih.gov"},
                )
                response.raise_for_status()
                return response.text
        except Exception as e:
            print(f"Error getting service ticket: {e}")
        
        return None
    
    async def search_concepts(
        self,
        query: str,
        search_type: str = "words",
        page_size: int = 50,
    ) -> List[Dict[str, Any]]:
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
        """Search for UMLS concepts."""
        ticket = await self._get_service_ticket()
        if not ticket:
            return []
        
<<<<<<< HEAD
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
=======
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/search/current",
                params={
                    "string": query,
                    "searchType": search_type,
                    "pageSize": page_size,
                    "ticket": ticket,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result", {}).get("results", [])
        except Exception as e:
            print(f"Error searching concepts: {e}")
            return []
    
    async def get_concept(self, cui: str) -> Optional[Dict[str, Any]]:
        """Get detailed concept information."""
        ticket = await self._get_service_ticket()
        if not ticket:
            return None
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/content/current/CUI/{cui}",
                params={"ticket": ticket},
            )
            response.raise_for_status()
            return response.json().get("result", {})
        except Exception as e:
            print(f"Error getting concept {cui}: {e}")
            return None
    
    async def get_concept_definitions(self, cui: str) -> List[str]:
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
        """Get definitions for a concept."""
        ticket = await self._get_service_ticket()
        if not ticket:
            return []
        
<<<<<<< HEAD
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
=======
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/content/current/CUI/{cui}/definitions",
                params={"ticket": ticket},
            )
            response.raise_for_status()
            results = response.json().get("result", [])
            return [r.get("value", "") for r in results if r.get("value")]
        except Exception as e:
            return []
    
    async def get_concept_relations(self, cui: str) -> List[Dict[str, str]]:
        """Get relations for a concept."""
        ticket = await self._get_service_ticket()
        if not ticket:
            return []
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/content/current/CUI/{cui}/relations",
                params={"ticket": ticket, "pageSize": 50},
            )
            response.raise_for_status()
            results = response.json().get("result", [])
            return [
                {
                    "relation": r.get("relationLabel", ""),
                    "related_cui": r.get("relatedId", "").split("/")[-1],
                    "related_name": r.get("relatedIdName", ""),
                }
                for r in results
            ]
        except Exception as e:
            return []
    
    async def get_semantic_types(self, cui: str) -> List[str]:
        """Get semantic types for a concept."""
        ticket = await self._get_service_ticket()
        if not ticket:
            return []
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/content/current/CUI/{cui}",
                params={"ticket": ticket},
            )
            response.raise_for_status()
            result = response.json().get("result", {})
            return [st.get("name", "") for st in result.get("semanticTypes", [])]
        except Exception as e:
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class UMLSIngestionPipeline:
    """
    Pipeline for ingesting UMLS concepts into UMI knowledge base.
    """
    
    # Medical terms to search for concepts
    SEARCH_TERMS = [
        # Diseases
        "diabetes mellitus", "hypertension", "coronary artery disease",
        "heart failure", "myocardial infarction", "stroke",
        "chronic kidney disease", "liver cirrhosis", "hepatitis",
        "asthma", "chronic obstructive pulmonary disease", "pneumonia",
        "cancer", "neoplasm", "carcinoma", "lymphoma", "leukemia",
        "Alzheimer disease", "Parkinson disease", "multiple sclerosis",
        "epilepsy", "migraine", "depression", "anxiety", "schizophrenia",
        "rheumatoid arthritis", "osteoarthritis", "lupus", "psoriasis",
        "inflammatory bowel disease", "Crohn disease", "ulcerative colitis",
        "HIV infection", "tuberculosis", "sepsis", "meningitis",
        "obesity", "metabolic syndrome", "hyperlipidemia", "gout",
        
        # Symptoms
        "chest pain", "dyspnea", "cough", "fever", "fatigue",
        "headache", "dizziness", "nausea", "vomiting", "diarrhea",
        "abdominal pain", "back pain", "joint pain", "muscle weakness",
        "edema", "weight loss", "weight gain", "insomnia", "anxiety",
        
        # Procedures
        "surgery", "biopsy", "endoscopy", "colonoscopy", "bronchoscopy",
        "angiography", "catheterization", "dialysis", "transplantation",
        "chemotherapy", "radiation therapy", "immunotherapy",
        
        # Diagnostics
        "blood test", "urinalysis", "imaging", "MRI", "CT scan",
        "ultrasound", "X-ray", "ECG", "EEG", "biopsy",
        
        # Anatomy
        "heart", "lung", "liver", "kidney", "brain", "pancreas",
        "stomach", "intestine", "colon", "bone", "muscle", "nerve",
        "blood vessel", "artery", "vein", "lymph node",
        
        # Pharmacology
        "drug", "medication", "antibiotic", "antiviral", "antifungal",
        "analgesic", "anti-inflammatory", "antihypertensive", "antidiabetic",
        "anticoagulant", "antidepressant", "antipsychotic", "chemotherapy agent",
        
        # Lab values
        "hemoglobin", "glucose", "creatinine", "bilirubin", "albumin",
        "white blood cell", "platelet", "cholesterol", "triglyceride",
        "sodium", "potassium", "calcium", "magnesium",
    ]
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/umls",
        api_key: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = UMLSClient(api_key=api_key)
        self.concepts: List[UMLSConcept] = []
        self.processed_cuis: Set[str] = set()
    
    async def fetch_concepts(self, term: str) -> List[UMLSConcept]:
        """Fetch concepts for a search term."""
        concepts = []
        
        # Search for concepts
        results = await self.client.search_concepts(term)
        
        for result in results[:10]:  # Limit per term
            cui = result.get("ui", "")
            if not cui or cui in self.processed_cuis:
                continue
            
            self.processed_cuis.add(cui)
            
            # Get additional details
            definitions = await self.client.get_concept_definitions(cui)
            relations = await self.client.get_concept_relations(cui)
            semantic_types = await self.client.get_semantic_types(cui)
            
            concept = UMLSConcept(
                cui=cui,
                name=result.get("name", ""),
                semantic_types=semantic_types,
                definitions=definitions,
                atoms=[],
                relations=relations[:20],
                source_vocabularies=result.get("rootSource", "").split(","),
            )
            concepts.append(concept)
            
            await asyncio.sleep(0.2)  # Rate limiting
        
        return concepts
    
    async def run(self) -> None:
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI UMLS Ingestion Pipeline")
        print("=" * 60)
        
        if not self.client.api_key:
            print("\nWARNING: UMLS API key not configured.")
            print("To use UMLS data:")
            print("  1. Register at https://uts.nlm.nih.gov/uts/")
            print("  2. Get your API key from your profile")
            print("  3. Set UMLS_API_KEY environment variable")
            print("\nSkipping UMLS ingestion...")
            return
        
        all_concepts = []
        
        for term in tqdm(self.SEARCH_TERMS, desc="Fetching concepts"):
            try:
                concepts = await self.fetch_concepts(term)
                all_concepts.extend(concepts)
                print(f"  {term}: {len(concepts)} concepts")
            except Exception as e:
                print(f"  Error: {e}")
            
            await asyncio.sleep(0.5)
        
        self.concepts = all_concepts
        print(f"\nTotal unique concepts: {len(self.concepts)}")
        
        # Save
        await self.save()
        
        # Close client
        await self.client.close()
    
    async def save(self) -> None:
        """Save concepts to disk."""
        output_file = self.output_dir / "concepts.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for concept in self.concepts:
                content_parts = [
                    f"Concept: {concept.name}",
                    f"CUI: {concept.cui}",
                    "",
                ]
                
                if concept.semantic_types:
                    content_parts.append(f"Semantic Types: {', '.join(concept.semantic_types)}")
                    content_parts.append("")
                
                if concept.definitions:
                    content_parts.append("Definitions:")
                    for defn in concept.definitions[:3]:
                        content_parts.append(f"- {defn}")
                    content_parts.append("")
                
                if concept.relations:
                    content_parts.append("Related Concepts:")
                    for rel in concept.relations[:10]:
                        content_parts.append(
                            f"- {rel.get('relation', '')}: {rel.get('related_name', '')} ({rel.get('related_cui', '')})"
                        )
                
                doc = {
                    "id": f"umls_{concept.cui}",
                    "title": concept.name,
                    "content": "\n".join(content_parts),
                    "metadata": {
                        "cui": concept.cui,
                        "semantic_types": concept.semantic_types,
                        "source_vocabularies": concept.source_vocabularies,
                        "source": "UMLS",
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save statistics
        stats = {
            "total_concepts": len(self.concepts),
            "ingestion_date": datetime.now().isoformat(),
            "search_terms": self.SEARCH_TERMS,
        }
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


async def main():
    """Run the UMLS ingestion pipeline."""
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    pipeline = UMLSIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
