"""
UMI SNOMED CT Data Ingestion Pipeline
<<<<<<< HEAD
Fetches clinical terminology from SNOMED CT via Snowstorm API - NO LIMITS
=======
Fetches clinical terminology from SNOMED CT via public APIs
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
"""

import asyncio
import json
<<<<<<< HEAD
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
=======
@dataclass
class SNOMEDConcept:
    """Represents a SNOMED CT concept."""
    concept_id: str
    fsn: str  # Fully Specified Name
    preferred_term: str
    semantic_tag: str
    definitions: List[str]
    parents: List[Dict[str, str]]
    children: List[Dict[str, str]]
    relationships: List[Dict[str, str]]


class SNOMEDClient:
    """
    Client for SNOMED CT Browser API (Snowstorm).
    https://browser.ihtsdotools.org/snowstorm/snomed-ct/
    """
    
    BASE_URL = "https://browser.ihtsdotools.org/snowstorm/snomed-ct"
    BRANCH = "MAIN/2024-01-01"  # International Edition
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search_concepts(
        self,
        query: str,
        limit: int = 100,
        semantic_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for SNOMED CT concepts."""
        params = {
            "term": query,
            "limit": limit,
            "activeFilter": "true",
            "termActive": "true",
        }
        
        if semantic_filter:
            params["semanticFilter"] = semantic_filter
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/browser/{self.BRANCH}/descriptions",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("items", [])
        except Exception as e:
            print(f"Error searching concepts: {e}")
            return []
    
    async def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed concept information."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/browser/{self.BRANCH}/concepts/{concept_id}",
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting concept {concept_id}: {e}")
            return None
    
    async def get_concept_parents(self, concept_id: str) -> List[Dict[str, str]]:
        """Get parent concepts."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/{self.BRANCH}/concepts/{concept_id}/parents",
            )
            response.raise_for_status()
            data = response.json()
            return [
                {"id": p.get("conceptId", ""), "term": p.get("pt", {}).get("term", "")}
                for p in data
            ]
        except Exception as e:
            return []
    
    async def get_concept_children(self, concept_id: str, limit: int = 50) -> List[Dict[str, str]]:
        """Get child concepts."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/{self.BRANCH}/concepts/{concept_id}/children",
                params={"limit": limit},
            )
            response.raise_for_status()
            data = response.json()
            return [
                {"id": c.get("conceptId", ""), "term": c.get("pt", {}).get("term", "")}
                for c in data
            ]
        except Exception as e:
            return []
    
    async def get_concept_relationships(self, concept_id: str) -> List[Dict[str, str]]:
        """Get concept relationships."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/browser/{self.BRANCH}/concepts/{concept_id}",
            )
            response.raise_for_status()
            data = response.json()
            
            relationships = []
            for rel in data.get("relationships", []):
                if rel.get("active"):
                    relationships.append({
                        "type": rel.get("type", {}).get("pt", {}).get("term", ""),
                        "target_id": rel.get("target", {}).get("conceptId", ""),
                        "target_term": rel.get("target", {}).get("pt", {}).get("term", ""),
                    })
            
            return relationships
        except Exception as e:
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class SNOMEDIngestionPipeline:
    """
    Pipeline for ingesting SNOMED CT concepts into UMI knowledge base.
    """
    
    # Clinical terms to search
    SEARCH_TERMS = [
        # Clinical findings
        "diabetes mellitus", "hypertension", "heart failure",
        "myocardial infarction", "stroke", "pneumonia",
        "asthma", "chronic obstructive pulmonary disease",
        "chronic kidney disease", "liver cirrhosis",
        "cancer", "neoplasm", "tumor", "carcinoma",
        "infection", "sepsis", "fever", "inflammation",
        "pain", "headache", "chest pain", "abdominal pain",
        "dyspnea", "cough", "edema", "fatigue",
        "anemia", "thrombocytopenia", "leukocytosis",
        "hyperglycemia", "hypoglycemia", "hyperlipidemia",
        "hypothyroidism", "hyperthyroidism",
        "depression", "anxiety", "psychosis", "dementia",
        "fracture", "wound", "burn", "laceration",
        
        # Procedures
        "surgery", "biopsy", "endoscopy", "catheterization",
        "dialysis", "transplantation", "chemotherapy",
        "radiation therapy", "blood transfusion",
        "intubation", "ventilation", "resuscitation",
        
        # Body structures
        "heart", "lung", "liver", "kidney", "brain",
        "pancreas", "stomach", "intestine", "colon",
        "bone", "muscle", "nerve", "blood vessel",
        "artery", "vein", "lymph node", "skin",
        
        # Substances
        "drug", "medication", "antibiotic", "insulin",
        "vaccine", "blood product", "contrast agent",
        
        # Organisms
        "bacteria", "virus", "fungus", "parasite",
        "Staphylococcus", "Streptococcus", "Escherichia coli",
        "influenza virus", "coronavirus", "HIV",
        
        # Observable entities
        "blood pressure", "heart rate", "temperature",
        "respiratory rate", "oxygen saturation",
        "body weight", "body mass index",
        "blood glucose", "hemoglobin", "creatinine",
        
        # Situations
        "pregnancy", "postoperative state", "immunocompromised",
        "bedridden", "wheelchair bound",
    ]
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/snomed",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = SNOMEDClient()
        self.concepts: List[SNOMEDConcept] = []
        self.processed_ids: Set[str] = set()
    
    async def fetch_concepts(self, term: str) -> List[SNOMEDConcept]:
        """Fetch concepts for a search term."""
        concepts = []
        
        # Search for concepts
        results = await self.client.search_concepts(term)
        
        for result in results[:15]:  # Limit per term
            concept_id = result.get("concept", {}).get("conceptId", "")
            if not concept_id or concept_id in self.processed_ids:
                continue
            
            self.processed_ids.add(concept_id)
            
            # Get detailed info
            concept_data = await self.client.get_concept(concept_id)
            if not concept_data:
                continue
            
            # Get relationships
            parents = await self.client.get_concept_parents(concept_id)
            children = await self.client.get_concept_children(concept_id)
            relationships = await self.client.get_concept_relationships(concept_id)
            
            # Extract FSN and semantic tag
            fsn = concept_data.get("fsn", {}).get("term", "")
            semantic_tag = ""
            if "(" in fsn and fsn.endswith(")"):
                semantic_tag = fsn[fsn.rfind("(") + 1:-1]
            
            # Get definitions
            definitions = []
            for desc in concept_data.get("descriptions", []):
                if desc.get("type") == "TEXT_DEFINITION" and desc.get("active"):
                    definitions.append(desc.get("term", ""))
            
            concept = SNOMEDConcept(
                concept_id=concept_id,
                fsn=fsn,
                preferred_term=concept_data.get("pt", {}).get("term", ""),
                semantic_tag=semantic_tag,
                definitions=definitions,
                parents=parents[:10],
                children=children[:20],
                relationships=relationships[:20],
            )
            concepts.append(concept)
            
            await asyncio.sleep(0.2)  # Rate limiting
        
        return concepts
    
    async def run(self) -> None:
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI SNOMED CT Ingestion Pipeline")
        print("=" * 60)
        
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
                    f"SNOMED CT Concept: {concept.preferred_term}",
                    f"Concept ID: {concept.concept_id}",
                    f"Fully Specified Name: {concept.fsn}",
                    f"Semantic Tag: {concept.semantic_tag}",
                    "",
                ]
                
                if concept.definitions:
                    content_parts.append("Definitions:")
                    for defn in concept.definitions:
                        content_parts.append(f"- {defn}")
                    content_parts.append("")
                
                if concept.parents:
                    content_parts.append("Parent Concepts:")
                    for parent in concept.parents:
                        content_parts.append(f"- {parent.get('term', '')} ({parent.get('id', '')})")
                    content_parts.append("")
                
                if concept.children:
                    content_parts.append("Child Concepts:")
                    for child in concept.children[:10]:
                        content_parts.append(f"- {child.get('term', '')} ({child.get('id', '')})")
                    content_parts.append("")
                
                if concept.relationships:
                    content_parts.append("Relationships:")
                    for rel in concept.relationships[:10]:
                        content_parts.append(
                            f"- {rel.get('type', '')}: {rel.get('target_term', '')} ({rel.get('target_id', '')})"
                        )
                
                doc = {
                    "id": f"snomed_{concept.concept_id}",
                    "title": concept.preferred_term,
                    "content": "\n".join(content_parts),
                    "metadata": {
                        "concept_id": concept.concept_id,
                        "fsn": concept.fsn,
                        "semantic_tag": concept.semantic_tag,
                        "parent_count": len(concept.parents),
                        "child_count": len(concept.children),
                        "source": "SNOMED CT",
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save statistics
        stats = {
            "total_concepts": len(self.concepts),
            "ingestion_date": datetime.now().isoformat(),
            "search_terms": self.SEARCH_TERMS,
            "semantic_tags": list(set(c.semantic_tag for c in self.concepts if c.semantic_tag)),
        }
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


async def main():
    """Run the SNOMED CT ingestion pipeline."""
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    pipeline = SNOMEDIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
