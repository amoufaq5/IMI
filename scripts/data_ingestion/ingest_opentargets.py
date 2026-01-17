"""
UMI Open Targets Data Ingestion Pipeline
Fetches drug-target-disease associations - NO LIMITS
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from tqdm import tqdm


class OpenTargetsClient:
    """Client for Open Targets GraphQL API."""
    
    BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def query(self, query: str, variables: Dict = None) -> Dict[str, Any]:
        """Execute GraphQL query."""
        try:
            response = await self.client.post(
                self.BASE_URL,
                json={"query": query, "variables": variables or {}},
            )
            response.raise_for_status()
            return response.json().get("data", {})
        except Exception as e:
            print(f"    API error: {e}")
            return {}
    
    async def search_diseases(self, query: str, size: int = 100) -> List[Dict]:
        """Search for diseases."""
        gql = """
        query SearchDiseases($query: String!, $size: Int!) {
            search(queryString: $query, entityNames: ["disease"], page: {size: $size, index: 0}) {
                hits {
                    id
                    name
                    description
                    entity
                }
            }
        }
        """
        data = await self.query(gql, {"query": query, "size": size})
        return data.get("search", {}).get("hits", [])
    
    async def get_disease(self, disease_id: str) -> Dict[str, Any]:
        """Get disease details."""
        gql = """
        query GetDisease($diseaseId: String!) {
            disease(efoId: $diseaseId) {
                id
                name
                description
                synonyms { terms }
                therapeuticAreas { id name }
                knownDrugs { uniqueDrugs uniqueTargets }
            }
        }
        """
        data = await self.query(gql, {"diseaseId": disease_id})
        return data.get("disease", {})
    
    async def get_disease_drugs(self, disease_id: str, size: int = 500) -> List[Dict]:
        """Get drugs associated with a disease."""
        gql = """
        query GetDiseaseDrugs($diseaseId: String!, $size: Int!) {
            disease(efoId: $diseaseId) {
                knownDrugs(size: $size) {
                    rows {
                        drug { id name drugType maximumClinicalTrialPhase mechanismsOfAction { actionType targetName } }
                        target { id approvedSymbol approvedName }
                        disease { id name }
                        phase
                        status
                        urls { name url }
                    }
                }
            }
        }
        """
        data = await self.query(gql, {"diseaseId": disease_id, "size": size})
        return data.get("disease", {}).get("knownDrugs", {}).get("rows", [])
    
    async def search_drugs(self, query: str, size: int = 100) -> List[Dict]:
        """Search for drugs."""
        gql = """
        query SearchDrugs($query: String!, $size: Int!) {
            search(queryString: $query, entityNames: ["drug"], page: {size: $size, index: 0}) {
                hits {
                    id
                    name
                    description
                    entity
                }
            }
        }
        """
        data = await self.query(gql, {"query": query, "size": size})
        return data.get("search", {}).get("hits", [])
    
    async def get_drug(self, drug_id: str) -> Dict[str, Any]:
        """Get drug details."""
        gql = """
        query GetDrug($drugId: String!) {
            drug(chemblId: $drugId) {
                id
                name
                drugType
                maximumClinicalTrialPhase
                hasBeenWithdrawn
                description
                synonyms
                tradeNames
                mechanismsOfAction { actionType targetName targets { id approvedSymbol } }
                indications { rows { disease { id name } maxPhaseForIndication } }
                linkedDiseases { rows { id name } }
                linkedTargets { rows { id approvedSymbol approvedName } }
            }
        }
        """
        data = await self.query(gql, {"drugId": drug_id})
        return data.get("drug", {})


DISEASE_QUERIES = [
    "cancer", "diabetes", "cardiovascular", "alzheimer", "parkinson",
    "arthritis", "asthma", "depression", "anxiety", "schizophrenia",
    "epilepsy", "multiple sclerosis", "lupus", "crohn", "colitis",
    "hepatitis", "cirrhosis", "kidney disease", "hypertension", "stroke",
    "leukemia", "lymphoma", "melanoma", "breast cancer", "lung cancer",
    "prostate cancer", "colorectal cancer", "pancreatic cancer", "ovarian cancer",
    "obesity", "osteoporosis", "psoriasis", "eczema", "migraine",
    "fibromyalgia", "chronic pain", "COPD", "pneumonia", "tuberculosis",
    "HIV", "malaria", "influenza", "COVID-19", "sepsis",
]

DRUG_QUERIES = [
    "aspirin", "metformin", "atorvastatin", "lisinopril", "amlodipine",
    "omeprazole", "sertraline", "gabapentin", "losartan", "metoprolol",
    "pembrolizumab", "nivolumab", "adalimumab", "rituximab", "trastuzumab",
    "insulin", "semaglutide", "ozempic", "humira", "keytruda",
]


class OpenTargetsIngestionPipeline:
    """Pipeline for ingesting Open Targets data - NO LIMITS."""
    
    def __init__(self, output_dir: str = "data/knowledge_base/opentargets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenTargetsClient()
        self.diseases: List[Dict] = []
        self.drugs: List[Dict] = []
    
    def _disease_to_document(self, disease: Dict, drug_associations: List = None) -> Dict[str, Any]:
        """Convert disease to document."""
        name = disease.get("name", "Unknown")
        disease_id = disease.get("id", "")
        
        content_parts = [disease.get("description", f"{name} is a medical condition")]
        
        synonyms = disease.get("synonyms", {})
        if synonyms and synonyms.get("terms"):
            content_parts.append(f"Also known as: {', '.join(synonyms['terms'][:10])}")
        
        therapeutic_areas = disease.get("therapeuticAreas", [])
        if therapeutic_areas:
            areas = [ta.get("name") for ta in therapeutic_areas[:5] if ta.get("name")]
            if areas:
                content_parts.append(f"Therapeutic areas: {', '.join(areas)}")
        
        known_drugs = disease.get("knownDrugs", {})
        if known_drugs:
            content_parts.append(f"Known drugs: {known_drugs.get('uniqueDrugs', 0)}, Targets: {known_drugs.get('uniqueTargets', 0)}")
        
        if drug_associations:
            drug_names = list(set([d.get("drug", {}).get("name") for d in drug_associations[:20] if d.get("drug", {}).get("name")]))
            if drug_names:
                content_parts.append(f"Associated drugs include: {', '.join(drug_names[:15])}")
        
        return {
            "id": f"opentargets_disease_{disease_id}",
            "title": name,
            "content": ". ".join(content_parts),
            "metadata": {
                "source": "OpenTargets",
                "disease_id": disease_id,
                "type": "disease",
            }
        }
    
    def _drug_to_document(self, drug: Dict) -> Dict[str, Any]:
        """Convert drug to document."""
        name = drug.get("name", "Unknown")
        drug_id = drug.get("id", "")
        
        content_parts = [drug.get("description") or f"{name} is a pharmaceutical drug"]
        
        drug_type = drug.get("drugType")
        if drug_type:
            content_parts.append(f"Drug type: {drug_type}")
        
        max_phase = drug.get("maximumClinicalTrialPhase")
        if max_phase:
            content_parts.append(f"Maximum clinical trial phase: {max_phase}")
        
        mechanisms = drug.get("mechanismsOfAction", [])
        if mechanisms:
            mech_list = [f"{m.get('actionType')} on {m.get('targetName')}" for m in mechanisms[:5] if m.get('actionType')]
            if mech_list:
                content_parts.append(f"Mechanisms: {'; '.join(mech_list)}")
        
        indications = drug.get("indications", {}).get("rows", [])
        if indications:
            ind_names = [i.get("disease", {}).get("name") for i in indications[:10] if i.get("disease", {}).get("name")]
            if ind_names:
                content_parts.append(f"Indications: {', '.join(ind_names)}")
        
        trade_names = drug.get("tradeNames", [])
        if trade_names:
            content_parts.append(f"Trade names: {', '.join(trade_names[:10])}")
        
        return {
            "id": f"opentargets_drug_{drug_id}",
            "title": name,
            "content": ". ".join(content_parts),
            "metadata": {
                "source": "OpenTargets",
                "drug_id": drug_id,
                "drug_type": drug_type,
                "max_phase": max_phase,
                "type": "drug",
            }
        }
    
    async def run(self) -> None:
        """Run the ingestion pipeline - NO LIMITS."""
        print("=" * 60)
        print("OPEN TARGETS DATA INGESTION")
        print("=" * 60)
        
        try:
            disease_docs = []
            drug_docs = []
            seen_diseases = set()
            seen_drugs = set()
            
            # 1. Search and fetch diseases
            print("\n1. Fetching diseases...")
            for query in tqdm(DISEASE_QUERIES, desc="Disease searches"):
                try:
                    results = await self.client.search_diseases(query, size=100)
                    
                    for hit in results:
                        disease_id = hit.get("id")
                        if not disease_id or disease_id in seen_diseases:
                            continue
                        
                        seen_diseases.add(disease_id)
                        
                        # Get full disease details
                        disease = await self.client.get_disease(disease_id)
                        if disease:
                            drug_assocs = await self.client.get_disease_drugs(disease_id, size=100)
                            doc = self._disease_to_document(disease, drug_assocs)
                            disease_docs.append(doc)
                            self.diseases.append(disease)
                        
                        await asyncio.sleep(0.2)
                    
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"   Warning: Error with {query}: {e}")
            
            print(f"   Total diseases: {len(disease_docs)}")
            
            # 2. Search and fetch drugs
            print("\n2. Fetching drugs...")
            for query in tqdm(DRUG_QUERIES, desc="Drug searches"):
                try:
                    results = await self.client.search_drugs(query, size=50)
                    
                    for hit in results:
                        drug_id = hit.get("id")
                        if not drug_id or drug_id in seen_drugs:
                            continue
                        
                        seen_drugs.add(drug_id)
                        
                        drug = await self.client.get_drug(drug_id)
                        if drug:
                            doc = self._drug_to_document(drug)
                            drug_docs.append(doc)
                            self.drugs.append(drug)
                        
                        await asyncio.sleep(0.2)
                    
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"   Warning: Error with {query}: {e}")
            
            print(f"   Total drugs: {len(drug_docs)}")
            
            # Save
            with open(self.output_dir / "diseases.jsonl", 'w', encoding='utf-8') as f:
                for doc in disease_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            with open(self.output_dir / "drugs.jsonl", 'w', encoding='utf-8') as f:
                for doc in drug_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            print(f"\n{'=' * 60}")
            print("OPEN TARGETS INGESTION COMPLETE")
            print(f"Diseases: {len(disease_docs)}")
            print(f"Drugs: {len(drug_docs)}")
            print(f"Output: {self.output_dir}")
            print("=" * 60)
            
        finally:
            await self.client.close()


async def main():
    pipeline = OpenTargetsIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
