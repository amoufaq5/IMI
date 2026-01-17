"""
UMI DisGeNET Data Ingestion Pipeline
Fetches gene-disease associations - NO LIMITS
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import httpx
from tqdm import tqdm


class DisGeNETClient:
    """Client for DisGeNET API."""
    
    BASE_URL = "https://www.disgenet.org/api"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("DISGENET_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        await self.client.aclose()
    
    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def search_diseases(self, query: str, limit: int = 100) -> List[Dict]:
        """Search for diseases."""
        url = f"{self.BASE_URL}/gda/disease/{query}"
        params = {"limit": limit}
        
        try:
            response = await self.client.get(url, headers=self._headers(), params=params)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"    API error: {e}")
            return []
    
    async def search_genes(self, gene: str, limit: int = 100) -> List[Dict]:
        """Search for gene-disease associations."""
        url = f"{self.BASE_URL}/gda/gene/{gene}"
        params = {"limit": limit}
        
        try:
            response = await self.client.get(url, headers=self._headers(), params=params)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"    API error: {e}")
            return []


DISEASE_QUERIES = [
    "cancer", "diabetes", "alzheimer", "parkinson", "cardiovascular",
    "asthma", "arthritis", "depression", "schizophrenia", "autism",
    "obesity", "hypertension", "stroke", "epilepsy", "leukemia",
]

GENE_QUERIES = [
    "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "BRAF", "PIK3CA",
    "APC", "PTEN", "RB1", "MYC", "ERBB2", "CDK4", "MDM2",
    "APOE", "APP", "MAPT", "SNCA", "LRRK2", "GBA",
    "CFTR", "DMD", "HTT", "FMR1", "SMN1",
    "INS", "GCK", "HNF1A", "HNF4A", "KCNJ11",
    "ACE", "AGT", "AGTR1", "NOS3", "ADRB1",
]


class DisGeNETIngestionPipeline:
    """Pipeline for ingesting DisGeNET gene-disease associations - NO LIMITS."""
    
    def __init__(self, output_dir: str = "data/knowledge_base/disgenet", api_key: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = DisGeNETClient(api_key)
        self.associations: List[Dict] = []
    
    def _association_to_document(self, assoc: Dict) -> Dict[str, Any]:
        """Convert gene-disease association to document."""
        gene = assoc.get("gene_symbol", "Unknown")
        disease = assoc.get("disease_name", "Unknown")
        score = assoc.get("score", 0)
        
        content_parts = [
            f"Gene-disease association: {gene} is associated with {disease}",
            f"Association score: {score:.3f}" if score else "",
        ]
        
        if assoc.get("source"):
            content_parts.append(f"Source: {assoc['source']}")
        
        if assoc.get("pmid"):
            content_parts.append(f"PubMed ID: {assoc['pmid']}")
        
        ei = assoc.get("ei")
        if ei:
            content_parts.append(f"Evidence index: {ei}")
        
        return {
            "id": f"disgenet_{gene}_{assoc.get('diseaseid', '')}",
            "title": f"{gene} - {disease} Association",
            "content": ". ".join([p for p in content_parts if p]),
            "metadata": {
                "source": "DisGeNET",
                "gene_symbol": gene,
                "disease_name": disease,
                "disease_id": assoc.get("diseaseid", ""),
                "score": score,
                "type": "gene_disease_association",
            }
        }
    
    async def run(self) -> None:
        """Run the ingestion pipeline - NO LIMITS."""
        print("=" * 60)
        print("DISGENET GENE-DISEASE ASSOCIATIONS INGESTION")
        print("=" * 60)
        
        if not self.client.api_key:
            print("WARNING: No DisGeNET API key provided. Results may be limited.")
            print("Set DISGENET_API_KEY environment variable for full access.")
        
        try:
            assoc_docs = []
            seen = set()
            
            # Search by disease
            print("\n1. Searching by disease...")
            for query in tqdm(DISEASE_QUERIES, desc="Disease searches"):
                try:
                    results = await self.client.search_diseases(query, limit=200)
                    
                    for assoc in results:
                        key = f"{assoc.get('gene_symbol')}_{assoc.get('diseaseid')}"
                        if key not in seen:
                            seen.add(key)
                            doc = self._association_to_document(assoc)
                            assoc_docs.append(doc)
                            self.associations.append(assoc)
                    
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"   Warning: Error with {query}: {e}")
            
            # Search by gene
            print("\n2. Searching by gene...")
            for gene in tqdm(GENE_QUERIES, desc="Gene searches"):
                try:
                    results = await self.client.search_genes(gene, limit=200)
                    
                    for assoc in results:
                        key = f"{assoc.get('gene_symbol')}_{assoc.get('diseaseid')}"
                        if key not in seen:
                            seen.add(key)
                            doc = self._association_to_document(assoc)
                            assoc_docs.append(doc)
                            self.associations.append(assoc)
                    
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"   Warning: Error with {gene}: {e}")
            
            # Save
            with open(self.output_dir / "gene_disease_associations.jsonl", 'w', encoding='utf-8') as f:
                for doc in assoc_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            print(f"\n{'=' * 60}")
            print("DISGENET INGESTION COMPLETE")
            print(f"Associations: {len(assoc_docs)}")
            print(f"Output: {self.output_dir}")
            print("=" * 60)
            
        finally:
            await self.client.close()


async def main():
    pipeline = DisGeNETIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
