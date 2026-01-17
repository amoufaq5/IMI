"""
UMI DisGeNET Data Ingestion Pipeline
<<<<<<< HEAD
Fetches gene-disease associations - NO LIMITS
=======
Fetches gene-disease associations from DisGeNET
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
"""

import asyncio
import json
<<<<<<< HEAD
import os
from pathlib import Path
from typing import Any, Dict, List
=======
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)

import httpx
from tqdm import tqdm


<<<<<<< HEAD
class DisGeNETClient:
    """Client for DisGeNET API."""
    
    BASE_URL = "https://www.disgenet.org/api"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("DISGENET_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        await self.client.aclose()
    
    def _headers(self) -> Dict[str, str]:
=======
@dataclass
class GeneDiseaseAssociation:
    """Represents a gene-disease association."""
    gene_id: str
    gene_symbol: str
    disease_id: str
    disease_name: str
    score: float
    evidence_count: int
    source: str


class DisGeNETClient:
    """
    Client for DisGeNET REST API.
    https://www.disgenet.org/api/
    """
    
    BASE_URL = "https://www.disgenet.org/api"
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.environ.get("DISGENET_API_KEY")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
<<<<<<< HEAD
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
=======
    async def search_diseases(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search for diseases."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/disease/search/{query}",
                headers=self._get_headers(),
                params={"limit": limit},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error searching diseases: {e}")
            return []
    
    async def get_disease_genes(self, disease_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get genes associated with a disease."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/gda/disease/{disease_id}",
                headers=self._get_headers(),
                params={"limit": limit},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return []
    
    async def get_gene_diseases(self, gene_symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get diseases associated with a gene."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/gda/gene/{gene_symbol}",
                headers=self._get_headers(),
                params={"limit": limit},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class DisGeNETIngestionPipeline:
    """
    Pipeline for ingesting DisGeNET gene-disease associations into UMI knowledge base.
    """
    
    # Diseases to fetch gene associations for
    DISEASE_QUERIES = [
        # Major diseases
        "diabetes mellitus", "type 2 diabetes", "type 1 diabetes",
        "hypertension", "coronary artery disease", "heart failure",
        "myocardial infarction", "stroke", "atrial fibrillation",
        "chronic kidney disease", "liver cirrhosis",
        "asthma", "chronic obstructive pulmonary disease",
        "breast cancer", "lung cancer", "colorectal cancer",
        "prostate cancer", "pancreatic cancer", "leukemia",
        "Alzheimer disease", "Parkinson disease", "multiple sclerosis",
        "schizophrenia", "bipolar disorder", "major depression",
        "rheumatoid arthritis", "systemic lupus erythematosus",
        "inflammatory bowel disease", "Crohn disease",
        "obesity", "metabolic syndrome",
        
        # Rare diseases
        "cystic fibrosis", "sickle cell disease", "hemophilia",
        "muscular dystrophy", "Huntington disease",
        "amyotrophic lateral sclerosis",
    ]
    
    # Key genes to fetch disease associations for
    GENE_QUERIES = [
        # Oncogenes and tumor suppressors
        "TP53", "BRCA1", "BRCA2", "KRAS", "EGFR", "BRAF",
        "PIK3CA", "PTEN", "APC", "RB1", "MYC", "HER2",
        
        # Cardiovascular genes
        "APOE", "LDLR", "PCSK9", "ACE", "AGT", "SCN5A",
        
        # Neurological genes
        "APP", "PSEN1", "PSEN2", "MAPT", "SNCA", "LRRK2",
        "HTT", "SOD1", "C9orf72",
        
        # Metabolic genes
        "INS", "GCK", "HNF1A", "PPARG", "TCF7L2",
        
        # Immune genes
        "HLA-DRB1", "IL6", "TNF", "IL1B", "CTLA4",
    ]
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/disgenet",
        api_key: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = DisGeNETClient(api_key=api_key)
        self.associations: List[Dict[str, Any]] = []
    
    async def fetch_disease_associations(self, query: str) -> List[Dict[str, Any]]:
        """Fetch gene associations for a disease."""
        # Search for disease
        diseases = await self.client.search_diseases(query, limit=5)
        
        all_associations = []
        for disease in diseases:
            disease_id = disease.get("diseaseId", disease.get("umls_cui", ""))
            if disease_id:
                associations = await self.client.get_disease_genes(disease_id)
                for assoc in associations:
                    assoc["disease_query"] = query
                all_associations.extend(associations)
            
            await asyncio.sleep(0.3)
        
        return all_associations
    
    async def fetch_gene_associations(self, gene: str) -> List[Dict[str, Any]]:
        """Fetch disease associations for a gene."""
        associations = await self.client.get_gene_diseases(gene)
        for assoc in associations:
            assoc["gene_query"] = gene
        return associations
    
    async def run(self) -> None:
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI DisGeNET Ingestion Pipeline")
        print("=" * 60)
        
        if not self.client.api_key:
            print("\nNote: DisGeNET API key not set.")
            print("Some features may be limited without an API key.")
            print("Get a free API key at: https://www.disgenet.org/api/")
        
        all_associations = []
        
        # Fetch by disease
        print("\nFetching disease-gene associations...")
        for query in tqdm(self.DISEASE_QUERIES, desc="Diseases"):
            try:
                associations = await self.fetch_disease_associations(query)
                all_associations.extend(associations)
                print(f"  {query}: {len(associations)} associations")
            except Exception as e:
                print(f"  Error: {e}")
            
            await asyncio.sleep(0.5)
        
        # Fetch by gene
        print("\nFetching gene-disease associations...")
        for gene in tqdm(self.GENE_QUERIES, desc="Genes"):
            try:
                associations = await self.fetch_gene_associations(gene)
                all_associations.extend(associations)
                print(f"  {gene}: {len(associations)} associations")
            except Exception as e:
                print(f"  Error: {e}")
            
            await asyncio.sleep(0.5)
        
        # Deduplicate
        seen_keys = set()
        unique_associations = []
        for assoc in all_associations:
            key = f"{assoc.get('geneId', '')}_{assoc.get('diseaseId', '')}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_associations.append(assoc)
        
        self.associations = unique_associations
        print(f"\nTotal unique associations: {len(self.associations)}")
        
        # Save
        await self.save()
        
        # Close client
        await self.client.close()
    
    async def save(self) -> None:
        """Save associations to disk."""
        output_file = self.output_dir / "gene_disease_associations.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for assoc in self.associations:
                gene_symbol = assoc.get("geneSymbol", assoc.get("gene_symbol", ""))
                disease_name = assoc.get("diseaseName", assoc.get("disease_name", ""))
                score = assoc.get("score", assoc.get("gda_score", 0))
                
                content = f"""Gene-Disease Association
Gene: {gene_symbol} (ID: {assoc.get('geneId', assoc.get('gene_id', ''))})
Disease: {disease_name} (ID: {assoc.get('diseaseId', assoc.get('disease_id', ''))})
Association Score: {score}
Evidence Index: {assoc.get('ei', assoc.get('evidence_index', ''))}
Year Initial: {assoc.get('yearInitial', '')}
Year Final: {assoc.get('yearFinal', '')}
"""
                
                doc = {
                    "id": f"disgenet_{assoc.get('geneId', '')}_{assoc.get('diseaseId', '')}",
                    "title": f"{gene_symbol} - {disease_name}",
                    "content": content,
                    "metadata": {
                        "gene_id": assoc.get("geneId", assoc.get("gene_id", "")),
                        "gene_symbol": gene_symbol,
                        "disease_id": assoc.get("diseaseId", assoc.get("disease_id", "")),
                        "disease_name": disease_name,
                        "score": score,
                        "source": "DisGeNET",
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save statistics
        stats = {
            "total_associations": len(self.associations),
            "ingestion_date": datetime.now().isoformat(),
            "disease_queries": self.DISEASE_QUERIES,
            "gene_queries": self.GENE_QUERIES,
        }
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


async def main():
    """Run the DisGeNET ingestion pipeline."""
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    pipeline = DisGeNETIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
