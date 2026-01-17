"""
UMI Orphanet Data Ingestion Pipeline
Fetches rare disease information - NO LIMITS
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
from tqdm import tqdm


class OrphanetClient:
    """Client for Orphanet API."""
    
    BASE_URL = "https://api.orphadata.com/rd-cross-referencing"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def search_diseases(self, query: str) -> List[Dict]:
        """Search for rare diseases."""
        url = f"{self.BASE_URL}/orphacodes"
        params = {"search": query}
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("data", []) if isinstance(data, dict) else data
        except Exception as e:
            print(f"    API error: {e}")
            return []
    
    async def get_disease(self, orpha_code: str) -> Dict:
        """Get disease details."""
        url = f"{self.BASE_URL}/orphacodes/{orpha_code}"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()
        except:
            return {}


RARE_DISEASE_QUERIES = [
    # Genetic disorders
    "cystic fibrosis", "huntington", "duchenne", "hemophilia", "sickle cell",
    "thalassemia", "phenylketonuria", "gaucher", "fabry", "pompe",
    
    # Rare cancers
    "neuroblastoma", "retinoblastoma", "osteosarcoma", "ewing sarcoma",
    "rhabdomyosarcoma", "wilms tumor", "medulloblastoma",
    
    # Metabolic disorders
    "maple syrup urine", "galactosemia", "glycogen storage", "lysosomal",
    "mitochondrial", "peroxisomal", "urea cycle",
    
    # Neurological
    "amyotrophic lateral sclerosis", "spinal muscular atrophy", "ataxia",
    "charcot marie tooth", "myasthenia gravis", "guillain barre",
    
    # Autoimmune
    "systemic sclerosis", "dermatomyositis", "polymyositis", "vasculitis",
    "sarcoidosis", "behcet", "sjogren",
    
    # Connective tissue
    "marfan", "ehlers danlos", "osteogenesis imperfecta", "epidermolysis bullosa",
    
    # Blood disorders
    "aplastic anemia", "myelodysplastic", "mastocytosis", "histiocytosis",
    
    # Pulmonary
    "pulmonary fibrosis", "pulmonary hypertension", "alpha-1 antitrypsin",
    
    # Cardiac
    "hypertrophic cardiomyopathy", "dilated cardiomyopathy", "long QT syndrome",
    "brugada syndrome", "arrhythmogenic right ventricular",
]


class OrphanetIngestionPipeline:
    """Pipeline for ingesting Orphanet rare disease data - NO LIMITS."""
    
    def __init__(self, output_dir: str = "data/knowledge_base/orphanet"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = OrphanetClient()
        self.diseases: List[Dict] = []
    
    def _disease_to_document(self, disease: Dict) -> Dict[str, Any]:
        """Convert rare disease to document."""
        orpha_code = disease.get("ORPHAcode", "")
        name = disease.get("Preferred term", disease.get("name", "Unknown"))
        
        content_parts = [f"{name} is a rare disease"]
        content_parts.append(f"Orphanet code: ORPHA:{orpha_code}")
        
        synonyms = disease.get("Synonym", [])
        if synonyms:
            syn_list = synonyms if isinstance(synonyms, list) else [synonyms]
            content_parts.append(f"Also known as: {', '.join(str(s) for s in syn_list[:10])}")
        
        definition = disease.get("Definition", "")
        if definition:
            content_parts.append(f"Definition: {definition}")
        
        genes = disease.get("Gene", [])
        if genes:
            gene_list = genes if isinstance(genes, list) else [genes]
            gene_names = [g.get("Name", str(g)) if isinstance(g, dict) else str(g) for g in gene_list[:10]]
            content_parts.append(f"Associated genes: {', '.join(gene_names)}")
        
        return {
            "id": f"orphanet_{orpha_code}",
            "title": name,
            "content": ". ".join(content_parts),
            "metadata": {
                "source": "Orphanet",
                "orpha_code": orpha_code,
                "type": "rare_disease",
            }
        }
    
    async def run(self) -> None:
        """Run the ingestion pipeline - NO LIMITS."""
        print("=" * 60)
        print("ORPHANET RARE DISEASE INGESTION")
        print("=" * 60)
        
        try:
            disease_docs = []
            seen = set()
            
            for query in tqdm(RARE_DISEASE_QUERIES, desc="Rare diseases"):
                try:
                    results = await self.client.search_diseases(query)
                    
                    for disease in results[:50]:
                        orpha = disease.get("ORPHAcode", "")
                        if orpha and orpha not in seen:
                            seen.add(orpha)
                            doc = self._disease_to_document(disease)
                            disease_docs.append(doc)
                            self.diseases.append(disease)
                    
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"   Warning: Error with {query}: {e}")
            
            # Save
            with open(self.output_dir / "rare_diseases.jsonl", 'w', encoding='utf-8') as f:
                for doc in disease_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            print(f"\n{'=' * 60}")
            print("ORPHANET INGESTION COMPLETE")
            print(f"Rare diseases: {len(disease_docs)}")
            print(f"Output: {self.output_dir}")
            print("=" * 60)
            
        finally:
            await self.client.close()


async def main():
    pipeline = OrphanetIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
