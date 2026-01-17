"""
UMI UniProt Data Ingestion Pipeline
Fetches protein and gene data - NO LIMITS
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
from tqdm import tqdm


class UniProtClient:
    """Client for UniProt REST API."""
    
    BASE_URL = "https://rest.uniprot.org/uniprotkb"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def search(self, query: str, size: int = 500, reviewed: bool = True) -> List[Dict]:
        """Search UniProt."""
        all_results = []
        
        params = {
            "query": f"{query} AND (reviewed:{str(reviewed).lower()})",
            "format": "json",
            "size": min(size, 500),
        }
        
        try:
            response = await self.client.get(f"{self.BASE_URL}/search", params=params)
            response.raise_for_status()
            data = response.json()
            all_results = data.get("results", [])
        except Exception as e:
            print(f"    API error: {e}")
        
        return all_results[:size]


PROTEIN_QUERIES = [
    # Disease-related proteins
    "disease:cancer", "disease:diabetes", "disease:alzheimer", "disease:parkinson",
    "disease:cardiovascular", "disease:inflammation", "disease:infection",
    
    # Drug targets
    "keyword:pharmaceutical", "keyword:drug target", "annotation:(type:pharmaceutical)",
    
    # Enzymes
    "ec:1.*", "ec:2.*", "ec:3.*",  # Oxidoreductases, Transferases, Hydrolases
    
    # Receptors
    "keyword:receptor", "family:GPCR", "family:kinase",
    
    # Specific proteins
    "gene:EGFR", "gene:HER2", "gene:BRCA1", "gene:BRCA2", "gene:TP53",
    "gene:ACE2", "gene:insulin", "gene:hemoglobin", "gene:albumin",
    
    # Organism specific
    "organism_id:9606",  # Human
]


class UniProtIngestionPipeline:
    """Pipeline for ingesting UniProt protein data - NO LIMITS."""
    
    def __init__(self, output_dir: str = "data/knowledge_base/uniprot"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = UniProtClient()
        self.proteins: List[Dict] = []
    
    def _protein_to_document(self, protein: Dict) -> Dict[str, Any]:
        """Convert protein to document."""
        entry = protein.get("primaryAccession", "")
        
        # Get protein name
        protein_desc = protein.get("proteinDescription", {})
        rec_name = protein_desc.get("recommendedName", {})
        name = rec_name.get("fullName", {}).get("value", "Unknown Protein")
        
        # Get gene name
        genes = protein.get("genes", [])
        gene_name = genes[0].get("geneName", {}).get("value", "") if genes else ""
        
        # Get organism
        organism = protein.get("organism", {}).get("scientificName", "")
        
        # Get function
        comments = protein.get("comments", [])
        function = ""
        for comment in comments:
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    function = texts[0].get("value", "")
                    break
        
        # Get disease associations
        diseases = []
        for comment in comments:
            if comment.get("commentType") == "DISEASE":
                disease = comment.get("disease", {})
                if disease.get("diseaseId"):
                    diseases.append(disease.get("diseaseId"))
        
        # Get keywords
        keywords = [kw.get("name", "") for kw in protein.get("keywords", [])[:10]]
        
        # Build content
        content_parts = [f"{name} (Gene: {gene_name})" if gene_name else name]
        
        if organism:
            content_parts.append(f"Organism: {organism}")
        
        if function:
            content_parts.append(f"Function: {function[:500]}")
        
        if diseases:
            content_parts.append(f"Associated diseases: {', '.join(diseases[:10])}")
        
        if keywords:
            content_parts.append(f"Keywords: {', '.join(keywords)}")
        
        return {
            "id": f"uniprot_{entry}",
            "title": name,
            "content": ". ".join(content_parts),
            "metadata": {
                "source": "UniProt",
                "accession": entry,
                "gene": gene_name,
                "organism": organism,
                "diseases": diseases[:10],
                "keywords": keywords,
            }
        }
    
    async def run(self) -> None:
        """Run the ingestion pipeline - NO LIMITS."""
        print("=" * 60)
        print("UNIPROT PROTEIN DATA INGESTION")
        print("=" * 60)
        
        try:
            protein_docs = []
            seen = set()
            
            for query in tqdm(PROTEIN_QUERIES, desc="Protein searches"):
                try:
                    results = await self.client.search(query, size=500)
                    
                    for protein in results:
                        acc = protein.get("primaryAccession")
                        if acc and acc not in seen:
                            seen.add(acc)
                            doc = self._protein_to_document(protein)
                            protein_docs.append(doc)
                            self.proteins.append(protein)
                    
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"   Warning: Error with {query}: {e}")
            
            # Save
            with open(self.output_dir / "proteins.jsonl", 'w', encoding='utf-8') as f:
                for doc in protein_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            print(f"\n{'=' * 60}")
            print("UNIPROT INGESTION COMPLETE")
            print(f"Proteins: {len(protein_docs)}")
            print(f"Output: {self.output_dir}")
            print("=" * 60)
            
        finally:
            await self.client.close()


async def main():
    pipeline = UniProtIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
