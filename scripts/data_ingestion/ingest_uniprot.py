"""
UMI UniProt Data Ingestion Pipeline
<<<<<<< HEAD
Fetches protein and gene data - NO LIMITS
=======
Fetches protein and gene data from UniProt database
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
from typing import Any, Dict, List, Optional
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)

import httpx
from tqdm import tqdm


<<<<<<< HEAD
class UniProtClient:
    """Client for UniProt REST API."""
=======
@dataclass
class Protein:
    """Represents a UniProt protein entry."""
    accession: str
    entry_name: str
    protein_name: str
    gene_names: List[str]
    organism: str
    function: str
    subcellular_location: List[str]
    disease_associations: List[str]
    go_terms: List[str]
    keywords: List[str]
    sequence_length: int


class UniProtClient:
    """
    Client for UniProt REST API.
    https://www.uniprot.org/help/api
    """
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    BASE_URL = "https://rest.uniprot.org/uniprotkb"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
    
<<<<<<< HEAD
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
=======
    async def search_proteins(
        self,
        query: str,
        limit: int = 100,
        organism: str = "9606",  # Human
    ) -> List[Dict[str, Any]]:
        """Search for proteins."""
        full_query = f"({query}) AND (organism_id:{organism})"
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/search",
                params={
                    "query": full_query,
                    "format": "json",
                    "size": limit,
                    "fields": "accession,id,protein_name,gene_names,organism_name,"
                             "cc_function,cc_subcellular_location,cc_disease,"
                             "go,keyword,length",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            print(f"Error searching proteins: {e}")
            return []
    
    async def get_protein(self, accession: str) -> Optional[Dict[str, Any]]:
        """Get detailed protein information."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/{accession}.json",
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return None
    
    async def get_disease_proteins(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Get proteins associated with diseases."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/search",
                params={
                    "query": "(cc_disease:*) AND (organism_id:9606) AND (reviewed:true)",
                    "format": "json",
                    "size": limit,
                    "fields": "accession,id,protein_name,gene_names,organism_name,"
                             "cc_function,cc_subcellular_location,cc_disease,"
                             "go,keyword,length",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            print(f"Error getting disease proteins: {e}")
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class UniProtIngestionPipeline:
    """
    Pipeline for ingesting UniProt protein data into UMI knowledge base.
    """
    
    # Protein/gene queries
    PROTEIN_QUERIES = [
        # Drug targets
        "kinase", "GPCR", "ion channel", "nuclear receptor",
        "protease", "phosphatase", "transporter",
        
        # Disease-related proteins
        "tumor suppressor", "oncogene", "apoptosis",
        "inflammation", "immune response", "cytokine",
        "growth factor", "hormone", "receptor",
        
        # Metabolic enzymes
        "oxidoreductase", "transferase", "hydrolase",
        "lyase", "isomerase", "ligase",
        
        # Specific protein families
        "insulin", "hemoglobin", "collagen", "actin", "myosin",
        "tubulin", "keratin", "albumin", "immunoglobulin",
        "complement", "coagulation factor", "cytochrome",
        
        # Signaling proteins
        "MAP kinase", "tyrosine kinase", "serine/threonine kinase",
        "phospholipase", "adenylate cyclase", "guanylate cyclase",
        
        # Transcription factors
        "transcription factor", "zinc finger", "helix-loop-helix",
        "leucine zipper", "homeobox",
        
        # Membrane proteins
        "channel", "pump", "carrier", "receptor tyrosine kinase",
        "G protein-coupled receptor", "toll-like receptor",
        
        # Structural proteins
        "extracellular matrix", "basement membrane", "cytoskeleton",
        "cell adhesion", "tight junction", "gap junction",
    ]
    
    # Specific genes of medical importance
    GENE_QUERIES = [
        # Oncogenes and tumor suppressors
        "TP53", "BRCA1", "BRCA2", "KRAS", "EGFR", "BRAF",
        "PIK3CA", "PTEN", "APC", "RB1", "MYC", "ERBB2",
        "ALK", "RET", "MET", "NRAS", "CDKN2A", "VHL",
        
        # Cardiovascular
        "APOE", "LDLR", "PCSK9", "ACE", "AGT", "AGTR1",
        "SCN5A", "KCNQ1", "KCNH2", "MYH7", "MYBPC3",
        
        # Neurological
        "APP", "PSEN1", "PSEN2", "MAPT", "SNCA", "LRRK2",
        "HTT", "SOD1", "C9orf72", "FUS", "TARDBP",
        
        # Metabolic
        "INS", "GCK", "HNF1A", "PPARG", "TCF7L2", "KCNJ11",
        "ABCC8", "SLC2A2", "GCKR", "MTNR1B",
        
        # Immune/Inflammatory
        "HLA-A", "HLA-B", "HLA-DRB1", "IL6", "TNF", "IL1B",
        "CTLA4", "PDCD1", "CD274", "FOXP3", "IL2RA",
        
        # Hematological
        "HBB", "HBA1", "F8", "F9", "VWF", "FGA", "FGB",
        "PROC", "PROS1", "SERPINC1",
    ]
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/uniprot",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = UniProtClient()
        self.proteins: List[Dict[str, Any]] = []
    
    async def fetch_proteins(self, query: str) -> List[Dict[str, Any]]:
        """Fetch proteins for a query."""
        proteins = await self.client.search_proteins(query, limit=100)
        return proteins
    
    async def run(self) -> None:
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI UniProt Ingestion Pipeline")
        print("=" * 60)
        
        all_proteins = []
        
        # Fetch disease-associated proteins (no limit)
        print("\nFetching disease-associated proteins...")
        disease_proteins = await self.client.get_disease_proteins(limit=1000)
        all_proteins.extend(disease_proteins)
        print(f"  Fetched {len(disease_proteins)} disease proteins")
        
        # Fetch by protein query
        print("\nFetching proteins by category...")
        for query in tqdm(self.PROTEIN_QUERIES, desc="Protein queries"):
            try:
                proteins = await self.fetch_proteins(query)
                all_proteins.extend(proteins)
                print(f"  {query}: {len(proteins)} proteins")
            except Exception as e:
                print(f"  Error: {e}")
            
            await asyncio.sleep(0.3)
        
        # Fetch by gene name
        print("\nFetching proteins by gene name...")
        for gene in tqdm(self.GENE_QUERIES, desc="Gene queries"):
            try:
                proteins = await self.client.search_proteins(f"gene:{gene}", limit=10)
                all_proteins.extend(proteins)
                print(f"  {gene}: {len(proteins)} proteins")
            except Exception as e:
                print(f"  Error: {e}")
            
            await asyncio.sleep(0.3)
        
        # Deduplicate
        seen_accessions = set()
        unique_proteins = []
        for protein in all_proteins:
            accession = protein.get("primaryAccession", "")
            if accession and accession not in seen_accessions:
                seen_accessions.add(accession)
                unique_proteins.append(protein)
        
        self.proteins = unique_proteins
        print(f"\nTotal unique proteins: {len(self.proteins)}")
        
        # Save
        await self.save()
        
        # Close client
        await self.client.close()
    
    def _extract_text(self, obj: Any, key: str = "value") -> str:
        """Extract text from nested UniProt structures."""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return obj.get(key, str(obj))
        if isinstance(obj, list):
            return "; ".join(self._extract_text(item, key) for item in obj)
        return str(obj)
    
    async def save(self) -> None:
        """Save proteins to disk."""
        output_file = self.output_dir / "proteins.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for protein in self.proteins:
                accession = protein.get("primaryAccession", "")
                
                # Extract protein name
                protein_desc = protein.get("proteinDescription", {})
                rec_name = protein_desc.get("recommendedName", {})
                full_name = rec_name.get("fullName", {})
                protein_name = self._extract_text(full_name, "value") if full_name else accession
                
                # Extract gene names
                genes = protein.get("genes", [])
                gene_names = []
                for gene in genes:
                    if gene.get("geneName"):
                        gene_names.append(gene["geneName"].get("value", ""))
                
                # Extract organism
                organism = protein.get("organism", {}).get("scientificName", "")
                
                content_parts = [
                    f"Protein: {protein_name}",
                    f"UniProt Accession: {accession}",
                    f"Entry Name: {protein.get('uniProtkbId', '')}",
                    f"Gene Names: {', '.join(gene_names)}",
                    f"Organism: {organism}",
                    f"Sequence Length: {protein.get('sequence', {}).get('length', '')} aa",
                    "",
                ]
                
                # Function
                comments = protein.get("comments", [])
                for comment in comments:
                    if comment.get("commentType") == "FUNCTION":
                        texts = comment.get("texts", [])
                        if texts:
                            content_parts.append("Function:")
                            for text in texts:
                                content_parts.append(f"  {text.get('value', '')}")
                            content_parts.append("")
                
                # Disease associations
                for comment in comments:
                    if comment.get("commentType") == "DISEASE":
                        disease = comment.get("disease", {})
                        disease_name = disease.get("diseaseId", "")
                        description = disease.get("description", "")
                        if disease_name:
                            content_parts.append(f"Disease: {disease_name}")
                            if description:
                                content_parts.append(f"  {description}")
                
                # Subcellular location
                for comment in comments:
                    if comment.get("commentType") == "SUBCELLULAR LOCATION":
                        locations = comment.get("subcellularLocations", [])
                        if locations:
                            content_parts.append("Subcellular Location:")
                            for loc in locations[:5]:
                                loc_val = loc.get("location", {}).get("value", "")
                                if loc_val:
                                    content_parts.append(f"  - {loc_val}")
                
                # Keywords
                keywords = protein.get("keywords", [])
                if keywords:
                    kw_values = [kw.get("name", "") for kw in keywords[:15]]
                    content_parts.append(f"\nKeywords: {', '.join(kw_values)}")
                
                doc = {
                    "id": f"uniprot_{accession}",
                    "title": protein_name,
                    "content": "\n".join(content_parts),
                    "metadata": {
                        "accession": accession,
                        "entry_name": protein.get("uniProtkbId", ""),
                        "gene_names": gene_names,
                        "organism": organism,
                        "length": protein.get("sequence", {}).get("length", 0),
                        "source": "UniProt",
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save statistics
        stats = {
            "total_proteins": len(self.proteins),
            "ingestion_date": datetime.now().isoformat(),
            "protein_queries": self.PROTEIN_QUERIES,
            "gene_queries": self.GENE_QUERIES,
        }
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


async def main():
    """Run the UniProt ingestion pipeline."""
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    pipeline = UniProtIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
