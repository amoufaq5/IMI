"""
UMI Orphanet Data Ingestion Pipeline
<<<<<<< HEAD
Fetches rare disease information - NO LIMITS
=======
Fetches rare disease information from Orphanet
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
class OrphanetClient:
    """Client for Orphanet API."""
=======
@dataclass
class RareDisease:
    """Represents a rare disease from Orphanet."""
    orpha_code: str
    name: str
    synonyms: List[str]
    definition: str
    prevalence: str
    inheritance: List[str]
    age_of_onset: List[str]
    genes: List[Dict[str, str]]
    phenotypes: List[str]
    icd10_codes: List[str]


class OrphanetClient:
    """
    Client for Orphanet API.
    https://www.orpha.net/
    """
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    BASE_URL = "https://api.orphadata.com/rd-cross-referencing"
    
    def __init__(self):
<<<<<<< HEAD
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
=======
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search_diseases(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for rare diseases."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/diseases",
                params={"name": query, "limit": limit},
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            print(f"Error searching diseases: {e}")
            return []
    
    async def get_disease(self, orpha_code: str) -> Optional[Dict[str, Any]]:
        """Get detailed disease information."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/diseases/{orpha_code}",
            )
            response.raise_for_status()
            return response.json().get("data", {})
        except Exception as e:
            return None
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class OrphanetIngestionPipeline:
    """
    Pipeline for ingesting Orphanet rare disease data into UMI knowledge base.
    """
    
    # Rare disease categories to search
    DISEASE_CATEGORIES = [
        # Genetic disorders
        "cystic fibrosis", "sickle cell disease", "hemophilia",
        "muscular dystrophy", "Duchenne muscular dystrophy",
        "spinal muscular atrophy", "Huntington disease",
        "Marfan syndrome", "Ehlers-Danlos syndrome",
        "neurofibromatosis", "tuberous sclerosis",
        "phenylketonuria", "galactosemia", "maple syrup urine disease",
        "Fabry disease", "Gaucher disease", "Pompe disease",
        "Tay-Sachs disease", "Niemann-Pick disease",
        "mucopolysaccharidosis", "glycogen storage disease",
        
        # Autoimmune rare diseases
        "myasthenia gravis", "Guillain-Barre syndrome",
        "chronic inflammatory demyelinating polyneuropathy",
        "autoimmune hepatitis", "primary biliary cholangitis",
        "systemic sclerosis", "dermatomyositis", "polymyositis",
        "Sjogren syndrome", "antiphospholipid syndrome",
        "Behcet disease", "Takayasu arteritis", "giant cell arteritis",
        
        # Rare cancers
        "neuroblastoma", "retinoblastoma", "Wilms tumor",
        "Ewing sarcoma", "osteosarcoma", "rhabdomyosarcoma",
        "medulloblastoma", "glioblastoma", "mesothelioma",
        "cholangiocarcinoma", "adrenocortical carcinoma",
        "pheochromocytoma", "paraganglioma",
        
        # Rare blood disorders
        "aplastic anemia", "Diamond-Blackfan anemia",
        "Fanconi anemia", "paroxysmal nocturnal hemoglobinuria",
        "thrombotic thrombocytopenic purpura",
        "hemolytic uremic syndrome", "thalassemia",
        "hereditary spherocytosis", "hereditary elliptocytosis",
        
        # Rare neurological disorders
        "amyotrophic lateral sclerosis", "Charcot-Marie-Tooth disease",
        "Friedreich ataxia", "spinocerebellar ataxia",
        "progressive supranuclear palsy", "multiple system atrophy",
        "Creutzfeldt-Jakob disease", "fatal familial insomnia",
        "Rett syndrome", "Angelman syndrome", "Prader-Willi syndrome",
        
        # Rare pulmonary disorders
        "pulmonary arterial hypertension", "idiopathic pulmonary fibrosis",
        "lymphangioleiomyomatosis", "pulmonary alveolar proteinosis",
        "alpha-1 antitrypsin deficiency",
        
        # Rare cardiac disorders
        "hypertrophic cardiomyopathy", "dilated cardiomyopathy",
        "arrhythmogenic right ventricular cardiomyopathy",
        "long QT syndrome", "Brugada syndrome",
        "catecholaminergic polymorphic ventricular tachycardia",
        
        # Rare renal disorders
        "polycystic kidney disease", "Alport syndrome",
        "Bartter syndrome", "Gitelman syndrome",
        "cystinosis", "primary hyperoxaluria",
        
        # Rare hepatic disorders
        "Wilson disease", "hemochromatosis",
        "alpha-1 antitrypsin deficiency", "Alagille syndrome",
        "progressive familial intrahepatic cholestasis",
        
        # Rare endocrine disorders
        "congenital adrenal hyperplasia", "Addison disease",
        "multiple endocrine neoplasia", "Cushing syndrome",
        "acromegaly", "hypopituitarism",
        
        # Rare immunodeficiencies
        "severe combined immunodeficiency",
        "common variable immunodeficiency",
        "X-linked agammaglobulinemia", "Wiskott-Aldrich syndrome",
        "chronic granulomatous disease", "complement deficiency",
        
        # Rare connective tissue disorders
        "osteogenesis imperfecta", "achondroplasia",
        "hypophosphatasia", "fibrodysplasia ossificans progressiva",
        "epidermolysis bullosa", "ichthyosis",
        
        # Rare eye disorders
        "retinitis pigmentosa", "Leber congenital amaurosis",
        "Stargardt disease", "Best disease", "choroideremia",
        "aniridia", "Usher syndrome",
    ]
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/orphanet",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = OrphanetClient()
        self.diseases: List[Dict[str, Any]] = []
    
    async def fetch_diseases(self, query: str) -> List[Dict[str, Any]]:
        """Fetch diseases for a search query."""
        results = await self.client.search_diseases(query)
        return results
    
    async def run(self) -> None:
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI Orphanet Rare Disease Ingestion Pipeline")
        print("=" * 60)
        
        all_diseases = []
        
        for query in tqdm(self.DISEASE_CATEGORIES, desc="Fetching diseases"):
            try:
                diseases = await self.fetch_diseases(query)
                all_diseases.extend(diseases)
                print(f"  {query}: {len(diseases)} diseases")
            except Exception as e:
                print(f"  Error: {e}")
            
            await asyncio.sleep(0.5)
        
        # Deduplicate by Orpha code
        seen_codes = set()
        unique_diseases = []
        for disease in all_diseases:
            code = disease.get("orphaCode", disease.get("id", ""))
            if code and code not in seen_codes:
                seen_codes.add(code)
                unique_diseases.append(disease)
        
        self.diseases = unique_diseases
        print(f"\nTotal unique diseases: {len(self.diseases)}")
        
        # Save
        await self.save()
        
        # Close client
        await self.client.close()
    
    async def save(self) -> None:
        """Save diseases to disk."""
        output_file = self.output_dir / "rare_diseases.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for disease in self.diseases:
                orpha_code = disease.get("orphaCode", disease.get("id", ""))
                name = disease.get("name", disease.get("preferredTerm", ""))
                
                content_parts = [
                    f"Rare Disease: {name}",
                    f"Orphanet Code: {orpha_code}",
                    "",
                ]
                
                definition = disease.get("definition", "")
                if definition:
                    content_parts.append("Definition:")
                    content_parts.append(definition)
                    content_parts.append("")
                
                synonyms = disease.get("synonyms", [])
                if synonyms:
                    content_parts.append("Synonyms:")
                    for syn in synonyms[:10]:
                        if isinstance(syn, dict):
                            content_parts.append(f"- {syn.get('name', syn.get('term', ''))}")
                        else:
                            content_parts.append(f"- {syn}")
                    content_parts.append("")
                
                genes = disease.get("genes", disease.get("associatedGenes", []))
                if genes:
                    content_parts.append("Associated Genes:")
                    for gene in genes[:10]:
                        if isinstance(gene, dict):
                            content_parts.append(f"- {gene.get('symbol', gene.get('name', ''))}")
                        else:
                            content_parts.append(f"- {gene}")
                
                doc = {
                    "id": f"orphanet_{orpha_code}",
                    "title": name,
                    "content": "\n".join(content_parts),
                    "metadata": {
                        "orpha_code": orpha_code,
                        "name": name,
                        "type": "rare_disease",
                        "source": "Orphanet",
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save statistics
        stats = {
            "total_diseases": len(self.diseases),
            "ingestion_date": datetime.now().isoformat(),
            "search_queries": self.DISEASE_CATEGORIES,
        }
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


async def main():
    """Run the Orphanet ingestion pipeline."""
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    pipeline = OrphanetIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
