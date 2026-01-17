"""
UMI ChEMBL Data Ingestion Pipeline
Fetches bioactivity and drug data from ChEMBL database - NO LIMITS
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from tqdm import tqdm


@dataclass
class ChEMBLCompound:
    """Represents a ChEMBL compound."""
    chembl_id: str
    pref_name: str
    molecule_type: str
    max_phase: int
    oral: bool
    therapeutic_flag: bool
    natural_product: bool
    first_approval: Optional[int]
    indication_class: str
    mechanism_of_action: List[str]
    targets: List[Dict[str, str]]
    activities: List[Dict[str, Any]]


class ChEMBLClient:
    """Client for ChEMBL REST API."""
    
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def _get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make GET request to ChEMBL API."""
        url = f"{self.BASE_URL}/{endpoint}.json"
        params = params or {}
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"    API error: {e}")
            return {}
    
    async def search_molecules(self, query: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Search for molecules."""
        all_results = []
        offset = 0
        
        while True:
            params = {
                "molecule_synonyms__molecule_synonym__icontains": query,
                "limit": min(limit - len(all_results), 100),
                "offset": offset,
            }
            
            data = await self._get("molecule", params)
            molecules = data.get("molecules", [])
            
            if not molecules:
                break
            
            all_results.extend(molecules)
            
            if len(all_results) >= limit or len(molecules) < 100:
                break
            
            offset += 100
            await asyncio.sleep(0.2)
        
        return all_results
    
    async def get_approved_drugs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all approved drugs (max_phase = 4)."""
        all_results = []
        offset = 0
        batch_size = 100
        
        while True:
            params = {
                "max_phase": 4,
                "limit": batch_size,
                "offset": offset,
            }
            
            data = await self._get("molecule", params)
            molecules = data.get("molecules", [])
            
            if not molecules:
                break
            
            all_results.extend(molecules)
            
            if limit and len(all_results) >= limit:
                break
            
            if len(molecules) < batch_size:
                break
            
            offset += batch_size
            await asyncio.sleep(0.2)
            
            if offset % 500 == 0:
                print(f"    Fetched {len(all_results)} approved drugs...")
        
        return all_results[:limit] if limit else all_results
    
    async def get_molecule_mechanisms(self, chembl_id: str) -> List[Dict[str, Any]]:
        """Get mechanism of action for a molecule."""
        params = {"molecule_chembl_id": chembl_id}
        data = await self._get("mechanism", params)
        return data.get("mechanisms", [])
    
    async def get_molecule_indications(self, chembl_id: str) -> List[Dict[str, Any]]:
        """Get drug indications."""
        params = {"molecule_chembl_id": chembl_id}
        data = await self._get("drug_indication", params)
        return data.get("drug_indications", [])
    
    async def get_target(self, target_chembl_id: str) -> Dict[str, Any]:
        """Get target information."""
        data = await self._get(f"target/{target_chembl_id}")
        return data
    
    async def search_targets(self, query: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Search for targets."""
        all_results = []
        offset = 0
        
        while len(all_results) < limit:
            params = {
                "pref_name__icontains": query,
                "limit": min(limit - len(all_results), 100),
                "offset": offset,
            }
            
            data = await self._get("target", params)
            targets = data.get("targets", [])
            
            if not targets:
                break
            
            all_results.extend(targets)
            offset += 100
            await asyncio.sleep(0.2)
        
        return all_results


# Drug categories to search
DRUG_QUERIES = [
    "aspirin", "ibuprofen", "acetaminophen", "metformin", "atorvastatin",
    "lisinopril", "amlodipine", "metoprolol", "omeprazole", "losartan",
    "gabapentin", "sertraline", "fluoxetine", "escitalopram", "duloxetine",
    "tramadol", "hydrocodone", "oxycodone", "morphine", "fentanyl",
    "warfarin", "rivaroxaban", "apixaban", "clopidogrel", "heparin",
    "insulin", "glipizide", "sitagliptin", "empagliflozin", "liraglutide",
    "pembrolizumab", "nivolumab", "trastuzumab", "rituximab", "bevacizumab",
    "adalimumab", "infliximab", "etanercept", "tocilizumab", "secukinumab",
    "remdesivir", "molnupiravir", "paxlovid", "dexamethasone", "baricitinib",
    "semaglutide", "tirzepatide", "ozempic", "wegovy", "mounjaro",
]

TARGET_QUERIES = [
    "kinase", "receptor", "enzyme", "transporter", "channel",
    "protease", "polymerase", "integrase", "topoisomerase", "synthase",
    "EGFR", "HER2", "VEGF", "PD-1", "PD-L1", "CTLA-4", "TNF",
    "ACE", "angiotensin", "dopamine", "serotonin", "GABA",
    "insulin receptor", "glucagon", "GLP-1", "SGLT2",
]


class ChEMBLIngestionPipeline:
    """Pipeline for ingesting ChEMBL bioactivity data - NO LIMITS."""
    
    def __init__(self, output_dir: str = "data/knowledge_base/chembl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = ChEMBLClient()
        self.molecules: List[Dict[str, Any]] = []
        self.targets: List[Dict[str, Any]] = []
    
    def _molecule_to_document(self, mol: Dict[str, Any], mechanisms: List = None, indications: List = None) -> Dict[str, Any]:
        """Convert molecule data to document format."""
        name = mol.get("pref_name") or mol.get("molecule_chembl_id", "Unknown")
        chembl_id = mol.get("molecule_chembl_id", "")
        
        # Build content
        content_parts = [f"{name} is a pharmaceutical compound"]
        
        mol_type = mol.get("molecule_type")
        if mol_type:
            content_parts.append(f"classified as a {mol_type}")
        
        max_phase = mol.get("max_phase")
        if max_phase:
            phase_desc = {4: "approved drug", 3: "Phase III", 2: "Phase II", 1: "Phase I"}
            content_parts.append(f"with development status: {phase_desc.get(max_phase, f'Phase {max_phase}')}")
        
        props = mol.get("molecule_properties", {}) or {}
        if props:
            if props.get("full_mwt"):
                content_parts.append(f"Molecular weight: {props['full_mwt']}")
            if props.get("alogp"):
                content_parts.append(f"LogP: {props['alogp']}")
        
        if mechanisms:
            mech_list = [m.get("mechanism_of_action", "") for m in mechanisms if m.get("mechanism_of_action")]
            if mech_list:
                content_parts.append(f"Mechanism of action: {'; '.join(mech_list[:5])}")
        
        if indications:
            ind_list = [i.get("mesh_heading", "") for i in indications if i.get("mesh_heading")]
            if ind_list:
                content_parts.append(f"Indications: {', '.join(ind_list[:10])}")
        
        content = ". ".join(content_parts) + "."
        
        return {
            "id": f"chembl_{chembl_id}",
            "title": name,
            "content": content,
            "metadata": {
                "source": "ChEMBL",
                "chembl_id": chembl_id,
                "molecule_type": mol_type,
                "max_phase": max_phase,
                "first_approval": mol.get("first_approval"),
                "oral": mol.get("oral"),
                "therapeutic_flag": mol.get("therapeutic_flag"),
            }
        }
    
    def _target_to_document(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Convert target data to document format."""
        name = target.get("pref_name", "Unknown Target")
        chembl_id = target.get("target_chembl_id", "")
        
        content_parts = [f"{name} is a drug target"]
        
        target_type = target.get("target_type")
        if target_type:
            content_parts.append(f"classified as {target_type}")
        
        organism = target.get("organism")
        if organism:
            content_parts.append(f"found in {organism}")
        
        components = target.get("target_components", [])
        if components:
            genes = [c.get("component_synonym", "") for c in components[:5] if c.get("component_synonym")]
            if genes:
                content_parts.append(f"Associated genes: {', '.join(genes)}")
        
        content = ". ".join(content_parts) + "."
        
        return {
            "id": f"chembl_target_{chembl_id}",
            "title": name,
            "content": content,
            "metadata": {
                "source": "ChEMBL",
                "target_chembl_id": chembl_id,
                "target_type": target_type,
                "organism": organism,
                "type": "drug_target",
            }
        }
    
    async def run(self) -> None:
        """Run the ingestion pipeline - NO LIMITS."""
        print("=" * 60)
        print("CHEMBL BIOACTIVITY DATA INGESTION")
        print("=" * 60)
        
        try:
            # 1. Fetch all approved drugs
            print("\n1. Fetching approved drugs (no limit)...")
            approved_drugs = await self.client.get_approved_drugs(limit=None)
            print(f"   Found {len(approved_drugs)} approved drugs")
            
            # 2. Search for additional drugs
            print("\n2. Searching for specific drugs...")
            for query in tqdm(DRUG_QUERIES, desc="Drug searches"):
                try:
                    results = await self.client.search_molecules(query, limit=100)
                    for mol in results:
                        if mol.get("molecule_chembl_id") not in [m.get("molecule_chembl_id") for m in approved_drugs]:
                            approved_drugs.append(mol)
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"   Warning: Error searching {query}: {e}")
            
            print(f"   Total molecules: {len(approved_drugs)}")
            
            # 3. Get mechanisms and indications for top drugs
            print("\n3. Fetching mechanisms and indications...")
            molecule_docs = []
            
            for mol in tqdm(approved_drugs[:2000], desc="Processing molecules"):
                chembl_id = mol.get("molecule_chembl_id")
                if not chembl_id:
                    continue
                
                try:
                    mechanisms = await self.client.get_molecule_mechanisms(chembl_id)
                    indications = await self.client.get_molecule_indications(chembl_id)
                    
                    doc = self._molecule_to_document(mol, mechanisms, indications)
                    molecule_docs.append(doc)
                    self.molecules.append(mol)
                    
                    await asyncio.sleep(0.1)
                except Exception as e:
                    # Still add basic document
                    doc = self._molecule_to_document(mol)
                    molecule_docs.append(doc)
                    self.molecules.append(mol)
            
            # 4. Fetch targets
            print("\n4. Fetching drug targets...")
            target_docs = []
            seen_targets = set()
            
            for query in tqdm(TARGET_QUERIES, desc="Target searches"):
                try:
                    targets = await self.client.search_targets(query, limit=200)
                    for target in targets:
                        tid = target.get("target_chembl_id")
                        if tid and tid not in seen_targets:
                            seen_targets.add(tid)
                            doc = self._target_to_document(target)
                            target_docs.append(doc)
                            self.targets.append(target)
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"   Warning: Error searching targets {query}: {e}")
            
            print(f"   Total targets: {len(target_docs)}")
            
            # Save molecules
            mol_file = self.output_dir / "molecules.jsonl"
            with open(mol_file, 'w', encoding='utf-8') as f:
                for doc in molecule_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            # Save targets
            target_file = self.output_dir / "targets.jsonl"
            with open(target_file, 'w', encoding='utf-8') as f:
                for doc in target_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            print(f"\n{'=' * 60}")
            print("CHEMBL INGESTION COMPLETE")
            print(f"Molecules: {len(molecule_docs)}")
            print(f"Targets: {len(target_docs)}")
            print(f"Output: {self.output_dir}")
            print("=" * 60)
            
        finally:
            await self.client.close()


async def main():
    pipeline = ChEMBLIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
