"""
UMI DrugBank/OpenFDA Data Ingestion Pipeline
Fetches drug information from open sources for RAG
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
class DrugInfo:
    """Represents drug information."""
    id: str
    name: str
    generic_name: str
    brand_names: List[str]
    drug_class: str
    description: str
    indications: List[str]
    mechanism: str
    dosage_forms: List[str]
    warnings: List[str]
    contraindications: List[str]
    interactions: List[Dict[str, str]]
    side_effects: List[str]
    pregnancy_category: Optional[str] = None
    source: str = "OpenFDA"


class OpenFDAClient:
    """
    Client for OpenFDA Drug API.
    https://open.fda.gov/apis/drug/
    """
    
    BASE_URL = "https://api.fda.gov/drug"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search_drugs(
        self,
        query: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for drugs by name or indication."""
        params = {
            "search": query,
            "limit": limit,
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/label.json",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return []
            raise
    
    async def get_drug_labels(self, limit: int = 1000, skip: int = 0) -> List[Dict[str, Any]]:
        """Get drug labels in bulk."""
        params = {
            "limit": min(limit, 100),  # API max is 100
            "skip": skip,
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/label.json",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            print(f"Error fetching labels: {e}")
            return []
    
    async def get_drug_interactions(self, drug_name: str) -> List[Dict[str, Any]]:
        """Get drug interactions for a specific drug."""
        params = {
            "search": f'drug_interactions:"{drug_name}"',
            "limit": 10,
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/label.json",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except:
            return []
    
    def parse_drug_label(self, label: Dict[str, Any]) -> Optional[DrugInfo]:
        """Parse FDA drug label into DrugInfo."""
        try:
            openfda = label.get("openfda", {})
            
            # Get drug name
            brand_names = openfda.get("brand_name", [])
            generic_names = openfda.get("generic_name", [])
            
            name = brand_names[0] if brand_names else (generic_names[0] if generic_names else None)
            if not name:
                return None
            
            generic_name = generic_names[0] if generic_names else name
            
            # Generate ID
            drug_id = openfda.get("product_ndc", [""])[0] or name.lower().replace(" ", "_")
            
            # Drug class
            pharm_class = openfda.get("pharm_class_epc", [])
            drug_class = pharm_class[0] if pharm_class else "Unknown"
            
            # Description
            description_parts = label.get("description", [])
            description = description_parts[0][:2000] if description_parts else ""
            
            # Indications
            indications_parts = label.get("indications_and_usage", [])
            indications = [ind[:500] for ind in indications_parts[:5]]
            
            # Mechanism
            mechanism_parts = label.get("mechanism_of_action", [])
            mechanism = mechanism_parts[0][:1000] if mechanism_parts else ""
            
            # Dosage forms
            dosage_parts = label.get("dosage_forms_and_strengths", [])
            dosage_forms = dosage_parts[:5] if dosage_parts else []
            
            # Warnings
            warnings_parts = label.get("warnings", []) or label.get("warnings_and_cautions", [])
            warnings = [w[:500] for w in warnings_parts[:5]]
            
            # Contraindications
            contra_parts = label.get("contraindications", [])
            contraindications = [c[:500] for c in contra_parts[:5]]
            
            # Side effects
            adverse_parts = label.get("adverse_reactions", [])
            side_effects = [a[:500] for a in adverse_parts[:5]]
            
            # Interactions
            interaction_parts = label.get("drug_interactions", [])
            interactions = []
            for inter in interaction_parts[:10]:
                interactions.append({
                    "description": inter[:500] if isinstance(inter, str) else str(inter)[:500],
                })
            
            # Pregnancy
            pregnancy_parts = label.get("pregnancy", [])
            pregnancy_category = None
            if pregnancy_parts:
                preg_text = pregnancy_parts[0]
                for cat in ["A", "B", "C", "D", "X"]:
                    if f"Category {cat}" in preg_text or f"category {cat}" in preg_text:
                        pregnancy_category = cat
                        break
            
            return DrugInfo(
                id=drug_id,
                name=name,
                generic_name=generic_name,
                brand_names=brand_names,
                drug_class=drug_class,
                description=description,
                indications=indications,
                mechanism=mechanism,
                dosage_forms=dosage_forms,
                warnings=warnings,
                contraindications=contraindications,
                interactions=interactions,
                side_effects=side_effects,
                pregnancy_category=pregnancy_category,
                source="OpenFDA",
            )
        
        except Exception as e:
            print(f"Error parsing drug label: {e}")
            return None
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class DrugIngestionPipeline:
    """
    Pipeline for ingesting drug information into UMI knowledge base.
    """
    
    # Comprehensive drug categories to fetch - expanded for maximum coverage
    DRUG_CATEGORIES = [
        # Pain & Inflammation
        "analgesic",
        "NSAID",
        "opioid",
        "opioid analgesic",
        "non-opioid analgesic",
        "muscle relaxant",
        "local anesthetic",
        
        # Cardiovascular
        "antihypertensive",
        "beta blocker",
        "ACE inhibitor",
        "angiotensin receptor blocker",
        "calcium channel blocker",
        "diuretic",
        "loop diuretic",
        "thiazide diuretic",
        "potassium-sparing diuretic",
        "statin",
        "HMG-CoA reductase inhibitor",
        "anticoagulant",
        "antiplatelet",
        "thrombolytic",
        "antiarrhythmic",
        "cardiac glycoside",
        "vasodilator",
        "nitrate",
        "fibrate",
        
        # Diabetes & Metabolic
        "antidiabetic",
        "insulin",
        "biguanide",
        "sulfonylurea",
        "DPP-4 inhibitor",
        "SGLT2 inhibitor",
        "GLP-1 agonist",
        "thiazolidinedione",
        "thyroid hormone",
        "antithyroid",
        
        # Respiratory
        "bronchodilator",
        "beta-2 agonist",
        "anticholinergic bronchodilator",
        "inhaled corticosteroid",
        "leukotriene inhibitor",
        "antihistamine",
        "H1 antihistamine",
        "decongestant",
        "antitussive",
        "expectorant",
        "mucolytic",
        
        # Gastrointestinal
        "proton pump inhibitor",
        "H2 blocker",
        "antacid",
        "antiemetic",
        "prokinetic",
        "laxative",
        "antidiarrheal",
        "antispasmodic",
        "5-HT3 antagonist",
        
        # Psychiatric & Neurological
        "antidepressant",
        "SSRI",
        "SNRI",
        "tricyclic antidepressant",
        "MAO inhibitor",
        "antipsychotic",
        "typical antipsychotic",
        "atypical antipsychotic",
        "anxiolytic",
        "benzodiazepine",
        "anticonvulsant",
        "antiepileptic",
        "mood stabilizer",
        "stimulant",
        "ADHD medication",
        "hypnotic",
        "sedative",
        "antiparkinson",
        "dopamine agonist",
        "cholinesterase inhibitor",
        
        # Anti-infective
        "antibiotic",
        "penicillin",
        "cephalosporin",
        "fluoroquinolone",
        "macrolide",
        "aminoglycoside",
        "tetracycline",
        "sulfonamide",
        "carbapenem",
        "glycopeptide",
        "antiviral",
        "antiretroviral",
        "antifungal",
        "azole antifungal",
        "antimalarial",
        "antiparasitic",
        "anthelmintic",
        "antiprotozoal",
        
        # Immunology & Rheumatology
        "corticosteroid",
        "glucocorticoid",
        "immunosuppressant",
        "DMARD",
        "biologic DMARD",
        "TNF inhibitor",
        "interleukin inhibitor",
        "JAK inhibitor",
        "immunomodulator",
        
        # Oncology
        "antineoplastic",
        "chemotherapy",
        "alkylating agent",
        "antimetabolite",
        "topoisomerase inhibitor",
        "mitotic inhibitor",
        "tyrosine kinase inhibitor",
        "monoclonal antibody",
        "hormone therapy",
        "aromatase inhibitor",
        "antiandrogen",
        
        # Hematology
        "anticoagulant",
        "direct oral anticoagulant",
        "heparin",
        "vitamin K antagonist",
        "erythropoietin",
        "colony stimulating factor",
        "iron supplement",
        "vitamin B12",
        "folic acid",
        
        # Dermatology
        "topical corticosteroid",
        "topical antibiotic",
        "topical antifungal",
        "retinoid",
        "keratolytic",
        "emollient",
        
        # Ophthalmology
        "ophthalmic antibiotic",
        "ophthalmic anti-inflammatory",
        "glaucoma medication",
        "mydriatic",
        
        # Urology
        "alpha blocker",
        "5-alpha reductase inhibitor",
        "phosphodiesterase inhibitor",
        "urinary antispasmodic",
        
        # Obstetrics & Gynecology
        "contraceptive",
        "estrogen",
        "progestin",
        "oxytocic",
        "tocolytic",
        
        # Miscellaneous
        "vitamin",
        "mineral supplement",
        "electrolyte",
        "antidote",
        "chelating agent",
        "vaccine",
    ]
    
    # Comprehensive OTC drugs to specifically include
    OTC_DRUGS = [
        # Pain relievers
        "acetaminophen",
        "ibuprofen",
        "aspirin",
        "naproxen",
        "ketoprofen",
        "magnesium salicylate",
        
        # Allergy & Cold
        "diphenhydramine",
        "loratadine",
        "cetirizine",
        "fexofenadine",
        "chlorpheniramine",
        "pseudoephedrine",
        "phenylephrine",
        "oxymetazoline",
        "guaifenesin",
        "dextromethorphan",
        "benzonatate",
        
        # Gastrointestinal
        "omeprazole",
        "esomeprazole",
        "lansoprazole",
        "famotidine",
        "ranitidine",
        "cimetidine",
        "calcium carbite",
        "magnesium hydroxide",
        "aluminum hydroxide",
        "bismuth subsalicylate",
        "loperamide",
        "simethicone",
        "docusate",
        "bisacodyl",
        "polyethylene glycol",
        "psyllium",
        "sennosides",
        
        # Topical
        "hydrocortisone",
        "bacitracin",
        "neomycin",
        "polymyxin B",
        "benzoyl peroxide",
        "salicylic acid",
        "clotrimazole",
        "miconazole",
        "terbinafine",
        "tolnaftate",
        "lidocaine",
        "benzocaine",
        "capsaicin",
        "menthol",
        "camphor",
        
        # Eye care
        "artificial tears",
        "tetrahydrozoline",
        "naphazoline",
        "ketotifen",
        
        # Sleep aids
        "melatonin",
        "doxylamine",
        "valerian",
        
        # Vitamins & Supplements
        "vitamin D",
        "vitamin C",
        "vitamin B12",
        "folic acid",
        "iron",
        "calcium",
        "magnesium",
        "zinc",
        "omega-3",
        "probiotics",
        "glucosamine",
        "chondroitin",
        
        # Motion sickness
        "dimenhydrinate",
        "meclizine",
        
        # Smoking cessation
        "nicotine",
        
        # Weight loss
        "orlistat",
        
        # Diabetes
        "glucose",
    ]
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/drugs",
        api_key: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenFDAClient(api_key=api_key)
        self.drugs: List[DrugInfo] = []
    
    async def fetch_category(self, category: str, limit: int = None) -> List[DrugInfo]:  # No limit by default
        """Fetch drugs for a specific category."""
        query = f'openfda.pharm_class_epc:"{category}"'
        
        try:
            labels = await self.client.search_drugs(query, limit=limit)
            
            drugs = []
            for label in labels:
                drug = self.client.parse_drug_label(label)
                if drug:
                    drugs.append(drug)
            
            return drugs
        except Exception as e:
            print(f"Error fetching category {category}: {e}")
            return []
    
    async def fetch_drug_by_name(self, name: str) -> Optional[DrugInfo]:
        """Fetch a specific drug by name."""
        query = f'openfda.generic_name:"{name}" OR openfda.brand_name:"{name}"'
        
        try:
            labels = await self.client.search_drugs(query, limit=1)
            
            if labels:
                return self.client.parse_drug_label(labels[0])
            return None
        except Exception as e:
            print(f"Error fetching drug {name}: {e}")
            return None
    
    async def run(self, max_per_category: int = None) -> None:  # No limit by default
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI Drug Data Ingestion Pipeline")
        print("=" * 60)
        
        all_drugs = []
        
        # Fetch by category
        for category in tqdm(self.DRUG_CATEGORIES, desc="Fetching categories"):
            try:
                drugs = await self.fetch_category(category, limit=max_per_category)
                all_drugs.extend(drugs)
                print(f"  {category}: {len(drugs)} drugs")
            except Exception as e:
                print(f"  Error: {e}")
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Fetch specific OTC drugs
        print("\nFetching OTC drugs...")
        for drug_name in tqdm(self.OTC_DRUGS, desc="OTC drugs"):
            try:
                drug = await self.fetch_drug_by_name(drug_name)
                if drug:
                    all_drugs.append(drug)
            except Exception as e:
                print(f"  Error fetching {drug_name}: {e}")
            
            await asyncio.sleep(0.3)
        
        # Deduplicate
        seen_ids = set()
        unique_drugs = []
        for drug in all_drugs:
            key = drug.generic_name.lower()
            if key not in seen_ids:
                seen_ids.add(key)
                unique_drugs.append(drug)
        
        self.drugs = unique_drugs
        print(f"\nTotal unique drugs: {len(self.drugs)}")
        
        # Save
        await self.save()
        
        # Close client
        await self.client.close()
    
    async def save(self) -> None:
        """Save drugs to disk."""
        # Save as JSONL for RAG indexing
        output_file = self.output_dir / "drugs.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for drug in self.drugs:
                # Create comprehensive content for RAG
                content_parts = [
                    f"Drug: {drug.name}",
                    f"Generic Name: {drug.generic_name}",
                    f"Drug Class: {drug.drug_class}",
                    "",
                    "Description:",
                    drug.description,
                    "",
                ]
                
                if drug.indications:
                    content_parts.append("Indications:")
                    content_parts.extend([f"- {ind}" for ind in drug.indications])
                    content_parts.append("")
                
                if drug.mechanism:
                    content_parts.append("Mechanism of Action:")
                    content_parts.append(drug.mechanism)
                    content_parts.append("")
                
                if drug.warnings:
                    content_parts.append("Warnings:")
                    content_parts.extend([f"- {w}" for w in drug.warnings])
                    content_parts.append("")
                
                if drug.contraindications:
                    content_parts.append("Contraindications:")
                    content_parts.extend([f"- {c}" for c in drug.contraindications])
                    content_parts.append("")
                
                if drug.side_effects:
                    content_parts.append("Side Effects:")
                    content_parts.extend([f"- {s}" for s in drug.side_effects])
                
                doc = {
                    "id": f"drug_{drug.id}",
                    "title": f"{drug.name} ({drug.generic_name})",
                    "content": "\n".join(content_parts),
                    "metadata": {
                        "drug_id": drug.id,
                        "name": drug.name,
                        "generic_name": drug.generic_name,
                        "brand_names": drug.brand_names,
                        "drug_class": drug.drug_class,
                        "pregnancy_category": drug.pregnancy_category,
                        "source": drug.source,
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save drug interactions separately
        interactions_file = self.output_dir / "interactions.jsonl"
        with open(interactions_file, 'w', encoding='utf-8') as f:
            for drug in self.drugs:
                if drug.interactions:
                    for interaction in drug.interactions:
                        doc = {
                            "drug": drug.generic_name,
                            "interaction": interaction,
                        }
                        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # Save statistics
        stats = {
            "total_drugs": len(self.drugs),
            "ingestion_date": datetime.now().isoformat(),
            "categories": self.DRUG_CATEGORIES,
            "otc_drugs": self.OTC_DRUGS,
        }
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


async def main():
    """Run the drug ingestion pipeline with no limits."""
    pipeline = DrugIngestionPipeline()
    # No limit - fetch all available drugs
    await pipeline.run(max_per_category=None)


if __name__ == "__main__":
    asyncio.run(main())
