"""
UMI RxNorm Data Ingestion Pipeline
Fetches drug terminology and relationships from NIH RxNorm API
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
from tqdm import tqdm


@dataclass
class RxNormDrug:
    """Represents an RxNorm drug concept."""
    rxcui: str
    name: str
    synonym: str
    tty: str  # Term type (e.g., SBD, SCD, IN, BN)
    ingredients: List[str]
    brand_names: List[str]
    dose_forms: List[str]
    strengths: List[str]
    ndc_codes: List[str]
    drug_classes: List[str]
    interactions: List[Dict[str, str]]
    related_drugs: List[Dict[str, str]]


class RxNormClient:
    """
    Client for NIH RxNorm REST API.
    https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
    """
    
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search_drugs(self, name: str) -> List[Dict[str, Any]]:
        """Search for drugs by name."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/drugs.json",
                params={"name": name},
            )
            response.raise_for_status()
            data = response.json()
            
            drug_group = data.get("drugGroup", {})
            concept_groups = drug_group.get("conceptGroup", [])
            
            results = []
            for group in concept_groups:
                concepts = group.get("conceptProperties", [])
                for concept in concepts:
                    results.append({
                        "rxcui": concept.get("rxcui", ""),
                        "name": concept.get("name", ""),
                        "synonym": concept.get("synonym", ""),
                        "tty": concept.get("tty", ""),
                    })
            
            return results
        except Exception as e:
            print(f"Error searching drugs: {e}")
            return []
    
    async def get_all_drugs_by_class(self, class_id: str, class_type: str = "ATC") -> List[Dict[str, Any]]:
        """Get all drugs in a drug class."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/rxclass/classMembers.json",
                params={
                    "classId": class_id,
                    "relaSource": class_type,
                },
            )
            response.raise_for_status()
            data = response.json()
            
            members = data.get("drugMemberGroup", {}).get("drugMember", [])
            results = []
            for member in members:
                node = member.get("minConcept", {})
                results.append({
                    "rxcui": node.get("rxcui", ""),
                    "name": node.get("name", ""),
                    "tty": node.get("tty", ""),
                })
            
            return results
        except Exception as e:
            print(f"Error getting class members: {e}")
            return []
    
    async def get_drug_properties(self, rxcui: str) -> Dict[str, Any]:
        """Get detailed properties for a drug."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/rxcui/{rxcui}/allProperties.json",
                params={"prop": "all"},
            )
            response.raise_for_status()
            data = response.json()
            
            properties = {}
            prop_concepts = data.get("propConceptGroup", {}).get("propConcept", [])
            for prop in prop_concepts:
                prop_name = prop.get("propName", "")
                prop_value = prop.get("propValue", "")
                if prop_name not in properties:
                    properties[prop_name] = []
                properties[prop_name].append(prop_value)
            
            return properties
        except Exception as e:
            return {}
    
    async def get_related_drugs(self, rxcui: str) -> List[Dict[str, str]]:
        """Get related drugs (brand names, generics, etc.)."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/rxcui/{rxcui}/related.json",
                params={"tty": "BN+IN+SBD+SCD"},
            )
            response.raise_for_status()
            data = response.json()
            
            related = []
            groups = data.get("relatedGroup", {}).get("conceptGroup", [])
            for group in groups:
                tty = group.get("tty", "")
                concepts = group.get("conceptProperties", [])
                for concept in concepts:
                    related.append({
                        "rxcui": concept.get("rxcui", ""),
                        "name": concept.get("name", ""),
                        "tty": tty,
                    })
            
            return related
        except Exception as e:
            return []
    
    async def get_drug_interactions(self, rxcui: str) -> List[Dict[str, str]]:
        """Get drug-drug interactions."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/interaction/interaction.json",
                params={"rxcui": rxcui},
            )
            response.raise_for_status()
            data = response.json()
            
            interactions = []
            interaction_groups = data.get("interactionTypeGroup", [])
            for group in interaction_groups:
                for itype in group.get("interactionType", []):
                    for pair in itype.get("interactionPair", []):
                        description = pair.get("description", "")
                        severity = pair.get("severity", "")
                        
                        # Get the interacting drug
                        concepts = pair.get("interactionConcept", [])
                        for concept in concepts:
                            drug_info = concept.get("minConceptItem", {})
                            if drug_info.get("rxcui") != rxcui:
                                interactions.append({
                                    "drug": drug_info.get("name", ""),
                                    "rxcui": drug_info.get("rxcui", ""),
                                    "description": description,
                                    "severity": severity,
                                })
            
            return interactions
        except Exception as e:
            return []
    
    async def get_ndc_codes(self, rxcui: str) -> List[str]:
        """Get NDC codes for a drug."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/rxcui/{rxcui}/ndcs.json",
            )
            response.raise_for_status()
            data = response.json()
            
            ndc_group = data.get("ndcGroup", {})
            ndc_list = ndc_group.get("ndcList", {}).get("ndc", [])
            return ndc_list  # No limit on NDCs
        except Exception as e:
            return []
    
    async def get_drug_classes(self, rxcui: str) -> List[str]:
        """Get drug classes for a drug."""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/rxclass/class/byRxcui.json",
                params={"rxcui": rxcui},
            )
            response.raise_for_status()
            data = response.json()
            
            classes = []
            class_info = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
            for info in class_info:
                class_name = info.get("rxclassMinConceptItem", {}).get("className", "")
                if class_name and class_name not in classes:
                    classes.append(class_name)
            
            return classes  # No limit on classes
        except Exception as e:
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class RxNormIngestionPipeline:
    """
    Pipeline for ingesting RxNorm drug data into UMI knowledge base.
    """
    
    # Common drug names to fetch (ingredients and brand names)
    DRUG_NAMES = [
        # Top prescribed medications
        "lisinopril", "atorvastatin", "metformin", "amlodipine", "metoprolol",
        "omeprazole", "simvastatin", "losartan", "gabapentin", "hydrochlorothiazide",
        "sertraline", "acetaminophen", "atenolol", "levothyroxine", "furosemide",
        "azithromycin", "amoxicillin", "alprazolam", "prednisone", "zolpidem",
        "clopidogrel", "pantoprazole", "escitalopram", "carvedilol", "trazodone",
        "fluticasone", "montelukast", "rosuvastatin", "tramadol", "tamsulosin",
        "duloxetine", "venlafaxine", "bupropion", "citalopram", "fluoxetine",
        "paroxetine", "mirtazapine", "quetiapine", "aripiprazole", "risperidone",
        "olanzapine", "clonazepam", "lorazepam", "diazepam", "buspirone",
        "warfarin", "rivaroxaban", "apixaban", "dabigatran", "enoxaparin",
        "insulin glargine", "insulin lispro", "insulin aspart", "sitagliptin", "empagliflozin",
        "liraglutide", "semaglutide", "dulaglutide", "glipizide", "glyburide",
        "albuterol", "budesonide", "tiotropium", "formoterol", "ipratropium",
        "cetirizine", "loratadine", "fexofenadine", "diphenhydramine", "hydroxyzine",
        "ibuprofen", "naproxen", "meloxicam", "celecoxib", "diclofenac",
        "oxycodone", "hydrocodone", "morphine", "fentanyl", "codeine",
        "cyclobenzaprine", "baclofen", "tizanidine", "methocarbamol",
        "ciprofloxacin", "levofloxacin", "doxycycline", "cephalexin", "amoxicillin-clavulanate",
        "sulfamethoxazole-trimethoprim", "nitrofurantoin", "metronidazole", "clindamycin",
        "fluconazole", "nystatin", "terbinafine", "ketoconazole",
        "acyclovir", "valacyclovir", "oseltamivir",
        "esomeprazole", "famotidine", "ranitidine", "sucralfate",
        "ondansetron", "promethazine", "metoclopramide", "prochlorperazine",
        "spironolactone", "triamterene", "chlorthalidone", "indapamide",
        "diltiazem", "verapamil", "nifedipine", "felodipine",
        "pravastatin", "lovastatin", "fluvastatin", "pitavastatin",
        "ezetimibe", "fenofibrate", "gemfibrozil", "niacin",
        "liothyronine", "methimazole", "propylthiouracil",
        "allopurinol", "colchicine", "febuxostat",
        "finasteride", "dutasteride", "sildenafil", "tadalafil",
        "sumatriptan", "rizatriptan", "topiramate", "valproic acid",
        "levetiracetam", "lamotrigine", "carbamazepine", "phenytoin", "oxcarbazepine",
        "lithium", "divalproex",
        "donepezil", "memantine", "rivastigmine", "galantamine",
        "carbidopa-levodopa", "pramipexole", "ropinirole", "rasagiline",
        "methylphenidate", "amphetamine", "lisdexamfetamine", "atomoxetine",
        "prednisone", "methylprednisolone", "dexamethasone", "hydrocortisone",
        "methotrexate", "hydroxychloroquine", "sulfasalazine", "leflunomide",
        "adalimumab", "etanercept", "infliximab", "certolizumab", "golimumab",
        "tacrolimus", "cyclosporine", "mycophenolate", "azathioprine",
    ]
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/rxnorm",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = RxNormClient()
        self.drugs: List[RxNormDrug] = []
        self.processed_rxcuis: Set[str] = set()
    
    async def fetch_drug(self, name: str) -> List[RxNormDrug]:
        """Fetch drug information by name."""
        drugs = []
        
        # Search for the drug
        search_results = await self.client.search_drugs(name)
        
        for result in search_results:  # No limit - process all results
            rxcui = result.get("rxcui", "")
            
            if not rxcui or rxcui in self.processed_rxcuis:
                continue
            
            self.processed_rxcuis.add(rxcui)
            
            # Get additional information
            properties = await self.client.get_drug_properties(rxcui)
            related = await self.client.get_related_drugs(rxcui)
            interactions = await self.client.get_drug_interactions(rxcui)
            ndc_codes = await self.client.get_ndc_codes(rxcui)
            drug_classes = await self.client.get_drug_classes(rxcui)
            
            # Extract ingredients
            ingredients = []
            for rel in related:
                if rel.get("tty") == "IN":
                    ingredients.append(rel.get("name", ""))
            
            # Extract brand names
            brand_names = []
            for rel in related:
                if rel.get("tty") == "BN":
                    brand_names.append(rel.get("name", ""))
            
            # Extract dose forms and strengths from properties
            dose_forms = properties.get("DOSE_FORM", [])
            strengths = properties.get("STRENGTH", [])
            
            drug = RxNormDrug(
                rxcui=rxcui,
                name=result.get("name", ""),
                synonym=result.get("synonym", ""),
                tty=result.get("tty", ""),
                ingredients=ingredients,
                brand_names=brand_names,
                dose_forms=dose_forms,
                strengths=strengths,
                ndc_codes=ndc_codes,
                drug_classes=drug_classes,
                interactions=interactions,  # No limit on interactions
                related_drugs=[r for r in related if r.get("tty") in ("SBD", "SCD")][:10],
            )
            drugs.append(drug)
            
            # Rate limiting
            await asyncio.sleep(0.2)
        
        return drugs
    
    async def run(self) -> None:
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI RxNorm Ingestion Pipeline")
        print("=" * 60)
        
        all_drugs = []
        
        for name in tqdm(self.DRUG_NAMES, desc="Fetching drugs"):
            try:
                drugs = await self.fetch_drug(name)
                all_drugs.extend(drugs)
                if drugs:
                    print(f"  {name}: {len(drugs)} entries")
            except Exception as e:
                print(f"  Error fetching {name}: {e}")
            
            await asyncio.sleep(0.3)  # Rate limiting
        
        self.drugs = all_drugs
        print(f"\nTotal drug entries: {len(self.drugs)}")
        
        # Save
        await self.save()
        
        # Close client
        await self.client.close()
    
    async def save(self) -> None:
        """Save drugs to disk."""
        output_file = self.output_dir / "drugs.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for drug in self.drugs:
                # Create comprehensive content for RAG
                content_parts = [
                    f"Drug: {drug.name}",
                    f"RxCUI: {drug.rxcui}",
                    f"Type: {drug.tty}",
                    "",
                ]
                
                if drug.ingredients:
                    content_parts.append(f"Active Ingredients: {', '.join(drug.ingredients)}")
                
                if drug.brand_names:
                    content_parts.append(f"Brand Names: {', '.join(drug.brand_names)}")
                
                if drug.drug_classes:
                    content_parts.append(f"Drug Classes: {', '.join(drug.drug_classes)}")
                
                if drug.dose_forms:
                    content_parts.append(f"Dose Forms: {', '.join(drug.dose_forms)}")
                
                if drug.strengths:
                    content_parts.append(f"Strengths: {', '.join(drug.strengths)}")
                
                if drug.interactions:
                    content_parts.append("")
                    content_parts.append("Drug Interactions:")
                    for interaction in drug.interactions[:10]:
                        severity = interaction.get("severity", "")
                        desc = interaction.get("description", "")[:200]
                        content_parts.append(f"- {interaction.get('drug', '')} ({severity}): {desc}")
                
                doc = {
                    "id": f"rxnorm_{drug.rxcui}",
                    "title": drug.name,
                    "content": "\n".join(content_parts),
                    "metadata": {
                        "rxcui": drug.rxcui,
                        "tty": drug.tty,
                        "ingredients": drug.ingredients,
                        "brand_names": drug.brand_names,
                        "drug_classes": drug.drug_classes,
                        "ndc_codes": drug.ndc_codes[:5],
                        "source": "RxNorm",
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save interactions separately for quick lookup
        interactions_file = self.output_dir / "interactions.jsonl"
        with open(interactions_file, 'w', encoding='utf-8') as f:
            for drug in self.drugs:
                for interaction in drug.interactions:
                    doc = {
                        "drug1": drug.name,
                        "drug1_rxcui": drug.rxcui,
                        "drug2": interaction.get("drug", ""),
                        "drug2_rxcui": interaction.get("rxcui", ""),
                        "severity": interaction.get("severity", ""),
                        "description": interaction.get("description", ""),
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # Save statistics
        stats = {
            "total_drugs": len(self.drugs),
            "total_interactions": sum(len(d.interactions) for d in self.drugs),
            "ingestion_date": datetime.now().isoformat(),
            "drug_names_searched": self.DRUG_NAMES,
        }
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


async def main():
    """Run the RxNorm ingestion pipeline."""
    pipeline = RxNormIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
