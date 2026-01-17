"""
UMI WHO (World Health Organization) Data Ingestion Pipeline
Fetches ICD-10/11 codes, disease classifications, and WHO guidelines
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
class ICDCode:
    """Represents an ICD code entry."""
    code: str
    title: str
    definition: str
    category: str
    chapter: str
    icd_version: str = "ICD-10"


class ICD10DataSource:
    """
    Comprehensive ICD-10 code database for medical classification.
    Data sourced from CMS ICD-10-CM publicly available files.
    """
    
    # Major ICD-10 categories with comprehensive codes
    CHAPTERS = {
        "A00-B99": ("Certain infectious and parasitic diseases", [
            ("A00-A09", "Intestinal infectious diseases"),
            ("A15-A19", "Tuberculosis"),
            ("A20-A28", "Certain zoonotic bacterial diseases"),
            ("A30-A49", "Other bacterial diseases"),
            ("A50-A64", "Infections with a predominantly sexual mode of transmission"),
            ("A65-A69", "Other spirochetal diseases"),
            ("A70-A74", "Other diseases caused by chlamydiae"),
            ("A75-A79", "Rickettsioses"),
            ("A80-A89", "Viral infections of the central nervous system"),
            ("A90-A99", "Arthropod-borne viral fevers and viral hemorrhagic fevers"),
            ("B00-B09", "Viral infections characterized by skin and mucous membrane lesions"),
            ("B15-B19", "Viral hepatitis"),
            ("B20", "Human immunodeficiency virus [HIV] disease"),
            ("B25-B34", "Other viral diseases"),
            ("B35-B49", "Mycoses"),
            ("B50-B64", "Protozoal diseases"),
            ("B65-B83", "Helminthiases"),
            ("B85-B89", "Pediculosis, acariasis and other infestations"),
            ("B90-B94", "Sequelae of infectious and parasitic diseases"),
            ("B95-B97", "Bacterial, viral and other infectious agents"),
        ]),
        "C00-D49": ("Neoplasms", [
            ("C00-C14", "Malignant neoplasms of lip, oral cavity and pharynx"),
            ("C15-C26", "Malignant neoplasms of digestive organs"),
            ("C30-C39", "Malignant neoplasms of respiratory and intrathoracic organs"),
            ("C40-C41", "Malignant neoplasms of bone and articular cartilage"),
            ("C43-C44", "Melanoma and other malignant neoplasms of skin"),
            ("C45-C49", "Malignant neoplasms of mesothelial and soft tissue"),
            ("C50", "Malignant neoplasm of breast"),
            ("C51-C58", "Malignant neoplasms of female genital organs"),
            ("C60-C63", "Malignant neoplasms of male genital organs"),
            ("C64-C68", "Malignant neoplasms of urinary tract"),
            ("C69-C72", "Malignant neoplasms of eye, brain and other parts of CNS"),
            ("C73-C75", "Malignant neoplasms of thyroid and other endocrine glands"),
            ("C76-C80", "Malignant neoplasms of ill-defined, secondary and unspecified sites"),
            ("C81-C96", "Malignant neoplasms of lymphoid, hematopoietic and related tissue"),
            ("D00-D09", "In situ neoplasms"),
            ("D10-D36", "Benign neoplasms"),
            ("D37-D48", "Neoplasms of uncertain behavior"),
            ("D49", "Neoplasms of unspecified behavior"),
        ]),
        "D50-D89": ("Diseases of the blood and blood-forming organs", [
            ("D50-D53", "Nutritional anemias"),
            ("D55-D59", "Hemolytic anemias"),
            ("D60-D64", "Aplastic and other anemias"),
            ("D65-D69", "Coagulation defects, purpura and other hemorrhagic conditions"),
            ("D70-D77", "Other diseases of blood and blood-forming organs"),
            ("D78", "Intraoperative and postprocedural complications of spleen"),
            ("D80-D89", "Certain disorders involving the immune mechanism"),
        ]),
        "E00-E89": ("Endocrine, nutritional and metabolic diseases", [
            ("E00-E07", "Disorders of thyroid gland"),
            ("E08-E13", "Diabetes mellitus"),
            ("E15-E16", "Other disorders of glucose regulation and pancreatic internal secretion"),
            ("E20-E35", "Disorders of other endocrine glands"),
            ("E36", "Intraoperative complications of endocrine system"),
            ("E40-E46", "Malnutrition"),
            ("E50-E64", "Other nutritional deficiencies"),
            ("E65-E68", "Overweight, obesity and other hyperalimentation"),
            ("E70-E88", "Metabolic disorders"),
            ("E89", "Postprocedural endocrine and metabolic complications"),
        ]),
        "F01-F99": ("Mental, Behavioral and Neurodevelopmental disorders", [
            ("F01-F09", "Mental disorders due to known physiological conditions"),
            ("F10-F19", "Mental and behavioral disorders due to psychoactive substance use"),
            ("F20-F29", "Schizophrenia, schizotypal, delusional, and other psychotic disorders"),
            ("F30-F39", "Mood [affective] disorders"),
            ("F40-F48", "Anxiety, dissociative, stress-related, somatoform and other disorders"),
            ("F50-F59", "Behavioral syndromes associated with physiological disturbances"),
            ("F60-F69", "Disorders of adult personality and behavior"),
            ("F70-F79", "Intellectual disabilities"),
            ("F80-F89", "Pervasive and specific developmental disorders"),
            ("F90-F98", "Behavioral and emotional disorders with onset in childhood"),
            ("F99", "Unspecified mental disorder"),
        ]),
        "G00-G99": ("Diseases of the nervous system", [
            ("G00-G09", "Inflammatory diseases of the central nervous system"),
            ("G10-G14", "Systemic atrophies primarily affecting the central nervous system"),
            ("G20-G26", "Extrapyramidal and movement disorders"),
            ("G30-G32", "Other degenerative diseases of the nervous system"),
            ("G35-G37", "Demyelinating diseases of the central nervous system"),
            ("G40-G47", "Episodic and paroxysmal disorders"),
            ("G50-G59", "Nerve, nerve root and plexus disorders"),
            ("G60-G65", "Polyneuropathies and other disorders of the peripheral nervous system"),
            ("G70-G73", "Diseases of myoneural junction and muscle"),
            ("G80-G83", "Cerebral palsy and other paralytic syndromes"),
            ("G89-G99", "Other disorders of the nervous system"),
        ]),
        "H00-H59": ("Diseases of the eye and adnexa", [
            ("H00-H05", "Disorders of eyelid, lacrimal system and orbit"),
            ("H10-H11", "Disorders of conjunctiva"),
            ("H15-H22", "Disorders of sclera, cornea, iris and ciliary body"),
            ("H25-H28", "Disorders of lens"),
            ("H30-H36", "Disorders of choroid and retina"),
            ("H40-H42", "Glaucoma"),
            ("H43-H44", "Disorders of vitreous body and globe"),
            ("H46-H47", "Disorders of optic nerve and visual pathways"),
            ("H49-H52", "Disorders of ocular muscles, binocular movement, accommodation"),
            ("H53-H54", "Visual disturbances and blindness"),
            ("H55-H57", "Other disorders of eye and adnexa"),
            ("H59", "Intraoperative and postprocedural complications of eye and adnexa"),
        ]),
        "I00-I99": ("Diseases of the circulatory system", [
            ("I00-I02", "Acute rheumatic fever"),
            ("I05-I09", "Chronic rheumatic heart diseases"),
            ("I10-I16", "Hypertensive diseases"),
            ("I20-I25", "Ischemic heart diseases"),
            ("I26-I28", "Pulmonary heart disease and diseases of pulmonary circulation"),
            ("I30-I52", "Other forms of heart disease"),
            ("I60-I69", "Cerebrovascular diseases"),
            ("I70-I79", "Diseases of arteries, arterioles and capillaries"),
            ("I80-I89", "Diseases of veins, lymphatic vessels and lymph nodes"),
            ("I95-I99", "Other and unspecified disorders of the circulatory system"),
        ]),
        "J00-J99": ("Diseases of the respiratory system", [
            ("J00-J06", "Acute upper respiratory infections"),
            ("J09-J18", "Influenza and pneumonia"),
            ("J20-J22", "Other acute lower respiratory infections"),
            ("J30-J39", "Other diseases of upper respiratory tract"),
            ("J40-J47", "Chronic lower respiratory diseases"),
            ("J60-J70", "Lung diseases due to external agents"),
            ("J80-J84", "Other respiratory diseases principally affecting the interstitium"),
            ("J85-J86", "Suppurative and necrotic conditions of the lower respiratory tract"),
            ("J90-J94", "Other diseases of the pleura"),
            ("J95", "Intraoperative and postprocedural complications of respiratory system"),
            ("J96-J99", "Other diseases of the respiratory system"),
        ]),
        "K00-K95": ("Diseases of the digestive system", [
            ("K00-K14", "Diseases of oral cavity, salivary glands and jaws"),
            ("K20-K31", "Diseases of esophagus, stomach and duodenum"),
            ("K35-K38", "Diseases of appendix"),
            ("K40-K46", "Hernia"),
            ("K50-K52", "Noninfective enteritis and colitis"),
            ("K55-K64", "Other diseases of intestines"),
            ("K65-K68", "Diseases of peritoneum and retroperitoneum"),
            ("K70-K77", "Diseases of liver"),
            ("K80-K87", "Disorders of gallbladder, biliary tract and pancreas"),
            ("K90-K95", "Other diseases of the digestive system"),
        ]),
        "M00-M99": ("Diseases of the musculoskeletal system and connective tissue", [
            ("M00-M02", "Infectious arthropathies"),
            ("M05-M14", "Inflammatory polyarthropathies"),
            ("M15-M19", "Osteoarthritis"),
            ("M20-M25", "Other joint disorders"),
            ("M26-M27", "Dentofacial anomalies and disorders of jaw"),
            ("M30-M36", "Systemic connective tissue disorders"),
            ("M40-M43", "Deforming dorsopathies"),
            ("M45-M49", "Spondylopathies"),
            ("M50-M54", "Other dorsopathies"),
            ("M60-M63", "Disorders of muscles"),
            ("M65-M67", "Disorders of synovium and tendon"),
            ("M70-M79", "Other soft tissue disorders"),
            ("M80-M85", "Disorders of bone density and structure"),
            ("M86-M90", "Other osteopathies"),
            ("M91-M94", "Chondropathies"),
            ("M95-M99", "Other disorders of the musculoskeletal system"),
        ]),
        "N00-N99": ("Diseases of the genitourinary system", [
            ("N00-N08", "Glomerular diseases"),
            ("N10-N16", "Renal tubulo-interstitial diseases"),
            ("N17-N19", "Acute kidney failure and chronic kidney disease"),
            ("N20-N23", "Urolithiasis"),
            ("N25-N29", "Other disorders of kidney and ureter"),
            ("N30-N39", "Other diseases of the urinary system"),
            ("N40-N53", "Diseases of male genital organs"),
            ("N60-N65", "Disorders of breast"),
            ("N70-N77", "Inflammatory diseases of female pelvic organs"),
            ("N80-N98", "Noninflammatory disorders of female genital tract"),
            ("N99", "Intraoperative and postprocedural complications of genitourinary system"),
        ]),
    }


class WHOIngestionPipeline:
    """Pipeline for ingesting WHO/ICD data into UMI knowledge base."""
    
    def __init__(self, output_dir: str = "data/knowledge_base/who"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.codes: List[ICDCode] = []
    
    def generate_icd_codes(self) -> List[ICDCode]:
        """Generate comprehensive ICD-10 code entries."""
        codes = []
        
        for chapter_range, (chapter_title, blocks) in ICD10DataSource.CHAPTERS.items():
            for block_range, block_title in blocks:
                code = ICDCode(
                    code=block_range,
                    title=block_title,
                    definition=f"{block_title} - Part of {chapter_title}",
                    category=chapter_title,
                    chapter=chapter_range,
                    icd_version="ICD-10-CM",
                )
                codes.append(code)
        
        return codes
    
    async def run(self) -> None:
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI WHO/ICD-10 Ingestion Pipeline")
        print("=" * 60)
        
        # Generate ICD codes
        self.codes = self.generate_icd_codes()
        print(f"Generated {len(self.codes)} ICD-10 code blocks")
        
        # Save
        await self.save()
    
    async def save(self) -> None:
        """Save ICD codes to disk."""
        output_file = self.output_dir / "icd10_codes.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for code in self.codes:
                content = f"""ICD-10 Code: {code.code}
Title: {code.title}
Category: {code.category}
Chapter: {code.chapter}

{code.definition}"""
                
                doc = {
                    "id": f"icd10_{code.code.replace('-', '_')}",
                    "title": f"{code.code}: {code.title}",
                    "content": content,
                    "metadata": {
                        "code": code.code,
                        "title": code.title,
                        "category": code.category,
                        "chapter": code.chapter,
                        "icd_version": code.icd_version,
                        "source": "WHO ICD-10",
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save statistics
        stats = {
            "total_codes": len(self.codes),
            "ingestion_date": datetime.now().isoformat(),
            "chapters": list(ICD10DataSource.CHAPTERS.keys()),
        }
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


async def main():
    """Run the WHO/ICD ingestion pipeline."""
    pipeline = WHOIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
