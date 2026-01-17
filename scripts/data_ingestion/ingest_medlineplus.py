"""
UMI MedlinePlus Data Ingestion Pipeline
<<<<<<< HEAD
Fetches consumer health information from NIH MedlinePlus - NO LIMITS
=======
Fetches health information from NIH MedlinePlus for RAG
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
"""

import asyncio
import json
<<<<<<< HEAD
from pathlib import Path
from typing import Any, Dict, List
=======
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)

import httpx
from tqdm import tqdm


<<<<<<< HEAD
class MedlinePlusClient:
    """Client for MedlinePlus API."""
    
    BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"
=======
@dataclass
class HealthTopic:
    """Represents a MedlinePlus health topic."""
    id: str
    title: str
    url: str
    language: str
    date_created: str
    full_summary: str
    also_called: List[str]
    primary_institute: str
    see_references: List[str]
    groups: List[str]
    mesh_headings: List[str]
    related_topics: List[str]


class MedlinePlusClient:
    """
    Client for MedlinePlus Connect and Health Topics API.
    https://medlineplus.gov/webservices.html
    """
    
    BASE_URL = "https://wsearch.nlm.nih.gov/ws"
    HEALTH_TOPICS_URL = "https://medlineplus.gov/xml"
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
    
<<<<<<< HEAD
    async def close(self):
        await self.client.aclose()
    
    async def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search MedlinePlus health topics."""
        results = []
        
        params = {
            "db": "healthTopics",
            "term": query,
            "retmax": min(max_results, 100),
            "rettype": "all",
        }
        
        try:
            response = await self.client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            # Parse XML response (simplified)
            text = response.text
            
            # Extract documents from XML
            import re
            docs = re.findall(r'<document[^>]*>(.*?)</document>', text, re.DOTALL)
            
            for doc in docs:
                title_match = re.search(r'<content name="title">(.*?)</content>', doc)
                snippet_match = re.search(r'<content name="FullSummary">(.*?)</content>', doc, re.DOTALL)
                url_match = re.search(r'<content name="url">(.*?)</content>', doc)
                
                if title_match:
                    results.append({
                        "title": title_match.group(1).strip(),
                        "summary": snippet_match.group(1).strip() if snippet_match else "",
                        "url": url_match.group(1).strip() if url_match else "",
                    })
        except Exception as e:
            print(f"    API error: {e}")
        
        return results


HEALTH_TOPICS = [
    # Common conditions
    "diabetes", "hypertension", "heart disease", "stroke", "cancer",
    "asthma", "COPD", "arthritis", "osteoporosis", "depression",
    "anxiety", "alzheimer", "parkinson", "epilepsy", "migraine",
    
    # Symptoms
    "chest pain", "headache", "fatigue", "fever", "cough",
    "shortness of breath", "dizziness", "nausea", "back pain", "joint pain",
    
    # Medications
    "aspirin", "ibuprofen", "acetaminophen", "antibiotics", "antidepressants",
    "blood pressure medication", "cholesterol medication", "diabetes medication",
    
    # Procedures
    "surgery", "MRI", "CT scan", "blood test", "colonoscopy",
    "mammogram", "vaccination", "physical therapy", "chemotherapy",
    
    # Wellness
    "nutrition", "exercise", "sleep", "stress management", "weight loss",
    "smoking cessation", "alcohol", "mental health", "preventive care",
    
    # Body systems
    "heart", "lungs", "liver", "kidney", "brain", "digestive system",
    "immune system", "endocrine system", "nervous system", "musculoskeletal",
    
    # Life stages
    "pregnancy", "infant health", "child health", "teen health",
    "women health", "men health", "senior health", "aging",
    
    # Diseases A-Z
    "anemia", "bronchitis", "celiac disease", "dementia", "eczema",
    "fibromyalgia", "gout", "hepatitis", "influenza", "jaundice",
    "kidney stones", "lupus", "meningitis", "neuropathy", "obesity",
    "pneumonia", "rheumatoid arthritis", "sepsis", "thyroid", "ulcer",
]


class MedlinePlusIngestionPipeline:
    """Pipeline for ingesting MedlinePlus health information - NO LIMITS."""
    
    def __init__(self, output_dir: str = "data/knowledge_base/medlineplus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = MedlinePlusClient()
        self.topics: List[Dict] = []
    
    def _topic_to_document(self, topic: Dict, query: str) -> Dict[str, Any]:
        """Convert health topic to document."""
        title = topic.get("title", "Health Topic")
        summary = topic.get("summary", "")
        url = topic.get("url", "")
        
        # Clean HTML from summary
        import re
        clean_summary = re.sub(r'<[^>]+>', '', summary)
        clean_summary = re.sub(r'\s+', ' ', clean_summary).strip()
        
        content = clean_summary if clean_summary else f"Health information about {title}"
        
        return {
            "id": f"medlineplus_{title.lower().replace(' ', '_')[:50]}",
            "title": title,
            "content": content,
            "metadata": {
                "source": "MedlinePlus",
                "url": url,
                "search_query": query,
                "type": "health_topic",
            }
        }
    
    async def run(self) -> None:
        """Run the ingestion pipeline - NO LIMITS."""
        print("=" * 60)
        print("MEDLINEPLUS HEALTH INFORMATION INGESTION")
        print("=" * 60)
        
        try:
            topic_docs = []
            seen_titles = set()
            
            for query in tqdm(HEALTH_TOPICS, desc="Health topics"):
                try:
                    results = await self.client.search(query, max_results=50)
                    
                    for topic in results:
                        title = topic.get("title", "").lower()
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            doc = self._topic_to_document(topic, query)
                            topic_docs.append(doc)
                            self.topics.append(topic)
                    
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"   Warning: Error with {query}: {e}")
            
            # Save
            with open(self.output_dir / "topics.jsonl", 'w', encoding='utf-8') as f:
                for doc in topic_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            print(f"\n{'=' * 60}")
            print("MEDLINEPLUS INGESTION COMPLETE")
            print(f"Topics: {len(topic_docs)}")
            print(f"Output: {self.output_dir}")
            print("=" * 60)
            
        finally:
            await self.client.close()


async def main():
=======
    async def search_health_topics(
        self,
        query: str,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for health topics."""
        params = {
            "db": "healthTopics",
            "term": query,
            "retmax": max_results,
        }
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/query",
                params=params,
            )
            response.raise_for_status()
            return self._parse_search_results(response.text)
        except Exception as e:
            print(f"Error searching health topics: {e}")
            return []
    
    def _parse_search_results(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse search results XML."""
        results = []
        try:
            root = ET.fromstring(xml_text)
            for doc in root.findall(".//document"):
                content = doc.find("content")
                if content is not None:
                    results.append({
                        "title": content.findtext("title", ""),
                        "url": content.findtext("FullSummary/@url", ""),
                        "snippet": content.findtext("snippet", ""),
                    })
        except Exception as e:
            print(f"Parse error: {e}")
        return results
    
    async def get_health_topics_xml(self) -> str:
        """Download the full health topics XML file."""
        try:
            response = await self.client.get(
                f"{self.HEALTH_TOPICS_URL}/mplus_topics_2024-01-01.xml",
                timeout=120.0,
            )
            response.raise_for_status()
            return response.text
        except Exception:
            # Try alternative URL
            try:
                response = await self.client.get(
                    "https://medlineplus.gov/xml/mplus_topics.xml",
                    timeout=120.0,
                )
                response.raise_for_status()
                return response.text
            except Exception as e:
                print(f"Error fetching health topics XML: {e}")
                return ""
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class MedlinePlusIngestionPipeline:
    """
    Pipeline for ingesting MedlinePlus health topics into UMI knowledge base.
    """
    
    # Comprehensive health topics to search
    HEALTH_TOPICS = [
        # Common conditions
        "diabetes", "hypertension", "heart disease", "stroke", "cancer",
        "asthma", "COPD", "arthritis", "osteoporosis", "obesity",
        "depression", "anxiety", "Alzheimer's disease", "Parkinson's disease",
        "multiple sclerosis", "epilepsy", "migraine", "chronic pain",
        
        # Infectious diseases
        "COVID-19", "influenza", "pneumonia", "tuberculosis", "HIV AIDS",
        "hepatitis", "meningitis", "sepsis", "urinary tract infection",
        "sexually transmitted diseases", "Lyme disease", "malaria",
        
        # Women's health
        "pregnancy", "menopause", "breast cancer", "ovarian cancer",
        "endometriosis", "polycystic ovary syndrome", "cervical cancer",
        "osteoporosis in women", "menstrual disorders",
        
        # Men's health
        "prostate cancer", "erectile dysfunction", "testicular cancer",
        "benign prostatic hyperplasia", "male infertility",
        
        # Children's health
        "childhood obesity", "ADHD", "autism", "childhood asthma",
        "childhood diabetes", "childhood vaccines", "developmental delays",
        
        # Mental health
        "bipolar disorder", "schizophrenia", "PTSD", "eating disorders",
        "substance abuse", "alcohol use disorder", "opioid addiction",
        "suicide prevention", "stress management",
        
        # Digestive health
        "irritable bowel syndrome", "Crohn's disease", "ulcerative colitis",
        "GERD", "celiac disease", "liver disease", "pancreatitis",
        "gallbladder disease", "hemorrhoids", "constipation",
        
        # Heart and blood
        "coronary artery disease", "heart failure", "arrhythmia",
        "atrial fibrillation", "high cholesterol", "anemia",
        "blood clots", "peripheral artery disease", "varicose veins",
        
        # Respiratory
        "lung cancer", "pulmonary fibrosis", "sleep apnea",
        "bronchitis", "emphysema", "cystic fibrosis", "allergies",
        
        # Kidney and urinary
        "chronic kidney disease", "kidney stones", "urinary incontinence",
        "bladder cancer", "kidney cancer", "dialysis",
        
        # Skin conditions
        "psoriasis", "eczema", "acne", "skin cancer", "melanoma",
        "rosacea", "vitiligo", "shingles", "wound care",
        
        # Eye health
        "glaucoma", "cataracts", "macular degeneration", "diabetic retinopathy",
        "dry eye", "vision problems",
        
        # Bone and joint
        "back pain", "neck pain", "fibromyalgia", "gout",
        "rheumatoid arthritis", "lupus", "scoliosis", "fractures",
        
        # Endocrine
        "thyroid disease", "hypothyroidism", "hyperthyroidism",
        "adrenal disorders", "pituitary disorders", "metabolic syndrome",
        
        # Medications
        "pain medications", "antibiotics", "blood thinners",
        "antidepressants", "blood pressure medications", "diabetes medications",
        "cholesterol medications", "drug interactions", "medication safety",
        
        # Procedures and tests
        "MRI", "CT scan", "blood tests", "colonoscopy", "mammogram",
        "biopsy", "surgery preparation", "anesthesia",
        
        # Lifestyle
        "nutrition", "exercise", "weight loss", "smoking cessation",
        "alcohol and health", "sleep disorders", "stress",
        
        # Emergency
        "first aid", "CPR", "choking", "burns", "poisoning",
        "heart attack symptoms", "stroke symptoms",
    ]
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/medlineplus",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = MedlinePlusClient()
        self.topics: List[Dict[str, Any]] = []
    
    async def fetch_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Fetch health information for a topic."""
        results = await self.client.search_health_topics(topic)
        return results
    
    async def run(self) -> None:
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI MedlinePlus Ingestion Pipeline")
        print("=" * 60)
        
        all_topics = []
        
        for topic in tqdm(self.HEALTH_TOPICS, desc="Fetching topics"):
            try:
                results = await self.fetch_topic(topic)
                for result in results:
                    result["search_term"] = topic
                all_topics.extend(results)
                print(f"  {topic}: {len(results)} results")
            except Exception as e:
                print(f"  Error fetching {topic}: {e}")
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Deduplicate by URL
        seen_urls = set()
        unique_topics = []
        for topic in all_topics:
            url = topic.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_topics.append(topic)
        
        self.topics = unique_topics
        print(f"\nTotal unique topics: {len(self.topics)}")
        
        # Save
        await self.save()
        
        # Close client
        await self.client.close()
    
    async def save(self) -> None:
        """Save topics to disk."""
        output_file = self.output_dir / "health_topics.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for topic in self.topics:
                doc = {
                    "id": f"medlineplus_{hash(topic.get('url', topic.get('title', '')))}",
                    "title": topic.get("title", ""),
                    "content": f"{topic.get('title', '')}\n\n{topic.get('snippet', '')}",
                    "metadata": {
                        "url": topic.get("url", ""),
                        "search_term": topic.get("search_term", ""),
                        "source": "MedlinePlus",
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save statistics
        stats = {
            "total_topics": len(self.topics),
            "ingestion_date": datetime.now().isoformat(),
            "search_terms": self.HEALTH_TOPICS,
        }
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


async def main():
    """Run the MedlinePlus ingestion pipeline."""
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    pipeline = MedlinePlusIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
