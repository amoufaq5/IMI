"""
UMI MedlinePlus Data Ingestion Pipeline
Fetches consumer health information from NIH MedlinePlus - NO LIMITS
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
from tqdm import tqdm


class MedlinePlusClient:
    """Client for MedlinePlus API."""
    
    BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
    
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
    pipeline = MedlinePlusIngestionPipeline()
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
