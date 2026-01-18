"""
IMI PubMed Scraper - Fetch medical literature from PubMed/NCBI
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from base_scraper import BaseScraper, ScrapedItem, logger


@dataclass
class PubMedArticle:
    """PubMed article data."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    mesh_terms: List[str]
    keywords: List[str]
    doi: Optional[str] = None


class PubMedScraper(BaseScraper[PubMedArticle]):
    """
    Scraper for PubMed medical literature.
    Uses NCBI E-utilities API: https://www.ncbi.nlm.nih.gov/books/NBK25500/
    """
    
    SOURCE_NAME = "pubmed"
    DEFAULT_RATE_LIMIT = 3.0  # 3 req/sec without API key, 10 with
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    # Medical topics to search
    MEDICAL_TOPICS = [
        "diabetes treatment",
        "hypertension management",
        "cancer immunotherapy",
        "cardiovascular disease",
        "infectious disease",
        "mental health disorders",
        "autoimmune diseases",
        "neurodegenerative diseases",
        "pediatric medicine",
        "geriatric care",
        "drug interactions",
        "clinical trials",
        "medical diagnosis",
        "surgical procedures",
        "emergency medicine",
        "preventive medicine",
        "pharmacology",
        "pathophysiology",
        "medical imaging",
        "genetic disorders",
    ]
    
    def __init__(self, output_dir: str, api_key: Optional[str] = None, email: Optional[str] = None):
        super().__init__(output_dir, api_key=api_key)
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.email = email or os.environ.get("NCBI_EMAIL", "imi@medical.ai")
        
        if self.api_key:
            self.rate_limiter.min_interval = 0.1  # 10 req/sec with API key
    
    async def search(self, query: str, max_results: int = 100) -> List[str]:
        """Search PubMed and return PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = await self.fetch(f"{self.BASE_URL}/esearch.fcgi", params=params)
        data = response.json()
        
        return data.get("esearchresult", {}).get("idlist", [])
    
    async def fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch full article details for given PMIDs."""
        if not pmids:
            return []
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = await self.fetch(f"{self.BASE_URL}/efetch.fcgi", params=params)
        
        return self._parse_xml(response.text)
    
    def _parse_xml(self, xml_text: str) -> List[PubMedArticle]:
        """Parse PubMed XML response."""
        articles = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    # PMID
                    pmid_elem = article_elem.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    
                    # Title
                    title_elem = article_elem.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else ""
                    
                    # Abstract
                    abstract_parts = []
                    for abs_elem in article_elem.findall(".//AbstractText"):
                        label = abs_elem.get("Label", "")
                        text = abs_elem.text or ""
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
                    abstract = "\n".join(abstract_parts)
                    
                    # Authors
                    authors = []
                    for author in article_elem.findall(".//Author"):
                        last = author.find("LastName")
                        first = author.find("ForeName")
                        if last is not None and first is not None:
                            authors.append(f"{first.text} {last.text}")
                    
                    # Journal
                    journal_elem = article_elem.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else ""
                    
                    # Publication date
                    pub_date_elem = article_elem.find(".//PubDate")
                    if pub_date_elem is not None:
                        year = pub_date_elem.find("Year")
                        month = pub_date_elem.find("Month")
                        pub_date = f"{year.text if year is not None else ''}-{month.text if month is not None else '01'}"
                    else:
                        pub_date = ""
                    
                    # MeSH terms
                    mesh_terms = [
                        mesh.find("DescriptorName").text
                        for mesh in article_elem.findall(".//MeshHeading")
                        if mesh.find("DescriptorName") is not None
                    ]
                    
                    # Keywords
                    keywords = [
                        kw.text for kw in article_elem.findall(".//Keyword")
                        if kw.text
                    ]
                    
                    # DOI
                    doi = None
                    for id_elem in article_elem.findall(".//ArticleId"):
                        if id_elem.get("IdType") == "doi":
                            doi = id_elem.text
                            break
                    
                    if pmid and (title or abstract):
                        articles.append(PubMedArticle(
                            pmid=pmid,
                            title=title or "Untitled",
                            abstract=abstract,
                            authors=authors,
                            journal=journal,
                            pub_date=pub_date,
                            mesh_terms=mesh_terms,
                            keywords=keywords,
                            doi=doi,
                        ))
                
                except Exception as e:
                    self.errors.append({"error": str(e), "type": "parse_error"})
        
        except ET.ParseError as e:
            self.errors.append({"error": str(e), "type": "xml_parse_error"})
        
        return articles
    
    async def scrape(
        self,
        topics: Optional[List[str]] = None,
        articles_per_topic: int = 200,
        **kwargs
    ) -> List[PubMedArticle]:
        """
        Scrape PubMed articles for medical topics.
        
        Args:
            topics: List of search topics (uses defaults if None)
            articles_per_topic: Max articles to fetch per topic
        """
        topics = topics or self.MEDICAL_TOPICS
        all_articles = []
        seen_pmids = set()
        
        for topic in topics:
            logger.info(f"Searching PubMed for: {topic}")
            
            try:
                pmids = await self.search(topic, max_results=articles_per_topic)
                
                # Filter already seen
                new_pmids = [p for p in pmids if p not in seen_pmids]
                seen_pmids.update(new_pmids)
                
                if new_pmids:
                    # Fetch in batches of 100
                    for i in range(0, len(new_pmids), 100):
                        batch = new_pmids[i:i+100]
                        articles = await self.fetch_articles(batch)
                        all_articles.extend(articles)
                        logger.info(f"  Fetched {len(articles)} articles for '{topic}'")
                
            except Exception as e:
                logger.error(f"Error scraping topic '{topic}': {e}")
                self.errors.append({"topic": topic, "error": str(e)})
        
        logger.info(f"Total PubMed articles scraped: {len(all_articles)}")
        return all_articles
    
    def item_to_scraped(self, item: PubMedArticle) -> ScrapedItem:
        """Convert PubMedArticle to ScrapedItem."""
        content = f"{item.title}\n\n{item.abstract}"
        
        return ScrapedItem(
            id=f"pubmed_{item.pmid}",
            title=item.title,
            content=content,
            source="pubmed",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{item.pmid}/",
            metadata={
                "pmid": item.pmid,
                "authors": item.authors,
                "journal": item.journal,
                "pub_date": item.pub_date,
                "mesh_terms": item.mesh_terms,
                "keywords": item.keywords,
                "doi": item.doi,
            }
        )


async def main():
    """Run PubMed scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape PubMed medical literature")
    parser.add_argument("--output", default="data/raw/pubmed", help="Output directory")
    parser.add_argument("--api-key", help="NCBI API key (optional, increases rate limit)")
    parser.add_argument("--articles-per-topic", type=int, default=200, help="Articles per topic")
    args = parser.parse_args()
    
    scraper = PubMedScraper(output_dir=args.output, api_key=args.api_key)
    result = await scraper.run(articles_per_topic=args.articles_per_topic)
    
    print(f"\nResults: {result}")


if __name__ == "__main__":
    asyncio.run(main())
