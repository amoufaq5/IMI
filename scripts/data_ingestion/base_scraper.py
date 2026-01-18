"""
IMI Base Scraper - Common functionality for all data scrapers
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Generic

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ScrapedItem:
    """Base class for scraped data items."""
    id: str
    title: str
    content: str
    source: str
    url: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
    
    async def wait(self):
        """Wait if needed to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class BaseScraper(ABC, Generic[T]):
    """
    Base class for all IMI data scrapers.
    Provides common functionality: HTTP client, rate limiting, saving, error handling.
    """
    
    SOURCE_NAME: str = "unknown"
    DEFAULT_RATE_LIMIT: float = 2.0  # requests per second
    
    def __init__(
        self,
        output_dir: str,
        rate_limit: Optional[float] = None,
        api_key: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rate_limiter = RateLimiter(rate_limit or self.DEFAULT_RATE_LIMIT)
        self.api_key = api_key
        
        self.client: Optional[httpx.AsyncClient] = None
        self.items: List[T] = []
        self.errors: List[Dict[str, Any]] = []
        
    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "IMI-Medical-Scraper/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def fetch(self, url: str, params: Optional[Dict] = None) -> httpx.Response:
        """Fetch URL with rate limiting and retries."""
        await self.rate_limiter.wait()
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def post(self, url: str, data: Optional[Dict] = None, json_data: Optional[Dict] = None) -> httpx.Response:
        """POST request with rate limiting and retries."""
        await self.rate_limiter.wait()
        response = await self.client.post(url, data=data, json=json_data)
        response.raise_for_status()
        return response
    
    @abstractmethod
    async def scrape(self, **kwargs) -> List[T]:
        """Main scraping method - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def item_to_scraped(self, item: T) -> ScrapedItem:
        """Convert source-specific item to ScrapedItem."""
        pass
    
    def save_jsonl(self, filename: str = "data.jsonl"):
        """Save scraped items to JSONL file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.items:
                scraped = self.item_to_scraped(item)
                f.write(scraped.to_json() + '\n')
        
        logger.info(f"Saved {len(self.items)} items to {output_path}")
        return output_path
    
    def save_errors(self):
        """Save error log."""
        if self.errors:
            error_path = self.output_dir / "errors.json"
            with open(error_path, 'w') as f:
                json.dump(self.errors, f, indent=2)
            logger.warning(f"Saved {len(self.errors)} errors to {error_path}")
    
    async def run(self, **kwargs) -> Dict[str, Any]:
        """Run the scraper and save results."""
        start_time = time.time()
        
        async with self:
            try:
                self.items = await self.scrape(**kwargs)
            except Exception as e:
                logger.error(f"Scraping failed: {e}")
                self.errors.append({"error": str(e), "type": "scrape_failure"})
        
        # Save results
        output_path = self.save_jsonl()
        self.save_errors()
        
        elapsed = time.time() - start_time
        
        return {
            "source": self.SOURCE_NAME,
            "items_scraped": len(self.items),
            "errors": len(self.errors),
            "output_path": str(output_path),
            "elapsed_seconds": round(elapsed, 2),
        }


class HuggingFaceDatasetScraper(BaseScraper):
    """Scraper that loads from HuggingFace datasets."""
    
    SOURCE_NAME = "huggingface"
    
    def __init__(self, output_dir: str, dataset_name: str, split: str = "train"):
        super().__init__(output_dir)
        self.dataset_name = dataset_name
        self.split = split
    
    async def scrape(self, max_samples: Optional[int] = None, **kwargs) -> List[Dict]:
        """Load dataset from HuggingFace."""
        from datasets import load_dataset
        
        logger.info(f"Loading {self.dataset_name} from HuggingFace...")
        dataset = load_dataset(self.dataset_name, split=self.split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        return list(dataset)
    
    def item_to_scraped(self, item: Dict) -> ScrapedItem:
        """Convert HF dataset item - override in subclass for specific datasets."""
        return ScrapedItem(
            id=str(item.get("id", hash(str(item)))),
            title=item.get("question", item.get("title", ""))[:200],
            content=json.dumps(item, ensure_ascii=False),
            source=self.dataset_name,
            metadata={"split": self.split}
        )
