"""
Embedding Service for RAG Pipeline

Generates embeddings for medical text using:
- Sentence transformers (default)
- OpenAI embeddings (optional)
- Medical-specific embeddings (PubMedBERT)
"""
import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates embeddings for text using various models
    
    Supports:
    - sentence-transformers (local, free)
    - OpenAI embeddings (API, paid)
    - PubMedBERT (medical-specific)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_medical_model: bool = False,
        openai_api_key: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.use_medical_model = use_medical_model
        self.openai_api_key = openai_api_key
        self.device = device
        self.model = None
        self.dimension = 384  # Default for MiniLM
        
        if use_medical_model:
            self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            self.dimension = 768
    
    def load_model(self):
        """Load the embedding model"""
        if self.model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.dimension}")
            
        except ImportError:
            logger.warning("sentence-transformers not installed, using fallback")
            self.model = None
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            text: Single string or list of strings
            
        Returns:
            numpy array of shape (n, dimension)
        """
        if isinstance(text, str):
            text = [text]
        
        if self.openai_api_key:
            return self._embed_openai(text)
        
        if self.model is None:
            self.load_model()
        
        if self.model is not None:
            embeddings = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embeddings
        else:
            # Fallback: random embeddings (for testing only)
            logger.warning("Using random embeddings - install sentence-transformers")
            return np.random.randn(len(text), self.dimension).astype(np.float32)
    
    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (may use different prefix for some models)"""
        return self.embed(query)[0]
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed multiple documents"""
        return self.embed(documents)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


# Singleton
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
