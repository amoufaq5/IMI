"""
UMI RAG Service
Retrieval-Augmented Generation for medical knowledge retrieval
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedDocument:
    """Retrieved document from vector database."""
    id: str
    title: str
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResult:
    """Result from RAG retrieval."""
    documents: List[RetrievedDocument]
    query_embedding: Optional[List[float]]
    retrieval_time_ms: float


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
    
    async def _load_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                # Use medical-specific embedding model
                model_name = settings.embedding_model
                logger.info("loading_embedding_model", model=model_name)
                
                self._model = SentenceTransformer(model_name)
                
                logger.info("embedding_model_loaded", model=model_name)
            except ImportError:
                logger.warning("sentence_transformers_not_installed")
                self._model = MockEmbeddingModel()
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        await self._load_model()
        
        if isinstance(self._model, MockEmbeddingModel):
            return await self._model.embed(text)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text).tolist()
        )
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        await self._load_model()
        
        if isinstance(self._model, MockEmbeddingModel):
            return [await self._model.embed(t) for t in texts]
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts).tolist()
        )
        
        return embeddings


class MockEmbeddingModel:
    """Mock embedding model for development."""
    
    async def embed(self, text: str) -> List[float]:
        """Generate mock embedding."""
        import hashlib
        
        # Generate deterministic pseudo-random embedding based on text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = [
            (b - 128) / 128.0 for b in hash_bytes[:settings.embedding_dimension // 8]
        ]
        
        # Pad to full dimension
        while len(embedding) < settings.embedding_dimension:
            embedding.extend(embedding[:settings.embedding_dimension - len(embedding)])
        
        return embedding[:settings.embedding_dimension]


class VectorStore:
    """Interface to vector database (Qdrant)."""
    
    def __init__(self):
        self._client = None
    
    async def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http import models
                
                self._client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
                )
                
                logger.info("qdrant_connected", url=settings.qdrant_url)
            except Exception as e:
                logger.warning("qdrant_connection_failed", error=str(e))
                self._client = MockVectorStore()
        
        return self._client
    
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Returns:
            List of (id, score, payload) tuples
        """
        client = await self._get_client()
        
        if isinstance(client, MockVectorStore):
            return await client.search(collection, query_vector, top_k, filters)
        
        try:
            from qdrant_client.http import models
            
            # Build filter if provided
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value),
                            )
                        )
                    else:
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value),
                            )
                        )
                qdrant_filter = models.Filter(must=conditions)
            
            results = client.search(
                collection_name=f"{settings.qdrant_collection_prefix}{collection}",
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
            )
            
            return [
                (str(hit.id), hit.score, hit.payload or {})
                for hit in results
            ]
        
        except Exception as e:
            logger.error("qdrant_search_error", error=str(e), collection=collection)
            return []
    
    async def upsert(
        self,
        collection: str,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
    ) -> bool:
        """
        Upsert vectors into collection.
        
        Args:
            collection: Collection name
            vectors: List of (id, vector, payload) tuples
        
        Returns:
            Success status
        """
        client = await self._get_client()
        
        if isinstance(client, MockVectorStore):
            return await client.upsert(collection, vectors)
        
        try:
            from qdrant_client.http import models
            
            points = [
                models.PointStruct(
                    id=vid,
                    vector=vec,
                    payload=payload,
                )
                for vid, vec, payload in vectors
            ]
            
            client.upsert(
                collection_name=f"{settings.qdrant_collection_prefix}{collection}",
                points=points,
            )
            
            return True
        
        except Exception as e:
            logger.error("qdrant_upsert_error", error=str(e), collection=collection)
            return False
    
    async def create_collection(
        self,
        collection: str,
        vector_size: int = None,
    ) -> bool:
        """Create a new collection."""
        client = await self._get_client()
        
        if isinstance(client, MockVectorStore):
            return True
        
        try:
            from qdrant_client.http import models
            
            vector_size = vector_size or settings.embedding_dimension
            
            client.create_collection(
                collection_name=f"{settings.qdrant_collection_prefix}{collection}",
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            
            logger.info("collection_created", collection=collection)
            return True
        
        except Exception as e:
            if "already exists" in str(e).lower():
                return True
            logger.error("collection_create_error", error=str(e), collection=collection)
            return False


class MockVectorStore:
    """Mock vector store for development."""
    
    def __init__(self):
        self._collections: Dict[str, List[Tuple[str, List[float], Dict[str, Any]]]] = {}
    
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Mock search returning sample results."""
        
        # Return mock medical documents
        mock_results = [
            (
                "doc_1",
                0.92,
                {
                    "title": "Common Cold - Clinical Guidelines",
                    "content": "The common cold is a viral infection of the upper respiratory tract...",
                    "source": "NICE Guidelines",
                }
            ),
            (
                "doc_2",
                0.87,
                {
                    "title": "Paracetamol - Drug Information",
                    "content": "Paracetamol (acetaminophen) is an analgesic and antipyretic...",
                    "source": "BNF",
                }
            ),
            (
                "doc_3",
                0.82,
                {
                    "title": "Headache - Differential Diagnosis",
                    "content": "Headaches can be classified as primary or secondary...",
                    "source": "BMJ Best Practice",
                }
            ),
        ]
        
        return mock_results[:top_k]
    
    async def upsert(
        self,
        collection: str,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
    ) -> bool:
        """Mock upsert."""
        if collection not in self._collections:
            self._collections[collection] = []
        self._collections[collection].extend(vectors)
        return True


class RAGService:
    """
    Retrieval-Augmented Generation service for medical knowledge.
    """
    
    # Knowledge base collections
    COLLECTIONS = {
        "medical_literature": "Research papers and medical literature",
        "drug_information": "Drug database and interactions",
        "clinical_guidelines": "Clinical practice guidelines",
        "regulatory_documents": "Regulatory compliance documents",
        "qa_qc_templates": "QA/QC document templates",
    }
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
    
    async def retrieve(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> RAGResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            collections: Collections to search (default: all)
            top_k: Number of results per collection
            filters: Metadata filters
            rerank: Whether to rerank results
        
        Returns:
            RAGResult with retrieved documents
        """
        start_time = datetime.now()
        
        # Generate query embedding
        query_embedding = await self.embedding_service.embed(query)
        
        # Search collections
        collections = collections or list(self.COLLECTIONS.keys())
        all_results = []
        
        for collection in collections:
            results = await self.vector_store.search(
                collection=collection,
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters,
            )
            
            for doc_id, score, payload in results:
                all_results.append(
                    RetrievedDocument(
                        id=doc_id,
                        title=payload.get("title", "Unknown"),
                        content=payload.get("content", ""),
                        source=payload.get("source", collection),
                        score=score,
                        metadata=payload,
                    )
                )
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Rerank if enabled (would use cross-encoder in production)
        if rerank and len(all_results) > top_k:
            all_results = await self._rerank(query, all_results, top_k)
        else:
            all_results = all_results[:top_k]
        
        retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            "rag_retrieval",
            query_length=len(query),
            collections=collections,
            results_count=len(all_results),
            retrieval_time_ms=retrieval_time,
        )
        
        return RAGResult(
            documents=all_results,
            query_embedding=query_embedding,
            retrieval_time_ms=retrieval_time,
        )
    
    async def _rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int,
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using cross-encoder.
        In production, this would use a trained reranker model.
        """
        # Simple reranking based on keyword overlap for now
        query_words = set(query.lower().split())
        
        for doc in documents:
            content_words = set(doc.content.lower().split())
            overlap = len(query_words & content_words)
            # Boost score based on keyword overlap
            doc.score = doc.score * (1 + overlap * 0.05)
        
        documents.sort(key=lambda x: x.score, reverse=True)
        return documents[:top_k]
    
    async def index_document(
        self,
        collection: str,
        doc_id: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Index a document into the vector store.
        
        Args:
            collection: Target collection
            doc_id: Document ID
            title: Document title
            content: Document content
            metadata: Additional metadata
        
        Returns:
            Success status
        """
        # Generate embedding
        text_to_embed = f"{title}\n\n{content}"
        embedding = await self.embedding_service.embed(text_to_embed)
        
        # Prepare payload
        payload = {
            "title": title,
            "content": content[:5000],  # Limit content size
            **(metadata or {}),
        }
        
        # Upsert to vector store
        success = await self.vector_store.upsert(
            collection=collection,
            vectors=[(doc_id, embedding, payload)],
        )
        
        if success:
            logger.info(
                "document_indexed",
                collection=collection,
                doc_id=doc_id,
                title=title[:50],
            )
        
        return success
    
    async def index_batch(
        self,
        collection: str,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Index multiple documents.
        
        Args:
            collection: Target collection
            documents: List of documents with id, title, content, metadata
        
        Returns:
            Number of successfully indexed documents
        """
        # Generate embeddings in batch
        texts = [f"{d['title']}\n\n{d['content']}" for d in documents]
        embeddings = await self.embedding_service.embed_batch(texts)
        
        # Prepare vectors
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            payload = {
                "title": doc["title"],
                "content": doc["content"][:5000],
                **(doc.get("metadata", {})),
            }
            vectors.append((doc["id"], embedding, payload))
        
        # Upsert in batches
        batch_size = 100
        indexed = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            success = await self.vector_store.upsert(collection, batch)
            if success:
                indexed += len(batch)
        
        logger.info(
            "batch_indexed",
            collection=collection,
            total=len(documents),
            indexed=indexed,
        )
        
        return indexed
    
    async def initialize_collections(self) -> None:
        """Initialize all knowledge base collections."""
        for collection in self.COLLECTIONS:
            await self.vector_store.create_collection(collection)
        
        logger.info("collections_initialized", collections=list(self.COLLECTIONS.keys()))
