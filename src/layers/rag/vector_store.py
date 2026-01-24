"""
Vector Store for RAG Pipeline

Stores and retrieves document embeddings using:
- ChromaDB (default, local)
- Pinecone (optional, cloud)
- FAISS (optional, local)
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for document embeddings
    
    Supports multiple backends:
    - ChromaDB (default): Local, persistent, easy setup
    - FAISS: Fast, local, good for large datasets
    - Pinecone: Cloud-hosted, scalable
    """
    
    def __init__(
        self,
        collection_name: str = "imi_medical_docs",
        persist_directory: Optional[str] = None,
        backend: str = "chroma",  # chroma, faiss, pinecone
        embedding_dimension: int = 384,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(
            Path(__file__).parent.parent.parent.parent / "data" / "vector_store"
        )
        self.backend = backend
        self.embedding_dimension = embedding_dimension
        
        self.client = None
        self.collection = None
        self.index = None  # For FAISS
        self.metadata_store: Dict[str, Dict] = {}  # For FAISS metadata
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the vector store backend"""
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        if self.backend == "chroma":
            self._init_chroma()
        elif self.backend == "faiss":
            self._init_faiss()
        else:
            logger.warning(f"Unknown backend {self.backend}, using in-memory store")
            self._init_memory()
    
    def _init_chroma(self):
        """Initialize ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            
            logger.info(f"ChromaDB initialized with {self.collection.count()} documents")
            
        except ImportError:
            logger.warning("chromadb not installed, falling back to memory store")
            self._init_memory()
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            import faiss
            
            # Create index
            self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product (cosine after normalization)
            
            # Load existing index if available
            index_path = Path(self.persist_directory) / "faiss.index"
            metadata_path = Path(self.persist_directory) / "faiss_metadata.json"
            
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.metadata_store = json.load(f)
            
        except ImportError:
            logger.warning("faiss not installed, falling back to memory store")
            self._init_memory()
    
    def _init_memory(self):
        """Initialize simple in-memory store"""
        self.backend = "memory"
        self.embeddings: List[np.ndarray] = []
        self.documents: List[Dict[str, Any]] = []
        
        # Load from disk if available
        store_path = Path(self.persist_directory) / "memory_store.json"
        if store_path.exists():
            with open(store_path) as f:
                data = json.load(f)
                self.documents = data.get("documents", [])
                self.embeddings = [
                    np.array(e, dtype=np.float32) 
                    for e in data.get("embeddings", [])
                ]
            logger.info(f"Loaded {len(self.documents)} documents from memory store")
    
    def add(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Add documents to the vector store
        
        Args:
            documents: List of document texts
            embeddings: numpy array of shape (n, dimension)
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
        """
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        if self.backend == "chroma" and self.collection:
            self.collection.add(
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids,
            )
            
        elif self.backend == "faiss" and self.index is not None:
            import faiss
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            
            # Store metadata
            for i, (doc_id, doc, meta) in enumerate(zip(ids, documents, metadatas)):
                self.metadata_store[str(self.index.ntotal - len(documents) + i)] = {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                }
            
            # Persist
            self._save_faiss()
            
        else:  # memory
            for doc, emb, meta, doc_id in zip(documents, embeddings, metadatas, ids):
                self.embeddings.append(emb)
                self.documents.append({
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                })
            self._save_memory()
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of dicts with 'document', 'metadata', 'score'
        """
        results = []
        
        if self.backend == "chroma" and self.collection:
            where = filter_metadata if filter_metadata else None
            
            query_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where,
            )
            
            if query_results["documents"]:
                for i, doc in enumerate(query_results["documents"][0]):
                    results.append({
                        "document": doc,
                        "metadata": query_results["metadatas"][0][i] if query_results["metadatas"] else {},
                        "score": 1 - query_results["distances"][0][i] if query_results["distances"] else 0,
                        "id": query_results["ids"][0][i] if query_results["ids"] else None,
                    })
                    
        elif self.backend == "faiss" and self.index is not None:
            import faiss
            
            # Normalize query
            query_norm = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_norm)
            
            # Search
            scores, indices = self.index.search(query_norm, top_k)
            
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and str(idx) in self.metadata_store:
                    item = self.metadata_store[str(idx)]
                    results.append({
                        "document": item["document"],
                        "metadata": item["metadata"],
                        "score": float(score),
                        "id": item["id"],
                    })
                    
        else:  # memory
            if not self.embeddings:
                return []
            
            # Compute similarities
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            similarities = []
            for i, emb in enumerate(self.embeddings):
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                sim = float(np.dot(query_norm, emb_norm))
                similarities.append((sim, i))
            
            # Sort and get top-k
            similarities.sort(reverse=True)
            
            for score, idx in similarities[:top_k]:
                doc = self.documents[idx]
                
                # Apply filter
                if filter_metadata:
                    match = all(
                        doc["metadata"].get(k) == v 
                        for k, v in filter_metadata.items()
                    )
                    if not match:
                        continue
                
                results.append({
                    "document": doc["document"],
                    "metadata": doc["metadata"],
                    "score": score,
                    "id": doc["id"],
                })
        
        return results
    
    def delete(self, ids: List[str]):
        """Delete documents by ID"""
        if self.backend == "chroma" and self.collection:
            self.collection.delete(ids=ids)
        elif self.backend == "memory":
            self.documents = [d for d in self.documents if d["id"] not in ids]
            # Note: embeddings not cleaned up in memory mode
            self._save_memory()
        
        logger.info(f"Deleted {len(ids)} documents")
    
    def count(self) -> int:
        """Get total document count"""
        if self.backend == "chroma" and self.collection:
            return self.collection.count()
        elif self.backend == "faiss" and self.index:
            return self.index.ntotal
        else:
            return len(self.documents)
    
    def _save_faiss(self):
        """Persist FAISS index"""
        import faiss
        
        index_path = Path(self.persist_directory) / "faiss.index"
        metadata_path = Path(self.persist_directory) / "faiss_metadata.json"
        
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata_store, f)
    
    def _save_memory(self):
        """Persist memory store"""
        store_path = Path(self.persist_directory) / "memory_store.json"
        
        with open(store_path, 'w') as f:
            json.dump({
                "documents": self.documents,
                "embeddings": [e.tolist() for e in self.embeddings],
            }, f)


# Singleton
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create vector store singleton"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
