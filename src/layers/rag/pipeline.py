"""
RAG Pipeline for IMI Medical Platform

Retrieval Augmented Generation pipeline that:
1. Ingests medical documents (PDFs, guidelines, literature)
2. Chunks and embeds documents
3. Retrieves relevant context via hybrid search (BM25 + dense vectors)
4. Re-ranks top candidates with a cross-encoder
5. Augments LLM prompts with retrieved knowledge

Hybrid retrieval strategy:
- Dense retrieval: semantic similarity via sentence-transformers embeddings
- BM25 retrieval: keyword overlap — critical for medical terminology, drug names, ICD codes
- Reciprocal Rank Fusion (RRF): merges both ranked lists without requiring score normalization
- Cross-encoder re-ranking: selects top-5 from merged top-20 using a fine-grained relevance model
"""
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json

from .embeddings import EmbeddingService, get_embedding_service
from .vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BM25 retriever (pure-Python, no extra server required)
# Falls back gracefully when rank_bm25 is not installed.
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed — BM25 hybrid search disabled. "
                   "Install with: pip install rank-bm25")

# ---------------------------------------------------------------------------
# Cross-encoder re-ranker
# Falls back gracefully when sentence-transformers is not installed.
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not installed — cross-encoder re-ranking disabled. "
                   "Install with: pip install sentence-transformers")


@dataclass
class RAGResult:
    """Result from RAG retrieval"""
    documents: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    sources: List[str] = field(default_factory=list)
    total_retrieved: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": self.documents,
            "context": self.context[:500] + "..." if len(self.context) > 500 else self.context,
            "sources": self.sources,
            "total_retrieved": self.total_retrieved,
        }


@dataclass
class Document:
    """Document for ingestion"""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            # Generate ID from content hash
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:16]


class RAGPipeline:
    """
    RAG Pipeline for medical knowledge retrieval

    Features:
    - Document ingestion with chunking
    - Hybrid search: BM25 (keyword) + dense embeddings (semantic)
    - Reciprocal Rank Fusion to merge retrieval lists
    - Cross-encoder re-ranking of top-20 → top-5
    - Source attribution and citation support
    - Medical-specific preprocessing
    """

    # Cross-encoder model — lightweight (22M params), fast on CPU
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        min_relevance_score: float = 0.5,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        rrf_k: int = 60,
        rerank_top_n: int = 20,
    ):
        self.embeddings = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        self.use_hybrid = use_hybrid and BM25_AVAILABLE
        self.use_reranker = use_reranker and CROSS_ENCODER_AVAILABLE
        self.rrf_k = rrf_k
        self.rerank_top_n = rerank_top_n

        # BM25 index — rebuilt on each ingest_* call
        self._bm25_corpus: List[str] = []
        self._bm25_ids: List[str] = []
        self._bm25_metadatas: List[Dict[str, Any]] = []
        self._bm25_index: Optional[Any] = None  # BM25Okapi instance

        # Cross-encoder — lazy-loaded on first rerank call
        self._cross_encoder: Optional[Any] = None

        # Document type handlers
        self.doc_handlers = {
            "pdf": self._process_pdf,
            "txt": self._process_text,
            "json": self._process_json,
            "md": self._process_markdown,
        }
    
    # =========================================================================
    # INGESTION
    # =========================================================================
    
    def ingest_document(
        self,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_type: str = "txt",
    ) -> int:
        """
        Ingest a single document
        
        Args:
            content: Document text content
            source: Source identifier (filename, URL, etc.)
            metadata: Optional metadata dict
            doc_type: Document type for preprocessing
            
        Returns:
            Number of chunks created
        """
        logger.info(f"Ingesting document: {source}")
        
        # Preprocess based on type
        if doc_type in self.doc_handlers:
            processed_content = self.doc_handlers[doc_type](content)
        else:
            processed_content = content
        
        # Chunk the document
        chunks = self._chunk_text(processed_content)
        
        if not chunks:
            logger.warning(f"No chunks created for {source}")
            return 0
        
        # Prepare metadata
        base_metadata = metadata or {}
        base_metadata["source"] = source
        base_metadata["doc_type"] = doc_type
        
        # Create documents with chunk metadata
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{hashlib.md5(source.encode()).hexdigest()[:8]}_{i}"
            
            documents.append(chunk)
            metadatas.append({
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
            ids.append(chunk_id)
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(documents)
        
        # Add to vector store
        self.vector_store.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        # Update BM25 index
        if self.use_hybrid:
            for chunk, meta, doc_id in zip(documents, metadatas, ids):
                self._bm25_corpus.append(chunk)
                self._bm25_ids.append(doc_id)
                self._bm25_metadatas.append(meta)
            self._rebuild_bm25_index()

        logger.info(f"Ingested {len(chunks)} chunks from {source}")
        return len(chunks)
    
    def ingest_directory(
        self,
        directory: str,
        extensions: List[str] = None,
        recursive: bool = True,
    ) -> Dict[str, int]:
        """
        Ingest all documents from a directory
        
        Args:
            directory: Path to directory
            extensions: List of file extensions to process
            recursive: Whether to process subdirectories
            
        Returns:
            Dict mapping filename to chunk count
        """
        if extensions is None:
            extensions = ["pdf", "txt", "md", "json"]
        
        directory = Path(directory)
        results = {}
        
        pattern = "**/*" if recursive else "*"
        
        for ext in extensions:
            for file_path in directory.glob(f"{pattern}.{ext}"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    chunks = self.ingest_document(
                        content=content,
                        source=str(file_path),
                        metadata={"filename": file_path.name},
                        doc_type=ext,
                    )
                    results[str(file_path)] = chunks
                    
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path}: {e}")
                    results[str(file_path)] = 0
        
        logger.info(f"Ingested {len(results)} files, {sum(results.values())} total chunks")
        return results
    
    def ingest_medical_guidelines(
        self,
        guidelines: List[Dict[str, Any]],
    ) -> int:
        """
        Ingest structured medical guidelines
        
        Args:
            guidelines: List of guideline dicts with 'title', 'content', 'source', 'category'
            
        Returns:
            Total chunks created
        """
        total_chunks = 0
        
        for guideline in guidelines:
            chunks = self.ingest_document(
                content=guideline["content"],
                source=guideline.get("source", guideline["title"]),
                metadata={
                    "title": guideline["title"],
                    "category": guideline.get("category", "general"),
                    "type": "medical_guideline",
                },
                doc_type="txt",
            )
            total_chunks += chunks
        
        return total_chunks
    
    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    
    def retrieve(
        self,
        query: str,
        patient_context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search + cross-encoder re-ranking.

        Pipeline:
          1. Dense retrieval (semantic embeddings) — top rerank_top_n candidates
          2. BM25 retrieval (keyword) — top rerank_top_n candidates [if available]
          3. RRF merge of both ranked lists
          4. Cross-encoder re-ranking of merged list → final top_k [if available]
          5. Fallback: dense-only if BM25/reranker unavailable

        Args:
            query: User query
            patient_context: Optional patient context for query expansion
            top_k: Number of results (default: self.top_k)
            filter_metadata: Optional metadata filter

        Returns:
            List of relevant documents with scores, ranked by relevance
        """
        top_k = top_k or self.top_k
        expanded_query = self._expand_query(query, patient_context)

        # ── Step 1: Dense retrieval ──────────────────────────────────────────
        query_embedding = self.embeddings.embed_query(expanded_query)
        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.rerank_top_n,
            filter_metadata=filter_metadata,
        )

        # ── Step 2 + 3: BM25 + RRF merge ────────────────────────────────────
        if self.use_hybrid and self._bm25_index is not None:
            bm25_results = self._bm25_search(expanded_query, top_k=self.rerank_top_n)
            merged = self._reciprocal_rank_fusion([dense_results, bm25_results])
        else:
            merged = dense_results

        # Filter by minimum relevance score before re-ranking
        merged = [r for r in merged if r.get("score", 1.0) >= self.min_relevance_score]

        if not merged:
            logger.info("Retrieved 0 documents after relevance filtering")
            return []

        # ── Step 4: Cross-encoder re-ranking ────────────────────────────────
        if self.use_reranker and len(merged) > top_k:
            final = self._rerank(query, merged, top_k)
        else:
            final = merged[:top_k]

        logger.info(f"Retrieved {len(final)} documents (hybrid={self.use_hybrid}, "
                    f"reranker={self.use_reranker})")
        return final

    # =========================================================================
    # HYBRID SEARCH HELPERS
    # =========================================================================

    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from current corpus."""
        if not BM25_AVAILABLE or not self._bm25_corpus:
            return
        tokenized = [doc.lower().split() for doc in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Run BM25 keyword search and return results in standard format."""
        if self._bm25_index is None:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "id": self._bm25_ids[idx],
                    "document": self._bm25_corpus[idx],
                    "metadata": self._bm25_metadatas[idx],
                    "score": float(scores[idx]),
                    "retrieval_method": "bm25",
                })
        return results

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[Dict[str, Any]]],
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion.

        RRF score = sum(1 / (rank + k)) across all lists.
        k=60 is the standard default from the original RRF paper.
        """
        k = k or self.rrf_k
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict[str, Any]] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                doc_id = doc.get("id", doc.get("document", "")[:32])
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + k)
                if doc_id not in doc_store:
                    doc_store[doc_id] = doc

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        merged = []
        for doc_id in sorted_ids:
            doc = dict(doc_store[doc_id])
            doc["score"] = rrf_scores[doc_id]
            doc["retrieval_method"] = "rrf_hybrid"
            merged.append(doc)
        return merged

    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Re-rank candidate documents using a cross-encoder model."""
        if not CROSS_ENCODER_AVAILABLE:
            return candidates[:top_k]

        if self._cross_encoder is None:
            logger.info(f"Loading cross-encoder: {self.CROSS_ENCODER_MODEL}")
            self._cross_encoder = CrossEncoder(self.CROSS_ENCODER_MODEL)

        pairs = [(query, doc["document"]) for doc in candidates]
        scores = self._cross_encoder.predict(pairs)

        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda d: d.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_k]
    
    def retrieve_with_context(
        self,
        query: str,
        patient_context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> RAGResult:
        """
        Retrieve and format as context string
        
        Args:
            query: User query
            patient_context: Optional patient context
            top_k: Number of results
            
        Returns:
            RAGResult with formatted context
        """
        documents = self.retrieve(query, patient_context, top_k)
        
        if not documents:
            return RAGResult()
        
        # Format as context string
        context_parts = []
        sources = []
        
        for i, doc in enumerate(documents, 1):
            source = doc["metadata"].get("source", "Unknown")
            title = doc["metadata"].get("title", source)
            
            context_parts.append(
                f"[{i}] {title}\n{doc['document']}"
            )
            
            if source not in sources:
                sources.append(source)
        
        context = "\n\n---\n\n".join(context_parts)
        
        return RAGResult(
            documents=documents,
            context=context,
            sources=sources,
            total_retrieved=len(documents),
        )
    
    def _expand_query(
        self,
        query: str,
        patient_context: Optional[Dict[str, Any]],
    ) -> str:
        """Expand query with patient context for better retrieval"""
        if not patient_context:
            return query
        
        expansions = []
        
        # Add conditions
        if patient_context.get("conditions"):
            expansions.append(f"Patient has: {', '.join(patient_context['conditions'])}")
        
        # Add medications
        if patient_context.get("medications"):
            expansions.append(f"Taking: {', '.join(patient_context['medications'])}")
        
        # Add age group
        if patient_context.get("age"):
            age = patient_context["age"]
            if age < 18:
                expansions.append("pediatric patient")
            elif age > 65:
                expansions.append("geriatric patient elderly")
        
        if expansions:
            return f"{query} {' '.join(expansions)}"
        
        return query
    
    # =========================================================================
    # DOCUMENT PROCESSING
    # =========================================================================
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Uses sentence-aware chunking to avoid breaking mid-sentence
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences (simple approach)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _process_pdf(self, content: str) -> str:
        """Process PDF content (assumes already extracted)"""
        # Clean up common PDF extraction artifacts
        import re
        
        # Remove page numbers
        content = re.sub(r'\n\s*\d+\s*\n', '\n', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        
        return content.strip()
    
    def _process_text(self, content: str) -> str:
        """Process plain text"""
        return content.strip()
    
    def _process_json(self, content: str) -> str:
        """Process JSON content"""
        try:
            data = json.loads(content)
            
            # Handle common formats
            if isinstance(data, dict):
                if "content" in data:
                    return data["content"]
                elif "text" in data:
                    return data["text"]
                else:
                    return json.dumps(data, indent=2)
            elif isinstance(data, list):
                return "\n\n".join(str(item) for item in data)
            else:
                return str(data)
                
        except json.JSONDecodeError:
            return content
    
    def _process_markdown(self, content: str) -> str:
        """Process Markdown content"""
        import re
        
        # Remove markdown formatting but keep structure
        # Remove headers markers but keep text
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        
        # Remove bold/italic markers
        content = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', content)
        content = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', content)
        
        # Remove links but keep text
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        
        # Remove code blocks
        content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        return content.strip()
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "total_documents": self.vector_store.count(),
            "embedding_dimension": self.embeddings.dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_store_backend": self.vector_store.backend,
        }
    
    def clear(self):
        """Clear all documents from the vector store"""
        # This would need to be implemented in VectorStore
        logger.warning("Clear operation not fully implemented")


# ============================================================================
# SINGLETON
# ============================================================================

_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline singleton"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
