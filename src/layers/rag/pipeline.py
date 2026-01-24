"""
RAG Pipeline for IMI Medical Platform

Retrieval Augmented Generation pipeline that:
1. Ingests medical documents (PDFs, guidelines, literature)
2. Chunks and embeds documents
3. Retrieves relevant context for queries
4. Augments LLM prompts with retrieved knowledge
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json

from .embeddings import EmbeddingService, get_embedding_service
from .vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


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
    - Semantic search with embeddings
    - Source attribution
    - Medical-specific preprocessing
    - Hybrid search (semantic + keyword)
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        min_relevance_score: float = 0.5,
    ):
        self.embeddings = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        
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
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            patient_context: Optional patient context for query expansion
            top_k: Number of results (default: self.top_k)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant documents with scores
        """
        top_k = top_k or self.top_k
        
        # Expand query with patient context
        expanded_query = self._expand_query(query, patient_context)
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(expanded_query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more, then filter
            filter_metadata=filter_metadata,
        )
        
        # Filter by relevance score
        filtered_results = [
            r for r in results 
            if r["score"] >= self.min_relevance_score
        ][:top_k]
        
        logger.info(f"Retrieved {len(filtered_results)} documents for query")
        
        return filtered_results
    
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
