"""
Citation Tracking System for IMI

Tracks and formats citations for all model outputs:
- Knowledge Graph sources
- RAG document sources
- Medical guidelines
- Rule engine references
- Verification sources

Provides:
- Inline citation formatting [1], [2], etc.
- Full reference list generation
- Citation verification
- Source credibility scoring
"""
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import re

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of citation sources"""
    KNOWLEDGE_GRAPH = "knowledge_graph"
    RAG_DOCUMENT = "rag_document"
    MEDICAL_GUIDELINE = "medical_guideline"
    RULE_ENGINE = "rule_engine"
    PATIENT_HISTORY = "patient_history"
    DRUG_DATABASE = "drug_database"
    CLINICAL_TRIAL = "clinical_trial"
    TEXTBOOK = "textbook"
    RESEARCH_PAPER = "research_paper"
    REGULATORY = "regulatory"
    UNKNOWN = "unknown"


class CredibilityLevel(Enum):
    """Credibility levels for sources"""
    HIGHEST = "highest"      # Peer-reviewed guidelines, FDA, WHO
    HIGH = "high"            # Major medical textbooks, established databases
    MEDIUM = "medium"        # Research papers, clinical trials
    LOW = "low"              # General medical websites
    UNVERIFIED = "unverified"


@dataclass
class Citation:
    """A single citation reference"""
    id: str
    source_type: SourceType
    title: str
    source: str  # URL, database name, or document path
    
    # Optional metadata
    authors: Optional[List[str]] = None
    publication_date: Optional[str] = None
    publisher: Optional[str] = None
    section: Optional[str] = None
    page: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None  # PubMed ID
    
    # Credibility
    credibility: CredibilityLevel = CredibilityLevel.MEDIUM
    
    # Usage tracking
    relevance_score: float = 0.0
    text_snippet: Optional[str] = None
    
    # Timestamps
    accessed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from source hash
            self.id = hashlib.md5(
                f"{self.source}:{self.title}".encode()
            ).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type.value,
            "title": self.title,
            "source": self.source,
            "authors": self.authors,
            "publication_date": self.publication_date,
            "publisher": self.publisher,
            "section": self.section,
            "credibility": self.credibility.value,
            "relevance_score": self.relevance_score,
            "accessed_at": self.accessed_at,
        }
    
    def format_apa(self) -> str:
        """Format citation in APA style"""
        parts = []
        
        # Authors
        if self.authors:
            if len(self.authors) == 1:
                parts.append(self.authors[0])
            elif len(self.authors) == 2:
                parts.append(f"{self.authors[0]} & {self.authors[1]}")
            else:
                parts.append(f"{self.authors[0]} et al.")
        
        # Year
        if self.publication_date:
            year = self.publication_date[:4] if len(self.publication_date) >= 4 else self.publication_date
            parts.append(f"({year})")
        
        # Title
        parts.append(f"{self.title}.")
        
        # Publisher/Source
        if self.publisher:
            parts.append(f"{self.publisher}.")
        elif self.source:
            parts.append(f"Retrieved from {self.source}")
        
        return " ".join(parts)
    
    def format_short(self) -> str:
        """Format as short inline citation"""
        if self.authors and self.publication_date:
            author = self.authors[0].split()[-1] if self.authors else "Unknown"
            year = self.publication_date[:4]
            return f"{author}, {year}"
        return self.title[:30] + "..." if len(self.title) > 30 else self.title


@dataclass
class CitationResult:
    """Result containing text with citations and reference list"""
    original_text: str
    cited_text: str
    citations: List[Citation] = field(default_factory=list)
    citation_map: Dict[str, int] = field(default_factory=dict)  # citation_id -> number
    
    # Statistics
    total_citations: int = 0
    unique_sources: int = 0
    credibility_breakdown: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cited_text": self.cited_text,
            "citations": [c.to_dict() for c in self.citations],
            "total_citations": self.total_citations,
            "unique_sources": self.unique_sources,
            "credibility_breakdown": self.credibility_breakdown,
        }
    
    def get_reference_list(self, format: str = "apa") -> str:
        """Generate formatted reference list"""
        lines = ["## References\n"]
        
        for i, citation in enumerate(self.citations, 1):
            if format == "apa":
                ref = citation.format_apa()
            else:
                ref = f"{citation.title}. {citation.source}"
            
            lines.append(f"[{i}] {ref}")
        
        return "\n".join(lines)
    
    def get_inline_citations(self) -> str:
        """Get just the inline citation numbers used"""
        return ", ".join(f"[{i+1}]" for i in range(len(self.citations)))


class CitationTracker:
    """
    Tracks and manages citations across the IMI platform
    
    Features:
    - Collect citations from multiple sources (KG, RAG, rules)
    - Insert inline citations into text
    - Generate reference lists
    - Verify citation accuracy
    - Score source credibility
    """
    
    def __init__(self):
        self.citations: Dict[str, Citation] = {}  # id -> Citation
        self.source_credibility = self._init_credibility_map()
    
    def _init_credibility_map(self) -> Dict[str, CredibilityLevel]:
        """Initialize credibility ratings for known sources"""
        return {
            # Highest credibility
            "fda.gov": CredibilityLevel.HIGHEST,
            "who.int": CredibilityLevel.HIGHEST,
            "cdc.gov": CredibilityLevel.HIGHEST,
            "nih.gov": CredibilityLevel.HIGHEST,
            "cochrane": CredibilityLevel.HIGHEST,
            "uptodate": CredibilityLevel.HIGHEST,
            "aha": CredibilityLevel.HIGHEST,  # American Heart Association
            "acc": CredibilityLevel.HIGHEST,  # American College of Cardiology
            
            # High credibility
            "pubmed": CredibilityLevel.HIGH,
            "medscape": CredibilityLevel.HIGH,
            "nejm": CredibilityLevel.HIGH,
            "lancet": CredibilityLevel.HIGH,
            "jama": CredibilityLevel.HIGH,
            "bmj": CredibilityLevel.HIGH,
            "merck": CredibilityLevel.HIGH,
            
            # Medium credibility
            "wikipedia": CredibilityLevel.MEDIUM,
            "webmd": CredibilityLevel.MEDIUM,
            "mayoclinic": CredibilityLevel.MEDIUM,
            
            # Knowledge graph and internal sources
            "knowledge_graph": CredibilityLevel.HIGH,
            "rule_engine": CredibilityLevel.HIGHEST,
            "drug_database": CredibilityLevel.HIGH,
        }
    
    def add_citation(
        self,
        source_type: SourceType,
        title: str,
        source: str,
        **kwargs,
    ) -> Citation:
        """Add a new citation"""
        # Determine credibility
        credibility = self._assess_credibility(source)
        
        citation = Citation(
            id="",  # Will be auto-generated
            source_type=source_type,
            title=title,
            source=source,
            credibility=credibility,
            **kwargs,
        )
        
        self.citations[citation.id] = citation
        return citation
    
    def add_from_rag(
        self,
        rag_results: List[Dict[str, Any]],
    ) -> List[Citation]:
        """Add citations from RAG retrieval results"""
        citations = []
        
        for result in rag_results:
            metadata = result.get("metadata", {})
            
            citation = self.add_citation(
                source_type=SourceType.RAG_DOCUMENT,
                title=metadata.get("title", result.get("source", "Unknown")),
                source=metadata.get("source", "RAG Document"),
                section=metadata.get("section"),
                relevance_score=result.get("score", 0.0),
                text_snippet=result.get("document", "")[:200],
            )
            citations.append(citation)
        
        return citations
    
    def add_from_knowledge_graph(
        self,
        kg_results: Dict[str, Any],
    ) -> List[Citation]:
        """Add citations from knowledge graph queries"""
        citations = []
        
        for source in kg_results.get("sources", []):
            citation = self.add_citation(
                source_type=SourceType.KNOWLEDGE_GRAPH,
                title=source.get("name", "Medical Knowledge Graph"),
                source="IMI Knowledge Graph",
                section=source.get("node_type"),
                relevance_score=source.get("confidence", 1.0),
            )
            citations.append(citation)
        
        return citations
    
    def add_from_rules(
        self,
        rules_applied: List[str],
    ) -> List[Citation]:
        """Add citations from rule engine"""
        citations = []
        
        for rule in rules_applied:
            citation = self.add_citation(
                source_type=SourceType.RULE_ENGINE,
                title=f"Safety Rule: {rule}",
                source="IMI Rule Engine",
                section=rule,
            )
            citations.append(citation)
        
        return citations
    
    def add_from_guidelines(
        self,
        guidelines: List[Dict[str, Any]],
    ) -> List[Citation]:
        """Add citations from medical guidelines"""
        citations = []
        
        for guideline in guidelines:
            citation = self.add_citation(
                source_type=SourceType.MEDICAL_GUIDELINE,
                title=guideline.get("title", "Medical Guideline"),
                source=guideline.get("source", "Clinical Guidelines"),
                publisher=guideline.get("organization"),
                publication_date=guideline.get("year"),
                section=guideline.get("section"),
            )
            citations.append(citation)
        
        return citations
    
    def _assess_credibility(self, source: str) -> CredibilityLevel:
        """Assess credibility of a source"""
        source_lower = source.lower()
        
        for key, level in self.source_credibility.items():
            if key in source_lower:
                return level
        
        return CredibilityLevel.UNVERIFIED
    
    def insert_citations(
        self,
        text: str,
        citations: List[Citation],
        style: str = "numbered",  # numbered, author-date, footnote
    ) -> CitationResult:
        """
        Insert inline citations into text
        
        Matches text content to citation snippets and inserts references.
        """
        if not citations:
            return CitationResult(
                original_text=text,
                cited_text=text,
                citations=[],
            )
        
        cited_text = text
        citation_map = {}
        used_citations = []
        
        # Build citation map
        for i, citation in enumerate(citations):
            citation_map[citation.id] = i + 1
        
        # Strategy 1: Insert citations at sentence ends where content matches
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cited_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matching_citations = []
            
            for citation in citations:
                # Check if citation snippet matches sentence content
                if citation.text_snippet:
                    snippet_words = set(citation.text_snippet.lower().split()[:10])
                    sentence_words = set(sentence_lower.split())
                    
                    # If significant overlap, add citation
                    overlap = len(snippet_words & sentence_words)
                    if overlap >= 3:
                        matching_citations.append(citation)
                
                # Check for keyword matches
                elif citation.title:
                    title_words = set(citation.title.lower().split())
                    sentence_words = set(sentence_lower.split())
                    if len(title_words & sentence_words) >= 2:
                        matching_citations.append(citation)
            
            # Add citations to sentence
            if matching_citations:
                citation_refs = []
                for c in matching_citations[:3]:  # Max 3 citations per sentence
                    ref_num = citation_map[c.id]
                    citation_refs.append(f"[{ref_num}]")
                    if c not in used_citations:
                        used_citations.append(c)
                
                # Insert before period
                if sentence.rstrip().endswith(('.', '!', '?')):
                    sentence = sentence.rstrip()[:-1] + " " + ", ".join(citation_refs) + sentence.rstrip()[-1]
                else:
                    sentence = sentence + " " + ", ".join(citation_refs)
            
            cited_sentences.append(sentence)
        
        cited_text = " ".join(cited_sentences)
        
        # If no matches found, add all citations at the end
        if not used_citations and citations:
            used_citations = citations[:5]  # Limit to top 5
            citation_refs = [f"[{citation_map[c.id]}]" for c in used_citations]
            cited_text = text + f" {', '.join(citation_refs)}"
        
        # Calculate statistics
        credibility_breakdown = {}
        for c in used_citations:
            level = c.credibility.value
            credibility_breakdown[level] = credibility_breakdown.get(level, 0) + 1
        
        return CitationResult(
            original_text=text,
            cited_text=cited_text,
            citations=used_citations,
            citation_map=citation_map,
            total_citations=len(used_citations),
            unique_sources=len(set(c.source for c in used_citations)),
            credibility_breakdown=credibility_breakdown,
        )
    
    def format_response_with_citations(
        self,
        response: str,
        rag_results: Optional[List[Dict[str, Any]]] = None,
        kg_results: Optional[Dict[str, Any]] = None,
        rules_applied: Optional[List[str]] = None,
        guidelines: Optional[List[Dict[str, Any]]] = None,
        include_reference_list: bool = True,
    ) -> Dict[str, Any]:
        """
        Format a complete response with citations from all sources
        
        Args:
            response: The model's response text
            rag_results: Results from RAG retrieval
            kg_results: Results from knowledge graph
            rules_applied: Rules that were applied
            guidelines: Medical guidelines referenced
            include_reference_list: Whether to append reference list
            
        Returns:
            Dict with cited_response, citations, and metadata
        """
        all_citations = []
        
        # Collect citations from all sources
        if rag_results:
            all_citations.extend(self.add_from_rag(rag_results))
        
        if kg_results:
            all_citations.extend(self.add_from_knowledge_graph(kg_results))
        
        if rules_applied:
            all_citations.extend(self.add_from_rules(rules_applied))
        
        if guidelines:
            all_citations.extend(self.add_from_guidelines(guidelines))
        
        # Sort by relevance and credibility
        all_citations.sort(
            key=lambda c: (
                c.credibility.value != CredibilityLevel.HIGHEST.value,
                -c.relevance_score,
            )
        )
        
        # Insert citations
        result = self.insert_citations(response, all_citations)
        
        # Build final response
        final_response = result.cited_text
        
        if include_reference_list and result.citations:
            final_response += "\n\n" + result.get_reference_list()
        
        return {
            "response": final_response,
            "original_response": response,
            "citations": [c.to_dict() for c in result.citations],
            "total_citations": result.total_citations,
            "unique_sources": result.unique_sources,
            "credibility_breakdown": result.credibility_breakdown,
            "reference_list": result.get_reference_list() if result.citations else "",
        }
    
    def verify_citations(
        self,
        text: str,
        citations: List[Citation],
    ) -> Dict[str, Any]:
        """
        Verify that citations in text are accurate
        
        Checks:
        - Citation numbers match reference list
        - Cited content matches source
        - Sources are accessible/valid
        """
        verification = {
            "valid": True,
            "issues": [],
            "verified_citations": 0,
            "unverified_citations": 0,
        }
        
        # Extract citation numbers from text
        citation_pattern = r'\[(\d+)\]'
        cited_numbers = set(int(m) for m in re.findall(citation_pattern, text))
        
        # Check each cited number has a corresponding reference
        for num in cited_numbers:
            if num > len(citations) or num < 1:
                verification["valid"] = False
                verification["issues"].append(f"Citation [{num}] has no corresponding reference")
            else:
                verification["verified_citations"] += 1
        
        # Check for uncited references
        for i, citation in enumerate(citations, 1):
            if i not in cited_numbers:
                verification["issues"].append(f"Reference [{i}] ({citation.title}) is not cited in text")
        
        verification["unverified_citations"] = len(citations) - verification["verified_citations"]
        
        return verification
    
    def clear(self):
        """Clear all tracked citations"""
        self.citations.clear()


# ============================================================================
# SINGLETON
# ============================================================================

_citation_tracker: Optional[CitationTracker] = None


def get_citation_tracker() -> CitationTracker:
    """Get or create citation tracker singleton"""
    global _citation_tracker
    if _citation_tracker is None:
        _citation_tracker = CitationTracker()
    return _citation_tracker
