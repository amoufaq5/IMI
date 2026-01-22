"""
PDF Ingestion Script for WHO & FDA Regulations

Extracts and processes text from regulatory PDF documents:
- WHO guidelines
- FDA regulations
- Converts to training format for regulatory QA adapter
"""
import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PDF libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. Run: pip install pymupdf")

try:
    from pdfminer.high_level import extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False


DATA_DIR = Path(__file__).parent.parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed" / "regulatory_qa"


@dataclass
class PDFDocument:
    """Represents a processed PDF document"""
    filename: str
    title: str
    source: str  # WHO, FDA, etc.
    sections: List[Dict[str, str]]
    total_pages: int


class PDFIngester:
    """Ingests and processes regulatory PDFs"""
    
    def __init__(self, pdf_dir: Path = PDF_DIR):
        self.pdf_dir = pdf_dir
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    def extract_text_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF"""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not available")
        
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def extract_text_pdfminer(self, pdf_path: Path) -> str:
        """Extract text using pdfminer"""
        if not PDFMINER_AVAILABLE:
            raise ImportError("pdfminer not available")
        return extract_text(pdf_path)
    
    def extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using available library"""
        if PYMUPDF_AVAILABLE:
            return self.extract_text_pymupdf(pdf_path)
        elif PDFMINER_AVAILABLE:
            return self.extract_text_pdfminer(pdf_path)
        else:
            raise ImportError("No PDF library available. Install pymupdf or pdfminer.six")
    
    def detect_source(self, filename: str, text: str) -> str:
        """Detect the source organization from filename or content"""
        filename_lower = filename.lower()
        text_lower = text[:5000].lower()
        
        if "who" in filename_lower or "world health organization" in text_lower:
            return "WHO"
        elif "fda" in filename_lower or "food and drug administration" in text_lower:
            return "FDA"
        elif "ema" in filename_lower or "european medicines agency" in text_lower:
            return "EMA"
        elif "ich" in filename_lower or "international council for harmonisation" in text_lower:
            return "ICH"
        elif "gmp" in filename_lower or "good manufacturing practice" in text_lower:
            return "GMP"
        else:
            return "REGULATORY"
    
    def split_into_sections(self, text: str) -> List[Dict[str, str]]:
        """Split document into logical sections"""
        sections = []
        
        # Common section patterns in regulatory documents
        section_patterns = [
            r'^(?:CHAPTER|SECTION|PART)\s+\d+[:\.]?\s*(.+)$',
            r'^(?:\d+\.)+\s*(.+)$',  # Numbered sections like 1.1, 2.3.1
            r'^[A-Z][A-Z\s]{5,50}$',  # ALL CAPS headers
        ]
        
        # Split by double newlines first
        paragraphs = text.split('\n\n')
        
        current_section = {"title": "Introduction", "content": ""}
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this is a section header
            is_header = False
            for pattern in section_patterns:
                if re.match(pattern, para, re.MULTILINE):
                    # Save previous section
                    if current_section["content"]:
                        sections.append(current_section.copy())
                    
                    current_section = {"title": para[:100], "content": ""}
                    is_header = True
                    break
            
            if not is_header:
                current_section["content"] += para + "\n\n"
        
        # Add last section
        if current_section["content"]:
            sections.append(current_section)
        
        return sections
    
    def create_qa_pairs(self, doc: PDFDocument) -> List[Dict[str, Any]]:
        """Create Q&A pairs from document sections"""
        qa_pairs = []
        
        for section in doc.sections:
            title = section["title"]
            content = section["content"].strip()
            
            if len(content) < 100:  # Skip very short sections
                continue
            
            # Create different types of Q&A pairs
            
            # 1. Section summary question
            qa_pairs.append({
                "instruction": f"Summarize the {doc.source} guidelines regarding: {title}",
                "input": "",
                "output": content[:2000],  # Truncate long content
                "source": f"{doc.source}_{doc.filename}",
                "adapter": "regulatory_qa",
            })
            
            # 2. Compliance question
            qa_pairs.append({
                "instruction": f"What are the {doc.source} requirements for {title}?",
                "input": "",
                "output": f"According to {doc.source} guidelines:\n\n{content[:1500]}",
                "source": f"{doc.source}_{doc.filename}",
                "adapter": "regulatory_qa",
            })
            
            # 3. Context-based question
            if len(content) > 500:
                qa_pairs.append({
                    "instruction": "Based on the following regulatory text, what are the key requirements?",
                    "input": content[:1000],
                    "output": f"Key requirements from this {doc.source} section on {title}:\n\n" + 
                             self._extract_key_points(content),
                    "source": f"{doc.source}_{doc.filename}",
                    "adapter": "regulatory_qa",
                })
        
        return qa_pairs
    
    def _extract_key_points(self, text: str) -> str:
        """Extract key points from text"""
        # Look for bullet points, numbered items, or "shall/must" statements
        key_points = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Check for requirement indicators
            if any(word in line.lower() for word in ['shall', 'must', 'required', 'ensure']):
                key_points.append(f"• {line}")
            # Check for bullet points
            elif line.startswith(('•', '-', '*', '○')):
                key_points.append(line)
            # Check for numbered items
            elif re.match(r'^\d+[\.\)]\s', line):
                key_points.append(line)
        
        if key_points:
            return '\n'.join(key_points[:10])  # Limit to 10 points
        else:
            # Return first few sentences
            sentences = text.split('.')[:5]
            return '. '.join(sentences) + '.'
    
    def process_pdf(self, pdf_path: Path) -> Optional[PDFDocument]:
        """Process a single PDF file"""
        logger.info(f"Processing: {pdf_path.name}")
        
        try:
            # Extract text
            text = self.extract_text(pdf_path)
            
            if not text or len(text) < 100:
                logger.warning(f"  No text extracted from {pdf_path.name}")
                return None
            
            # Detect source
            source = self.detect_source(pdf_path.name, text)
            
            # Split into sections
            sections = self.split_into_sections(text)
            
            # Get page count
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                doc.close()
            else:
                total_pages = 0
            
            logger.info(f"  Source: {source}, Pages: {total_pages}, Sections: {len(sections)}")
            
            return PDFDocument(
                filename=pdf_path.name,
                title=pdf_path.stem,
                source=source,
                sections=sections,
                total_pages=total_pages,
            )
            
        except Exception as e:
            logger.error(f"  Error processing {pdf_path.name}: {e}")
            return None
    
    def ingest_all(self):
        """Process all PDFs in the directory"""
        logger.info("=" * 60)
        logger.info("PDF Ingestion for Regulatory Documents")
        logger.info("=" * 60)
        logger.info(f"PDF Directory: {self.pdf_dir}")
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            logger.info("Please add your WHO/FDA PDF files to this directory.")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        all_qa_pairs = []
        stats = {"processed": 0, "failed": 0, "qa_pairs": 0}
        
        for pdf_path in pdf_files:
            doc = self.process_pdf(pdf_path)
            
            if doc:
                qa_pairs = self.create_qa_pairs(doc)
                all_qa_pairs.extend(qa_pairs)
                stats["processed"] += 1
                stats["qa_pairs"] += len(qa_pairs)
                logger.info(f"  Generated {len(qa_pairs)} Q&A pairs")
            else:
                stats["failed"] += 1
        
        # Save processed data
        if all_qa_pairs:
            output_path = PROCESSED_DIR / "regulatory_pdfs.json"
            with open(output_path, 'w') as f:
                json.dump(all_qa_pairs, f, indent=2)
            logger.info(f"\nSaved {len(all_qa_pairs)} Q&A pairs to {output_path}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Ingestion Summary")
        logger.info("=" * 60)
        logger.info(f"PDFs processed: {stats['processed']}")
        logger.info(f"PDFs failed: {stats['failed']}")
        logger.info(f"Total Q&A pairs: {stats['qa_pairs']}")


def main():
    ingester = PDFIngester()
    ingester.ingest_all()


if __name__ == "__main__":
    main()
