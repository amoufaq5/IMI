"""
Google Drive File Downloader for IMI Training Data

Downloads pharma QA files, SOPs, and regulatory documents from a shared
Google Drive folder and converts them into training-ready format.

Usage:
    # Download from a shared folder link
    python scripts/data_collection/download_gdrive.py \
        --folder-url "https://drive.google.com/drive/folders/FOLDER_ID" \
        --adapter regulatory_qa

    # Download using folder ID directly
    python scripts/data_collection/download_gdrive.py \
        --folder-id "1aBcDeFgHiJkLmNoPqRsTuVwXyZ" \
        --adapter regulatory_qa

    # Download to a custom directory
    python scripts/data_collection/download_gdrive.py \
        --folder-id "FOLDER_ID" \
        --output-dir ./data/gdrive/pharma_qa

Requirements:
    pip install gdown
"""
import os
import re
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

DATA_DIR = Path(__file__).parent.parent.parent / "data"
GDRIVE_DIR = DATA_DIR / "gdrive"
PROCESSED_DIR = DATA_DIR / "processed"


def extract_folder_id(url_or_id: str) -> str:
    """Extract Google Drive folder ID from URL or return ID directly."""
    # Match: https://drive.google.com/drive/folders/FOLDER_ID?...
    match = re.search(r"folders/([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    # Assume it's already a folder ID
    return url_or_id.strip()


def download_folder(folder_id: str, output_dir: Path) -> List[Path]:
    """Download all files from a Google Drive folder using gdown."""
    if not GDOWN_AVAILABLE:
        raise ImportError(
            "gdown is not installed. Run: pip install gdown\n"
            "Then re-run this script."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    logger.info(f"Downloading from Google Drive folder: {folder_id}")
    logger.info(f"Output directory: {output_dir}")

    downloaded = gdown.download_folder(
        url=url,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
    )

    if not downloaded:
        logger.warning("No files were downloaded. Check that the folder is publicly shared.")
        return []

    paths = [Path(p) for p in downloaded]
    logger.info(f"Downloaded {len(paths)} files")
    return paths


def download_single_file(file_url: str, output_dir: Path) -> Optional[Path]:
    """Download a single file from Google Drive."""
    if not GDOWN_AVAILABLE:
        raise ImportError("gdown is not installed. Run: pip install gdown")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = gdown.download(file_url, output=str(output_dir) + "/", fuzzy=True)
    if output_path:
        return Path(output_path)
    return None


def process_downloaded_files(
    files: List[Path],
    adapter: str,
) -> List[Dict[str, Any]]:
    """
    Convert downloaded files into instruction-tuning format.

    Supports:
      - .json / .jsonl  — expects instruction/input/output keys or raw text
      - .txt            — wraps as regulatory Q&A pairs
      - .pdf            — delegates to ingest_pdfs.py logic
      - .csv            — basic row-per-example parsing
    """
    qa_pairs: List[Dict[str, Any]] = []

    for fpath in files:
        ext = fpath.suffix.lower()
        logger.info(f"Processing {fpath.name} ({ext})")

        try:
            if ext in (".json", ".jsonl"):
                qa_pairs.extend(_process_json(fpath, adapter))
            elif ext == ".txt":
                qa_pairs.extend(_process_text(fpath, adapter))
            elif ext == ".pdf":
                qa_pairs.extend(_process_pdf(fpath, adapter))
            elif ext == ".csv":
                qa_pairs.extend(_process_csv(fpath, adapter))
            else:
                logger.warning(f"  Skipping unsupported file type: {ext}")
        except Exception as e:
            logger.error(f"  Error processing {fpath.name}: {e}")

    return qa_pairs


def _process_json(fpath: Path, adapter: str) -> List[Dict[str, Any]]:
    """Process JSON/JSONL files."""
    items = []
    if fpath.suffix == ".jsonl":
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    else:
        with open(fpath) as f:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]

    results = []
    for item in items:
        # If already in instruction format, use as-is
        if "instruction" in item and "output" in item:
            item.setdefault("input", "")
            item.setdefault("adapter", adapter)
            item.setdefault("source", f"gdrive_{fpath.stem}")
            results.append(item)
        # Otherwise try to convert Q&A format
        elif "question" in item and "answer" in item:
            results.append({
                "instruction": item["question"],
                "input": item.get("context", ""),
                "output": item["answer"],
                "adapter": adapter,
                "source": f"gdrive_{fpath.stem}",
            })
        # Raw text field
        elif "text" in item:
            results.append({
                "instruction": f"Explain the following regulatory content from {fpath.stem}:",
                "input": item["text"][:2000],
                "output": item["text"][:2000],
                "adapter": adapter,
                "source": f"gdrive_{fpath.stem}",
            })

    logger.info(f"  → {len(results)} examples from {fpath.name}")
    return results


def _process_text(fpath: Path, adapter: str) -> List[Dict[str, Any]]:
    """Process plain text files (SOPs, guidelines, memos)."""
    text = fpath.read_text(encoding="utf-8", errors="ignore")
    if len(text) < 50:
        return []

    # Split into chunks of ~1500 chars at paragraph boundaries
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > 1500 and current:
            chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())

    results = []
    for i, chunk in enumerate(chunks):
        if len(chunk) < 50:
            continue
        results.append({
            "instruction": f"Summarize the following pharmaceutical QA document section from {fpath.stem}:",
            "input": chunk,
            "output": chunk,
            "adapter": adapter,
            "source": f"gdrive_{fpath.stem}_chunk{i}",
        })

    logger.info(f"  → {len(results)} chunks from {fpath.name}")
    return results


def _process_pdf(fpath: Path, adapter: str) -> List[Dict[str, Any]]:
    """Process PDF files by delegating to the existing PDF ingester."""
    try:
        from scripts.data_collection.ingest_pdfs import PDFIngester
        ingester = PDFIngester(pdf_dir=fpath.parent)
        doc = ingester.process_pdf(fpath)
        if doc:
            return ingester.create_qa_pairs(doc)
    except ImportError:
        logger.warning("  Could not import PDFIngester — processing PDF as text fallback")
        try:
            import fitz
            text = ""
            pdf_doc = fitz.open(fpath)
            for page in pdf_doc:
                text += page.get_text()
            pdf_doc.close()
            # Re-use text processor
            tmp = fpath.with_suffix(".txt")
            tmp.write_text(text)
            results = _process_text(tmp, adapter)
            tmp.unlink()
            return results
        except ImportError:
            logger.error("  No PDF library available. Install pymupdf: pip install pymupdf")
    return []


def _process_csv(fpath: Path, adapter: str) -> List[Dict[str, Any]]:
    """Process CSV files — expects columns like question,answer or instruction,output."""
    import csv

    results = []
    with open(fpath, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "instruction" in row and "output" in row:
                results.append({
                    "instruction": row["instruction"],
                    "input": row.get("input", ""),
                    "output": row["output"],
                    "adapter": adapter,
                    "source": f"gdrive_{fpath.stem}",
                })
            elif "question" in row and "answer" in row:
                results.append({
                    "instruction": row["question"],
                    "input": row.get("context", ""),
                    "output": row["answer"],
                    "adapter": adapter,
                    "source": f"gdrive_{fpath.stem}",
                })

    logger.info(f"  → {len(results)} rows from {fpath.name}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download pharma QA files from Google Drive for IMI training"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--folder-url",
        help="Google Drive shared folder URL",
    )
    group.add_argument(
        "--folder-id",
        help="Google Drive folder ID",
    )
    group.add_argument(
        "--file-url",
        help="Google Drive single file URL",
    )
    parser.add_argument(
        "--adapter",
        default="regulatory_qa",
        choices=[
            "patient_triage", "clinical_pharmacist", "clinical_decision",
            "education", "regulatory_qa", "research",
        ],
        help="Target adapter for this data (default: regulatory_qa)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (default: data/gdrive/<adapter>)",
    )

    args = parser.parse_args()

    # Resolve output directory
    output_dir = args.output_dir or (GDRIVE_DIR / args.adapter)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download files
    if args.file_url:
        path = download_single_file(args.file_url, output_dir)
        downloaded_files = [path] if path else []
    else:
        folder_id = args.folder_id or extract_folder_id(args.folder_url)
        downloaded_files = download_folder(folder_id, output_dir)

    if not downloaded_files:
        logger.error("No files downloaded. Exiting.")
        return

    # Process into training format
    qa_pairs = process_downloaded_files(downloaded_files, args.adapter)

    if not qa_pairs:
        logger.warning("No training examples generated from downloaded files.")
        return

    # Save processed data
    proc_dir = PROCESSED_DIR / args.adapter
    proc_dir.mkdir(parents=True, exist_ok=True)
    output_path = proc_dir / "gdrive_data.json"

    # Append to existing if present
    existing = []
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        logger.info(f"Appending to existing {len(existing)} examples")

    combined = existing + qa_pairs
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Google Drive Ingestion Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Files downloaded: {len(downloaded_files)}")
    logger.info(f"New examples: {len(qa_pairs)}")
    logger.info(f"Total examples: {len(combined)}")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"Target adapter: {args.adapter}")


if __name__ == "__main__":
    main()
