"""
UMI Kaggle Medical Datasets Ingestion Pipeline
Downloads and processes medical datasets from Kaggle with no limits
"""

import asyncio
import json
import os
import subprocess
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import io

import httpx
from tqdm import tqdm


# Curated list of high-quality medical datasets on Kaggle
MEDICAL_DATASETS = [
    # Disease & Diagnosis
    "uciml/heart-disease",
    "fedesoriano/heart-failure-prediction",
    "rashikrahmanpritom/heart-attack-analysis-prediction-dataset",
    "johnsmith88/heart-disease-dataset",
    "sulianova/cardiovascular-disease-dataset",
    "andrewmvd/heart-failure-clinical-data",
    
    # Diabetes
    "uciml/pima-indians-diabetes-database",
    "mathchi/diabetes-data-set",
    "alexteboul/diabetes-health-indicators-dataset",
    "iammustafatz/diabetes-prediction-dataset",
    
    # Cancer
    "uciml/breast-cancer-wisconsin-data",
    "erdemtaha/cancer-data",
    "obulisainaren/multi-cancer",
    "andrewmvd/lung-and-colon-cancer-histopathological-images",
    "paultimothymooney/chest-xray-pneumonia",
    "kmader/skin-cancer-mnist-ham10000",
    "adityamahimkar/iqothnccd-lung-cancer-dataset",
    
    # Medical Imaging
    "tawsifurrahman/covid19-radiography-database",
    "nih-chest-xrays/data",
    "paultimothymooney/blood-cells",
    "andrewmvd/medical-mnist",
    "navoneel/brain-mri-images-for-brain-tumor-detection",
    "masoudnickparvar/brain-tumor-mri-dataset",
    "ahmedhamada0/brain-tumor-detection",
    
    # Drug & Pharma
    "jessicali9530/kuc-hackathon-winter-2018",
    "jithinanievarghese/drugs-side-effects-and-medical-condition",
    "prathamtripathi/drug-classification",
    "rohanharode07/webmd-drug-reviews-dataset",
    "jessicali9530/celeba-dataset",
    
    # Clinical & Patient Data
    "sudalairajkumar/novel-corona-virus-2019-dataset",
    "allen-institute-for-ai/CORD-19-research-challenge",
    "roche-data-science-coalition/uncover",
    "nehaprabhavalkar/av-healthcare-analytics-ii",
    "hongseoi/sepsis-survival-minimal-clinical-records",
    
    # Mental Health
    "osmi/mental-health-in-tech-survey",
    "osmi/mental-health-in-tech-2016",
    "thedevastator/medical-student-mental-health",
    
    # Genomics & Biomedical
    "crawford/gene-expression",
    "uciml/human-activity-recognition-with-smartphones",
    "muonneutrino/us-census-demographic-data",
    
    # Healthcare Operations
    "nehaprabhavalkar/av-healthcare-analytics-ii",
    "brandao/diabetes",
    "joniarroba/noshowappointments",
    "prasad22/healthcare-dataset",
    
    # Medical Text & NLP
    "tboyle10/medicaltranscriptions",
    "xhlulu/medal-emnlp",
    "finalepoch/medical-ner",
    
    # Vital Signs & Monitoring
    "shayanfazeli/heartbeat",
    "kinguistics/heartbeat-sounds",
    "saurabh00007/diabetescsv",
    
    # Radiology
    "ywchrome/chest-xray-dataset",
    "prashant268/chest-xray-covid19-pneumonia",
    "tolgadincer/labeled-chest-xray-images",
    
    # Ophthalmology
    "benjaminwarner/resized-2015-2019-blindness-detection-images",
    "mariaherrerot/eyepacspreprocess",
    
    # Dermatology
    "shubhamgoel27/dermnet",
    "nodoubttome/skin-cancer9-classesisic",
    
    # Additional Medical Datasets
    "mirichoi0218/insurance",
    "uciml/indian-liver-patient-records",
    "fedesoriano/stroke-prediction-dataset",
    "andrewmvd/fetal-health-classification",
    "uciml/chronic-kidney-disease",
]


@dataclass
class KaggleDataset:
    """Represents a Kaggle dataset."""
    ref: str
    title: str
    size: int
    download_count: int
    description: str
    files: List[str] = field(default_factory=list)
    columns: Dict[str, List[str]] = field(default_factory=dict)


class KaggleClient:
    """Client for Kaggle API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._setup_credentials()
    
    def _setup_credentials(self) -> None:
        """Setup Kaggle credentials."""
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if self.api_key and not kaggle_json.exists():
            kaggle_dir.mkdir(exist_ok=True)
            # Parse API key format: username:key or just key
            if ":" in self.api_key:
                username, key = self.api_key.split(":", 1)
            else:
                # Assume it's a token format like KGAT_xxx
                username = "kaggle_user"
                key = self.api_key
            
            with open(kaggle_json, 'w') as f:
                json.dump({"username": username, "key": key}, f)
            os.chmod(kaggle_json, 0o600)
    
    def download_dataset(self, dataset_ref: str, output_dir: Path) -> bool:
        """Download a dataset from Kaggle."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(output_dir), "--unzip"],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            
            if result.returncode != 0:
                print(f"  Warning: Failed to download {dataset_ref}: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"  Warning: Timeout downloading {dataset_ref}")
            return False
        except FileNotFoundError:
            print("  Warning: Kaggle CLI not installed. Run: pip install kaggle")
            return False
        except Exception as e:
            print(f"  Warning: Error downloading {dataset_ref}: {e}")
            return False
    
    def get_dataset_info(self, dataset_ref: str) -> Optional[Dict[str, Any]]:
        """Get dataset metadata."""
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "metadata", "-d", dataset_ref],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode == 0:
                # Parse the output
                return {"ref": dataset_ref, "description": result.stdout}
            return None
            
        except Exception:
            return None


class KaggleIngestionPipeline:
    """
    Pipeline for ingesting medical datasets from Kaggle.
    No limits on data fetching.
    """
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/kaggle",
        download_dir: str = "data/raw/kaggle",
        api_key: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.download_dir = Path(download_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Use environment variable if not provided
        if api_key is None:
            api_key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_KEY")
        
        self.client = KaggleClient(api_key)
        self.datasets_processed: List[KaggleDataset] = []
    
    def _identify_medical_columns(self, df_columns: List[str]) -> List[str]:
        """Identify columns that contain medical information."""
        medical_keywords = [
            'age', 'sex', 'gender', 'blood', 'pressure', 'cholesterol', 'glucose',
            'heart', 'rate', 'bmi', 'weight', 'height', 'diagnosis', 'disease',
            'symptom', 'medication', 'drug', 'dose', 'treatment', 'patient',
            'cancer', 'tumor', 'diabetes', 'insulin', 'hemoglobin', 'platelet',
            'white', 'red', 'cell', 'count', 'level', 'test', 'result', 'lab',
            'ecg', 'ekg', 'mri', 'ct', 'xray', 'scan', 'imaging', 'radiology',
            'surgery', 'procedure', 'hospital', 'admission', 'discharge', 'icu',
            'mortality', 'survival', 'outcome', 'prognosis', 'stage', 'grade',
        ]
        
        medical_cols = []
        for col in df_columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in medical_keywords):
                medical_cols.append(col)
        
        return medical_cols if medical_cols else df_columns[:10]
    
    def _process_csv_file(self, file_path: Path, dataset_ref: str) -> List[Dict[str, Any]]:
        """Process a CSV file and extract medical information."""
        documents = []
        
        try:
            # Read CSV with various encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        # Read first few lines to detect structure
                        sample = f.read(10000)
                        f.seek(0)
                        
                        # Detect delimiter
                        sniffer = csv.Sniffer()
                        try:
                            dialect = sniffer.sniff(sample)
                            delimiter = dialect.delimiter
                        except:
                            delimiter = ','
                        
                        reader = csv.DictReader(f, delimiter=delimiter)
                        columns = reader.fieldnames or []
                        
                        if not columns:
                            continue
                        
                        # Get medical columns
                        medical_cols = self._identify_medical_columns(columns)
                        
                        # Create dataset overview document
                        overview_doc = {
                            "id": f"kaggle_{dataset_ref.replace('/', '_')}_{file_path.stem}_overview",
                            "title": f"Medical Dataset: {dataset_ref} - {file_path.name}",
                            "content": f"This medical dataset contains the following data fields: {', '.join(columns)}. "
                                      f"Key medical columns include: {', '.join(medical_cols)}. "
                                      f"Source: Kaggle dataset {dataset_ref}.",
                            "metadata": {
                                "source": "kaggle",
                                "dataset_ref": dataset_ref,
                                "file": file_path.name,
                                "columns": columns,
                                "medical_columns": medical_cols,
                                "type": "dataset_overview",
                            }
                        }
                        documents.append(overview_doc)
                        
                        # Process rows (sample for large files)
                        row_count = 0
                        for row in reader:
                            row_count += 1
                            
                            # Create document from row data
                            if row_count <= 1000:  # First 1000 rows as examples
                                content_parts = []
                                for col in medical_cols[:10]:
                                    if col in row and row[col]:
                                        content_parts.append(f"{col}: {row[col]}")
                                
                                if content_parts:
                                    doc = {
                                        "id": f"kaggle_{dataset_ref.replace('/', '_')}_{file_path.stem}_row{row_count}",
                                        "title": f"Medical Record from {dataset_ref}",
                                        "content": "Medical data record: " + "; ".join(content_parts),
                                        "metadata": {
                                            "source": "kaggle",
                                            "dataset_ref": dataset_ref,
                                            "file": file_path.name,
                                            "row": row_count,
                                            "type": "data_record",
                                        }
                                    }
                                    documents.append(doc)
                        
                        # Add statistics document
                        stats_doc = {
                            "id": f"kaggle_{dataset_ref.replace('/', '_')}_{file_path.stem}_stats",
                            "title": f"Dataset Statistics: {dataset_ref}",
                            "content": f"The dataset {dataset_ref} file {file_path.name} contains {row_count} records "
                                      f"with {len(columns)} columns. Medical-relevant columns: {', '.join(medical_cols)}.",
                            "metadata": {
                                "source": "kaggle",
                                "dataset_ref": dataset_ref,
                                "file": file_path.name,
                                "row_count": row_count,
                                "column_count": len(columns),
                                "type": "statistics",
                            }
                        }
                        documents.append(stats_doc)
                        
                        break  # Successfully read file
                        
                except UnicodeDecodeError:
                    continue
                    
        except Exception as e:
            print(f"    Warning: Error processing {file_path}: {e}")
        
        return documents
    
    def _process_dataset(self, dataset_ref: str) -> List[Dict[str, Any]]:
        """Download and process a single dataset."""
        documents = []
        dataset_dir = self.download_dir / dataset_ref.replace("/", "_")
        
        # Download if not already present
        if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
            print(f"  Downloading {dataset_ref}...")
            success = self.client.download_dataset(dataset_ref, dataset_dir)
            if not success:
                return documents
        
        # Process all CSV files
        for file_path in dataset_dir.rglob("*.csv"):
            print(f"    Processing {file_path.name}...")
            docs = self._process_csv_file(file_path, dataset_ref)
            documents.extend(docs)
        
        # Also check for JSON files
        for file_path in dataset_dir.rglob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    for i, item in enumerate(data[:100]):  # First 100 items
                        if isinstance(item, dict):
                            doc = {
                                "id": f"kaggle_{dataset_ref.replace('/', '_')}_{file_path.stem}_{i}",
                                "title": f"Medical Data from {dataset_ref}",
                                "content": json.dumps(item, indent=2)[:2000],
                                "metadata": {
                                    "source": "kaggle",
                                    "dataset_ref": dataset_ref,
                                    "file": file_path.name,
                                    "type": "json_record",
                                }
                            }
                            documents.append(doc)
            except Exception as e:
                print(f"    Warning: Error processing JSON {file_path}: {e}")
        
        return documents
    
    async def run(self, max_datasets: Optional[int] = None) -> None:
        """Run the ingestion pipeline with no limits."""
        print("=" * 60)
        print("KAGGLE MEDICAL DATASETS INGESTION")
        print(f"Total datasets to process: {len(MEDICAL_DATASETS)}")
        print("=" * 60)
        
        all_documents = []
        datasets_to_process = MEDICAL_DATASETS if max_datasets is None else MEDICAL_DATASETS[:max_datasets]
        
        for i, dataset_ref in enumerate(tqdm(datasets_to_process, desc="Processing datasets")):
            print(f"\n[{i+1}/{len(datasets_to_process)}] {dataset_ref}")
            
            try:
                docs = self._process_dataset(dataset_ref)
                all_documents.extend(docs)
                
                self.datasets_processed.append(KaggleDataset(
                    ref=dataset_ref,
                    title=dataset_ref.split("/")[-1],
                    size=0,
                    download_count=0,
                    description="",
                    files=[],
                ))
                
                print(f"    Generated {len(docs)} documents")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
        
        # Save all documents
        output_file = self.output_dir / "datasets.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in all_documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"\n{'=' * 60}")
        print(f"KAGGLE INGESTION COMPLETE")
        print(f"Datasets processed: {len(self.datasets_processed)}")
        print(f"Documents generated: {len(all_documents)}")
        print(f"Output: {output_file}")
        print("=" * 60)


async def main():
    """Run the Kaggle ingestion pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kaggle Medical Datasets Ingestion")
    parser.add_argument("--output-dir", default="data/knowledge_base/kaggle")
    parser.add_argument("--download-dir", default="data/raw/kaggle")
    parser.add_argument("--api-key", default=None, help="Kaggle API key")
    parser.add_argument("--max-datasets", type=int, default=None, help="Max datasets (None=all)")
    
    args = parser.parse_args()
    
    pipeline = KaggleIngestionPipeline(
        output_dir=args.output_dir,
        download_dir=args.download_dir,
        api_key=args.api_key,
    )
    
    await pipeline.run(max_datasets=args.max_datasets)


if __name__ == "__main__":
    asyncio.run(main())
