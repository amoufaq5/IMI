"""
UMI Kaggle Medical Datasets Ingestion Pipeline
<<<<<<< HEAD
Downloads and processes medical datasets from Kaggle with no limits
=======
Automatically downloads and processes medical datasets from Kaggle
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
"""

import asyncio
import json
import os
<<<<<<< HEAD
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
=======
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm


@dataclass
class KaggleDataset:
    """Represents a Kaggle dataset."""
    name: str
    owner: str
    title: str
    description: str
    size: str
    download_count: int
    vote_count: int
    usability_rating: float
    tags: List[str]
    files: List[str]


class KaggleClient:
    """
    Client for Kaggle API.
    Requires kaggle.json credentials in ~/.kaggle/
    """
    
    def __init__(self):
        self.kaggle_dir = Path.home() / ".kaggle"
        self._check_credentials()
    
    def _check_credentials(self):
        """Check if Kaggle credentials exist."""
        creds_file = self.kaggle_dir / "kaggle.json"
        if not creds_file.exists():
            print("WARNING: Kaggle credentials not found at ~/.kaggle/kaggle.json")
            print("To use Kaggle datasets:")
            print("  1. Go to https://www.kaggle.com/settings")
            print("  2. Click 'Create New Token' under API section")
            print("  3. Place the downloaded kaggle.json in ~/.kaggle/")
            print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
    
    def search_datasets(
        self,
        query: str,
        max_results: int = 50,
        sort_by: str = "hottest",
    ) -> List[Dict[str, Any]]:
        """Search for datasets on Kaggle."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            datasets = api.dataset_list(
                search=query,
                sort_by=sort_by,
                max_size=None,
                file_type="csv",
            )
            
            results = []
            for ds in datasets[:max_results]:
                results.append({
                    "ref": ds.ref,
                    "title": ds.title,
                    "size": ds.size,
                    "lastUpdated": str(ds.lastUpdated),
                    "downloadCount": ds.downloadCount,
                    "voteCount": ds.voteCount,
                    "usabilityRating": ds.usabilityRating,
                })
            
            return results
        except Exception as e:
            print(f"Error searching datasets: {e}")
            return []
    
    def download_dataset(
        self,
        dataset_ref: str,
        output_dir: Path,
        unzip: bool = True,
    ) -> bool:
        """Download a dataset from Kaggle."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            api.dataset_download_files(
                dataset_ref,
                path=str(output_dir),
                unzip=unzip,
            )
            
            return True
        except Exception as e:
            print(f"Error downloading dataset {dataset_ref}: {e}")
            return False
    
    def download_competition_data(
        self,
        competition: str,
        output_dir: Path,
    ) -> bool:
        """Download competition data from Kaggle."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            api.competition_download_files(
                competition,
                path=str(output_dir),
            )
            
            # Unzip if needed
            for zip_file in output_dir.glob("*.zip"):
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(output_dir)
                zip_file.unlink()
            
            return True
        except Exception as e:
            print(f"Error downloading competition {competition}: {e}")
            return False


class KaggleMedicalDatasets:
    """
    Curated list of high-quality medical datasets on Kaggle.
    All datasets are publicly available and free to download.
    """
    
    # Format: (dataset_ref, description, category)
    DATASETS = [
        # Disease Prediction & Diagnosis
        ("uciml/heart-disease-uci", "UCI Heart Disease Dataset", "cardiology"),
        ("fedesoriano/heart-failure-prediction", "Heart Failure Prediction Dataset", "cardiology"),
        ("johnsmith88/heart-disease-dataset", "Heart Disease Cleveland Dataset", "cardiology"),
        ("sulianova/cardiovascular-disease-dataset", "Cardiovascular Disease Dataset", "cardiology"),
        ("andrewmvd/heart-failure-clinical-data", "Heart Failure Clinical Records", "cardiology"),
        
        ("uciml/pima-indians-diabetes-database", "Pima Indians Diabetes Database", "diabetes"),
        ("mathchi/diabetes-data-set", "Diabetes Dataset", "diabetes"),
        ("alexteboul/diabetes-health-indicators-dataset", "Diabetes Health Indicators", "diabetes"),
        ("iammustafatz/diabetes-prediction-dataset", "Diabetes Prediction Dataset", "diabetes"),
        
        ("uciml/breast-cancer-wisconsin-data", "Breast Cancer Wisconsin Dataset", "oncology"),
        ("erdemtaha/cancer-data", "Cancer Data", "oncology"),
        ("rishidamarla/cancer-patients-data", "Cancer Patients Data", "oncology"),
        ("blurredmachine/lung-cancer", "Lung Cancer Dataset", "oncology"),
        ("yasserh/lung-cancer-dataset", "Lung Cancer Prediction", "oncology"),
        ("adityamahimkar/iqothnccd-lung-cancer-dataset", "Lung Cancer CT Images", "oncology"),
        ("andrewmvd/lung-and-colon-cancer-histopathological-images", "Lung & Colon Cancer Images", "oncology"),
        ("obulisainaren/multi-cancer", "Multi-Cancer Dataset", "oncology"),
        
        ("fedesoriano/stroke-prediction-dataset", "Stroke Prediction Dataset", "neurology"),
        ("shashwatwork/dementia-prediction-dataset", "Dementia Prediction Dataset", "neurology"),
        ("jillanisofttech/brain-stroke-dataset", "Brain Stroke Dataset", "neurology"),
        ("tourist55/alzheimers-dataset-4-class-of-images", "Alzheimer's MRI Dataset", "neurology"),
        ("sachinkumar413/alzheimer-mri-dataset", "Alzheimer MRI Preprocessed", "neurology"),
        
        ("uciml/chronic-kidney-disease", "Chronic Kidney Disease Dataset", "nephrology"),
        ("mansoordaku/ckdisease", "Chronic Kidney Disease", "nephrology"),
        
        ("prashant111/liver-patient-dataset", "Indian Liver Patient Dataset", "hepatology"),
        ("uciml/indian-liver-patient-records", "Indian Liver Patient Records", "hepatology"),
        
        # Medical Imaging
        ("paultimothymooney/chest-xray-pneumonia", "Chest X-Ray Pneumonia", "radiology"),
        ("tawsifurrahman/covid19-radiography-database", "COVID-19 Radiography Database", "radiology"),
        ("pranavraikoern/covid19-image-dataset", "COVID-19 Image Dataset", "radiology"),
        ("andrewmvd/covid19-ct-scans", "COVID-19 CT Scans", "radiology"),
        ("nih-chest-xrays/data", "NIH Chest X-rays", "radiology"),
        ("kmader/rsna-pneumonia-detection-challenge", "RSNA Pneumonia Detection", "radiology"),
        ("andrewmvd/medical-mnist", "Medical MNIST", "radiology"),
        ("masoudnickparvar/brain-tumor-mri-dataset", "Brain Tumor MRI Dataset", "radiology"),
        ("navoneel/brain-mri-images-for-brain-tumor-detection", "Brain MRI Tumor Detection", "radiology"),
        ("sartajbhuvaji/brain-tumor-classification-mri", "Brain Tumor Classification MRI", "radiology"),
        ("ahmedhamada0/brain-tumor-detection", "Brain Tumor Detection", "radiology"),
        ("mateuszbuda/lgg-mri-segmentation", "Brain MRI Segmentation", "radiology"),
        
        # Dermatology
        ("kmader/skin-cancer-mnist-ham10000", "Skin Cancer MNIST HAM10000", "dermatology"),
        ("fanconic/skin-cancer-malignant-vs-benign", "Skin Cancer Malignant vs Benign", "dermatology"),
        ("nodoubttome/skin-cancer9-classesisic", "Skin Cancer 9 Classes ISIC", "dermatology"),
        
        # Ophthalmology
        ("andrewmvd/ocular-disease-recognition-odir5k", "Ocular Disease Recognition", "ophthalmology"),
        ("gunavenkatdoddi/eye-diseases-classification", "Eye Diseases Classification", "ophthalmology"),
        ("jr2ngb/cataractdataset", "Cataract Dataset", "ophthalmology"),
        ("mariaherrerot/eyepacspreprocess", "Diabetic Retinopathy Detection", "ophthalmology"),
        
        # Mental Health
        ("osmi/mental-health-in-tech-survey", "Mental Health in Tech Survey", "psychiatry"),
        ("osmi/mental-health-in-tech-2016", "Mental Health in Tech 2016", "psychiatry"),
        ("arashnic/depression-twitter-dataset-imbalanced-data", "Depression Twitter Dataset", "psychiatry"),
        
        # Drug & Pharmaceutical
        ("jessicali9530/kuc-hackathon-winter-2018", "Drug Review Dataset", "pharmacology"),
        ("rohanharode07/webmd-drug-reviews-dataset", "WebMD Drug Reviews", "pharmacology"),
        ("jithinanievarghese/drugs-side-effects-and-medical-condition", "Drug Side Effects", "pharmacology"),
        
        # Clinical & EHR Data
        ("mimic-iii/mimic-iii-clinical-database-demo", "MIMIC-III Demo", "clinical"),
        ("drscarlat/medication", "Medication Dataset", "clinical"),
        ("nehaprabhavalkar/av-healthcare-analytics-ii", "Healthcare Analytics", "clinical"),
        
        # Genomics & Bioinformatics
        ("crawford/gene-expression", "Gene Expression Dataset", "genomics"),
        ("uciml/breast-cancer-wisconsin-original", "Breast Cancer Wisconsin Original", "genomics"),
        
        # COVID-19 Specific
        ("sudalairajkumar/novel-corona-virus-2019-dataset", "Novel Corona Virus 2019 Dataset", "covid19"),
        ("imdevskp/corona-virus-report", "Corona Virus Report", "covid19"),
        ("allen-institute-for-ai/CORD-19-research-challenge", "CORD-19 Research Challenge", "covid19"),
        ("roche-data-science-coalition/uncover", "UNCOVER COVID-19 Challenge", "covid19"),
        
        # Symptoms & Diagnosis
        ("kaushil268/disease-prediction-using-machine-learning", "Disease Prediction ML", "diagnosis"),
        ("itachi9604/disease-symptom-description-dataset", "Disease Symptom Description", "diagnosis"),
        ("niyarrbarman/symptom2disease", "Symptom to Disease", "diagnosis"),
        
        # Vital Signs & Monitoring
        ("shayanfazeli/heartbeat", "Heartbeat ECG Dataset", "cardiology"),
        ("kinguistics/heartbeat-sounds", "Heartbeat Sounds", "cardiology"),
        
        # Nutrition & Lifestyle
        ("niharika41298/gym-exercise-data", "Gym Exercise Data", "fitness"),
        ("uciml/human-activity-recognition-with-smartphones", "Human Activity Recognition", "fitness"),
        
        # Pediatrics
        ("aryashah2k/breast-cancer-dataset", "Breast Cancer Dataset", "oncology"),
        
        # Emergency Medicine
        ("saurabhshahane/road-traffic-accidents", "Road Traffic Accidents", "emergency"),
        
        # Dental
        ("deepcontractor/tufts-dental-database", "Tufts Dental Database", "dental"),
    ]
    
    # Kaggle Competitions with medical data
    COMPETITIONS = [
        ("rsna-pneumonia-detection-challenge", "RSNA Pneumonia Detection Challenge"),
        ("rsna-intracranial-hemorrhage-detection", "RSNA Intracranial Hemorrhage Detection"),
        ("rsna-miccai-brain-tumor-radiogenomic-classification", "RSNA Brain Tumor Classification"),
        ("siim-isic-melanoma-classification", "SIIM-ISIC Melanoma Classification"),
        ("ranzcr-clip-catheter-line-classification", "RANZCR CLiP Catheter Classification"),
        ("vinbigdata-chest-xray-abnormalities-detection", "VinBigData Chest X-ray Detection"),
        ("aptos2019-blindness-detection", "APTOS Blindness Detection"),
        ("diabetic-retinopathy-detection", "Diabetic Retinopathy Detection"),
    ]
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)


class KaggleIngestionPipeline:
    """
<<<<<<< HEAD
    Pipeline for ingesting medical datasets from Kaggle.
    No limits on data fetching.
=======
    Pipeline for ingesting Kaggle medical datasets into UMI knowledge base.
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
    """
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/kaggle",
        download_dir: str = "data/raw/kaggle",
<<<<<<< HEAD
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
=======
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.client = KaggleClient()
        self.datasets_processed: List[Dict[str, Any]] = []
    
    def process_csv_to_knowledge(
        self,
        csv_path: Path,
        dataset_ref: str,
        description: str,
        category: str,
    ) -> List[Dict[str, Any]]:
        """Process a CSV file into knowledge base documents."""
        documents = []
        
        try:
            # Read CSV with pandas
            df = pd.read_csv(csv_path, nrows=10000)  # Sample for metadata
            
            # Create dataset overview document
            columns_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].count()
                unique = df[col].nunique()
                columns_info.append(f"- {col}: {dtype}, {non_null} non-null, {unique} unique values")
            
            overview_content = f"""Dataset: {dataset_ref}
Description: {description}
Category: {category}
Rows: {len(df)}
Columns: {len(df.columns)}

Column Information:
{chr(10).join(columns_info)}

Sample Statistics:
{df.describe().to_string()}
"""
            
            documents.append({
                "id": f"kaggle_{dataset_ref.replace('/', '_')}_overview",
                "title": f"Kaggle Dataset: {description}",
                "content": overview_content,
                "metadata": {
                    "source": "Kaggle",
                    "dataset_ref": dataset_ref,
                    "category": category,
                    "type": "dataset_overview",
                    "rows": len(df),
                    "columns": list(df.columns),
                },
            })
            
            # Create column-specific documents for medical relevance
            medical_columns = self._identify_medical_columns(df)
            for col, col_type in medical_columns.items():
                if df[col].dtype in ['object', 'category']:
                    value_counts = df[col].value_counts().head(20).to_dict()
                    col_content = f"""Column: {col}
Dataset: {dataset_ref}
Type: {col_type}
Data Type: {df[col].dtype}

Value Distribution:
{json.dumps(value_counts, indent=2)}
"""
                else:
                    stats = df[col].describe().to_dict()
                    col_content = f"""Column: {col}
Dataset: {dataset_ref}
Type: {col_type}
Data Type: {df[col].dtype}

Statistics:
{json.dumps(stats, indent=2)}
"""
                
                documents.append({
                    "id": f"kaggle_{dataset_ref.replace('/', '_')}_{col}",
                    "title": f"{col} - {description}",
                    "content": col_content,
                    "metadata": {
                        "source": "Kaggle",
                        "dataset_ref": dataset_ref,
                        "category": category,
                        "type": "column_analysis",
                        "column": col,
                    },
                })
        
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
        
        return documents
    
    def _identify_medical_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Identify columns with medical relevance."""
        medical_keywords = {
            "diagnosis": ["diagnosis", "disease", "condition", "disorder"],
            "symptom": ["symptom", "sign", "complaint"],
            "treatment": ["treatment", "therapy", "medication", "drug"],
            "vital": ["blood_pressure", "heart_rate", "temperature", "pulse", "bp", "hr"],
            "lab": ["glucose", "cholesterol", "hemoglobin", "creatinine", "bilirubin"],
            "demographic": ["age", "gender", "sex", "race", "ethnicity"],
            "outcome": ["outcome", "survival", "death", "mortality", "prognosis"],
            "imaging": ["image", "scan", "xray", "mri", "ct"],
        }
        
        medical_cols = {}
        for col in df.columns:
            col_lower = col.lower()
            for med_type, keywords in medical_keywords.items():
                if any(kw in col_lower for kw in keywords):
                    medical_cols[col] = med_type
                    break
        
        return medical_cols
    
    async def download_and_process_dataset(
        self,
        dataset_ref: str,
        description: str,
        category: str,
    ) -> List[Dict[str, Any]]:
        """Download and process a single dataset."""
        documents = []
        
        # Create dataset-specific directory
        dataset_dir = self.download_dir / dataset_ref.replace("/", "_")
        
        # Download dataset
        success = self.client.download_dataset(dataset_ref, dataset_dir)
        
        if success:
            # Process all CSV files in the dataset
            for csv_file in dataset_dir.glob("**/*.csv"):
                docs = self.process_csv_to_knowledge(
                    csv_file, dataset_ref, description, category
                )
                documents.extend(docs)
        
        return documents
    
    async def run(
        self,
        max_datasets: Optional[int] = None,
        categories: Optional[List[str]] = None,
    ) -> None:
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI Kaggle Medical Datasets Ingestion Pipeline")
        print("=" * 60)
        
        all_documents = []
        datasets_to_process = KaggleMedicalDatasets.DATASETS
        
        # Filter by category if specified
        if categories:
            datasets_to_process = [
                d for d in datasets_to_process if d[2] in categories
            ]
        
        # Limit if specified (no limit by default)
        if max_datasets:
            datasets_to_process = datasets_to_process[:max_datasets]
        
        for dataset_ref, description, category in tqdm(datasets_to_process, desc="Processing datasets"):
            try:
                print(f"\n  Processing: {dataset_ref}")
                docs = await self.download_and_process_dataset(
                    dataset_ref, description, category
                )
                all_documents.extend(docs)
                
                self.datasets_processed.append({
                    "ref": dataset_ref,
                    "description": description,
                    "category": category,
                    "documents": len(docs),
                })
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)
                
                print(f"    Generated {len(docs)} documents")
                
            except Exception as e:
<<<<<<< HEAD
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
=======
                print(f"    Error: {e}")
            
            await asyncio.sleep(1)  # Rate limiting
        
        # Save documents
        await self.save(all_documents)
        
        print(f"\nTotal datasets processed: {len(self.datasets_processed)}")
        print(f"Total documents generated: {len(all_documents)}")
    
    async def save(self, documents: List[Dict[str, Any]]) -> None:
        """Save documents to disk."""
        output_file = self.output_dir / "kaggle_datasets.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save statistics
        stats = {
            "total_documents": len(documents),
            "datasets_processed": self.datasets_processed,
            "ingestion_date": datetime.now().isoformat(),
            "available_datasets": len(KaggleMedicalDatasets.DATASETS),
            "available_competitions": len(KaggleMedicalDatasets.COMPETITIONS),
        }
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save dataset catalog
        catalog_file = self.output_dir / "dataset_catalog.json"
        with open(catalog_file, 'w') as f:
            json.dump({
                "datasets": [
                    {"ref": d[0], "description": d[1], "category": d[2]}
                    for d in KaggleMedicalDatasets.DATASETS
                ],
                "competitions": [
                    {"ref": c[0], "description": c[1]}
                    for c in KaggleMedicalDatasets.COMPETITIONS
                ],
            }, f, indent=2)


async def main():
    """Run the Kaggle ingestion pipeline with no limits."""
    pipeline = KaggleIngestionPipeline()
    # No max_datasets limit - process all available datasets
    await pipeline.run(max_datasets=None)
>>>>>>> 66878e9 (Expand data ingestion: 9 new scrapers, Kaggle support, remove caps, robust error handling)


if __name__ == "__main__":
    asyncio.run(main())
