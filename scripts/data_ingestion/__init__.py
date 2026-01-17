# Data Ingestion Pipelines
"""
UMI Data Ingestion Pipelines

Comprehensive data collection from open-source medical databases:
- PubMed: Medical literature and research papers
- OpenFDA: Drug labels, interactions, and safety information
- ClinicalTrials.gov: Clinical trial data
- RxNorm: Drug terminology and relationships
- WHO/ICD-10: Disease classification codes
"""

from .ingest_pubmed import PubMedClient, PubMedIngestionPipeline, PubMedArticle
from .ingest_drugbank import OpenFDAClient, DrugIngestionPipeline, DrugInfo
from .ingest_clinicaltrials import ClinicalTrialsClient, ClinicalTrialsIngestionPipeline, ClinicalTrial
from .ingest_rxnorm import RxNormClient, RxNormIngestionPipeline, RxNormDrug
from .ingest_who import WHOIngestionPipeline, ICDCode
from .ingest_all import run_all_pipelines

__all__ = [
    # PubMed
    "PubMedClient",
    "PubMedIngestionPipeline",
    "PubMedArticle",
    # OpenFDA/DrugBank
    "OpenFDAClient",
    "DrugIngestionPipeline",
    "DrugInfo",
    # ClinicalTrials.gov
    "ClinicalTrialsClient",
    "ClinicalTrialsIngestionPipeline",
    "ClinicalTrial",
    # RxNorm
    "RxNormClient",
    "RxNormIngestionPipeline",
    "RxNormDrug",
    # WHO/ICD
    "WHOIngestionPipeline",
    "ICDCode",
    # Master pipeline
    "run_all_pipelines",
]
