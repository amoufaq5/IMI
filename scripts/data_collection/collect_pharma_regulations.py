"""
Pharmaceutical Regulations, QA/QC, Manufacturing & Inspection Data Collector

Downloads open-source data covering pharmaceutical company regulations,
Good Manufacturing Practice (GMP), quality assurance/control, and inspection records.

Sources:
  - FDA (cGMP, 483s, warning letters, recalls, inspections, NDAs)
  - EMA (European Medicines Agency guidelines)
  - ICH (International Council for Harmonisation — Q-series guidelines)
  - WHO (prequalification, GMP annexes)
  - USP (pharmacopeial standards references)
  - EPA (pharmaceutical environmental compliance)

Categories:
  gmp             — Good Manufacturing Practice regulations & guidance
  inspection      — FDA 483 observations, warning letters, facility inspections
  quality         — QA/QC testing, stability, validation, CAPA
  manufacturing   — Drug manufacturing, process validation, scale-up
  recalls         — Product recalls, market withdrawals, safety alerts
  labeling        — Drug labeling, packaging, NDC, approval data
  pharmacovigilance — Adverse events, REMS, post-market surveillance

Usage:
    pip install requests
    python scripts/data_collection/collect_pharma_regulations.py              # download all
    python scripts/data_collection/collect_pharma_regulations.py --category gmp
    python scripts/data_collection/collect_pharma_regulations.py --list
"""
import io
import csv
import json
import logging
import zipfile
import argparse
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "pharma_regulations"
PROCESSED_DIR = DATA_DIR / "processed" / "regulatory_qa"

HEADERS = {
    "User-Agent": "IMI-Medical-AI-Research/1.0 (academic research)",
    "Accept": "application/json, text/csv, text/plain",
}


# ============================================================================
# PHARMACEUTICAL REGULATION DATA SOURCES
#
# Format: (name, url, file_format, category, est_size, description)
# ============================================================================

PHARMA_SOURCES = [

    # =====================================================================
    # GMP — Good Manufacturing Practice
    # =====================================================================

    ("fda_cgmp_guidance_drugs",
     "https://api.fda.gov/drug/label.json?search=openfda.product_type:\"HUMAN+PRESCRIPTION+DRUG\"+AND+_exists_:spl_product_data_elements&limit=1000",
     "api_json", "gmp", "1K",
     "FDA cGMP drug product labeling with manufacturing data elements"),

    ("fda_cgmp_guidance_biologics",
     "https://api.fda.gov/drug/label.json?search=openfda.product_type:\"HUMAN+PRESCRIPTION+DRUG\"+AND+_exists_:manufacturing&limit=1000",
     "api_json", "gmp", "1K",
     "FDA biologics manufacturing labeling data"),

    ("ich_q7_gmp_api",
     "https://raw.githubusercontent.com/nickloman/WHO-EML/master/WHO_EML_2023.csv",
     "reference", "gmp", "ref",
     "ICH Q7 — GMP Guide for Active Pharmaceutical Ingredients (reference)"),

    # =====================================================================
    # INSPECTION — FDA 483s, Warning Letters, Facility Data
    # =====================================================================

    ("fda_warning_letters",
     "https://api.fda.gov/drug/enforcement.json?search=report_date:[20200101+TO+20241231]&limit=1000",
     "api_json", "inspection", "1K",
     "FDA drug enforcement actions and warning letters 2020-2024"),

    ("fda_facility_inspections",
     "https://api.fda.gov/drug/enforcement.json?search=classification:\"Class+I\"&limit=1000",
     "api_json", "inspection", "1K",
     "FDA Class I enforcement actions (most serious violations)"),

    ("fda_class2_enforcement",
     "https://api.fda.gov/drug/enforcement.json?search=classification:\"Class+II\"&limit=1000",
     "api_json", "inspection", "1K",
     "FDA Class II enforcement actions"),

    ("fda_class3_enforcement",
     "https://api.fda.gov/drug/enforcement.json?search=classification:\"Class+III\"&limit=1000",
     "api_json", "inspection", "1K",
     "FDA Class III enforcement actions (least serious)"),

    ("fda_inspections_domestic",
     "https://datadashboard.fda.gov/api/dataset/Inspection%20Classification%20Search",
     "skip", "inspection", "50K",
     "FDA inspection classification search (requires dashboard access)"),

    ("fda_debarment_list",
     "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/compliance-actions-and-activities/debarment-list-drug-product-applications",
     "skip", "inspection", "200",
     "FDA debarment list (requires HTML scraping)"),

    # =====================================================================
    # QUALITY — QA/QC, Testing, Stability, Validation
    # =====================================================================

    ("fda_drug_quality_sampling",
     "https://api.fda.gov/drug/enforcement.json?search=reason_for_recall:\"failed+quality\"&limit=1000",
     "api_json", "quality", "1K",
     "FDA recalls due to quality failures (QC testing failures)"),

    ("fda_sterility_recalls",
     "https://api.fda.gov/drug/enforcement.json?search=reason_for_recall:\"sterility\"&limit=1000",
     "api_json", "quality", "500",
     "FDA recalls related to sterility assurance failures"),

    ("fda_stability_recalls",
     "https://api.fda.gov/drug/enforcement.json?search=reason_for_recall:\"stability\"&limit=1000",
     "api_json", "quality", "300",
     "FDA recalls due to stability testing failures (out-of-spec)"),

    ("fda_dissolution_recalls",
     "https://api.fda.gov/drug/enforcement.json?search=reason_for_recall:\"dissolution\"&limit=1000",
     "api_json", "quality", "200",
     "FDA recalls due to dissolution testing failures"),

    ("fda_impurity_recalls",
     "https://api.fda.gov/drug/enforcement.json?search=reason_for_recall:\"impurity\"&limit=1000",
     "api_json", "quality", "300",
     "FDA recalls due to impurity/contamination findings"),

    ("fda_potency_recalls",
     "https://api.fda.gov/drug/enforcement.json?search=reason_for_recall:\"potency\"&limit=1000",
     "api_json", "quality", "200",
     "FDA recalls due to potency/assay failures"),

    ("fda_labeling_error_recalls",
     "https://api.fda.gov/drug/enforcement.json?search=reason_for_recall:\"mislabel\"&limit=1000",
     "api_json", "quality", "300",
     "FDA recalls due to labeling/packaging errors"),

    # =====================================================================
    # MANUFACTURING — Process, Validation, Scale-up
    # =====================================================================

    ("fda_mfg_cgmp_recalls",
     "https://api.fda.gov/drug/enforcement.json?search=reason_for_recall:\"cGMP\"&limit=1000",
     "api_json", "manufacturing", "500",
     "FDA recalls citing cGMP deviations in manufacturing"),

    ("fda_contamination_recalls",
     "https://api.fda.gov/drug/enforcement.json?search=reason_for_recall:\"contamination\"&limit=1000",
     "api_json", "manufacturing", "500",
     "FDA recalls due to manufacturing contamination"),

    ("fda_cross_contamination",
     "https://api.fda.gov/drug/enforcement.json?search=reason_for_recall:\"cross-contamination\"&limit=1000",
     "api_json", "manufacturing", "200",
     "FDA recalls due to cross-contamination during manufacturing"),

    ("fda_particulate_recalls",
     "https://api.fda.gov/drug/enforcement.json?search=reason_for_recall:\"particulate\"&limit=1000",
     "api_json", "manufacturing", "300",
     "FDA recalls due to visible/sub-visible particulate matter"),

    ("openfda_establishment_registrations",
     "https://api.fda.gov/drug/drugsfda.json?limit=1000",
     "api_json", "manufacturing", "1K",
     "FDA approved drug products with manufacturing info"),

    ("fda_anda_approvals",
     "https://api.fda.gov/drug/drugsfda.json?search=submissions.submission_type:\"ANDA\"&limit=1000",
     "api_json", "manufacturing", "1K",
     "FDA ANDA (generic drug) approvals with manufacturing data"),

    ("fda_nda_approvals",
     "https://api.fda.gov/drug/drugsfda.json?search=submissions.submission_type:\"NDA\"&limit=1000",
     "api_json", "manufacturing", "1K",
     "FDA NDA (new drug) approvals"),

    # =====================================================================
    # RECALLS — Product Recalls, Market Withdrawals, Safety Alerts
    # =====================================================================

    ("fda_all_drug_recalls_2024",
     "https://api.fda.gov/drug/enforcement.json?search=report_date:[20240101+TO+20241231]&limit=1000",
     "api_json", "recalls", "1K",
     "All FDA drug recalls reported in 2024"),

    ("fda_all_drug_recalls_2023",
     "https://api.fda.gov/drug/enforcement.json?search=report_date:[20230101+TO+20231231]&limit=1000",
     "api_json", "recalls", "1K",
     "All FDA drug recalls reported in 2023"),

    ("fda_all_drug_recalls_2022",
     "https://api.fda.gov/drug/enforcement.json?search=report_date:[20220101+TO+20221231]&limit=1000",
     "api_json", "recalls", "1K",
     "All FDA drug recalls reported in 2022"),

    ("fda_voluntary_recalls",
     "https://api.fda.gov/drug/enforcement.json?search=voluntary_mandated:\"Voluntary:+Firm+Initiated\"&limit=1000",
     "api_json", "recalls", "1K",
     "FDA voluntary (firm-initiated) drug recalls"),

    ("fda_mandated_recalls",
     "https://api.fda.gov/drug/enforcement.json?search=voluntary_mandated:\"FDA+Mandated\"&limit=1000",
     "api_json", "recalls", "500",
     "FDA mandated drug recalls"),

    ("fda_device_recalls",
     "https://api.fda.gov/device/recall.json?limit=1000",
     "api_json", "recalls", "1K",
     "FDA medical device recalls"),

    ("fda_food_recalls",
     "https://api.fda.gov/food/enforcement.json?limit=1000",
     "api_json", "recalls", "1K",
     "FDA food/supplement recalls (dietary supplements with drug claims)"),

    # =====================================================================
    # LABELING — Drug Labeling, NDC, Approval Data
    # =====================================================================

    ("fda_prescription_labeling",
     "https://api.fda.gov/drug/label.json?search=openfda.product_type:\"HUMAN+PRESCRIPTION+DRUG\"&limit=1000",
     "api_json", "labeling", "1K",
     "FDA prescription drug labeling (indications, dosing, warnings)"),

    ("fda_otc_labeling",
     "https://api.fda.gov/drug/label.json?search=openfda.product_type:\"HUMAN+OTC+DRUG\"&limit=1000",
     "api_json", "labeling", "1K",
     "FDA OTC drug labeling (Drug Facts)"),

    ("fda_ndc_directory",
     "https://download.open.fda.gov/drug/ndc/drug-ndc-0001-of-0001.json.zip",
     "zip_json", "labeling", "300K",
     "FDA National Drug Code directory — all marketed drugs"),

    ("fda_orange_book_products",
     "https://www.fda.gov/media/76860/download",
     "zip_csv", "labeling", "50K",
     "FDA Orange Book — approved drug products with therapeutic equivalence"),

    ("dailymed_rxterms",
     "https://lhncbc.nlm.nih.gov/RxTerms/RxTerms202401.zip",
     "zip_csv", "labeling", "30K",
     "NLM RxTerms drug terminology 2024"),

    # =====================================================================
    # PHARMACOVIGILANCE — Adverse Events, REMS, Post-Market
    # =====================================================================

    ("fda_adverse_events_recent",
     "https://api.fda.gov/drug/event.json?search=receivedate:[20240101+TO+20241231]&limit=1000",
     "api_json", "pharmacovigilance", "1K",
     "FDA FAERS adverse events reported 2024"),

    ("fda_serious_adverse_events",
     "https://api.fda.gov/drug/event.json?search=serious:1&limit=1000",
     "api_json", "pharmacovigilance", "1K",
     "FDA serious adverse event reports (hospitalization/death)"),

    ("fda_death_adverse_events",
     "https://api.fda.gov/drug/event.json?search=seriousnessdeath:1&limit=1000",
     "api_json", "pharmacovigilance", "1K",
     "FDA adverse event reports resulting in death"),

    ("fda_device_adverse_events",
     "https://api.fda.gov/device/event.json?search=date_received:[20240101+TO+20241231]&limit=1000",
     "api_json", "pharmacovigilance", "1K",
     "FDA medical device adverse events 2024"),

    ("sider_side_effects",
     "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz",
     "tsv_gz", "pharmacovigilance", "310K",
     "SIDER database — drug side effects (MedDRA coded)"),

    ("sider_indications",
     "http://sideeffects.embl.de/media/download/meddra_all_indications.tsv.gz",
     "tsv_gz", "pharmacovigilance", "100K",
     "SIDER database — drug indications"),

    ("sider_frequencies",
     "http://sideeffects.embl.de/media/download/meddra_freq.tsv.gz",
     "tsv_gz", "pharmacovigilance", "170K",
     "SIDER database — side effect frequencies"),

    # =====================================================================
    # REFERENCE — ICH, WHO, Regulatory Standards (metadata/indexes)
    # =====================================================================

    ("ich_guidelines_index",
     "https://raw.githubusercontent.com/nickloman/WHO-EML/master/WHO_EML_2023.csv",
     "reference", "gmp", "ref",
     "ICH Quality Guidelines reference index"),

    ("usp_reference_standards",
     "https://raw.githubusercontent.com/nickloman/WHO-EML/master/WHO_EML_2023.csv",
     "reference", "quality", "ref",
     "USP Reference Standards catalog (reference)"),
]


# ============================================================================
# ICH QUALITY GUIDELINES — Structured as Q&A for training
# These are manually curated from ICH.org public summaries
# ============================================================================

ICH_QUALITY_GUIDELINES = [
    {
        "instruction": "What is ICH Q1A and what does it cover?",
        "output": "ICH Q1A (R2) covers Stability Testing of New Drug Substances and Products. It establishes guidelines for the stability testing protocol including temperature, humidity, and duration requirements for long-term, intermediate, and accelerated storage conditions. Key requirements: long-term studies at 25C/60%RH or 30C/65%RH for 12 months minimum, accelerated at 40C/75%RH for 6 months.",
        "category": "quality"
    },
    {
        "instruction": "What is ICH Q1B guideline about?",
        "output": "ICH Q1B covers Photostability Testing of New Drug Substances and Products. It provides guidance on the photostability testing to demonstrate that light exposure does not result in unacceptable changes. Tests include forced degradation studies and confirmatory studies using a light source with both UV and visible output (Option 1: D65 standard or Option 2: cool white fluorescent and near UV lamp).",
        "category": "quality"
    },
    {
        "instruction": "What does ICH Q2 (R1) address?",
        "output": "ICH Q2 (R1) addresses Validation of Analytical Procedures: Text and Methodology. It defines the characteristics for validation including: Accuracy, Precision (repeatability and intermediate precision), Specificity, Detection Limit (LOD), Quantitation Limit (LOQ), Linearity, and Range. Each analytical procedure (identification, quantitative tests for impurities, assay) has specific validation requirements.",
        "category": "quality"
    },
    {
        "instruction": "What is ICH Q3A and what are the impurity thresholds?",
        "output": "ICH Q3A (R2) covers Impurities in New Drug Substances. Reporting thresholds: 0.05% for max daily dose >=2g, 0.03% for <2g. Identification thresholds: 0.10% for >=2g, 0.05% for <2g. Qualification thresholds: 0.15% for >=2g, 0.05% for <2g. Each impurity above the threshold must be identified, qualified for safety, and controlled in the specification.",
        "category": "quality"
    },
    {
        "instruction": "What does ICH Q3B cover regarding impurities in drug products?",
        "output": "ICH Q3B (R2) covers Impurities in New Drug Products (degradation products). Reporting threshold: 0.1% for max daily dose <=1g, 0.05% for >1g. Identification threshold: 0.2% for <=1g dose, scaling down to 0.05% for >2g. Qualification threshold: 0.2% for <=10mg dose up to 0.2% for >2g. Degradation products must be characterized and their safety qualified.",
        "category": "quality"
    },
    {
        "instruction": "Explain ICH Q3C and residual solvent classification.",
        "output": "ICH Q3C (R8) covers Residual Solvents. Solvents are classified: Class 1 (avoid — known carcinogens: benzene, carbon tetrachloride, 1,2-dichloroethane), Class 2 (limit — non-genotoxic with PDE limits, e.g., methanol PDE 30mg/day, dichloromethane PDE 6mg/day), Class 3 (low toxicity — PDE 50mg/day or more, e.g., acetone, ethanol, ethyl acetate). Testing uses GC headspace methods.",
        "category": "quality"
    },
    {
        "instruction": "What is ICH Q3D and how does it address elemental impurities?",
        "output": "ICH Q3D (R2) covers Elemental Impurities. Elements classified by toxicity and likelihood of occurrence: Class 1 (highly toxic — As, Cd, Hg, Pb — always controlled), Class 2A (high probability — Co, Ni, V — route-specific PDEs), Class 2B (low probability — Ag, Au, Ir, Os, Pd, Pt, Rh, Ru, Se, Tl), Class 3 (minimal toxicity — Ba, Cr, Cu, Li, Mo, Sb, Sn). PDEs differ by route: oral, parenteral, inhalation.",
        "category": "quality"
    },
    {
        "instruction": "What does ICH Q5C cover for biotechnology products?",
        "output": "ICH Q5C covers Quality of Biotechnological Products: Stability Testing. Specific to biotech/biological products: requires real-time, real-condition stability studies. Must test potency, purity, molecular characterization (SDS-PAGE, HPLC), sterility, and container closure integrity. Accelerated studies at elevated temperature to support shelf-life. Degradation pathways (aggregation, deamidation, oxidation) must be characterized.",
        "category": "quality"
    },
    {
        "instruction": "What is ICH Q6A and what are pharmaceutical specifications?",
        "output": "ICH Q6A covers Specifications: Test Procedures and Acceptance Criteria for New Drug Substances and New Drug Products. Universal tests for drug substance: description, identification, assay, impurities. Specific tests: particle size, polymorphic form, water content, residual solvents, microbial limits. Drug product tests: dissolution, uniformity of dosage units, hardness, friability, disintegration, moisture, microbial limits.",
        "category": "quality"
    },
    {
        "instruction": "Explain ICH Q7 GMP for Active Pharmaceutical Ingredients.",
        "output": "ICH Q7 covers Good Manufacturing Practice for Active Pharmaceutical Ingredients. Key areas: Quality Management (quality unit independence, CAPA), Personnel (training, hygiene), Buildings/Facilities (design to prevent contamination), Process Equipment (calibration, cleaning validation), Documentation (batch records, specifications), Materials Management (supplier qualification), Production/In-Process Controls, Packaging/Labeling, Storage/Distribution, Laboratory Controls (OOS investigations), Validation (process, cleaning, method), Change Control, Rejection/Reuse, Complaints/Recalls, Contract Manufacturing.",
        "category": "gmp"
    },
    {
        "instruction": "What is ICH Q8 and the concept of Quality by Design?",
        "output": "ICH Q8 (R2) covers Pharmaceutical Development and introduces Quality by Design (QbD). Key concepts: Quality Target Product Profile (QTPP) — prospective summary of desired quality characteristics. Critical Quality Attributes (CQAs) — properties that must be controlled. Design Space — multidimensional combination of input variables and process parameters demonstrated to provide assurance of quality. Risk Assessment to identify critical parameters. Control Strategy linking CQAs to process controls.",
        "category": "manufacturing"
    },
    {
        "instruction": "What does ICH Q9 cover regarding quality risk management?",
        "output": "ICH Q9 covers Quality Risk Management (QRM). Principles: risk evaluation based on scientific knowledge, level of effort proportional to risk level. Process: Risk Assessment (identification, analysis, evaluation using FMEA, HACCP, FTA), Risk Control (reduction and acceptance), Risk Review (ongoing monitoring). Tools: FMEA (Failure Mode Effects Analysis), FTA (Fault Tree Analysis), HACCP (Hazard Analysis Critical Control Points), Risk Ranking and Filtering. Applications: facility design, supplier management, change control, deviations.",
        "category": "quality"
    },
    {
        "instruction": "What is ICH Q10 and the Pharmaceutical Quality System?",
        "output": "ICH Q10 covers the Pharmaceutical Quality System (PQS). Applies throughout product lifecycle: development, technology transfer, commercial manufacturing, product discontinuation. Elements: Process Performance and Product Quality Monitoring, CAPA (Corrective and Preventive Action), Change Management, Management Review. Enablers: Knowledge Management and Quality Risk Management. Builds on GMP (ICH Q7), Quality by Design (Q8), and QRM (Q9). Senior management responsibility for quality policy and resource allocation.",
        "category": "gmp"
    },
    {
        "instruction": "What does ICH Q11 cover for drug substance development?",
        "output": "ICH Q11 covers Development and Manufacture of Drug Substances (Chemical and Biotechnological/Biological Entities). Key topics: selection of starting materials and source materials for biotech, process development approaches (traditional vs enhanced/QbD), definition of design space for manufacturing process, control strategy for drug substance CQAs, process validation/verification lifecycle approach, submission of manufacturing process information in CTD.",
        "category": "manufacturing"
    },
    {
        "instruction": "What is ICH Q12 and lifecycle management?",
        "output": "ICH Q12 covers Technical and Regulatory Considerations for Pharmaceutical Product Lifecycle Management. Key concepts: Established Conditions (ECs) — elements in the dossier which if changed require regulatory action. PACMP (Post-Approval Change Management Protocol) — pre-agreed protocols for changes. Product Lifecycle Management (PLCM) document — living document. Operational Flexibility — ability to make changes within approved ranges without prior approval. Structured approach to post-approval changes based on risk.",
        "category": "gmp"
    },
    {
        "instruction": "What is ICH Q13 about continuous manufacturing?",
        "output": "ICH Q13 covers Continuous Manufacturing of Drug Substances and Drug Products. Key concepts: system dynamics and process control strategy for continuous processes, residence time distribution (RTD) characterization, material traceability to define batch/lot, disturbance management and diversion strategy, real-time release testing (RTRT) using PAT (Process Analytical Technology), startup/shutdown procedures, integration of drug substance and drug product manufacturing.",
        "category": "manufacturing"
    },
    {
        "instruction": "What is ICH Q14 and analytical procedure development?",
        "output": "ICH Q14 covers Analytical Procedure Development. Introduces the Analytical Target Profile (ATP) — predefined set of requirements for the analytical procedure. Concepts: minimal vs enhanced approaches to development, method operable design region (MODR) similar to design space, risk-based assessment of method performance, lifecycle approach to method management, knowledge-rich vs minimal submissions, reporting of analytical procedure development information in CTD Module 3.",
        "category": "quality"
    },
    {
        "instruction": "What are FDA cGMP requirements under 21 CFR Parts 210 and 211?",
        "output": "21 CFR 210/211 — Current Good Manufacturing Practice for finished pharmaceuticals. Part 210: General provisions. Part 211 covers: Subpart B (Organization and Personnel — quality control unit, qualifications), Subpart C (Buildings and Facilities — design, lighting, ventilation, plumbing), Subpart D (Equipment — maintenance, calibration), Subpart E (Control of Components and Containers), Subpart F (Production and Process Controls), Subpart G (Packaging and Labeling), Subpart H (Holding and Distribution), Subpart I (Laboratory Controls), Subpart J (Records and Reports), Subpart K (Returned and Salvaged Products).",
        "category": "gmp"
    },
    {
        "instruction": "What is FDA 21 CFR Part 11 about?",
        "output": "21 CFR Part 11 covers Electronic Records; Electronic Signatures. Requirements for: electronic records to be trustworthy, reliable, and equivalent to paper records. Key requirements: validated systems, audit trails (who, what, when), copy-able and human-readable records, record protection and retention, limited system access, authority checks, device checks, unique user IDs, electronic signatures (at least two distinct identification components), signature manifestations linked to records. Applies to all FDA-regulated electronic records.",
        "category": "gmp"
    },
    {
        "instruction": "What is an FDA Form 483 and what happens during an inspection?",
        "output": "FDA Form 483 documents inspectional observations — conditions or practices that may violate the FD&C Act. Process: FDA investigators inspect facilities (announced or unannounced), review batch records, SOPs, training records, deviation reports, CAPA systems, lab data, cleaning validation. At closeout, 483 observations are presented to management. Company must respond within 15 business days with corrective actions. Repeat or serious observations can lead to Warning Letters, consent decrees, injunctions, or facility shutdown. Common 483 citations: inadequate investigations, failure to follow procedures, inadequate cleaning validation, data integrity issues.",
        "category": "inspection"
    },
    {
        "instruction": "What is pharmaceutical data integrity and ALCOA+ principles?",
        "output": "Data Integrity ensures data is complete, consistent, and accurate throughout the data lifecycle. ALCOA+ principles: Attributable (who performed/recorded), Legible (readable, permanent), Contemporaneous (recorded at time of activity), Original (first capture or certified copy), Accurate (no errors, editing documented). Plus: Complete (all data including reprocessed), Consistent (timestamp sequence logical), Enduring (recorded permanently), Available (accessible for review throughout retention period). Violations are top FDA/EMA 483 observations. Requires audit trails, backup, access controls, and periodic review.",
        "category": "quality"
    },
    {
        "instruction": "What is process validation in pharmaceutical manufacturing?",
        "output": "Process Validation (FDA 2011 Guidance) — lifecycle approach in 3 stages: Stage 1 (Process Design): define commercial process based on development data, identify CQAs and CPPs, establish control strategy. Stage 2 (Process Qualification): facility/equipment qualification (IQ/OQ/PQ), process performance qualification (PPQ) — typically 3 consecutive conforming batches with enhanced sampling. Stage 3 (Continued Process Verification): ongoing monitoring using statistical process control (SPC), trending of CQAs, periodic review. Each stage is documented and approved by the quality unit.",
        "category": "manufacturing"
    },
    {
        "instruction": "What is cleaning validation in pharmaceutical manufacturing?",
        "output": "Cleaning Validation demonstrates that cleaning procedures effectively remove product residues, cleaning agents, and microbial contamination to predetermined acceptable levels. Key elements: worst-case product (lowest therapeutic dose, hardest to clean), equipment grouping, sampling methods (swab and rinse), acceptance criteria based on 1/1000th dose or 10 ppm, visual inspection. Requires: validated analytical methods for residue detection, recovery studies, cleaning procedure documentation, revalidation triggers (new products, equipment changes). Hold time studies for dirty and clean equipment.",
        "category": "manufacturing"
    },
    {
        "instruction": "What is a pharmaceutical CAPA system?",
        "output": "CAPA (Corrective and Preventive Action) is a systematic approach to investigating, correcting, and preventing quality issues. Corrective Action: eliminates the cause of an existing nonconformity. Preventive Action: eliminates the cause of a potential nonconformity. Process: 1) Identify issue from complaints, deviations, OOS, audits, 2) Root cause analysis (Ishikawa, 5-Why, FTA), 3) Determine corrective/preventive actions, 4) Implement actions with responsible parties and timelines, 5) Verify effectiveness, 6) Close and document. CAPA effectiveness must be measured. Trending of CAPAs reviewed by management. Required by FDA 21 CFR 211, ICH Q10, and ISO 13485.",
        "category": "quality"
    },
    {
        "instruction": "What are pharmaceutical Out-of-Specification (OOS) investigation requirements?",
        "output": "OOS Investigation (FDA 2006 Guidance): Phase 1 (Laboratory Investigation): within 3 days, analyst and supervisor review calculation, method, equipment, sample integrity. If lab error confirmed, result invalidated and documented. If no lab error found: Phase 2 (Full-Scale Investigation): production review, batch record examination, process parameter review, other batch testing, additional lab testing (re-sampling allowed with justification). Requires: written SOP, timely investigation (30-day target), root cause determination, CAPA, QA approval to release/reject. Retesting rules: original OOS cannot be averaged with passing results.",
        "category": "quality"
    },
]


# ============================================================================
# DOWNLOAD & PROCESSING LOGIC
# ============================================================================

def download_source(name, url, fmt, dest):
    """Download a pharma regulation data source"""
    if fmt in ("skip", "reference"):
        logger.info(f"  [SKIP] {name} — {'requires manual access' if fmt == 'skip' else 'reference material'}")
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning(f"  [RETRY] {name} — trying without custom headers: {e}")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

    content = resp.content
    data = []

    if fmt == "api_json":
        raw = json.loads(content)
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict):
            if "results" in raw:
                data = raw["results"]
            elif "result" in raw:
                data = raw["result"] if isinstance(raw["result"], list) else [raw["result"]]
            elif "data" in raw:
                data = raw["data"]
            else:
                data = [raw]

    elif fmt == "csv":
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        data = [dict(row) for row in reader]

    elif fmt == "tsv_gz":
        import gzip
        text = gzip.decompress(content).decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        data = [dict(row) for row in reader]

    elif fmt == "zip_json":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name_in_zip in zf.namelist():
                if name_in_zip.endswith(".json"):
                    with zf.open(name_in_zip) as f:
                        raw = json.load(f)
                        data = raw.get("results", raw) if isinstance(raw, dict) else raw
                    break

    elif fmt == "zip_csv":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name_in_zip in zf.namelist():
                if name_in_zip.endswith(".csv") or name_in_zip.endswith(".txt"):
                    with zf.open(name_in_zip) as f:
                        text = f.read().decode("utf-8", errors="replace")
                        reader = csv.DictReader(io.StringIO(text))
                        data.extend(dict(row) for row in reader)
                    break

    if data:
        with open(dest, "w") as f:
            json.dump(data, f, default=str)
    return data


def to_pharma_qa_format(item, source_name, category):
    """Convert pharma regulation data to instruction/output format"""
    instruction = ""
    output = ""

    # FDA enforcement/recall format
    if "reason_for_recall" in item:
        product = item.get("product_description", "unknown product")[:150]
        reason = item.get("reason_for_recall", "")
        classification = item.get("classification", "")
        firm = item.get("recalling_firm", "")
        status = item.get("status", "")
        voluntary = item.get("voluntary_mandated", "")
        dist = item.get("distribution_pattern", "")

        instruction = f"What is the FDA recall/enforcement information for: {product}?"
        parts = [f"Recalling firm: {firm}"] if firm else []
        if reason:
            parts.append(f"Reason: {reason}")
        if classification:
            parts.append(f"Classification: {classification}")
        if status:
            parts.append(f"Status: {status}")
        if voluntary:
            parts.append(f"Type: {voluntary}")
        if dist:
            parts.append(f"Distribution: {dist}")
        output = "\n".join(parts)

    # FDA drug labeling format
    elif "indications_and_usage" in item or "warnings" in item or "dosage_and_administration" in item:
        brand = ""
        generic = ""
        if "openfda" in item and isinstance(item["openfda"], dict):
            brands = item["openfda"].get("brand_name", [])
            brand = brands[0] if brands else ""
            generics = item["openfda"].get("generic_name", [])
            generic = generics[0] if generics else ""

        name = brand or generic or "this drug"

        def _first(val):
            if isinstance(val, list):
                return val[0] if val else ""
            return str(val) if val else ""

        indications = _first(item.get("indications_and_usage", ""))
        warnings = _first(item.get("warnings", ""))
        dosage = _first(item.get("dosage_and_administration", ""))
        contraindications = _first(item.get("contraindications", ""))
        storage = _first(item.get("storage_and_handling", item.get("how_supplied", "")))

        if name and (indications or warnings):
            instruction = f"What are the regulatory labeling details for {name}?"
            parts = []
            if indications:
                parts.append(f"Indications: {indications[:500]}")
            if dosage:
                parts.append(f"Dosage & Administration: {dosage[:500]}")
            if warnings:
                parts.append(f"Warnings: {warnings[:500]}")
            if contraindications:
                parts.append(f"Contraindications: {contraindications[:300]}")
            if storage:
                parts.append(f"Storage: {storage[:200]}")
            output = "\n\n".join(parts)

    # FDA adverse event format
    elif "patient" in item and isinstance(item.get("patient"), dict):
        drugs = item.get("patient", {}).get("drug", [])
        reactions = item.get("patient", {}).get("reaction", [])
        if drugs and reactions:
            drug_names = [d.get("medicinalproduct", "") for d in drugs[:5] if d.get("medicinalproduct")]
            reaction_terms = [r.get("reactionmeddrapt", "") for r in reactions if r.get("reactionmeddrapt")]
            if drug_names and reaction_terms:
                instruction = f"What adverse events have been reported for {', '.join(drug_names[:3])}?"
                parts = [f"Reported reactions: {', '.join(reaction_terms)}"]
                serious = item.get("serious", "")
                if str(serious) == "1":
                    parts.append("Classification: Serious adverse event")
                outcome = item.get("patient", {}).get("patientonsetage", "")
                if outcome:
                    parts.append(f"Patient onset age: {outcome}")
                output = "\n".join(parts)

    # FDA DrugsFDA format (approvals)
    elif "products" in item or "submissions" in item:
        products = item.get("products", [])
        submissions = item.get("submissions", [])
        sponsor = item.get("sponsor_name", "")
        app_no = item.get("application_number", "")
        if products:
            prod = products[0]
            brand = prod.get("brand_name", "")
            active = prod.get("active_ingredients", [])
            active_str = ", ".join([a.get("name", "") for a in active]) if active else ""
            dosage_form = prod.get("dosage_form", "")
            route = prod.get("route", "")
            instruction = f"What is the FDA approval information for {brand or app_no}?"
            parts = []
            if brand:
                parts.append(f"Brand name: {brand}")
            if active_str:
                parts.append(f"Active ingredient(s): {active_str}")
            if dosage_form:
                parts.append(f"Dosage form: {dosage_form}")
            if route:
                parts.append(f"Route: {route}")
            if sponsor:
                parts.append(f"Sponsor: {sponsor}")
            if submissions:
                sub = submissions[0]
                parts.append(f"Submission type: {sub.get('submission_type', '')} {sub.get('submission_number', '')}")
            output = "\n".join(parts)

    # SIDER format
    elif len(item) <= 6 and any(k for k in item.keys() if k.startswith(("0", "1", "2", "3", "4", "5"))):
        vals = list(item.values())
        if len(vals) >= 5:
            instruction = f"What are the known side effects for drug {vals[0]}?"
            output = f"Side effect: {vals[-1]} (MedDRA term)"

    # Generic fallback
    if not instruction:
        text_fields = {k: v for k, v in item.items()
                       if isinstance(v, str) and len(str(v)) > 15}
        if len(text_fields) >= 2:
            keys = list(text_fields.keys())
            instruction = f"Describe the following pharmaceutical regulatory information: {str(text_fields[keys[0]])[:200]}"
            output = str(text_fields[keys[1]])[:500]

    if not instruction or not output:
        return None

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "source": source_name,
        "adapter": "regulatory_qa",
        "category": category,
    }


def download_all(filter_category=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "skip": 0, "fail": 0, "total_examples": 0}

    # --- API/URL sources ---
    sources = PHARMA_SOURCES
    if filter_category:
        sources = [s for s in PHARMA_SOURCES if s[3] == filter_category]
        logger.info(f"Filtering to category: {filter_category} ({len(sources)} sources)")

    logger.info(f"\n{'='*60}")
    logger.info(f"PHARMA REGULATION SOURCES ({len(sources)} API/URL sources)")
    logger.info(f"{'='*60}")

    for name, url, fmt, category, est_size, desc in sources:
        dest = RAW_DIR / f"{name}.json"
        if dest.exists():
            logger.info(f"[SKIP] {name} ({est_size}) — already exists")
            stats["skip"] += 1
            continue
        try:
            logger.info(f"[DL] {name} ({est_size}) <- {url[:70]}...")
            data = download_source(name, url, fmt, dest)
            if data is None:
                stats["skip"] += 1
                continue
            logger.info(f"  -> {len(data):,} raw records")
            stats["ok"] += 1

            proc_path = PROCESSED_DIR / f"{name}.json"
            processed = []
            for item in data:
                if isinstance(item, dict):
                    result = to_pharma_qa_format(item, name, category)
                    if result:
                        processed.append(result)
            if processed:
                with open(proc_path, "w") as f:
                    json.dump(processed, f, default=str)
                logger.info(f"  -> {len(processed):,} processed -> regulatory_qa/{name}.json")
                stats["total_examples"] += len(processed)

        except Exception as e:
            logger.error(f"  X FAILED {name}: {e}")
            stats["fail"] += 1

    # --- ICH curated guidelines ---
    if filter_category in (None, "gmp", "quality", "manufacturing"):
        logger.info(f"\n{'='*60}")
        logger.info(f"ICH QUALITY GUIDELINES ({len(ICH_QUALITY_GUIDELINES)} curated Q&A)")
        logger.info(f"{'='*60}")

        ich_path = PROCESSED_DIR / "ich_quality_guidelines.json"
        if ich_path.exists():
            logger.info("[SKIP] ich_quality_guidelines — already exists")
            stats["skip"] += 1
        else:
            ich_processed = []
            for item in ICH_QUALITY_GUIDELINES:
                ich_processed.append({
                    "instruction": item["instruction"],
                    "input": "",
                    "output": item["output"],
                    "source": "ich_guidelines_curated",
                    "adapter": "regulatory_qa",
                    "category": item["category"],
                })
            with open(ich_path, "w") as f:
                json.dump(ich_processed, f)
            logger.info(f"  -> {len(ich_processed)} ICH Q&A pairs -> regulatory_qa/ich_quality_guidelines.json")
            stats["ok"] += 1
            stats["total_examples"] += len(ich_processed)

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info("PHARMA REGULATIONS DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Downloaded:  {stats['ok']}")
    logger.info(f"  Skipped:     {stats['skip']}")
    logger.info(f"  Failed:      {stats['fail']}")
    logger.info(f"  Total processed: {stats['total_examples']:,}")

    categories = {}
    if PROCESSED_DIR.exists():
        for f in PROCESSED_DIR.glob("*.json"):
            try:
                with open(f) as fp:
                    items = json.load(fp)
                    if items and isinstance(items[0], dict):
                        cat = items[0].get("category", "unknown")
                        categories[cat] = categories.get(cat, 0) + len(items)
            except Exception:
                pass
    logger.info("\nBy category:")
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat:25s} {count:>10,} examples")
    logger.info(f"{'='*60}")


def list_sources():
    """Print all available pharma regulation data sources"""
    print(f"\n{'='*70}")
    print("PHARMACEUTICAL REGULATION DATA SOURCES")
    print(f"{'='*70}")
    print(f"{'#':>3}  {'Name':35s}  {'Category':20s}  {'Size':>6s}  Description")
    print("-" * 110)
    for i, (name, url, fmt, category, est_size, desc) in enumerate(PHARMA_SOURCES, 1):
        skip = " [manual]" if fmt in ("skip", "reference") else ""
        print(f"{i:>3}  {name:35s}  {category:20s}  {est_size:>6s}  {desc}{skip}")
    print(f"\n  + {len(ICH_QUALITY_GUIDELINES)} curated ICH Q-series guideline Q&A pairs")
    print(f"\nTotal: {len(PHARMA_SOURCES)} API sources + {len(ICH_QUALITY_GUIDELINES)} ICH Q&A")

    cats = {}
    for _, _, _, cat, _, _ in PHARMA_SOURCES:
        cats[cat] = cats.get(cat, 0) + 1
    print("\nBy category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat:20s} {count} sources")


def main():
    parser = argparse.ArgumentParser(
        description="Download pharmaceutical regulations, QA/QC, manufacturing & inspection data")
    parser.add_argument("--category", type=str, default=None,
                        choices=["gmp", "inspection", "quality", "manufacturing",
                                 "recalls", "labeling", "pharmacovigilance"],
                        help="Filter by category")
    parser.add_argument("--list", action="store_true",
                        help="List all available sources without downloading")
    args = parser.parse_args()

    if args.list:
        list_sources()
    else:
        download_all(filter_category=args.category)


if __name__ == "__main__":
    main()
