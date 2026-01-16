"""
UMI Medical Knowledge Models
Disease, Drug, and Clinical Guideline models
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base


class Disease(Base):
    """Disease/condition information from medical knowledge base."""
    
    __tablename__ = "diseases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Identification
    icd_code: Mapped[str] = mapped_column(String(20), unique=True, index=True)  # ICD-11
    snomed_code: Mapped[Optional[str]] = mapped_column(String(50), index=True)  # SNOMED CT
    
    # Basic info
    name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Classification
    category: Mapped[Optional[str]] = mapped_column(String(200))  # e.g., "Infectious", "Cardiovascular"
    subcategory: Mapped[Optional[str]] = mapped_column(String(200))
    body_system: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Clinical info (stored as JSON arrays)
    symptoms: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    causes: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    risk_factors: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    complications: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    
    # Diagnosis
    diagnostic_criteria: Mapped[Optional[str]] = mapped_column(Text)
    diagnostic_tests: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    
    # Treatment
    treatment_overview: Mapped[Optional[str]] = mapped_column(Text)
    first_line_treatments: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    
    # Epidemiology
    prevalence: Mapped[Optional[str]] = mapped_column(String(200))
    incidence: Mapped[Optional[str]] = mapped_column(String(200))
    affected_demographics: Mapped[Optional[str]] = mapped_column(Text)
    
    # Prognosis
    prognosis: Mapped[Optional[str]] = mapped_column(Text)
    mortality_rate: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Flags
    is_chronic: Mapped[bool] = mapped_column(Boolean, default=False)
    is_infectious: Mapped[bool] = mapped_column(Boolean, default=False)
    is_emergency: Mapped[bool] = mapped_column(Boolean, default=False)
    requires_specialist: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Metadata
    source: Mapped[Optional[str]] = mapped_column(String(200))  # Data source
    last_reviewed: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return f"<Disease {self.name} ({self.icd_code})>"


class DrugClass(str, Enum):
    """Drug classification types."""
    OTC = "otc"  # Over-the-counter
    PRESCRIPTION = "prescription"
    CONTROLLED = "controlled"
    INVESTIGATIONAL = "investigational"


class Drug(Base):
    """Drug/medication information."""
    
    __tablename__ = "drugs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Identification
    rxnorm_id: Mapped[Optional[str]] = mapped_column(String(50), unique=True, index=True)
    drugbank_id: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    atc_code: Mapped[Optional[str]] = mapped_column(String(20), index=True)  # ATC classification
    
    # Names
    name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)  # Brand name
    generic_name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    synonyms: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    
    # Classification
    drug_class: Mapped[DrugClass] = mapped_column(
        SQLEnum(DrugClass),
        default=DrugClass.PRESCRIPTION,
    )
    therapeutic_class: Mapped[Optional[str]] = mapped_column(String(200))
    pharmacological_class: Mapped[Optional[str]] = mapped_column(String(200))
    
    # Description
    description: Mapped[Optional[str]] = mapped_column(Text)
    mechanism_of_action: Mapped[Optional[str]] = mapped_column(Text)
    
    # Indications & Usage
    indications: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    off_label_uses: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    
    # Dosing
    dosage_forms: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB, default=list)
    # Structure: [{"form": "tablet", "strengths": ["100mg", "200mg"]}]
    
    standard_dosage: Mapped[Optional[str]] = mapped_column(Text)
    max_daily_dose: Mapped[Optional[str]] = mapped_column(String(200))
    
    # Administration
    routes: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)  # oral, IV, topical, etc.
    administration_instructions: Mapped[Optional[str]] = mapped_column(Text)
    
    # Safety
    contraindications: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    warnings: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    precautions: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    black_box_warning: Mapped[Optional[str]] = mapped_column(Text)
    
    # Side Effects
    common_side_effects: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    serious_side_effects: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    
    # Special Populations
    pregnancy_category: Mapped[Optional[str]] = mapped_column(String(50))
    pregnancy_warnings: Mapped[Optional[str]] = mapped_column(Text)
    lactation_warnings: Mapped[Optional[str]] = mapped_column(Text)
    pediatric_use: Mapped[Optional[str]] = mapped_column(Text)
    geriatric_use: Mapped[Optional[str]] = mapped_column(Text)
    renal_impairment: Mapped[Optional[str]] = mapped_column(Text)
    hepatic_impairment: Mapped[Optional[str]] = mapped_column(Text)
    
    # Pharmacokinetics
    absorption: Mapped[Optional[str]] = mapped_column(Text)
    distribution: Mapped[Optional[str]] = mapped_column(Text)
    metabolism: Mapped[Optional[str]] = mapped_column(Text)
    elimination: Mapped[Optional[str]] = mapped_column(Text)
    half_life: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Storage
    storage_conditions: Mapped[Optional[str]] = mapped_column(Text)
    
    # Regulatory
    fda_approval_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    mhra_approval_status: Mapped[Optional[str]] = mapped_column(String(100))
    uae_approval_status: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Metadata
    source: Mapped[Optional[str]] = mapped_column(String(200))
    last_reviewed: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    interactions_as_drug1: Mapped[List["DrugInteraction"]] = relationship(
        "DrugInteraction",
        foreign_keys="DrugInteraction.drug_id_1",
        back_populates="drug1",
        cascade="all, delete-orphan",
    )
    interactions_as_drug2: Mapped[List["DrugInteraction"]] = relationship(
        "DrugInteraction",
        foreign_keys="DrugInteraction.drug_id_2",
        back_populates="drug2",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Drug {self.generic_name}>"


class InteractionSeverity(str, Enum):
    """Drug interaction severity levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONTRAINDICATED = "contraindicated"


class DrugInteraction(Base):
    """Drug-drug interactions."""
    
    __tablename__ = "drug_interactions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    drug_id_1: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("drugs.id", ondelete="CASCADE"),
        nullable=False,
    )
    drug_id_2: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("drugs.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    severity: Mapped[InteractionSeverity] = mapped_column(
        SQLEnum(InteractionSeverity),
        nullable=False,
    )
    
    description: Mapped[str] = mapped_column(Text, nullable=False)
    mechanism: Mapped[Optional[str]] = mapped_column(Text)
    clinical_effects: Mapped[Optional[str]] = mapped_column(Text)
    management: Mapped[Optional[str]] = mapped_column(Text)  # How to manage the interaction
    
    # Evidence
    evidence_level: Mapped[Optional[str]] = mapped_column(String(50))  # established, theoretical, etc.
    references: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    
    source: Mapped[Optional[str]] = mapped_column(String(200))
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    drug1: Mapped["Drug"] = relationship(
        "Drug",
        foreign_keys=[drug_id_1],
        back_populates="interactions_as_drug1",
    )
    drug2: Mapped["Drug"] = relationship(
        "Drug",
        foreign_keys=[drug_id_2],
        back_populates="interactions_as_drug2",
    )

    def __repr__(self) -> str:
        return f"<DrugInteraction {self.drug_id_1} <-> {self.drug_id_2} ({self.severity})>"


class ClinicalGuideline(Base):
    """Clinical practice guidelines."""
    
    __tablename__ = "clinical_guidelines"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Identification
    guideline_id: Mapped[Optional[str]] = mapped_column(String(100), unique=True)
    
    # Basic info
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    condition: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Source
    guideline_body: Mapped[str] = mapped_column(String(200), nullable=False)  # NICE, WHO, etc.
    country: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Content
    summary: Mapped[Optional[str]] = mapped_column(Text)
    recommendations: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB, default=list)
    # Structure: [{"recommendation": "...", "strength": "strong/weak", "evidence": "high/moderate/low"}]
    
    # Applicability
    target_population: Mapped[Optional[str]] = mapped_column(Text)
    exclusions: Mapped[Optional[str]] = mapped_column(Text)
    
    # Version
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    published_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_updated: Mapped[Optional[datetime]] = mapped_column(DateTime)
    review_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Links
    source_url: Mapped[Optional[str]] = mapped_column(String(500))
    pdf_url: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Status
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return f"<ClinicalGuideline {self.title}>"


class ResearchPaper(Base):
    """Medical research papers from PubMed and other sources."""
    
    __tablename__ = "research_papers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Identification
    pmid: Mapped[Optional[str]] = mapped_column(String(50), unique=True, index=True)  # PubMed ID
    doi: Mapped[Optional[str]] = mapped_column(String(200), index=True)
    pmc_id: Mapped[Optional[str]] = mapped_column(String(50))  # PubMed Central ID
    
    # Basic info
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    
    # Authors
    authors: Mapped[Optional[List[Dict[str, str]]]] = mapped_column(JSONB, default=list)
    # Structure: [{"name": "...", "affiliation": "..."}]
    
    # Publication
    journal: Mapped[Optional[str]] = mapped_column(String(500))
    publication_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    volume: Mapped[Optional[str]] = mapped_column(String(50))
    issue: Mapped[Optional[str]] = mapped_column(String(50))
    pages: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Classification
    mesh_terms: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    keywords: Mapped[Optional[List[str]]] = mapped_column(JSONB, default=list)
    publication_type: Mapped[Optional[str]] = mapped_column(String(100))  # review, clinical trial, etc.
    
    # Full text
    full_text_available: Mapped[bool] = mapped_column(Boolean, default=False)
    full_text_url: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Embedding for RAG
    embedding_id: Mapped[Optional[str]] = mapped_column(String(100))  # Reference to vector DB
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return f"<ResearchPaper {self.pmid}: {self.title[:50]}...>"
