"""
UMI Medical Knowledge Service
Disease information, clinical guidelines, and research paper access
"""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import NotFoundError
from src.core.logging import get_logger
from src.models.medical import Disease, ClinicalGuideline, ResearchPaper

logger = get_logger(__name__)


class MedicalKnowledgeService:
    """Service for accessing medical knowledge base."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # =========================================================================
    # Disease Information
    # =========================================================================
    
    async def search_diseases(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Disease]:
        """Search for diseases by name or symptoms."""
        search_term = f"%{query.lower()}%"
        
        result = await self.db.execute(
            select(Disease)
            .where(
                or_(
                    func.lower(Disease.name).like(search_term),
                    func.lower(Disease.icd_code).like(search_term),
                    Disease.symptoms.cast(String).ilike(search_term),
                )
            )
            .limit(limit)
        )
        
        return list(result.scalars().all())
    
    async def get_disease(self, disease_id: uuid.UUID) -> Disease:
        """Get a disease by ID."""
        result = await self.db.execute(
            select(Disease).where(Disease.id == disease_id)
        )
        disease = result.scalar_one_or_none()
        
        if not disease:
            raise NotFoundError("Disease", disease_id)
        
        return disease
    
    async def get_disease_by_icd(self, icd_code: str) -> Optional[Disease]:
        """Get a disease by ICD code."""
        result = await self.db.execute(
            select(Disease).where(Disease.icd_code == icd_code)
        )
        return result.scalar_one_or_none()
    
    async def get_diseases_by_symptom(
        self,
        symptom: str,
        limit: int = 10,
    ) -> List[Disease]:
        """Find diseases that match a symptom."""
        search_term = f"%{symptom.lower()}%"
        
        result = await self.db.execute(
            select(Disease)
            .where(Disease.symptoms.cast(String).ilike(search_term))
            .limit(limit)
        )
        
        return list(result.scalars().all())
    
    async def get_differential_diagnosis(
        self,
        symptoms: List[str],
        age: Optional[int] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generate differential diagnosis based on symptoms.
        
        Args:
            symptoms: List of symptoms
            age: Patient age
            limit: Maximum diagnoses to return
        
        Returns:
            List of possible diagnoses with scores
        """
        # Score diseases by symptom match
        all_diseases = []
        
        for symptom in symptoms:
            diseases = await self.get_diseases_by_symptom(symptom, limit=20)
            all_diseases.extend(diseases)
        
        # Count occurrences (more symptom matches = higher score)
        disease_scores: Dict[uuid.UUID, Dict[str, Any]] = {}
        
        for disease in all_diseases:
            if disease.id not in disease_scores:
                disease_scores[disease.id] = {
                    "disease": disease,
                    "score": 0,
                    "matched_symptoms": [],
                }
            
            disease_scores[disease.id]["score"] += 1
        
        # Sort by score
        sorted_diseases = sorted(
            disease_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:limit]
        
        # Format results
        results = []
        for item in sorted_diseases:
            disease = item["disease"]
            results.append({
                "disease_id": str(disease.id),
                "name": disease.name,
                "icd_code": disease.icd_code,
                "score": item["score"] / len(symptoms),  # Normalize
                "symptoms": disease.symptoms[:5] if disease.symptoms else [],
                "is_emergency": disease.is_emergency,
                "requires_specialist": disease.requires_specialist,
            })
        
        return results
    
    # =========================================================================
    # Clinical Guidelines
    # =========================================================================
    
    async def search_guidelines(
        self,
        query: str,
        guideline_body: Optional[str] = None,
        limit: int = 20,
    ) -> List[ClinicalGuideline]:
        """Search for clinical guidelines."""
        search_term = f"%{query.lower()}%"
        
        stmt = select(ClinicalGuideline).where(
            or_(
                func.lower(ClinicalGuideline.title).like(search_term),
                func.lower(ClinicalGuideline.condition).like(search_term),
            )
        )
        
        if guideline_body:
            stmt = stmt.where(ClinicalGuideline.guideline_body == guideline_body)
        
        stmt = stmt.where(ClinicalGuideline.is_current == True).limit(limit)
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def get_guideline(self, guideline_id: uuid.UUID) -> ClinicalGuideline:
        """Get a guideline by ID."""
        result = await self.db.execute(
            select(ClinicalGuideline).where(ClinicalGuideline.id == guideline_id)
        )
        guideline = result.scalar_one_or_none()
        
        if not guideline:
            raise NotFoundError("ClinicalGuideline", guideline_id)
        
        return guideline
    
    async def get_guidelines_for_condition(
        self,
        condition: str,
        country: Optional[str] = None,
    ) -> List[ClinicalGuideline]:
        """Get guidelines for a specific condition."""
        search_term = f"%{condition.lower()}%"
        
        stmt = select(ClinicalGuideline).where(
            func.lower(ClinicalGuideline.condition).like(search_term),
            ClinicalGuideline.is_current == True,
        )
        
        if country:
            stmt = stmt.where(ClinicalGuideline.country == country)
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    # =========================================================================
    # Research Papers
    # =========================================================================
    
    async def search_papers(
        self,
        query: str,
        publication_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[ResearchPaper]:
        """Search for research papers."""
        search_term = f"%{query.lower()}%"
        
        stmt = select(ResearchPaper).where(
            or_(
                func.lower(ResearchPaper.title).like(search_term),
                func.lower(ResearchPaper.abstract).like(search_term),
                ResearchPaper.mesh_terms.cast(String).ilike(search_term),
            )
        )
        
        if publication_type:
            stmt = stmt.where(ResearchPaper.publication_type == publication_type)
        
        stmt = stmt.order_by(ResearchPaper.publication_date.desc()).limit(limit)
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def get_paper(self, paper_id: uuid.UUID) -> ResearchPaper:
        """Get a paper by ID."""
        result = await self.db.execute(
            select(ResearchPaper).where(ResearchPaper.id == paper_id)
        )
        paper = result.scalar_one_or_none()
        
        if not paper:
            raise NotFoundError("ResearchPaper", paper_id)
        
        return paper
    
    async def get_paper_by_pmid(self, pmid: str) -> Optional[ResearchPaper]:
        """Get a paper by PubMed ID."""
        result = await self.db.execute(
            select(ResearchPaper).where(ResearchPaper.pmid == pmid)
        )
        return result.scalar_one_or_none()
    
    async def get_related_papers(
        self,
        paper_id: uuid.UUID,
        limit: int = 5,
    ) -> List[ResearchPaper]:
        """Get papers related to a given paper (by MeSH terms)."""
        paper = await self.get_paper(paper_id)
        
        if not paper.mesh_terms:
            return []
        
        # Find papers with overlapping MeSH terms
        related = []
        for term in paper.mesh_terms[:3]:  # Use top 3 terms
            result = await self.db.execute(
                select(ResearchPaper)
                .where(ResearchPaper.id != paper_id)
                .where(ResearchPaper.mesh_terms.cast(String).ilike(f"%{term}%"))
                .limit(5)
            )
            related.extend(result.scalars().all())
        
        # Deduplicate
        seen = set()
        unique = []
        for p in related:
            if p.id not in seen:
                seen.add(p.id)
                unique.append(p)
        
        return unique[:limit]


# Import String for type casting
from sqlalchemy import String
