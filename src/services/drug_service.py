"""
UMI Drug Service
Drug information, interactions, and recommendations
"""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import NotFoundError
from src.core.logging import get_logger
from src.models.medical import Drug, DrugInteraction, InteractionSeverity

logger = get_logger(__name__)


class DrugService:
    """Service for drug information and interaction checking."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def search_drugs(
        self,
        query: str,
        limit: int = 20,
        otc_only: bool = False,
    ) -> List[Drug]:
        """
        Search for drugs by name.
        
        Args:
            query: Search query
            limit: Maximum results
            otc_only: Filter to OTC drugs only
        
        Returns:
            List of matching drugs
        """
        search_term = f"%{query.lower()}%"
        
        stmt = select(Drug).where(
            or_(
                func.lower(Drug.name).like(search_term),
                func.lower(Drug.generic_name).like(search_term),
            )
        )
        
        if otc_only:
            from src.models.medical import DrugClass
            stmt = stmt.where(Drug.drug_class == DrugClass.OTC)
        
        stmt = stmt.limit(limit)
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def get_drug(self, drug_id: uuid.UUID) -> Drug:
        """Get a drug by ID."""
        result = await self.db.execute(
            select(Drug).where(Drug.id == drug_id)
        )
        drug = result.scalar_one_or_none()
        
        if not drug:
            raise NotFoundError("Drug", drug_id)
        
        return drug
    
    async def get_drug_by_name(self, name: str) -> Optional[Drug]:
        """Get a drug by generic name."""
        result = await self.db.execute(
            select(Drug).where(
                func.lower(Drug.generic_name) == name.lower()
            )
        )
        return result.scalar_one_or_none()
    
    async def check_interactions(
        self,
        drug_ids: List[uuid.UUID],
    ) -> List[DrugInteraction]:
        """
        Check for interactions between multiple drugs.
        
        Args:
            drug_ids: List of drug UUIDs to check
        
        Returns:
            List of interactions found
        """
        if len(drug_ids) < 2:
            return []
        
        interactions = []
        
        # Check all pairs
        for i, drug1_id in enumerate(drug_ids):
            for drug2_id in drug_ids[i + 1:]:
                result = await self.db.execute(
                    select(DrugInteraction).where(
                        or_(
                            (DrugInteraction.drug_id_1 == drug1_id) & 
                            (DrugInteraction.drug_id_2 == drug2_id),
                            (DrugInteraction.drug_id_1 == drug2_id) & 
                            (DrugInteraction.drug_id_2 == drug1_id),
                        )
                    )
                )
                interaction = result.scalar_one_or_none()
                if interaction:
                    interactions.append(interaction)
        
        # Sort by severity
        severity_order = {
            InteractionSeverity.CONTRAINDICATED: 0,
            InteractionSeverity.MAJOR: 1,
            InteractionSeverity.MODERATE: 2,
            InteractionSeverity.MINOR: 3,
        }
        interactions.sort(key=lambda x: severity_order.get(x.severity, 99))
        
        return interactions
    
    async def check_interactions_by_name(
        self,
        drug_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Check interactions by drug names.
        
        Args:
            drug_names: List of drug names
        
        Returns:
            List of interaction details
        """
        # Find drug IDs
        drug_ids = []
        drug_map = {}
        
        for name in drug_names:
            drug = await self.get_drug_by_name(name)
            if drug:
                drug_ids.append(drug.id)
                drug_map[drug.id] = drug.generic_name
        
        if len(drug_ids) < 2:
            return []
        
        interactions = await self.check_interactions(drug_ids)
        
        # Format results
        results = []
        for interaction in interactions:
            results.append({
                "drug1": drug_map.get(interaction.drug_id_1, "Unknown"),
                "drug2": drug_map.get(interaction.drug_id_2, "Unknown"),
                "severity": interaction.severity.value,
                "description": interaction.description,
                "mechanism": interaction.mechanism,
                "management": interaction.management,
            })
        
        return results
    
    async def get_drug_alternatives(
        self,
        drug_id: uuid.UUID,
        limit: int = 5,
    ) -> List[Drug]:
        """
        Get alternative drugs in the same therapeutic class.
        
        Args:
            drug_id: Drug to find alternatives for
            limit: Maximum alternatives
        
        Returns:
            List of alternative drugs
        """
        drug = await self.get_drug(drug_id)
        
        if not drug.therapeutic_class:
            return []
        
        result = await self.db.execute(
            select(Drug)
            .where(Drug.therapeutic_class == drug.therapeutic_class)
            .where(Drug.id != drug_id)
            .limit(limit)
        )
        
        return list(result.scalars().all())
    
    async def get_otc_recommendations(
        self,
        symptoms: List[str],
        age: Optional[int] = None,
        allergies: Optional[List[str]] = None,
        current_medications: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get OTC drug recommendations based on symptoms.
        
        Args:
            symptoms: List of symptoms
            age: Patient age
            allergies: Known allergies
            current_medications: Current medications
        
        Returns:
            List of recommendations with warnings
        """
        from src.models.medical import DrugClass
        
        recommendations = []
        symptoms_lower = [s.lower() for s in symptoms]
        
        # Symptom to indication mapping
        symptom_indications = {
            "headache": ["pain", "headache", "analgesic"],
            "pain": ["pain", "analgesic", "anti-inflammatory"],
            "fever": ["fever", "antipyretic", "analgesic"],
            "cold": ["cold", "decongestant", "antihistamine"],
            "cough": ["cough", "antitussive", "expectorant"],
            "allergy": ["allergy", "antihistamine"],
            "heartburn": ["heartburn", "antacid", "acid reducer"],
            "diarrhea": ["diarrhea", "antidiarrheal"],
            "constipation": ["constipation", "laxative"],
            "nausea": ["nausea", "antiemetic"],
        }
        
        # Find matching indications
        target_indications = set()
        for symptom in symptoms_lower:
            for key, indications in symptom_indications.items():
                if key in symptom:
                    target_indications.update(indications)
        
        if not target_indications:
            return []
        
        # Search for OTC drugs
        for indication in target_indications:
            result = await self.db.execute(
                select(Drug)
                .where(Drug.drug_class == DrugClass.OTC)
                .where(
                    or_(
                        func.lower(Drug.therapeutic_class).like(f"%{indication}%"),
                        Drug.indications.cast(String).ilike(f"%{indication}%"),
                    )
                )
                .limit(3)
            )
            
            drugs = result.scalars().all()
            
            for drug in drugs:
                # Check for contraindications
                warnings = []
                skip = False
                
                # Age checks
                if age and age < 12 and drug.pediatric_use:
                    if "not recommended" in drug.pediatric_use.lower():
                        skip = True
                    else:
                        warnings.append(f"Pediatric dosing: {drug.pediatric_use[:100]}")
                
                if age and age > 65 and drug.geriatric_use:
                    warnings.append(f"Geriatric consideration: {drug.geriatric_use[:100]}")
                
                # Allergy checks
                if allergies:
                    for allergy in allergies:
                        if allergy.lower() in drug.generic_name.lower():
                            skip = True
                            break
                
                if skip:
                    continue
                
                recommendations.append({
                    "drug_id": str(drug.id),
                    "name": drug.name,
                    "generic_name": drug.generic_name,
                    "indication": drug.therapeutic_class,
                    "dosage": drug.standard_dosage,
                    "warnings": warnings + (drug.warnings[:3] if drug.warnings else []),
                    "contraindications": drug.contraindications[:3] if drug.contraindications else [],
                })
        
        # Deduplicate
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec["generic_name"] not in seen:
                seen.add(rec["generic_name"])
                unique_recs.append(rec)
        
        return unique_recs[:5]


# Import String for type casting in queries
from sqlalchemy import String
