"""
UMI Medical NLP Service
Specialized NLP for medical text processing, entity extraction, and analysis
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.logging import get_logger

logger = get_logger(__name__)


class EntityType(str, Enum):
    """Medical entity types."""
    SYMPTOM = "symptom"
    DISEASE = "disease"
    DRUG = "drug"
    BODY_PART = "body_part"
    PROCEDURE = "procedure"
    LAB_TEST = "lab_test"
    DOSAGE = "dosage"
    DURATION = "duration"
    FREQUENCY = "frequency"


@dataclass
class MedicalEntity:
    """Extracted medical entity."""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float
    normalized_form: Optional[str] = None
    codes: Optional[Dict[str, str]] = None  # ICD, SNOMED, RxNorm codes


@dataclass
class NLPResult:
    """Result from medical NLP processing."""
    entities: List[MedicalEntity]
    negations: List[Tuple[str, bool]]  # (entity, is_negated)
    severity_indicators: List[Tuple[str, str]]  # (entity, severity)
    temporal_info: List[Tuple[str, str]]  # (entity, temporal_context)


class MedicalLexicon:
    """Medical terminology lexicon for entity recognition."""
    
    SYMPTOMS = {
        "headache", "pain", "fever", "cough", "fatigue", "nausea", "vomiting",
        "diarrhea", "constipation", "dizziness", "shortness of breath", "chest pain",
        "abdominal pain", "back pain", "joint pain", "muscle pain", "sore throat",
        "runny nose", "congestion", "rash", "itching", "swelling", "bleeding",
        "numbness", "tingling", "weakness", "confusion", "anxiety", "depression",
        "insomnia", "loss of appetite", "weight loss", "weight gain", "palpitations",
        "sweating", "chills", "blurred vision", "hearing loss", "tinnitus",
    }
    
    BODY_PARTS = {
        "head", "neck", "chest", "abdomen", "back", "arm", "leg", "hand", "foot",
        "shoulder", "elbow", "wrist", "hip", "knee", "ankle", "throat", "ear",
        "eye", "nose", "mouth", "tongue", "stomach", "liver", "kidney", "heart",
        "lung", "brain", "spine", "skin", "muscle", "bone", "joint", "nerve",
    }
    
    SEVERITY_MODIFIERS = {
        "severe": "severe",
        "mild": "mild",
        "moderate": "moderate",
        "intense": "severe",
        "slight": "mild",
        "extreme": "severe",
        "terrible": "severe",
        "awful": "severe",
        "unbearable": "severe",
        "minor": "mild",
        "major": "severe",
        "chronic": "chronic",
        "acute": "acute",
        "persistent": "chronic",
        "intermittent": "intermittent",
        "constant": "constant",
        "occasional": "intermittent",
    }
    
    NEGATION_CUES = {
        "no", "not", "without", "denies", "denied", "negative", "absent",
        "never", "none", "neither", "nor", "doesn't", "don't", "didn't",
        "hasn't", "haven't", "hadn't", "isn't", "aren't", "wasn't", "weren't",
        "cannot", "can't", "won't", "wouldn't", "shouldn't",
    }
    
    TEMPORAL_CUES = {
        "today": "recent",
        "yesterday": "recent",
        "last week": "recent",
        "last month": "past",
        "years ago": "historical",
        "since": "ongoing",
        "for": "duration",
        "started": "onset",
        "began": "onset",
        "suddenly": "acute_onset",
        "gradually": "gradual_onset",
        "worsening": "progressive",
        "improving": "resolving",
        "constant": "continuous",
        "comes and goes": "intermittent",
    }
    
    COMMON_DRUGS = {
        "paracetamol", "acetaminophen", "ibuprofen", "aspirin", "naproxen",
        "amoxicillin", "penicillin", "metformin", "lisinopril", "amlodipine",
        "omeprazole", "pantoprazole", "atorvastatin", "simvastatin", "metoprolol",
        "losartan", "gabapentin", "sertraline", "fluoxetine", "escitalopram",
        "levothyroxine", "prednisone", "albuterol", "montelukast", "cetirizine",
        "loratadine", "diphenhydramine", "ranitidine", "famotidine",
    }
    
    DOSAGE_PATTERNS = [
        r"\d+\s*mg",
        r"\d+\s*g",
        r"\d+\s*ml",
        r"\d+\s*mcg",
        r"\d+\s*iu",
        r"\d+\s*units?",
    ]
    
    FREQUENCY_PATTERNS = [
        r"once\s+(?:a\s+)?day",
        r"twice\s+(?:a\s+)?day",
        r"three\s+times\s+(?:a\s+)?day",
        r"\d+\s+times\s+(?:a\s+)?day",
        r"every\s+\d+\s+hours?",
        r"daily",
        r"weekly",
        r"monthly",
        r"as\s+needed",
        r"prn",
        r"bid",
        r"tid",
        r"qid",
        r"qd",
    ]


class MedicalNLPService:
    """
    Medical NLP service for text processing and entity extraction.
    """
    
    def __init__(self):
        self.lexicon = MedicalLexicon()
        self._nlp_model = None
    
    async def _load_model(self):
        """Lazy load NLP model."""
        if self._nlp_model is None:
            try:
                import spacy
                
                # Try to load medical NLP model
                try:
                    self._nlp_model = spacy.load("en_core_sci_md")
                    logger.info("loaded_scispacy_model")
                except OSError:
                    # Fall back to standard model
                    try:
                        self._nlp_model = spacy.load("en_core_web_sm")
                        logger.info("loaded_standard_spacy_model")
                    except OSError:
                        logger.warning("no_spacy_model_available")
                        self._nlp_model = MockNLPModel()
            
            except ImportError:
                logger.warning("spacy_not_installed")
                self._nlp_model = MockNLPModel()
    
    async def process(self, text: str) -> NLPResult:
        """
        Process medical text and extract entities.
        
        Args:
            text: Input text to process
        
        Returns:
            NLPResult with extracted entities and analysis
        """
        await self._load_model()
        
        # Extract entities
        entities = await self._extract_entities(text)
        
        # Detect negations
        negations = self._detect_negations(text, entities)
        
        # Extract severity indicators
        severity = self._extract_severity(text, entities)
        
        # Extract temporal information
        temporal = self._extract_temporal(text, entities)
        
        return NLPResult(
            entities=entities,
            negations=negations,
            severity_indicators=severity,
            temporal_info=temporal,
        )
    
    async def _extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text."""
        entities = []
        text_lower = text.lower()
        
        # Extract symptoms
        for symptom in self.lexicon.SYMPTOMS:
            for match in re.finditer(r'\b' + re.escape(symptom) + r'\b', text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type=EntityType.SYMPTOM,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                ))
        
        # Extract body parts
        for body_part in self.lexicon.BODY_PARTS:
            for match in re.finditer(r'\b' + re.escape(body_part) + r'\b', text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type=EntityType.BODY_PART,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                ))
        
        # Extract drugs
        for drug in self.lexicon.COMMON_DRUGS:
            for match in re.finditer(r'\b' + re.escape(drug) + r'\b', text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type=EntityType.DRUG,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                ))
        
        # Extract dosages
        for pattern in self.lexicon.DOSAGE_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type=EntityType.DOSAGE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                ))
        
        # Extract frequencies
        for pattern in self.lexicon.FREQUENCY_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                entities.append(MedicalEntity(
                    text=text[match.start():match.end()],
                    entity_type=EntityType.FREQUENCY,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                ))
        
        # Remove duplicates and sort by position
        seen = set()
        unique_entities = []
        for entity in sorted(entities, key=lambda x: x.start):
            key = (entity.start, entity.end, entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _detect_negations(
        self,
        text: str,
        entities: List[MedicalEntity],
    ) -> List[Tuple[str, bool]]:
        """Detect negated entities."""
        negations = []
        text_lower = text.lower()
        
        for entity in entities:
            # Check for negation cues before the entity
            context_start = max(0, entity.start - 30)
            context = text_lower[context_start:entity.start]
            
            is_negated = any(
                cue in context.split()
                for cue in self.lexicon.NEGATION_CUES
            )
            
            negations.append((entity.text, is_negated))
        
        return negations
    
    def _extract_severity(
        self,
        text: str,
        entities: List[MedicalEntity],
    ) -> List[Tuple[str, str]]:
        """Extract severity indicators for symptoms."""
        severity_info = []
        text_lower = text.lower()
        
        for entity in entities:
            if entity.entity_type != EntityType.SYMPTOM:
                continue
            
            # Check for severity modifiers near the entity
            context_start = max(0, entity.start - 20)
            context_end = min(len(text), entity.end + 20)
            context = text_lower[context_start:context_end]
            
            severity = "unspecified"
            for modifier, level in self.lexicon.SEVERITY_MODIFIERS.items():
                if modifier in context:
                    severity = level
                    break
            
            severity_info.append((entity.text, severity))
        
        return severity_info
    
    def _extract_temporal(
        self,
        text: str,
        entities: List[MedicalEntity],
    ) -> List[Tuple[str, str]]:
        """Extract temporal information for entities."""
        temporal_info = []
        text_lower = text.lower()
        
        for entity in entities:
            # Check for temporal cues near the entity
            context_start = max(0, entity.start - 50)
            context_end = min(len(text), entity.end + 50)
            context = text_lower[context_start:context_end]
            
            temporal = "unspecified"
            for cue, category in self.lexicon.TEMPORAL_CUES.items():
                if cue in context:
                    temporal = category
                    break
            
            temporal_info.append((entity.text, temporal))
        
        return temporal_info
    
    async def extract_symptoms(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract symptoms with context.
        
        Returns:
            List of symptom dictionaries with details
        """
        result = await self.process(text)
        
        symptoms = []
        negation_map = dict(result.negations)
        severity_map = dict(result.severity_indicators)
        temporal_map = dict(result.temporal_info)
        
        for entity in result.entities:
            if entity.entity_type == EntityType.SYMPTOM:
                symptoms.append({
                    "symptom": entity.text,
                    "is_negated": negation_map.get(entity.text, False),
                    "severity": severity_map.get(entity.text, "unspecified"),
                    "temporal": temporal_map.get(entity.text, "unspecified"),
                    "confidence": entity.confidence,
                })
        
        return symptoms
    
    async def extract_medications(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medications with dosage and frequency.
        
        Returns:
            List of medication dictionaries
        """
        result = await self.process(text)
        
        medications = []
        
        # Group drugs with nearby dosages and frequencies
        drug_entities = [e for e in result.entities if e.entity_type == EntityType.DRUG]
        dosage_entities = [e for e in result.entities if e.entity_type == EntityType.DOSAGE]
        frequency_entities = [e for e in result.entities if e.entity_type == EntityType.FREQUENCY]
        
        for drug in drug_entities:
            med = {
                "drug": drug.text,
                "dosage": None,
                "frequency": None,
                "confidence": drug.confidence,
            }
            
            # Find nearby dosage
            for dosage in dosage_entities:
                if abs(dosage.start - drug.end) < 30:
                    med["dosage"] = dosage.text
                    break
            
            # Find nearby frequency
            for freq in frequency_entities:
                if abs(freq.start - drug.end) < 50:
                    med["frequency"] = freq.text
                    break
            
            medications.append(med)
        
        return medications
    
    async def check_drug_mentions(
        self,
        text: str,
        drug_list: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Check if specific drugs are mentioned in text.
        
        Args:
            text: Input text
            drug_list: List of drug names to check
        
        Returns:
            List of found drugs with context
        """
        found = []
        text_lower = text.lower()
        
        for drug in drug_list:
            drug_lower = drug.lower()
            if drug_lower in text_lower:
                # Get context around the mention
                idx = text_lower.find(drug_lower)
                context_start = max(0, idx - 50)
                context_end = min(len(text), idx + len(drug) + 50)
                context = text[context_start:context_end]
                
                found.append({
                    "drug": drug,
                    "context": context,
                    "position": idx,
                })
        
        return found
    
    def normalize_symptom(self, symptom: str) -> str:
        """Normalize symptom text to standard form."""
        # Simple normalization - in production would use SNOMED CT mapping
        symptom_lower = symptom.lower().strip()
        
        normalizations = {
            "headache": "headache",
            "head ache": "headache",
            "head pain": "headache",
            "stomach ache": "abdominal pain",
            "stomachache": "abdominal pain",
            "tummy ache": "abdominal pain",
            "belly pain": "abdominal pain",
            "throwing up": "vomiting",
            "being sick": "vomiting",
            "feeling sick": "nausea",
            "runny nose": "rhinorrhea",
            "stuffy nose": "nasal congestion",
            "blocked nose": "nasal congestion",
            "short of breath": "dyspnea",
            "breathlessness": "dyspnea",
            "difficulty breathing": "dyspnea",
            "tired": "fatigue",
            "tiredness": "fatigue",
            "exhaustion": "fatigue",
            "exhausted": "fatigue",
        }
        
        return normalizations.get(symptom_lower, symptom_lower)


class MockNLPModel:
    """Mock NLP model for development."""
    
    def __call__(self, text: str):
        """Process text (mock)."""
        return MockDoc(text)


class MockDoc:
    """Mock spaCy Doc."""
    
    def __init__(self, text: str):
        self.text = text
        self.ents = []
