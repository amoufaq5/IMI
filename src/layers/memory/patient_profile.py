"""
Patient Profile - HIPAA-compliant longitudinal patient data management

Stores and manages:
- Medical history
- Medications
- Allergies
- Conditions
- Treatment outcomes
- Interaction history
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field
import uuid

from src.core.security.encryption import EncryptionService, get_encryption_service
from src.core.security.hipaa import HIPAAComplianceService, get_hipaa_service


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class BloodType(str, Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"
    UNKNOWN = "unknown"


class MedicationStatus(str, Enum):
    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"


class Medication(BaseModel):
    """Patient medication record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    dosage: str
    frequency: str
    route: str
    start_date: date
    end_date: Optional[date] = None
    status: MedicationStatus = MedicationStatus.ACTIVE
    prescriber: Optional[str] = None
    indication: Optional[str] = None
    notes: Optional[str] = None


class Allergy(BaseModel):
    """Patient allergy record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    allergen: str
    reaction: str
    severity: str  # mild, moderate, severe, life-threatening
    onset_date: Optional[date] = None
    verified: bool = False
    notes: Optional[str] = None


class Condition(BaseModel):
    """Patient medical condition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    icd10_code: Optional[str] = None
    onset_date: Optional[date] = None
    resolution_date: Optional[date] = None
    status: str = "active"  # active, resolved, chronic
    severity: Optional[str] = None
    notes: Optional[str] = None


class Encounter(BaseModel):
    """Patient encounter/visit record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date: datetime
    type: str  # office_visit, er, telehealth, etc.
    chief_complaint: Optional[str] = None
    diagnosis: List[str] = Field(default_factory=list)
    procedures: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    provider: Optional[str] = None
    facility: Optional[str] = None


class VitalSigns(BaseModel):
    """Patient vital signs record"""
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    temperature_f: Optional[float] = None
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    bmi: Optional[float] = None


class LabResult(BaseModel):
    """Patient lab result"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    test_name: str
    loinc_code: Optional[str] = None
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    is_abnormal: bool = False
    collected_at: datetime
    resulted_at: Optional[datetime] = None
    notes: Optional[str] = None


class PatientProfile(BaseModel):
    """Complete patient profile - HIPAA compliant"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Demographics (encrypted at rest)
    first_name: str
    last_name: str
    date_of_birth: date
    gender: Optional[Gender] = None
    blood_type: Optional[BloodType] = BloodType.UNKNOWN
    
    # Contact (encrypted)
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    
    # Emergency contact (encrypted)
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    emergency_contact_relationship: Optional[str] = None
    
    # Medical data
    medications: List[Medication] = Field(default_factory=list)
    allergies: List[Allergy] = Field(default_factory=list)
    conditions: List[Condition] = Field(default_factory=list)
    encounters: List[Encounter] = Field(default_factory=list)
    vital_signs_history: List[VitalSigns] = Field(default_factory=list)
    lab_results: List[LabResult] = Field(default_factory=list)
    
    # Lifestyle
    smoking_status: Optional[str] = None
    alcohol_use: Optional[str] = None
    exercise_frequency: Optional[str] = None
    diet_restrictions: List[str] = Field(default_factory=list)
    
    # Insurance (encrypted)
    insurance_provider: Optional[str] = None
    insurance_id: Optional[str] = None
    insurance_group: Optional[str] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None
    
    # Consent
    consent_given: bool = False
    consent_date: Optional[datetime] = None
    data_sharing_preferences: Dict[str, bool] = Field(default_factory=dict)
    
    @property
    def age(self) -> int:
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
    
    @property
    def active_medications(self) -> List[Medication]:
        return [m for m in self.medications if m.status == MedicationStatus.ACTIVE]
    
    @property
    def active_conditions(self) -> List[Condition]:
        return [c for c in self.conditions if c.status in ["active", "chronic"]]
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
    
    def get_latest_vitals(self) -> Optional[VitalSigns]:
        if self.vital_signs_history:
            return max(self.vital_signs_history, key=lambda v: v.recorded_at)
        return None
    
    def get_medication_list(self) -> List[str]:
        return [m.name for m in self.active_medications]
    
    def get_allergy_list(self) -> List[str]:
        return [a.allergen for a in self.allergies]
    
    def get_condition_list(self) -> List[str]:
        return [c.name for c in self.active_conditions]


class PatientProfileManager:
    """Manages patient profiles with HIPAA compliance"""
    
    def __init__(self):
        self.encryption = get_encryption_service()
        self.hipaa = get_hipaa_service()
        self._profiles: Dict[str, PatientProfile] = {}  # In-memory for demo
    
    def create_profile(
        self,
        first_name: str,
        last_name: str,
        date_of_birth: date,
        **kwargs,
    ) -> PatientProfile:
        """Create a new patient profile"""
        profile = PatientProfile(
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            **kwargs,
        )
        self._profiles[profile.id] = profile
        return profile
    
    def get_profile(self, profile_id: str) -> Optional[PatientProfile]:
        """Get a patient profile by ID"""
        profile = self._profiles.get(profile_id)
        if profile:
            profile.last_accessed_at = datetime.utcnow()
        return profile
    
    def update_profile(
        self,
        profile_id: str,
        updates: Dict[str, Any],
    ) -> Optional[PatientProfile]:
        """Update a patient profile"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return None
        
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.updated_at = datetime.utcnow()
        return profile
    
    def add_medication(
        self,
        profile_id: str,
        medication: Medication,
    ) -> bool:
        """Add medication to patient profile"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        
        profile.medications.append(medication)
        profile.updated_at = datetime.utcnow()
        return True
    
    def add_allergy(
        self,
        profile_id: str,
        allergy: Allergy,
    ) -> bool:
        """Add allergy to patient profile"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        
        profile.allergies.append(allergy)
        profile.updated_at = datetime.utcnow()
        return True
    
    def add_condition(
        self,
        profile_id: str,
        condition: Condition,
    ) -> bool:
        """Add condition to patient profile"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        
        profile.conditions.append(condition)
        profile.updated_at = datetime.utcnow()
        return True
    
    def add_encounter(
        self,
        profile_id: str,
        encounter: Encounter,
    ) -> bool:
        """Add encounter to patient profile"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        
        profile.encounters.append(encounter)
        profile.updated_at = datetime.utcnow()
        return True
    
    def add_vital_signs(
        self,
        profile_id: str,
        vitals: VitalSigns,
    ) -> bool:
        """Add vital signs to patient profile"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        
        profile.vital_signs_history.append(vitals)
        profile.updated_at = datetime.utcnow()
        return True
    
    def get_encrypted_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get profile with PHI encrypted"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return None
        
        return self.hipaa.encrypt_phi(profile.model_dump())
    
    def get_anonymized_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get anonymized profile for research"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return None
        
        return self.hipaa.anonymize_phi(profile.model_dump())
    
    def get_patient_summary(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of patient profile for clinical use"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return None
        
        return {
            "id": profile.id,
            "age": profile.age,
            "gender": profile.gender.value if profile.gender else None,
            "active_medications": profile.get_medication_list(),
            "allergies": profile.get_allergy_list(),
            "active_conditions": profile.get_condition_list(),
            "latest_vitals": profile.get_latest_vitals().model_dump() if profile.get_latest_vitals() else None,
        }
