"""
Entity Profile - Profiles for non-patient entities

Supports:
- Pharmaceutical companies
- Hospitals
- Research organizations
- Healthcare providers
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class EntityType(str, Enum):
    """Types of entities"""
    PHARMACEUTICAL_COMPANY = "pharmaceutical_company"
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    RESEARCH_INSTITUTION = "research_institution"
    INSURANCE_PROVIDER = "insurance_provider"
    REGULATORY_BODY = "regulatory_body"
    HEALTHCARE_PROVIDER = "healthcare_provider"


class FacilityStatus(str, Enum):
    """Facility operational status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNDER_INSPECTION = "under_inspection"
    SUSPENDED = "suspended"


class RegulatoryAccreditation(BaseModel):
    """Regulatory accreditation record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    body: str  # FDA, EMA, SFDA, MHRA
    accreditation_type: str
    certificate_number: Optional[str] = None
    issue_date: date
    expiry_date: date
    status: str = "active"
    scope: Optional[str] = None
    notes: Optional[str] = None


class QADocument(BaseModel):
    """QA document record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_type: str  # SOP, validation_protocol, batch_record, etc.
    title: str
    version: str
    effective_date: date
    review_date: Optional[date] = None
    status: str = "current"  # current, superseded, draft
    department: Optional[str] = None
    author: Optional[str] = None
    approver: Optional[str] = None
    file_path: Optional[str] = None


class ValidationRecord(BaseModel):
    """Validation record for pharmaceutical facilities"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    validation_type: str  # cleaning, process, equipment, method
    protocol_number: str
    equipment_or_process: str
    start_date: date
    completion_date: Optional[date] = None
    status: str = "in_progress"
    result: Optional[str] = None  # passed, failed, conditional
    next_revalidation: Optional[date] = None
    deviations: List[str] = Field(default_factory=list)


class InspectionRecord(BaseModel):
    """Regulatory inspection record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    inspector_body: str  # FDA, EMA, etc.
    inspection_type: str  # routine, for_cause, pre_approval
    start_date: date
    end_date: Optional[date] = None
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    classification: Optional[str] = None  # NAI, VAI, OAI
    response_due_date: Optional[date] = None
    response_submitted: bool = False
    closed: bool = False


class Product(BaseModel):
    """Pharmaceutical product record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    generic_name: Optional[str] = None
    ndc_code: Optional[str] = None
    dosage_form: str
    strength: str
    route: str
    approval_date: Optional[date] = None
    patent_expiry: Optional[date] = None
    status: str = "active"


class SalesRecord(BaseModel):
    """Product sales record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    product_id: str
    period: str  # YYYY-MM or YYYY-Q#
    region: str
    units_sold: int
    revenue: float
    marketing_spend: Optional[float] = None
    notes: Optional[str] = None


class EntityProfile(BaseModel):
    """Profile for organizational entities"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType
    name: str
    legal_name: Optional[str] = None
    
    # Contact
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    
    # Regulatory
    accreditations: List[RegulatoryAccreditation] = Field(default_factory=list)
    facility_status: FacilityStatus = FacilityStatus.ACTIVE
    license_number: Optional[str] = None
    
    # QA/QC (for pharma)
    qa_documents: List[QADocument] = Field(default_factory=list)
    validation_records: List[ValidationRecord] = Field(default_factory=list)
    inspection_records: List[InspectionRecord] = Field(default_factory=list)
    
    # Products (for pharma)
    products: List[Product] = Field(default_factory=list)
    sales_records: List[SalesRecord] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Custom fields
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def active_accreditations(self) -> List[RegulatoryAccreditation]:
        today = date.today()
        return [
            a for a in self.accreditations
            if a.status == "active" and a.expiry_date >= today
        ]
    
    @property
    def pending_validations(self) -> List[ValidationRecord]:
        return [v for v in self.validation_records if v.status == "in_progress"]
    
    @property
    def open_inspections(self) -> List[InspectionRecord]:
        return [i for i in self.inspection_records if not i.closed]
    
    def get_expiring_accreditations(self, days: int = 90) -> List[RegulatoryAccreditation]:
        """Get accreditations expiring within specified days"""
        from datetime import timedelta
        cutoff = date.today() + timedelta(days=days)
        return [
            a for a in self.active_accreditations
            if a.expiry_date <= cutoff
        ]
    
    def get_overdue_validations(self) -> List[ValidationRecord]:
        """Get validations past their revalidation date"""
        today = date.today()
        return [
            v for v in self.validation_records
            if v.next_revalidation and v.next_revalidation < today
        ]
    
    def get_sales_by_product(self, product_id: str) -> List[SalesRecord]:
        """Get sales records for a specific product"""
        return [s for s in self.sales_records if s.product_id == product_id]
    
    def get_total_revenue(self, period: Optional[str] = None) -> float:
        """Get total revenue, optionally filtered by period"""
        records = self.sales_records
        if period:
            records = [s for s in records if s.period == period]
        return sum(s.revenue for s in records)


class EntityProfileManager:
    """Manages entity profiles"""
    
    def __init__(self):
        self._profiles: Dict[str, EntityProfile] = {}
    
    def create_profile(
        self,
        entity_type: EntityType,
        name: str,
        **kwargs,
    ) -> EntityProfile:
        """Create a new entity profile"""
        profile = EntityProfile(
            entity_type=entity_type,
            name=name,
            **kwargs,
        )
        self._profiles[profile.id] = profile
        return profile
    
    def get_profile(self, profile_id: str) -> Optional[EntityProfile]:
        """Get an entity profile by ID"""
        return self._profiles.get(profile_id)
    
    def update_profile(
        self,
        profile_id: str,
        updates: Dict[str, Any],
    ) -> Optional[EntityProfile]:
        """Update an entity profile"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return None
        
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.updated_at = datetime.utcnow()
        return profile
    
    def add_accreditation(
        self,
        profile_id: str,
        accreditation: RegulatoryAccreditation,
    ) -> bool:
        """Add accreditation to entity"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        
        profile.accreditations.append(accreditation)
        profile.updated_at = datetime.utcnow()
        return True
    
    def add_qa_document(
        self,
        profile_id: str,
        document: QADocument,
    ) -> bool:
        """Add QA document to entity"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        
        profile.qa_documents.append(document)
        profile.updated_at = datetime.utcnow()
        return True
    
    def add_validation_record(
        self,
        profile_id: str,
        validation: ValidationRecord,
    ) -> bool:
        """Add validation record to entity"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        
        profile.validation_records.append(validation)
        profile.updated_at = datetime.utcnow()
        return True
    
    def add_inspection_record(
        self,
        profile_id: str,
        inspection: InspectionRecord,
    ) -> bool:
        """Add inspection record to entity"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        
        profile.inspection_records.append(inspection)
        profile.updated_at = datetime.utcnow()
        return True
    
    def add_product(
        self,
        profile_id: str,
        product: Product,
    ) -> bool:
        """Add product to entity"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        
        profile.products.append(product)
        profile.updated_at = datetime.utcnow()
        return True
    
    def add_sales_record(
        self,
        profile_id: str,
        sales: SalesRecord,
    ) -> bool:
        """Add sales record to entity"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        
        profile.sales_records.append(sales)
        profile.updated_at = datetime.utcnow()
        return True
    
    def get_compliance_summary(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get compliance summary for an entity"""
        profile = self._profiles.get(profile_id)
        if not profile:
            return None
        
        return {
            "entity_name": profile.name,
            "facility_status": profile.facility_status.value,
            "active_accreditations": len(profile.active_accreditations),
            "expiring_accreditations_90d": len(profile.get_expiring_accreditations(90)),
            "pending_validations": len(profile.pending_validations),
            "overdue_validations": len(profile.get_overdue_validations()),
            "open_inspections": len(profile.open_inspections),
            "total_products": len(profile.products),
        }
