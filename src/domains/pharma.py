"""
Pharmaceutical Company Domain Service

Provides pharma-specific functionality:
- QA/QC document management
- Regulatory compliance
- Sales tracking
- Facility management
- FDA/EMA/SFDA/MHRA compliance
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field

from src.layers.memory.entity_profile import (
    EntityProfile, EntityProfileManager, EntityType,
    QADocument, ValidationRecord, InspectionRecord,
    RegulatoryAccreditation, Product, SalesRecord
)
from src.core.security.audit import AuditLogger, get_audit_logger, AuditAction


class RegulatoryBody(str, Enum):
    """Regulatory bodies"""
    FDA = "fda"
    EMA = "ema"
    SFDA = "sfda"  # Saudi FDA
    MHRA = "mhra"
    TGA = "tga"
    PMDA = "pmda"
    HEALTH_CANADA = "health_canada"


class DocumentType(str, Enum):
    """QA document types"""
    SOP = "sop"
    BATCH_RECORD = "batch_record"
    VALIDATION_PROTOCOL = "validation_protocol"
    VALIDATION_REPORT = "validation_report"
    DEVIATION_REPORT = "deviation_report"
    CAPA = "capa"
    CHANGE_CONTROL = "change_control"
    CLEANING_VALIDATION = "cleaning_validation"
    EQUIPMENT_QUALIFICATION = "equipment_qualification"
    STABILITY_PROTOCOL = "stability_protocol"
    TECHNICAL_SHEET = "technical_sheet"
    MANUFACTURING_LOGBOOK = "manufacturing_logbook"


class ValidationType(str, Enum):
    """Validation types"""
    CLEANING = "cleaning"
    PROCESS = "process"
    EQUIPMENT = "equipment"
    METHOD = "method"
    COMPUTER_SYSTEM = "computer_system"
    HVAC = "hvac"
    WATER_SYSTEM = "water_system"


class ComplianceCheckResult(BaseModel):
    """Result of compliance check"""
    is_compliant: bool
    regulation: str
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    required_actions: List[str] = Field(default_factory=list)
    deadline: Optional[date] = None


class PharmaService:
    """
    Pharmaceutical company support service
    
    Capabilities:
    - QA document management and generation
    - Regulatory compliance checking
    - Validation tracking
    - Sales and marketing analytics
    - Facility accreditation management
    """
    
    # GMP requirements by regulatory body
    GMP_REQUIREMENTS = {
        RegulatoryBody.FDA: {
            "name": "21 CFR Parts 210, 211",
            "key_requirements": [
                "Quality Unit independence",
                "Written procedures for production and control",
                "Equipment qualification and validation",
                "Process validation",
                "Stability testing program",
                "Laboratory controls",
                "Records and reports",
                "Returned and salvaged products",
            ],
        },
        RegulatoryBody.EMA: {
            "name": "EU GMP Annex 1-20",
            "key_requirements": [
                "Pharmaceutical Quality System",
                "Personnel qualifications",
                "Premises and equipment",
                "Documentation",
                "Production",
                "Quality Control",
                "Outsourced activities",
                "Complaints and recalls",
            ],
        },
        RegulatoryBody.SFDA: {
            "name": "SFDA GMP Guidelines",
            "key_requirements": [
                "Quality Management System",
                "Personnel",
                "Premises",
                "Equipment",
                "Materials",
                "Documentation",
                "Production",
                "Quality Control",
            ],
        },
    }
    
    def __init__(
        self,
        entity_manager: Optional[EntityProfileManager] = None,
        llm_service=None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.entity_manager = entity_manager or EntityProfileManager()
        self.llm = llm_service
        self.audit = audit_logger or get_audit_logger()
    
    # QA Document Management
    async def generate_document(
        self,
        document_type: DocumentType,
        template_data: Dict[str, Any],
        entity_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a QA document from template"""
        document = {
            "type": document_type.value,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "draft",
            "content": self._generate_document_content(document_type, template_data),
            "metadata": template_data,
        }
        
        self.audit.log(
            action=AuditAction.CREATE,
            description=f"QA document generated: {document_type.value}",
            user_id=user_id,
            resource_type="qa_document",
            details={"document_type": document_type.value},
        )
        
        return document
    
    def _generate_document_content(
        self,
        document_type: DocumentType,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate document content based on type"""
        templates = {
            DocumentType.SOP: {
                "sections": [
                    {"title": "Purpose", "content": data.get("purpose", "")},
                    {"title": "Scope", "content": data.get("scope", "")},
                    {"title": "Responsibilities", "content": data.get("responsibilities", "")},
                    {"title": "Procedure", "content": data.get("procedure", "")},
                    {"title": "References", "content": data.get("references", "")},
                    {"title": "Revision History", "content": ""},
                ],
            },
            DocumentType.DEVIATION_REPORT: {
                "sections": [
                    {"title": "Deviation Description", "content": data.get("description", "")},
                    {"title": "Root Cause Analysis", "content": ""},
                    {"title": "Impact Assessment", "content": ""},
                    {"title": "Corrective Actions", "content": ""},
                    {"title": "Preventive Actions", "content": ""},
                    {"title": "Effectiveness Check", "content": ""},
                ],
            },
            DocumentType.CLEANING_VALIDATION: {
                "sections": [
                    {"title": "Equipment Description", "content": data.get("equipment", "")},
                    {"title": "Cleaning Procedure", "content": data.get("procedure", "")},
                    {"title": "Acceptance Criteria", "content": data.get("criteria", "")},
                    {"title": "Sampling Plan", "content": ""},
                    {"title": "Analytical Methods", "content": ""},
                    {"title": "Results", "content": ""},
                    {"title": "Conclusion", "content": ""},
                ],
            },
        }
        
        return templates.get(document_type, {"sections": []})
    
    async def check_compliance(
        self,
        entity_id: str,
        regulatory_body: RegulatoryBody,
        check_areas: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> ComplianceCheckResult:
        """Check compliance against regulatory requirements"""
        entity = self.entity_manager.get_profile(entity_id)
        if not entity:
            return ComplianceCheckResult(
                is_compliant=False,
                regulation=regulatory_body.value,
                findings=[{"issue": "Entity not found"}],
            )
        
        findings = []
        recommendations = []
        required_actions = []
        
        # Check accreditations
        active_accreds = entity.active_accreditations
        has_accreditation = any(
            a.body.lower() == regulatory_body.value
            for a in active_accreds
        )
        
        if not has_accreditation:
            findings.append({
                "area": "Accreditation",
                "issue": f"No active {regulatory_body.value.upper()} accreditation",
                "severity": "critical",
            })
            required_actions.append(f"Obtain {regulatory_body.value.upper()} accreditation")
        
        # Check expiring accreditations
        expiring = entity.get_expiring_accreditations(90)
        for accred in expiring:
            findings.append({
                "area": "Accreditation",
                "issue": f"{accred.body} accreditation expiring on {accred.expiry_date}",
                "severity": "warning",
            })
            recommendations.append(f"Renew {accred.body} accreditation before {accred.expiry_date}")
        
        # Check overdue validations
        overdue = entity.get_overdue_validations()
        for validation in overdue:
            findings.append({
                "area": "Validation",
                "issue": f"{validation.validation_type} validation overdue for {validation.equipment_or_process}",
                "severity": "major",
            })
            required_actions.append(f"Complete revalidation for {validation.equipment_or_process}")
        
        # Check open inspections
        open_inspections = entity.open_inspections
        for inspection in open_inspections:
            if inspection.response_due_date and inspection.response_due_date < date.today():
                findings.append({
                    "area": "Inspection",
                    "issue": f"Overdue response for {inspection.inspector_body} inspection",
                    "severity": "critical",
                })
                required_actions.append(f"Submit response to {inspection.inspector_body}")
        
        is_compliant = not any(f.get("severity") == "critical" for f in findings)
        
        self.audit.log(
            action=AuditAction.VIEW,
            description="Compliance check performed",
            user_id=user_id,
            resource_type="entity",
            resource_id=entity_id,
            details={
                "regulatory_body": regulatory_body.value,
                "is_compliant": is_compliant,
                "findings_count": len(findings),
            },
        )
        
        return ComplianceCheckResult(
            is_compliant=is_compliant,
            regulation=self.GMP_REQUIREMENTS.get(regulatory_body, {}).get("name", regulatory_body.value),
            findings=findings,
            recommendations=recommendations,
            required_actions=required_actions,
        )
    
    # Validation Management
    def create_validation_record(
        self,
        entity_id: str,
        validation_type: ValidationType,
        protocol_number: str,
        equipment_or_process: str,
        user_id: Optional[str] = None,
    ) -> Optional[ValidationRecord]:
        """Create a new validation record"""
        validation = ValidationRecord(
            validation_type=validation_type.value,
            protocol_number=protocol_number,
            equipment_or_process=equipment_or_process,
            start_date=date.today(),
        )
        
        success = self.entity_manager.add_validation_record(entity_id, validation)
        
        if success:
            self.audit.log(
                action=AuditAction.CREATE,
                description="Validation record created",
                user_id=user_id,
                resource_type="validation",
                resource_id=validation.id,
                details={
                    "type": validation_type.value,
                    "protocol": protocol_number,
                },
            )
            return validation
        return None
    
    def complete_validation(
        self,
        entity_id: str,
        validation_id: str,
        result: str,
        next_revalidation: date,
        deviations: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Complete a validation record"""
        entity = self.entity_manager.get_profile(entity_id)
        if not entity:
            return False
        
        for validation in entity.validation_records:
            if validation.id == validation_id:
                validation.completion_date = date.today()
                validation.status = "completed"
                validation.result = result
                validation.next_revalidation = next_revalidation
                if deviations:
                    validation.deviations = deviations
                
                self.audit.log(
                    action=AuditAction.UPDATE,
                    description="Validation completed",
                    user_id=user_id,
                    resource_type="validation",
                    resource_id=validation_id,
                    details={"result": result},
                )
                return True
        
        return False
    
    # Sales Tracking
    def add_sales_record(
        self,
        entity_id: str,
        product_id: str,
        period: str,
        region: str,
        units_sold: int,
        revenue: float,
        marketing_spend: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> Optional[SalesRecord]:
        """Add a sales record"""
        sales = SalesRecord(
            product_id=product_id,
            period=period,
            region=region,
            units_sold=units_sold,
            revenue=revenue,
            marketing_spend=marketing_spend,
        )
        
        success = self.entity_manager.add_sales_record(entity_id, sales)
        
        if success:
            self.audit.log(
                action=AuditAction.CREATE,
                description="Sales record added",
                user_id=user_id,
                resource_type="sales",
                details={
                    "product_id": product_id,
                    "period": period,
                    "revenue": revenue,
                },
            )
            return sales
        return None
    
    def get_sales_analytics(
        self,
        entity_id: str,
        product_id: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get sales analytics"""
        entity = self.entity_manager.get_profile(entity_id)
        if not entity:
            return {}
        
        records = entity.sales_records
        
        if product_id:
            records = [r for r in records if r.product_id == product_id]
        
        if period_start:
            records = [r for r in records if r.period >= period_start]
        if period_end:
            records = [r for r in records if r.period <= period_end]
        
        total_revenue = sum(r.revenue for r in records)
        total_units = sum(r.units_sold for r in records)
        total_marketing = sum(r.marketing_spend or 0 for r in records)
        
        # Group by region
        by_region = {}
        for r in records:
            if r.region not in by_region:
                by_region[r.region] = {"revenue": 0, "units": 0}
            by_region[r.region]["revenue"] += r.revenue
            by_region[r.region]["units"] += r.units_sold
        
        return {
            "total_revenue": total_revenue,
            "total_units": total_units,
            "total_marketing_spend": total_marketing,
            "roi": (total_revenue / total_marketing) if total_marketing > 0 else None,
            "by_region": by_region,
            "record_count": len(records),
        }
    
    # Facility Management
    def get_facility_status(self, entity_id: str) -> Dict[str, Any]:
        """Get comprehensive facility status"""
        summary = self.entity_manager.get_compliance_summary(entity_id)
        if not summary:
            return {}
        
        entity = self.entity_manager.get_profile(entity_id)
        
        return {
            **summary,
            "products_count": len(entity.products) if entity else 0,
            "qa_documents_count": len(entity.qa_documents) if entity else 0,
            "recent_inspections": [
                {
                    "body": i.inspector_body,
                    "date": i.start_date.isoformat(),
                    "classification": i.classification,
                }
                for i in (entity.inspection_records[-3:] if entity else [])
            ],
        }
    
    async def get_regulatory_guidance(
        self,
        topic: str,
        regulatory_body: RegulatoryBody,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get regulatory guidance on a topic"""
        requirements = self.GMP_REQUIREMENTS.get(regulatory_body, {})
        
        return {
            "topic": topic,
            "regulatory_body": regulatory_body.value,
            "regulation_name": requirements.get("name", ""),
            "key_requirements": requirements.get("key_requirements", []),
            "guidance": f"Regulatory guidance for {topic} under {regulatory_body.value}...",
            "references": [],
        }
