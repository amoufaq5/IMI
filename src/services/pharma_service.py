"""
UMI Pharmaceutical Service
QA/QC Document Generation, Compliance Tracking, and Facility Management
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import NotFoundError, ValidationError
from src.core.logging import get_logger
from src.models.pharma import (
    Facility,
    FacilityStatus,
    Document,
    DocumentType,
    DocumentStatus,
    ComplianceCheck,
    ComplianceStatus,
    ProductionBatch,
    BatchStatus,
)
from src.schemas.pharma import (
    FacilityCreate,
    FacilityUpdate,
    DocumentCreate,
    DocumentGenerateRequest,
    DocumentUpdate,
    ComplianceCheckCreate,
    ComplianceCheckUpdate,
    ProductionBatchCreate,
    ProductionBatchUpdate,
)

logger = get_logger(__name__)


class DocumentTemplates:
    """QA/QC Document Templates for AI generation."""
    
    TEMPLATES = {
        DocumentType.CLEANING_VALIDATION: {
            "sections": [
                "1. Purpose and Scope",
                "2. Responsibilities",
                "3. Equipment Description",
                "4. Cleaning Procedure",
                "5. Acceptance Criteria",
                "6. Sampling Procedures",
                "7. Analytical Methods",
                "8. Validation Protocol",
                "9. Results and Discussion",
                "10. Conclusion",
                "11. References",
                "Appendix A: Sampling Locations",
                "Appendix B: Analytical Data",
                "Appendix C: Deviation Reports",
            ],
            "required_fields": [
                "equipment_name",
                "equipment_id",
                "cleaning_method",
                "acceptance_criteria",
            ],
        },
        DocumentType.PROCESS_VALIDATION: {
            "sections": [
                "1. Introduction",
                "2. Scope",
                "3. Process Description",
                "4. Critical Process Parameters",
                "5. Critical Quality Attributes",
                "6. Validation Strategy",
                "7. Sampling Plan",
                "8. Acceptance Criteria",
                "9. Protocol Execution",
                "10. Results",
                "11. Conclusion",
                "12. References",
            ],
            "required_fields": [
                "product_name",
                "process_name",
                "batch_size",
            ],
        },
        DocumentType.HVAC_QUALIFICATION: {
            "sections": [
                "1. Purpose",
                "2. Scope",
                "3. System Description",
                "4. Design Qualification (DQ)",
                "5. Installation Qualification (IQ)",
                "6. Operational Qualification (OQ)",
                "7. Performance Qualification (PQ)",
                "8. Acceptance Criteria",
                "9. Test Results",
                "10. Conclusion",
                "11. Appendices",
            ],
            "required_fields": [
                "system_id",
                "area_classification",
                "temperature_range",
                "humidity_range",
                "air_changes_per_hour",
            ],
        },
        DocumentType.WATER_SYSTEM_VALIDATION: {
            "sections": [
                "1. Introduction",
                "2. System Description",
                "3. Water Quality Specifications",
                "4. Sampling Points",
                "5. Sampling Frequency",
                "6. Test Methods",
                "7. Alert and Action Limits",
                "8. Validation Protocol",
                "9. Results",
                "10. Trend Analysis",
                "11. Conclusion",
            ],
            "required_fields": [
                "water_type",
                "system_capacity",
                "sampling_points",
            ],
        },
        DocumentType.SOP: {
            "sections": [
                "1. Purpose",
                "2. Scope",
                "3. Responsibilities",
                "4. Definitions",
                "5. Materials and Equipment",
                "6. Procedure",
                "7. Documentation",
                "8. References",
                "9. Revision History",
            ],
            "required_fields": [
                "procedure_name",
                "department",
            ],
        },
        DocumentType.BATCH_RECORD: {
            "sections": [
                "1. Product Information",
                "2. Batch Information",
                "3. Raw Materials",
                "4. Equipment",
                "5. Manufacturing Instructions",
                "6. In-Process Controls",
                "7. Yield Calculations",
                "8. Packaging Instructions",
                "9. QC Sampling",
                "10. Batch Release",
                "11. Signatures",
            ],
            "required_fields": [
                "product_name",
                "batch_size",
                "batch_number",
            ],
        },
        DocumentType.DEVIATION_REPORT: {
            "sections": [
                "1. Deviation Description",
                "2. Discovery Details",
                "3. Immediate Actions",
                "4. Impact Assessment",
                "5. Root Cause Analysis",
                "6. Corrective Actions",
                "7. Preventive Actions",
                "8. Effectiveness Check",
                "9. Approval",
            ],
            "required_fields": [
                "deviation_description",
                "discovery_date",
                "affected_batches",
            ],
        },
        DocumentType.CAPA: {
            "sections": [
                "1. Problem Statement",
                "2. Immediate Containment",
                "3. Root Cause Investigation",
                "4. Corrective Actions",
                "5. Preventive Actions",
                "6. Implementation Plan",
                "7. Effectiveness Verification",
                "8. Closure",
            ],
            "required_fields": [
                "problem_description",
                "source",
            ],
        },
    }
    
    @classmethod
    def get_template(cls, doc_type: DocumentType) -> Optional[Dict[str, Any]]:
        """Get template for document type."""
        return cls.TEMPLATES.get(doc_type)
    
    @classmethod
    def validate_parameters(cls, doc_type: DocumentType, parameters: Dict[str, Any]) -> List[str]:
        """Validate required parameters for document generation."""
        template = cls.get_template(doc_type)
        if not template:
            return [f"No template found for document type: {doc_type}"]
        
        missing = []
        for field in template.get("required_fields", []):
            if field not in parameters or not parameters[field]:
                missing.append(field)
        
        return missing


class PharmaService:
    """Service for pharmaceutical facility and document management."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.templates = DocumentTemplates()
    
    # =========================================================================
    # Facility Management
    # =========================================================================
    
    async def create_facility(
        self,
        organization_id: uuid.UUID,
        data: FacilityCreate,
    ) -> Facility:
        """Create a new facility."""
        
        # Generate facility code if not provided
        facility_code = data.facility_code
        if not facility_code:
            count = await self._get_facility_count(organization_id)
            facility_code = f"FAC-{count + 1:04d}"
        
        facility = Facility(
            organization_id=organization_id,
            facility_code=facility_code,
            name=data.name,
            type=data.type,
            status=FacilityStatus.ACTIVE,
            address=data.address,
            city=data.city,
            country=data.country,
            postal_code=data.postal_code,
            phone=data.phone,
            email=data.email,
            facility_manager=data.facility_manager,
            license_number=data.license_number,
            license_expiry=data.license_expiry,
            regulatory_body=data.regulatory_body,
            certifications=[c.model_dump() for c in data.certifications] if data.certifications else [],
            total_area_sqm=data.total_area_sqm,
            production_capacity=data.production_capacity,
            cleanroom_classes=data.cleanroom_classes,
            product_types=data.product_types,
        )
        
        self.db.add(facility)
        await self.db.flush()
        
        logger.info(
            "facility_created",
            facility_id=str(facility.id),
            organization_id=str(organization_id),
            name=facility.name,
        )
        
        return facility
    
    async def get_facility(
        self,
        facility_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> Facility:
        """Get a facility by ID."""
        
        result = await self.db.execute(
            select(Facility).where(
                Facility.id == facility_id,
                Facility.organization_id == organization_id,
            )
        )
        facility = result.scalar_one_or_none()
        
        if not facility:
            raise NotFoundError("Facility", facility_id)
        
        return facility
    
    async def update_facility(
        self,
        facility_id: uuid.UUID,
        organization_id: uuid.UUID,
        data: FacilityUpdate,
    ) -> Facility:
        """Update a facility."""
        
        facility = await self.get_facility(facility_id, organization_id)
        
        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if field == "certifications" and value:
                value = [c.model_dump() if hasattr(c, 'model_dump') else c for c in value]
            setattr(facility, field, value)
        
        await self.db.flush()
        
        return facility
    
    async def list_facilities(
        self,
        organization_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """List facilities for an organization."""
        
        # Count total
        count_result = await self.db.execute(
            select(func.count(Facility.id)).where(
                Facility.organization_id == organization_id
            )
        )
        total = count_result.scalar()
        
        # Get page
        offset = (page - 1) * page_size
        result = await self.db.execute(
            select(Facility)
            .where(Facility.organization_id == organization_id)
            .order_by(Facility.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        facilities = result.scalars().all()
        
        return {
            "items": facilities,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size if total else 0,
        }
    
    async def _get_facility_count(self, organization_id: uuid.UUID) -> int:
        """Get count of facilities for organization."""
        result = await self.db.execute(
            select(func.count(Facility.id)).where(
                Facility.organization_id == organization_id
            )
        )
        return result.scalar() or 0
    
    # =========================================================================
    # Document Management
    # =========================================================================
    
    async def create_document(
        self,
        facility_id: uuid.UUID,
        organization_id: uuid.UUID,
        data: DocumentCreate,
    ) -> Document:
        """Create a document manually."""
        
        # Verify facility belongs to organization
        await self.get_facility(facility_id, organization_id)
        
        document = Document(
            facility_id=facility_id,
            document_number=data.document_number,
            title=data.title,
            type=data.type,
            version="1.0",
            status=DocumentStatus.DRAFT,
            content=data.content,
            content_text=data.content_text,
            regulation_reference=data.regulation_reference,
            effective_date=data.effective_date,
            review_date=data.review_date,
            ai_generated=False,
        )
        
        self.db.add(document)
        await self.db.flush()
        
        logger.info(
            "document_created",
            document_id=str(document.id),
            facility_id=str(facility_id),
            type=data.type.value,
        )
        
        return document
    
    async def generate_document(
        self,
        organization_id: uuid.UUID,
        request: DocumentGenerateRequest,
    ) -> Document:
        """Generate a QA/QC document using AI."""
        
        # Verify facility
        facility = await self.get_facility(request.facility_id, organization_id)
        
        # Validate parameters
        missing = self.templates.validate_parameters(request.type, request.parameters)
        if missing:
            raise ValidationError(
                f"Missing required parameters: {', '.join(missing)}",
                details={"missing_fields": missing},
            )
        
        # Get template
        template = self.templates.get_template(request.type)
        
        # Generate document number
        doc_count = await self._get_document_count(request.facility_id, request.type)
        doc_number = f"{request.type.value.upper()[:3]}-{facility.facility_code}-{doc_count + 1:04d}"
        
        # Generate title if not provided
        title = request.title
        if not title:
            title = self._generate_title(request.type, request.parameters)
        
        # Generate content structure
        content = self._generate_document_content(
            doc_type=request.type,
            template=template,
            parameters=request.parameters,
            facility=facility,
            regulation=request.regulation,
            regulatory_body=request.regulatory_body,
        )
        
        # Create document
        document = Document(
            facility_id=request.facility_id,
            document_number=doc_number,
            title=title,
            type=request.type,
            version="1.0",
            status=DocumentStatus.DRAFT,
            content=content,
            content_text=self._content_to_text(content),
            template_id=f"template_{request.type.value}",
            regulation_reference=f"{request.regulatory_body} {request.regulation}",
            ai_generated=True,
            ai_model_version="umi-pharma-v1",
            generation_prompt=str(request.parameters),
        )
        
        self.db.add(document)
        await self.db.flush()
        
        logger.info(
            "document_generated",
            document_id=str(document.id),
            facility_id=str(request.facility_id),
            type=request.type.value,
            ai_generated=True,
        )
        
        return document
    
    async def get_document(
        self,
        document_id: uuid.UUID,
        facility_id: uuid.UUID,
    ) -> Document:
        """Get a document by ID."""
        
        result = await self.db.execute(
            select(Document).where(
                Document.id == document_id,
                Document.facility_id == facility_id,
            )
        )
        document = result.scalar_one_or_none()
        
        if not document:
            raise NotFoundError("Document", document_id)
        
        return document
    
    async def update_document(
        self,
        document_id: uuid.UUID,
        facility_id: uuid.UUID,
        data: DocumentUpdate,
    ) -> Document:
        """Update a document."""
        
        document = await self.get_document(document_id, facility_id)
        
        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(document, field, value)
        
        # Update content_text if content changed
        if "content" in update_data and document.content:
            document.content_text = self._content_to_text(document.content)
        
        await self.db.flush()
        
        return document
    
    async def approve_document(
        self,
        document_id: uuid.UUID,
        facility_id: uuid.UUID,
        approved_by: str,
    ) -> Document:
        """Approve a document."""
        
        document = await self.get_document(document_id, facility_id)
        
        if document.status not in [DocumentStatus.DRAFT, DocumentStatus.UNDER_REVIEW]:
            raise ValidationError(f"Cannot approve document in status: {document.status}")
        
        document.status = DocumentStatus.APPROVED
        document.approved_by = approved_by
        document.approved_date = datetime.now(timezone.utc)
        
        await self.db.flush()
        
        return document
    
    async def list_documents(
        self,
        facility_id: uuid.UUID,
        doc_type: Optional[DocumentType] = None,
        status: Optional[DocumentStatus] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """List documents for a facility."""
        
        query = select(Document).where(Document.facility_id == facility_id)
        count_query = select(func.count(Document.id)).where(Document.facility_id == facility_id)
        
        if doc_type:
            query = query.where(Document.type == doc_type)
            count_query = count_query.where(Document.type == doc_type)
        
        if status:
            query = query.where(Document.status == status)
            count_query = count_query.where(Document.status == status)
        
        # Count total
        count_result = await self.db.execute(count_query)
        total = count_result.scalar()
        
        # Get page
        offset = (page - 1) * page_size
        result = await self.db.execute(
            query.order_by(Document.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        documents = result.scalars().all()
        
        return {
            "items": documents,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size if total else 0,
        }
    
    async def _get_document_count(
        self,
        facility_id: uuid.UUID,
        doc_type: DocumentType,
    ) -> int:
        """Get count of documents of a type for facility."""
        result = await self.db.execute(
            select(func.count(Document.id)).where(
                Document.facility_id == facility_id,
                Document.type == doc_type,
            )
        )
        return result.scalar() or 0
    
    def _generate_title(self, doc_type: DocumentType, parameters: Dict[str, Any]) -> str:
        """Generate document title from type and parameters."""
        type_names = {
            DocumentType.CLEANING_VALIDATION: "Cleaning Validation Protocol",
            DocumentType.PROCESS_VALIDATION: "Process Validation Protocol",
            DocumentType.HVAC_QUALIFICATION: "HVAC System Qualification",
            DocumentType.WATER_SYSTEM_VALIDATION: "Water System Validation",
            DocumentType.SOP: "Standard Operating Procedure",
            DocumentType.BATCH_RECORD: "Batch Manufacturing Record",
            DocumentType.DEVIATION_REPORT: "Deviation Report",
            DocumentType.CAPA: "CAPA Report",
        }
        
        base_title = type_names.get(doc_type, doc_type.value.replace("_", " ").title())
        
        # Add context from parameters
        if "equipment_name" in parameters:
            return f"{base_title} - {parameters['equipment_name']}"
        elif "product_name" in parameters:
            return f"{base_title} - {parameters['product_name']}"
        elif "procedure_name" in parameters:
            return f"{base_title} - {parameters['procedure_name']}"
        
        return base_title
    
    def _generate_document_content(
        self,
        doc_type: DocumentType,
        template: Dict[str, Any],
        parameters: Dict[str, Any],
        facility: Facility,
        regulation: str,
        regulatory_body: str,
    ) -> Dict[str, Any]:
        """Generate structured document content."""
        
        content = {
            "metadata": {
                "document_type": doc_type.value,
                "facility_name": facility.name,
                "facility_code": facility.facility_code,
                "regulation": regulation,
                "regulatory_body": regulatory_body,
                "generated_date": datetime.now(timezone.utc).isoformat(),
            },
            "parameters": parameters,
            "sections": {},
        }
        
        # Generate section content based on type
        for section in template.get("sections", []):
            section_key = section.lower().replace(" ", "_").replace(".", "")
            content["sections"][section_key] = {
                "title": section,
                "content": self._generate_section_content(doc_type, section, parameters, facility),
            }
        
        return content
    
    def _generate_section_content(
        self,
        doc_type: DocumentType,
        section: str,
        parameters: Dict[str, Any],
        facility: Facility,
    ) -> str:
        """Generate content for a specific section."""
        # This would normally call the AI service
        # For now, return placeholder with parameters
        
        section_lower = section.lower()
        
        if "purpose" in section_lower:
            return f"This document establishes the protocol for {doc_type.value.replace('_', ' ')} at {facility.name}."
        
        if "scope" in section_lower:
            return f"This protocol applies to {parameters.get('equipment_name', parameters.get('product_name', 'the specified system'))} at {facility.name} ({facility.facility_code})."
        
        if "responsibilities" in section_lower:
            return (
                "- Quality Assurance: Protocol approval and oversight\n"
                "- Production: Execution of validation activities\n"
                "- Quality Control: Analytical testing and results review\n"
                "- Engineering: Equipment support and maintenance"
            )
        
        if "acceptance criteria" in section_lower:
            criteria = parameters.get("acceptance_criteria", {})
            if criteria:
                return "\n".join([f"- {k}: {v}" for k, v in criteria.items()])
            return "Acceptance criteria to be defined based on product and process requirements."
        
        # Default placeholder
        return f"[Content to be generated for: {section}]"
    
    def _content_to_text(self, content: Dict[str, Any]) -> str:
        """Convert structured content to plain text."""
        lines = []
        
        # Add metadata
        meta = content.get("metadata", {})
        lines.append(f"Document Type: {meta.get('document_type', 'N/A')}")
        lines.append(f"Facility: {meta.get('facility_name', 'N/A')}")
        lines.append(f"Regulation: {meta.get('regulatory_body', '')} {meta.get('regulation', '')}")
        lines.append("")
        
        # Add sections
        for section_key, section_data in content.get("sections", {}).items():
            lines.append(section_data.get("title", section_key))
            lines.append("-" * len(section_data.get("title", section_key)))
            lines.append(section_data.get("content", ""))
            lines.append("")
        
        return "\n".join(lines)
    
    # =========================================================================
    # Compliance Management
    # =========================================================================
    
    async def create_compliance_check(
        self,
        facility_id: uuid.UUID,
        organization_id: uuid.UUID,
        data: ComplianceCheckCreate,
    ) -> ComplianceCheck:
        """Create a compliance check."""
        
        await self.get_facility(facility_id, organization_id)
        
        # Generate check number
        count = await self._get_compliance_count(facility_id)
        check_number = f"CC-{count + 1:04d}"
        
        check = ComplianceCheck(
            facility_id=facility_id,
            check_number=check_number,
            title=data.title,
            regulation=data.regulation,
            regulation_section=data.regulation_section,
            regulatory_body=data.regulatory_body,
            check_type=data.check_type,
            scope=data.scope,
            status=ComplianceStatus.PENDING,
            scheduled_date=data.scheduled_date,
            auditor=data.auditor,
        )
        
        self.db.add(check)
        await self.db.flush()
        
        return check
    
    async def _get_compliance_count(self, facility_id: uuid.UUID) -> int:
        """Get count of compliance checks for facility."""
        result = await self.db.execute(
            select(func.count(ComplianceCheck.id)).where(
                ComplianceCheck.facility_id == facility_id
            )
        )
        return result.scalar() or 0
    
    # =========================================================================
    # Production Batch Management
    # =========================================================================
    
    async def create_batch(
        self,
        facility_id: uuid.UUID,
        organization_id: uuid.UUID,
        data: ProductionBatchCreate,
    ) -> ProductionBatch:
        """Create a production batch record."""
        
        await self.get_facility(facility_id, organization_id)
        
        batch = ProductionBatch(
            facility_id=facility_id,
            batch_number=data.batch_number,
            product_name=data.product_name,
            product_code=data.product_code,
            status=BatchStatus.PLANNED,
            batch_size=data.batch_size,
            batch_size_unit=data.batch_size_unit,
            theoretical_yield=data.theoretical_yield,
            planned_start=data.planned_start,
            planned_end=data.planned_end,
            production_supervisor=data.production_supervisor,
        )
        
        self.db.add(batch)
        await self.db.flush()
        
        return batch
    
    async def update_batch(
        self,
        batch_id: uuid.UUID,
        facility_id: uuid.UUID,
        data: ProductionBatchUpdate,
    ) -> ProductionBatch:
        """Update a production batch."""
        
        result = await self.db.execute(
            select(ProductionBatch).where(
                ProductionBatch.id == batch_id,
                ProductionBatch.facility_id == facility_id,
            )
        )
        batch = result.scalar_one_or_none()
        
        if not batch:
            raise NotFoundError("ProductionBatch", batch_id)
        
        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if field in ["raw_materials", "qc_tests"] and value:
                value = [item.model_dump() if hasattr(item, 'model_dump') else item for item in value]
            setattr(batch, field, value)
        
        # Calculate yield percentage if both values present
        if batch.actual_yield and batch.theoretical_yield:
            batch.yield_percentage = (batch.actual_yield / batch.theoretical_yield) * 100
        
        await self.db.flush()
        
        return batch
