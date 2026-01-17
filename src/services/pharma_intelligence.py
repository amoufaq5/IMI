"""
UMI Pharmaceutical Intelligence Service
Advanced features for pharma companies:
- Drug Pipeline Tracking
- Clinical Trial Monitoring
- Regulatory Intelligence
- Competitive Analysis
- Market Intelligence
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger

logger = get_logger(__name__)


class DrugPhase(str, Enum):
    """Drug development phases."""
    DISCOVERY = "discovery"
    PRECLINICAL = "preclinical"
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    NDA_SUBMITTED = "nda_submitted"
    APPROVED = "approved"
    MARKETED = "marketed"
    WITHDRAWN = "withdrawn"


class RegulatoryStatus(str, Enum):
    """Regulatory submission status."""
    NOT_SUBMITTED = "not_submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    CONDITIONAL_APPROVAL = "conditional_approval"


@dataclass
class DrugCandidate:
    """Drug candidate in development pipeline."""
    id: str
    name: str
    generic_name: Optional[str] = None
    molecule_type: str = "small_molecule"  # small_molecule, biologic, gene_therapy, cell_therapy
    therapeutic_area: str = ""
    indication: str = ""
    mechanism_of_action: str = ""
    phase: DrugPhase = DrugPhase.DISCOVERY
    target: Optional[str] = None
    
    # Development timeline
    discovery_date: Optional[datetime] = None
    ind_filing_date: Optional[datetime] = None
    phase1_start: Optional[datetime] = None
    phase2_start: Optional[datetime] = None
    phase3_start: Optional[datetime] = None
    nda_submission_date: Optional[datetime] = None
    approval_date: Optional[datetime] = None
    launch_date: Optional[datetime] = None
    
    # Regulatory
    regulatory_status: Dict[str, RegulatoryStatus] = field(default_factory=dict)  # {region: status}
    
    # Commercial
    peak_sales_estimate: Optional[float] = None
    patent_expiry: Optional[datetime] = None
    
    # Metadata
    notes: str = ""
    risks: List[str] = field(default_factory=list)
    milestones: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ClinicalTrialAlert:
    """Alert for clinical trial updates."""
    trial_id: str
    title: str
    alert_type: str  # new_trial, status_change, results_posted, enrollment_update
    description: str
    severity: str  # info, warning, critical
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegulatoryAlert:
    """Alert for regulatory updates."""
    id: str
    region: str
    agency: str
    alert_type: str  # guidance_update, approval, rejection, warning_letter, recall
    title: str
    description: str
    severity: str
    timestamp: datetime
    affected_products: List[str] = field(default_factory=list)
    url: Optional[str] = None


@dataclass
class CompetitorIntel:
    """Competitive intelligence data."""
    company_name: str
    drug_name: str
    indication: str
    phase: DrugPhase
    estimated_launch: Optional[datetime] = None
    peak_sales_estimate: Optional[float] = None
    differentiation: str = ""
    threat_level: str = "medium"  # low, medium, high
    notes: str = ""


class DrugPipelineService:
    """
    Service for tracking drug development pipeline.
    Monitors internal and competitor pipelines.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._pipeline: Dict[str, DrugCandidate] = {}
    
    async def add_drug_candidate(
        self,
        organization_id: uuid.UUID,
        candidate: DrugCandidate,
    ) -> DrugCandidate:
        """Add a new drug candidate to the pipeline."""
        candidate.id = str(uuid.uuid4())
        self._pipeline[candidate.id] = candidate
        
        logger.info(
            "drug_candidate_added",
            candidate_id=candidate.id,
            name=candidate.name,
            phase=candidate.phase.value,
        )
        
        return candidate
    
    async def update_phase(
        self,
        candidate_id: str,
        new_phase: DrugPhase,
        notes: str = "",
    ) -> DrugCandidate:
        """Update drug candidate phase."""
        candidate = self._pipeline.get(candidate_id)
        if not candidate:
            raise ValueError(f"Drug candidate not found: {candidate_id}")
        
        old_phase = candidate.phase
        candidate.phase = new_phase
        
        # Record milestone
        candidate.milestones.append({
            "date": datetime.now(timezone.utc).isoformat(),
            "event": f"Phase transition: {old_phase.value} -> {new_phase.value}",
            "notes": notes,
        })
        
        # Update timeline dates
        now = datetime.now(timezone.utc)
        if new_phase == DrugPhase.PHASE_1 and not candidate.phase1_start:
            candidate.phase1_start = now
        elif new_phase == DrugPhase.PHASE_2 and not candidate.phase2_start:
            candidate.phase2_start = now
        elif new_phase == DrugPhase.PHASE_3 and not candidate.phase3_start:
            candidate.phase3_start = now
        elif new_phase == DrugPhase.NDA_SUBMITTED and not candidate.nda_submission_date:
            candidate.nda_submission_date = now
        elif new_phase == DrugPhase.APPROVED and not candidate.approval_date:
            candidate.approval_date = now
        
        return candidate
    
    async def get_pipeline_summary(
        self,
        organization_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """Get summary of drug pipeline by phase."""
        summary = {phase.value: [] for phase in DrugPhase}
        
        for candidate in self._pipeline.values():
            summary[candidate.phase.value].append({
                "id": candidate.id,
                "name": candidate.name,
                "indication": candidate.indication,
                "therapeutic_area": candidate.therapeutic_area,
            })
        
        return {
            "total_candidates": len(self._pipeline),
            "by_phase": summary,
            "phase_counts": {phase: len(candidates) for phase, candidates in summary.items()},
        }
    
    async def get_timeline_forecast(
        self,
        candidate_id: str,
    ) -> Dict[str, Any]:
        """Forecast development timeline for a drug candidate."""
        candidate = self._pipeline.get(candidate_id)
        if not candidate:
            raise ValueError(f"Drug candidate not found: {candidate_id}")
        
        # Average phase durations (in months)
        phase_durations = {
            DrugPhase.DISCOVERY: 48,
            DrugPhase.PRECLINICAL: 18,
            DrugPhase.PHASE_1: 12,
            DrugPhase.PHASE_2: 24,
            DrugPhase.PHASE_3: 36,
            DrugPhase.NDA_SUBMITTED: 12,
        }
        
        current_phase = candidate.phase
        forecast = {
            "current_phase": current_phase.value,
            "estimated_milestones": [],
        }
        
        # Calculate remaining timeline
        phases = list(DrugPhase)
        current_idx = phases.index(current_phase)
        
        estimated_date = datetime.now(timezone.utc)
        for phase in phases[current_idx:]:
            if phase in phase_durations:
                duration = phase_durations[phase]
                estimated_date += timedelta(days=duration * 30)
                forecast["estimated_milestones"].append({
                    "phase": phase.value,
                    "estimated_completion": estimated_date.isoformat(),
                    "duration_months": duration,
                })
        
        if forecast["estimated_milestones"]:
            forecast["estimated_approval"] = forecast["estimated_milestones"][-1]["estimated_completion"]
        
        return forecast


class ClinicalTrialMonitorService:
    """
    Service for monitoring clinical trials.
    Tracks own trials and competitor trials.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._alerts: List[ClinicalTrialAlert] = []
        self._watched_trials: Dict[str, Dict[str, Any]] = {}
    
    async def watch_trial(
        self,
        organization_id: uuid.UUID,
        nct_id: str,
        watch_config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Add a clinical trial to watch list."""
        config = watch_config or {
            "notify_status_change": True,
            "notify_results": True,
            "notify_enrollment": True,
        }
        
        self._watched_trials[nct_id] = {
            "nct_id": nct_id,
            "organization_id": str(organization_id),
            "config": config,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        
        return self._watched_trials[nct_id]
    
    async def get_trial_alerts(
        self,
        organization_id: uuid.UUID,
        since: Optional[datetime] = None,
        severity: Optional[str] = None,
    ) -> List[ClinicalTrialAlert]:
        """Get clinical trial alerts."""
        alerts = self._alerts
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    async def analyze_competitive_trials(
        self,
        therapeutic_area: str,
        indication: str,
    ) -> Dict[str, Any]:
        """Analyze competitive landscape for clinical trials."""
        # This would integrate with ClinicalTrials.gov data
        return {
            "therapeutic_area": therapeutic_area,
            "indication": indication,
            "total_active_trials": 0,
            "phase_distribution": {
                "phase_1": 0,
                "phase_2": 0,
                "phase_3": 0,
            },
            "top_sponsors": [],
            "enrollment_trends": [],
            "analysis_date": datetime.now(timezone.utc).isoformat(),
        }


class RegulatoryIntelligenceService:
    """
    Service for regulatory intelligence.
    Monitors FDA, EMA, MHRA, and other agencies.
    """
    
    AGENCIES = {
        "US": {"name": "FDA", "url": "https://www.fda.gov"},
        "EU": {"name": "EMA", "url": "https://www.ema.europa.eu"},
        "UK": {"name": "MHRA", "url": "https://www.gov.uk/mhra"},
        "UAE": {"name": "UAE MOH", "url": "https://mohap.gov.ae"},
        "Japan": {"name": "PMDA", "url": "https://www.pmda.go.jp"},
        "China": {"name": "NMPA", "url": "https://www.nmpa.gov.cn"},
    }
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._alerts: List[RegulatoryAlert] = []
        self._tracked_guidances: List[Dict[str, Any]] = []
    
    async def get_regulatory_alerts(
        self,
        organization_id: uuid.UUID,
        region: Optional[str] = None,
        alert_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[RegulatoryAlert]:
        """Get regulatory alerts."""
        alerts = self._alerts
        
        if region:
            alerts = [a for a in alerts if a.region == region]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    async def track_guidance(
        self,
        organization_id: uuid.UUID,
        region: str,
        guidance_id: str,
        title: str,
    ) -> Dict[str, Any]:
        """Track a regulatory guidance document."""
        tracking = {
            "id": str(uuid.uuid4()),
            "organization_id": str(organization_id),
            "region": region,
            "guidance_id": guidance_id,
            "title": title,
            "tracked_at": datetime.now(timezone.utc).isoformat(),
        }
        
        self._tracked_guidances.append(tracking)
        return tracking
    
    async def get_submission_requirements(
        self,
        region: str,
        product_type: str,
    ) -> Dict[str, Any]:
        """Get regulatory submission requirements for a region."""
        # Common requirements by region
        requirements = {
            "US": {
                "agency": "FDA",
                "submission_types": ["IND", "NDA", "BLA", "ANDA"],
                "key_documents": [
                    "Module 1: Administrative Information",
                    "Module 2: CTD Summaries",
                    "Module 3: Quality",
                    "Module 4: Nonclinical Study Reports",
                    "Module 5: Clinical Study Reports",
                ],
                "review_timeline_days": 365,
                "user_fees": True,
            },
            "EU": {
                "agency": "EMA",
                "submission_types": ["Centralised", "Decentralised", "Mutual Recognition"],
                "key_documents": [
                    "Module 1: EU Administrative Information",
                    "Module 2: CTD Summaries",
                    "Module 3: Quality",
                    "Module 4: Nonclinical Study Reports",
                    "Module 5: Clinical Study Reports",
                ],
                "review_timeline_days": 210,
                "user_fees": True,
            },
            "UK": {
                "agency": "MHRA",
                "submission_types": ["National", "Reliance"],
                "key_documents": [
                    "CTD Format",
                    "UK-specific Module 1",
                ],
                "review_timeline_days": 150,
                "user_fees": True,
            },
        }
        
        return requirements.get(region, {
            "agency": self.AGENCIES.get(region, {}).get("name", "Unknown"),
            "submission_types": [],
            "key_documents": [],
            "review_timeline_days": None,
        })
    
    async def generate_regulatory_strategy(
        self,
        product_name: str,
        product_type: str,
        target_regions: List[str],
        indication: str,
    ) -> Dict[str, Any]:
        """Generate regulatory strategy recommendations."""
        strategy = {
            "product": product_name,
            "product_type": product_type,
            "indication": indication,
            "target_regions": target_regions,
            "recommendations": [],
            "timeline": [],
            "risks": [],
        }
        
        # Add recommendations based on regions
        if "US" in target_regions:
            strategy["recommendations"].append({
                "region": "US",
                "pathway": "Standard NDA" if product_type == "small_molecule" else "BLA",
                "considerations": [
                    "Consider Breakthrough Therapy designation if applicable",
                    "Fast Track designation may accelerate review",
                    "Pre-IND meeting recommended",
                ],
            })
        
        if "EU" in target_regions:
            strategy["recommendations"].append({
                "region": "EU",
                "pathway": "Centralised Procedure",
                "considerations": [
                    "PRIME designation for priority medicines",
                    "Scientific advice from EMA recommended",
                    "Rapporteur selection important",
                ],
            })
        
        # Risk assessment
        strategy["risks"] = [
            "Clinical trial design may need region-specific modifications",
            "Manufacturing site inspections required for each region",
            "Labeling requirements vary by region",
        ]
        
        return strategy


class CompetitiveIntelligenceService:
    """
    Service for competitive intelligence.
    Tracks competitor activities, market dynamics.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._competitors: Dict[str, List[CompetitorIntel]] = {}
    
    async def add_competitor_drug(
        self,
        organization_id: uuid.UUID,
        intel: CompetitorIntel,
    ) -> CompetitorIntel:
        """Add competitor drug intelligence."""
        org_key = str(organization_id)
        if org_key not in self._competitors:
            self._competitors[org_key] = []
        
        self._competitors[org_key].append(intel)
        return intel
    
    async def get_competitive_landscape(
        self,
        organization_id: uuid.UUID,
        therapeutic_area: Optional[str] = None,
        indication: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get competitive landscape analysis."""
        org_key = str(organization_id)
        competitors = self._competitors.get(org_key, [])
        
        if therapeutic_area:
            competitors = [c for c in competitors if c.indication and therapeutic_area.lower() in c.indication.lower()]
        
        if indication:
            competitors = [c for c in competitors if c.indication and indication.lower() in c.indication.lower()]
        
        # Analyze by phase
        phase_analysis = {}
        for phase in DrugPhase:
            phase_competitors = [c for c in competitors if c.phase == phase]
            phase_analysis[phase.value] = {
                "count": len(phase_competitors),
                "drugs": [{"company": c.company_name, "drug": c.drug_name} for c in phase_competitors],
            }
        
        # Threat assessment
        high_threats = [c for c in competitors if c.threat_level == "high"]
        
        return {
            "total_competitors": len(competitors),
            "by_phase": phase_analysis,
            "high_threat_count": len(high_threats),
            "high_threats": [
                {
                    "company": c.company_name,
                    "drug": c.drug_name,
                    "indication": c.indication,
                    "phase": c.phase.value,
                    "differentiation": c.differentiation,
                }
                for c in high_threats
            ],
            "analysis_date": datetime.now(timezone.utc).isoformat(),
        }
    
    async def generate_competitive_report(
        self,
        organization_id: uuid.UUID,
        indication: str,
    ) -> Dict[str, Any]:
        """Generate comprehensive competitive report."""
        landscape = await self.get_competitive_landscape(organization_id, indication=indication)
        
        report = {
            "title": f"Competitive Intelligence Report: {indication}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "executive_summary": "",
            "market_overview": {
                "indication": indication,
                "total_competitors": landscape["total_competitors"],
                "market_dynamics": "Analysis pending",
            },
            "competitive_landscape": landscape,
            "strategic_recommendations": [
                "Monitor Phase 3 competitors closely",
                "Differentiate on efficacy/safety profile",
                "Consider combination therapy opportunities",
                "Evaluate pricing strategy based on competitive positioning",
            ],
            "key_events_to_watch": [],
        }
        
        return report


class PharmaIntelligenceService:
    """
    Unified service for all pharmaceutical intelligence features.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.pipeline = DrugPipelineService(db)
        self.trials = ClinicalTrialMonitorService(db)
        self.regulatory = RegulatoryIntelligenceService(db)
        self.competitive = CompetitiveIntelligenceService(db)
    
    async def get_dashboard(
        self,
        organization_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """Get unified intelligence dashboard."""
        pipeline_summary = await self.pipeline.get_pipeline_summary(organization_id)
        trial_alerts = await self.trials.get_trial_alerts(
            organization_id,
            since=datetime.now(timezone.utc) - timedelta(days=7),
        )
        regulatory_alerts = await self.regulatory.get_regulatory_alerts(
            organization_id,
            since=datetime.now(timezone.utc) - timedelta(days=7),
        )
        competitive_landscape = await self.competitive.get_competitive_landscape(organization_id)
        
        return {
            "pipeline": {
                "total_candidates": pipeline_summary["total_candidates"],
                "phase_counts": pipeline_summary["phase_counts"],
            },
            "trials": {
                "recent_alerts": len(trial_alerts),
                "critical_alerts": len([a for a in trial_alerts if a.severity == "critical"]),
            },
            "regulatory": {
                "recent_alerts": len(regulatory_alerts),
                "critical_alerts": len([a for a in regulatory_alerts if a.severity == "critical"]),
            },
            "competitive": {
                "total_competitors": competitive_landscape["total_competitors"],
                "high_threats": competitive_landscape["high_threat_count"],
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
