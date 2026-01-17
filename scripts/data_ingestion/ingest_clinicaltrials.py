"""
UMI ClinicalTrials.gov Data Ingestion Pipeline
Fetches clinical trial data from ClinicalTrials.gov API for RAG
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from tqdm import tqdm


@dataclass
class ClinicalTrial:
    """Represents a clinical trial."""
    nct_id: str
    title: str
    brief_summary: str
    detailed_description: str
    conditions: List[str]
    interventions: List[Dict[str, str]]
    phase: str
    status: str
    enrollment: int
    start_date: str
    completion_date: str
    sponsor: str
    locations: List[str]
    eligibility_criteria: str
    primary_outcomes: List[str]
    secondary_outcomes: List[str]
    study_type: str
    keywords: List[str] = field(default_factory=list)


class ClinicalTrialsClient:
    """
    Client for ClinicalTrials.gov API v2.
    https://clinicaltrials.gov/data-api/api
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def search_studies(
        self,
        query: str,
        max_results: int = None,  # No limit by default
        status: Optional[List[str]] = None,
        phase: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for clinical trials.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            status: Filter by status (e.g., RECRUITING, COMPLETED)
            phase: Filter by phase (e.g., PHASE1, PHASE2, PHASE3)
        
        Returns:
            List of study data
        """
        all_studies = []
        page_size = 100  # API max per page
        page_token = None
        
        while max_results is None or len(all_studies) < max_results:
            params = {
                "query.term": query,
                "pageSize": page_size,
                "fields": "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,"
                         "Condition,InterventionName,InterventionType,Phase,OverallStatus,"
                         "EnrollmentCount,StartDate,CompletionDate,LeadSponsorName,"
                         "LocationCity,LocationCountry,EligibilityCriteria,"
                         "PrimaryOutcomeMeasure,SecondaryOutcomeMeasure,StudyType,Keyword",
            }
            
            if page_token:
                params["pageToken"] = page_token
            
            if status:
                params["filter.overallStatus"] = ",".join(status)
            
            if phase:
                params["filter.phase"] = ",".join(phase)
            
            try:
                response = await self.client.get(
                    f"{self.BASE_URL}/studies",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
                
                studies = data.get("studies", [])
                all_studies.extend(studies)
                
                page_token = data.get("nextPageToken")
                if not page_token or not studies:
                    break
                    
            except Exception as e:
                print(f"Error fetching studies: {e}")
                break
            
            await asyncio.sleep(0.3)  # Rate limiting
        
        return all_studies if max_results is None else all_studies[:max_results]
    
    def parse_study(self, study_data: Dict[str, Any]) -> Optional[ClinicalTrial]:
        """Parse study data into ClinicalTrial object."""
        try:
            protocol = study_data.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            desc_module = protocol.get("descriptionModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            design_module = protocol.get("designModule", {})
            status_module = protocol.get("statusModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            eligibility_module = protocol.get("eligibilityModule", {})
            outcomes_module = protocol.get("outcomesModule", {})
            arms_module = protocol.get("armsInterventionsModule", {})
            contacts_module = protocol.get("contactsLocationsModule", {})
            
            # NCT ID
            nct_id = id_module.get("nctId", "")
            if not nct_id:
                return None
            
            # Title
            title = id_module.get("officialTitle") or id_module.get("briefTitle", "")
            
            # Descriptions
            brief_summary = desc_module.get("briefSummary", "")
            detailed_description = desc_module.get("detailedDescription", "")
            
            # Conditions
            conditions = conditions_module.get("conditions", [])
            keywords = conditions_module.get("keywords", [])
            
            # Interventions
            interventions = []
            for intervention in arms_module.get("interventions", []):
                interventions.append({
                    "name": intervention.get("name", ""),
                    "type": intervention.get("type", ""),
                    "description": intervention.get("description", ""),
                })
            
            # Phase
            phases = design_module.get("phases", [])
            phase = ", ".join(phases) if phases else "N/A"
            
            # Status
            status = status_module.get("overallStatus", "")
            
            # Enrollment
            enrollment_info = design_module.get("enrollmentInfo", {})
            enrollment = enrollment_info.get("count", 0) or 0
            
            # Dates
            start_date_struct = status_module.get("startDateStruct", {})
            start_date = start_date_struct.get("date", "")
            
            completion_date_struct = status_module.get("completionDateStruct", {})
            completion_date = completion_date_struct.get("date", "")
            
            # Sponsor
            lead_sponsor = sponsor_module.get("leadSponsor", {})
            sponsor = lead_sponsor.get("name", "")
            
            # Locations
            locations = []
            for location in contacts_module.get("locations", [])[:10]:
                city = location.get("city", "")
                country = location.get("country", "")
                if city or country:
                    locations.append(f"{city}, {country}".strip(", "))
            
            # Eligibility
            eligibility_criteria = eligibility_module.get("eligibilityCriteria", "")
            
            # Outcomes
            primary_outcomes = []
            for outcome in outcomes_module.get("primaryOutcomes", []):
                measure = outcome.get("measure", "")
                if measure:
                    primary_outcomes.append(measure)
            
            secondary_outcomes = []
            for outcome in outcomes_module.get("secondaryOutcomes", []):
                measure = outcome.get("measure", "")
                if measure:
                    secondary_outcomes.append(measure)
            
            # Study type
            study_type = design_module.get("studyType", "")
            
            return ClinicalTrial(
                nct_id=nct_id,
                title=title,
                brief_summary=brief_summary,
                detailed_description=detailed_description,
                conditions=conditions,
                interventions=interventions,
                phase=phase,
                status=status,
                enrollment=enrollment,
                start_date=start_date,
                completion_date=completion_date,
                sponsor=sponsor,
                locations=locations,
                eligibility_criteria=eligibility_criteria,
                primary_outcomes=primary_outcomes,
                secondary_outcomes=secondary_outcomes,
                study_type=study_type,
                keywords=keywords,
            )
        
        except Exception as e:
            print(f"Error parsing study: {e}")
            return None
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class ClinicalTrialsIngestionPipeline:
    """
    Pipeline for ingesting clinical trials into UMI knowledge base.
    """
    
    # Medical conditions to search for trials
    SEARCH_TERMS = [
        # Major diseases
        "diabetes",
        "hypertension",
        "heart failure",
        "coronary artery disease",
        "atrial fibrillation",
        "stroke",
        "chronic kidney disease",
        "COPD",
        "asthma",
        "pneumonia",
        "COVID-19",
        "influenza",
        
        # Cancer
        "breast cancer",
        "lung cancer",
        "colorectal cancer",
        "prostate cancer",
        "leukemia",
        "lymphoma",
        "melanoma",
        "pancreatic cancer",
        "ovarian cancer",
        "brain tumor",
        
        # Mental health
        "depression",
        "anxiety",
        "schizophrenia",
        "bipolar disorder",
        "PTSD",
        "ADHD",
        "autism",
        "Alzheimer disease",
        "Parkinson disease",
        "multiple sclerosis",
        
        # Autoimmune
        "rheumatoid arthritis",
        "lupus",
        "psoriasis",
        "inflammatory bowel disease",
        "Crohn disease",
        "ulcerative colitis",
        "multiple sclerosis",
        
        # Infectious diseases
        "HIV",
        "hepatitis B",
        "hepatitis C",
        "tuberculosis",
        "malaria",
        "sepsis",
        
        # Metabolic
        "obesity",
        "metabolic syndrome",
        "hyperlipidemia",
        "thyroid disorders",
        
        # Pain
        "chronic pain",
        "migraine",
        "fibromyalgia",
        "neuropathic pain",
        
        # Pediatric
        "pediatric asthma",
        "pediatric diabetes",
        "childhood cancer",
        "neonatal",
        
        # Women's health
        "pregnancy",
        "preeclampsia",
        "endometriosis",
        "polycystic ovary syndrome",
        "menopause",
        
        # Rare diseases
        "cystic fibrosis",
        "sickle cell disease",
        "hemophilia",
        "muscular dystrophy",
        
        # Drug classes (for intervention-based searches)
        "immunotherapy",
        "gene therapy",
        "CAR-T",
        "monoclonal antibody",
        "vaccine",
        "stem cell",
    ]
    
    def __init__(
        self,
        output_dir: str = "data/knowledge_base/clinical_trials",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = ClinicalTrialsClient()
        self.trials: List[ClinicalTrial] = []
    
    async def fetch_trials_for_term(
        self,
        term: str,
        max_results: int = None,  # No limit by default
    ) -> List[ClinicalTrial]:
        """Fetch trials for a specific search term."""
        trials = []
        
        # Fetch recruiting and recently completed trials
        studies = await self.client.search_studies(
            query=term,
            max_results=max_results,
            status=["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"],
        )
        
        for study in studies:
            trial = self.client.parse_study(study)
            if trial:
                trials.append(trial)
        
        return trials
    
    async def run(self, max_per_term: int = None) -> None:  # No limit by default
        """Run the full ingestion pipeline."""
        print("=" * 60)
        print("UMI ClinicalTrials.gov Ingestion Pipeline")
        print("=" * 60)
        
        all_trials = []
        
        for term in tqdm(self.SEARCH_TERMS, desc="Fetching trials"):
            try:
                trials = await self.fetch_trials_for_term(term, max_results=max_per_term)
                all_trials.extend(trials)
                print(f"  {term}: {len(trials)} trials")
            except Exception as e:
                print(f"  Error fetching {term}: {e}")
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Deduplicate by NCT ID
        seen_ids = set()
        unique_trials = []
        for trial in all_trials:
            if trial.nct_id not in seen_ids:
                seen_ids.add(trial.nct_id)
                unique_trials.append(trial)
        
        self.trials = unique_trials
        print(f"\nTotal unique trials: {len(self.trials)}")
        
        # Save
        await self.save()
        
        # Close client
        await self.client.close()
    
    async def save(self) -> None:
        """Save trials to disk."""
        output_file = self.output_dir / "trials.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for trial in self.trials:
                # Create comprehensive content for RAG
                content_parts = [
                    f"Clinical Trial: {trial.title}",
                    f"NCT ID: {trial.nct_id}",
                    f"Phase: {trial.phase}",
                    f"Status: {trial.status}",
                    f"Study Type: {trial.study_type}",
                    "",
                    "Summary:",
                    trial.brief_summary,
                    "",
                ]
                
                if trial.conditions:
                    content_parts.append("Conditions:")
                    content_parts.extend([f"- {c}" for c in trial.conditions])
                    content_parts.append("")
                
                if trial.interventions:
                    content_parts.append("Interventions:")
                    for intervention in trial.interventions:
                        content_parts.append(f"- {intervention.get('type', '')}: {intervention.get('name', '')}")
                    content_parts.append("")
                
                if trial.primary_outcomes:
                    content_parts.append("Primary Outcomes:")
                    content_parts.extend([f"- {o}" for o in trial.primary_outcomes[:5]])
                    content_parts.append("")
                
                if trial.eligibility_criteria:
                    content_parts.append("Eligibility Criteria:")
                    content_parts.append(trial.eligibility_criteria[:2000])
                
                doc = {
                    "id": f"trial_{trial.nct_id}",
                    "title": trial.title,
                    "content": "\n".join(content_parts),
                    "metadata": {
                        "nct_id": trial.nct_id,
                        "phase": trial.phase,
                        "status": trial.status,
                        "conditions": trial.conditions,
                        "sponsor": trial.sponsor,
                        "enrollment": trial.enrollment,
                        "start_date": trial.start_date,
                        "completion_date": trial.completion_date,
                        "locations": trial.locations[:5],
                        "source": "ClinicalTrials.gov",
                    },
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Saved to: {output_file}")
        
        # Save statistics
        stats = {
            "total_trials": len(self.trials),
            "ingestion_date": datetime.now().isoformat(),
            "search_terms": self.SEARCH_TERMS,
            "status_breakdown": {},
            "phase_breakdown": {},
        }
        
        # Calculate breakdowns
        for trial in self.trials:
            stats["status_breakdown"][trial.status] = stats["status_breakdown"].get(trial.status, 0) + 1
            stats["phase_breakdown"][trial.phase] = stats["phase_breakdown"].get(trial.phase, 0) + 1
        
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


async def main():
    """Run the ClinicalTrials.gov ingestion pipeline with no limits."""
    pipeline = ClinicalTrialsIngestionPipeline()
    await pipeline.run(max_per_term=None)  # Fetch all available trials


if __name__ == "__main__":
    asyncio.run(main())
