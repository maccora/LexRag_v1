import requests
from typing import List, Dict, Optional
import json
from datetime import datetime


class GovInfoIngestion:
    """
    Data ingestion from GovInfo API for federal regulations and statutes.
    API: https://api.govinfo.gov/docs/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.govinfo.gov"
        self.api_key = api_key
    
    def search_regulations(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Search for federal regulations via GovInfo API.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of regulation documents
        """
        endpoint = f"{self.base_url}/search"
        params = {
            "query": query,
            "pageSize": max_results,
            "offsetMark": "*",
            "collection": "CFR"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", []):
                results.append(self._parse_regulation(item))
            
            return results
        except Exception as e:
            print(f"Error fetching from GovInfo: {e}")
            return []
    
    def _parse_regulation(self, item: Dict) -> Dict:
        """Parse GovInfo regulation into standardized schema."""
        return {
            "id": item.get("packageId", ""),
            "case_name": item.get("title", "Unknown Regulation"),
            "citation": item.get("citation", "N/A"),
            "court": "govinfo",
            "jurisdiction": "federal",
            "date_filed": item.get("dateIssued", ""),
            "text": item.get("summary", item.get("title", "")),
            "snippet": item.get("summary", "")[:200],
            "url": item.get("packageLink", ""),
            "document_type": "regulation"
        }


class ECFRIngestion:
    """
    Data ingestion from eCFR (Electronic Code of Federal Regulations).
    API: https://www.ecfr.gov/developers/documentation/api/v1
    """
    
    def __init__(self):
        self.base_url = "https://www.ecfr.gov/api/search/v1"
    
    def search_cfr(self, query: str, title: Optional[int] = None, 
                   max_results: int = 20) -> List[Dict]:
        """
        Search the Code of Federal Regulations.
        
        Args:
            query: Search query
            title: CFR title number (e.g., 29 for Labor)
            max_results: Maximum number of results
            
        Returns:
            List of CFR documents
        """
        params = {
            "query": query,
            "per_page": min(max_results, 100)
        }
        
        if title:
            params["title"] = title
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(self._parse_cfr(item))
            
            return results
        except Exception as e:
            print(f"Error fetching from eCFR: {e}")
            return []
    
    def _parse_cfr(self, item: Dict) -> Dict:
        """Parse eCFR document into standardized schema."""
        title = item.get("title_number", "")
        section = item.get("section_number", "")
        citation = f"{title} CFR {section}" if title and section else "N/A"
        
        return {
            "id": f"ecfr_{title}_{section}",
            "case_name": item.get("section_title", "Unknown CFR Section"),
            "citation": citation,
            "court": "ecfr",
            "jurisdiction": "federal",
            "date_filed": item.get("effective_date", ""),
            "text": item.get("full_text", item.get("section_title", "")),
            "snippet": item.get("snippet", "")[:200],
            "url": item.get("html_url", ""),
            "document_type": "regulation"
        }


class RegulationsGovIngestion:
    """
    Data ingestion from Regulations.gov API.
    API: https://open.gsa.gov/api/regulationsgov/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.regulations.gov/v4"
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["X-Api-Key"] = api_key
    
    def search_documents(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Search regulations and comments.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of regulatory documents
        """
        endpoint = f"{self.base_url}/documents"
        params = {
            "filter[searchTerm]": query,
            "page[size]": min(max_results, 250)
        }
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("data", [])[:max_results]:
                results.append(self._parse_regulation_doc(item))
            
            return results
        except Exception as e:
            print(f"Error fetching from Regulations.gov: {e}")
            return []
    
    def _parse_regulation_doc(self, item: Dict) -> Dict:
        """Parse Regulations.gov document into standardized schema."""
        attributes = item.get("attributes", {})
        
        return {
            "id": item.get("id", ""),
            "case_name": attributes.get("title", "Unknown Document"),
            "citation": attributes.get("documentId", "N/A"),
            "court": "regulations_gov",
            "jurisdiction": "federal",
            "date_filed": attributes.get("postedDate", ""),
            "text": attributes.get("summary", attributes.get("title", "")),
            "snippet": attributes.get("summary", "")[:200],
            "url": f"https://www.regulations.gov/document/{item.get('id', '')}",
            "document_type": "regulatory_comment"
        }


def create_sample_regulatory_data() -> List[Dict]:
    """
    Create sample regulatory documents for demonstration.
    """
    return [
        {
            "id": "reg_sample_1",
            "case_name": "29 CFR § 1630.2 - Definitions (ADA Employment Regulations)",
            "citation": "29 CFR § 1630.2",
            "court": "ecfr",
            "jurisdiction": "federal",
            "date_filed": "2023-01-01",
            "text": "The Americans with Disabilities Act defines disability as a physical or mental impairment that substantially limits one or more major life activities. Employers must provide reasonable accommodations unless doing so would impose undue hardship on business operations. Major life activities include caring for oneself, performing manual tasks, seeing, hearing, eating, sleeping, walking, standing, lifting, bending, speaking, breathing, learning, reading, concentrating, thinking, communicating, and working.",
            "snippet": "ADA employment regulations defining disability and accommodation requirements...",
            "url": "https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XIV/part-1630",
            "document_type": "regulation"
        },
        {
            "id": "reg_sample_2",
            "case_name": "17 CFR § 240.10b-5 - Employment of manipulative and deceptive devices",
            "citation": "17 CFR § 240.10b-5",
            "court": "ecfr",
            "jurisdiction": "federal",
            "date_filed": "2022-06-15",
            "text": "It is unlawful for any person to employ any device, scheme, or artifice to defraud in connection with the purchase or sale of any security. This includes making untrue statements of material fact or omitting material facts necessary to make statements not misleading. The rule also prohibits engaging in any act, practice, or course of business which operates as fraud or deceit upon any person. This is the primary antifraud provision under federal securities law.",
            "snippet": "Securities fraud prohibition under SEC Rule 10b-5...",
            "url": "https://www.ecfr.gov/current/title-17/chapter-II/part-240/section-240.10b-5",
            "document_type": "regulation"
        },
        {
            "id": "reg_sample_3",
            "case_name": "16 CFR Part 312 - Children's Online Privacy Protection Rule (COPPA)",
            "citation": "16 CFR Part 312",
            "court": "ecfr",
            "jurisdiction": "federal",
            "date_filed": "2023-03-20",
            "text": "Operators of websites or online services directed to children under 13 must obtain verifiable parental consent before collecting personal information. The rule requires clear privacy policies, reasonable security measures, and limits on data collection to what is necessary. Parents have the right to review and delete their child's information. Violations can result in civil penalties. This rule implements the Children's Online Privacy Protection Act.",
            "snippet": "COPPA requirements for protecting children's online privacy...",
            "url": "https://www.ecfr.gov/current/title-16/chapter-I/subchapter-C/part-312",
            "document_type": "regulation"
        },
        {
            "id": "reg_sample_4",
            "case_name": "40 CFR § 52.21 - Prevention of significant deterioration of air quality",
            "citation": "40 CFR § 52.21",
            "court": "ecfr",
            "jurisdiction": "federal",
            "date_filed": "2022-11-08",
            "text": "New major sources and major modifications at existing sources must obtain permits demonstrating use of best available control technology (BACT). The rule protects air quality in areas meeting National Ambient Air Quality Standards. Applicants must conduct air quality analyses and demonstrate that emissions will not cause or contribute to violations. Public notice and comment periods are required. This implements the Clean Air Act's PSD program.",
            "snippet": "EPA air quality prevention of significant deterioration rules...",
            "url": "https://www.ecfr.gov/current/title-40/chapter-I/subchapter-C/part-52/subpart-A/section-52.21",
            "document_type": "regulation"
        }
    ]
