import requests
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import os


class CourtListenerIngestion:
    """
    Data ingestion module for CourtListener API.
    Fetches legal opinions with jurisdiction metadata.
    """
    
    def __init__(self, api_token: Optional[str] = None):
        self.base_url = "https://www.courtlistener.com/api/rest/v3"
        self.api_token = api_token
        self.headers = {}
        if api_token:
            self.headers["Authorization"] = f"Token {api_token}"
    
    def search_opinions(self, query: str, jurisdiction: Optional[str] = None, 
                       max_results: int = 20) -> List[Dict]:
        """
        Search for legal opinions using CourtListener API.
        
        Args:
            query: Search query
            jurisdiction: Filter by jurisdiction (e.g., 'fed', 'ca', 'ny')
            max_results: Maximum number of results to fetch
            
        Returns:
            List of opinion documents with metadata
        """
        endpoint = f"{self.base_url}/search/"
        params = {
            "q": query,
            "type": "o",  # opinions
            "order_by": "score desc",
        }
        
        if jurisdiction:
            params["court"] = jurisdiction
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(self._parse_opinion(item))
            
            return results
        except Exception as e:
            print(f"Error fetching from CourtListener: {e}")
            return []
    
    def _parse_opinion(self, item: Dict) -> Dict:
        """
        Parse CourtListener opinion into standardized schema.
        
        Returns:
            Normalized document with metadata
        """
        return {
            "id": item.get("id", ""),
            "case_name": item.get("caseName", "Unknown Case"),
            "citation": item.get("citation", ["N/A"])[0] if item.get("citation") else "N/A",
            "court": item.get("court", "Unknown Court"),
            "jurisdiction": self._extract_jurisdiction(item.get("court", "")),
            "date_filed": item.get("dateFiled", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("absolute_url", ""),
            "text": item.get("text", item.get("snippet", "")),
            "document_type": "case_law"
        }
    
    def _extract_jurisdiction(self, court: str) -> str:
        """
        Extract jurisdiction from court identifier.
        Maps court codes to federal/state jurisdictions.
        """
        federal_courts = ["scotus", "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", 
                         "ca7", "ca8", "ca9", "ca10", "ca11", "cadc", "cafc"]
        
        if any(fed in court.lower() for fed in federal_courts):
            return "federal"
        return "state"
    
    def fetch_sample_corpus(self, topics: List[str] = None, 
                           docs_per_topic: int = 10) -> List[Dict]:
        """
        Fetch a sample corpus of legal documents across multiple topics.
        
        Args:
            topics: List of legal topics to search
            docs_per_topic: Number of documents per topic
            
        Returns:
            Combined list of documents
        """
        if topics is None:
            topics = [
                "contract dispute",
                "employment law",
                "intellectual property",
                "constitutional rights",
                "criminal procedure"
            ]
        
        all_docs = []
        for topic in topics:
            print(f"Fetching documents for: {topic}")
            docs = self.search_opinions(topic, max_results=docs_per_topic)
            all_docs.extend(docs)
        
        return all_docs
    
    def save_to_jsonl(self, documents: List[Dict], filepath: str = "legal_corpus.jsonl"):
        """
        Save documents to JSONL format for persistence.
        
        Args:
            documents: List of document dictionaries
            filepath: Output file path
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        print(f"Saved {len(documents)} documents to {filepath}")
    
    def load_from_jsonl(self, filepath: str = "legal_corpus.jsonl") -> List[Dict]:
        """
        Load documents from JSONL file.
        
        Args:
            filepath: Input file path
            
        Returns:
            List of document dictionaries
        """
        documents = []
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    documents.append(json.loads(line))
            print(f"Loaded {len(documents)} documents from {filepath}")
        return documents


def create_sample_legal_data() -> List[Dict]:
    """
    Create sample legal documents for demonstration purposes.
    This provides immediate functionality without API dependency.
    """
    return [
        {
            "id": "sample_1",
            "case_name": "Smith v. Jones",
            "citation": "123 F.3d 456 (9th Cir. 2020)",
            "court": "ca9",
            "jurisdiction": "federal",
            "date_filed": "2020-03-15",
            "text": "The Ninth Circuit held that employment contracts must be interpreted according to their plain meaning. When an employee handbook explicitly states that employment is at-will, courts should not infer additional job security protections absent clear and unambiguous language to the contrary. The court emphasized that employers have the right to modify policies, but must provide adequate notice to employees.",
            "snippet": "Employment contracts interpreted by plain meaning...",
            "url": "https://example.com/sample1",
            "document_type": "case_law"
        },
        {
            "id": "sample_2",
            "case_name": "TechCorp v. Innovation Labs",
            "citation": "567 F. Supp. 3d 890 (N.D. Cal. 2021)",
            "court": "californiad",
            "jurisdiction": "federal",
            "date_filed": "2021-06-22",
            "text": "The district court ruled that trade secret misappropriation requires proof that the defendant acquired information through improper means. Mere similarity between products is insufficient. The plaintiff must demonstrate that the defendant knew or should have known the information was obtained through breach of confidentiality. The court applied the Defend Trade Secrets Act and found that independent development is a complete defense.",
            "snippet": "Trade secret misappropriation requires improper acquisition...",
            "url": "https://example.com/sample2",
            "document_type": "case_law"
        },
        {
            "id": "sample_3",
            "case_name": "Miranda v. Arizona",
            "citation": "384 U.S. 436 (1966)",
            "court": "scotus",
            "jurisdiction": "federal",
            "date_filed": "1966-06-13",
            "text": "The Supreme Court established that criminal suspects must be informed of their constitutional rights before custodial interrogation. The Fifth Amendment privilege against self-incrimination requires law enforcement to advise individuals of their right to remain silent and right to counsel. Any statements obtained without proper Miranda warnings are inadmissible in court. This landmark decision fundamentally changed criminal procedure across the United States.",
            "snippet": "Constitutional rights during custodial interrogation...",
            "url": "https://example.com/sample3",
            "document_type": "case_law"
        },
        {
            "id": "sample_4",
            "case_name": "Johnson v. State Board of Education",
            "citation": "234 P.3d 567 (Cal. 2019)",
            "court": "cal",
            "jurisdiction": "state",
            "date_filed": "2019-09-10",
            "text": "The California Supreme Court held that state education regulations must comply with equal protection guarantees. School districts cannot implement policies that disproportionately burden students based on protected characteristics without demonstrating a compelling state interest. The court applied strict scrutiny review and found that less restrictive alternatives existed. This decision reinforced California's commitment to educational equity and non-discrimination.",
            "snippet": "Equal protection in education policy...",
            "url": "https://example.com/sample4",
            "document_type": "case_law"
        },
        {
            "id": "sample_5",
            "case_name": "United States v. Digital Privacy Foundation",
            "citation": "789 F.3d 123 (2nd Cir. 2022)",
            "court": "ca2",
            "jurisdiction": "federal",
            "date_filed": "2022-11-08",
            "text": "The Second Circuit addressed Fourth Amendment protections for digital communications. The court held that warrantless searches of electronic devices violate the Fourth Amendment absent exigent circumstances. Cloud-stored data receives the same constitutional protection as physical documents. Law enforcement must obtain a warrant supported by probable cause before accessing personal digital information. The decision balanced privacy rights with law enforcement needs in the digital age.",
            "snippet": "Fourth Amendment protections for digital data...",
            "url": "https://example.com/sample5",
            "document_type": "case_law"
        },
        {
            "id": "sample_6",
            "case_name": "Green Construction v. Workers Union Local 45",
            "citation": "456 N.Y.S.2d 789 (N.Y. App. Div. 2020)",
            "court": "nyappdiv",
            "jurisdiction": "state",
            "date_filed": "2020-04-17",
            "text": "The New York Appellate Division ruled on collective bargaining disputes in the construction industry. The court held that employers must bargain in good faith with certified union representatives. Unilateral changes to working conditions during active negotiations constitute unfair labor practices. The decision emphasized that labor law seeks to balance employer interests with worker protections and promote industrial peace through structured negotiations.",
            "snippet": "Collective bargaining and good faith negotiations...",
            "url": "https://example.com/sample6",
            "document_type": "case_law"
        },
        {
            "id": "sample_7",
            "case_name": "Environmental Defense Fund v. State EPA",
            "citation": "890 F.3d 234 (D.C. Cir. 2023)",
            "court": "cadc",
            "jurisdiction": "federal",
            "date_filed": "2023-02-28",
            "text": "The D.C. Circuit reviewed environmental regulations under the Clean Air Act. The court held that federal agencies must base regulations on scientific evidence and cannot ignore significant environmental harms. When data demonstrates public health risks, the EPA has a statutory duty to act. The decision reinforced the importance of evidence-based policymaking and judicial deference to agency expertise within statutory boundaries.",
            "snippet": "Environmental regulation and scientific evidence...",
            "url": "https://example.com/sample7",
            "document_type": "case_law"
        },
        {
            "id": "sample_8",
            "case_name": "Martinez v. Landlord Property Management",
            "citation": "345 Cal. Rptr. 3d 678 (Cal. Ct. App. 2021)",
            "court": "calctapp",
            "jurisdiction": "state",
            "date_filed": "2021-07-14",
            "text": "The California Court of Appeal addressed tenant rights under state housing law. The court held that landlords must maintain habitable premises and cannot retaliate against tenants who report code violations. Constructive eviction occurs when conditions become so poor that reasonable tenants are forced to leave. The decision protected vulnerable renters and clarified remedies available under California's tenant protection statutes.",
            "snippet": "Tenant rights and habitability requirements...",
            "url": "https://example.com/sample8",
            "document_type": "case_law"
        }
    ]
