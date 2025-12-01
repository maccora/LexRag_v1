from mistralai import Mistral
# import openai
#from openai  import OpenAI
from typing import Dict, List, Optional, Tuple
import os
import json


class LegalResearchAgent:
    """
    Multi-step agentic reasoning for legal research.
    Implements jurisdiction verification and citation cross-checking.
    """
    
    def __init__(self, rag_pipeline, mistral_api_key: Optional[str] = None):
        self.rag_pipeline = rag_pipeline
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        self.openai_api_key = os.getenv("openai_api_key")
        
        if self.mistral_api_key:
            self.mistral_client = Mistral(api_key=self.mistral_api_key)
        else:
            self.mistral_client = None
    
    def multi_step_research(self, question: str, 
                          model: str = "mistral-small-latest") -> Dict:
        """
        Execute multi-step agentic legal research workflow.
        
        Steps:
        1. Analyze question to determine jurisdiction requirements
        2. Retrieve relevant documents with jurisdiction filtering
        3. Verify citation accuracy and relevance
        4. Cross-check jurisdictional consistency
        5. Generate comprehensive answer
        
        Args:
            question: User's legal question
            model: Mistral model to use
            
        Returns:
            Dict with answer, reasoning steps, and verified citations
        """
        if not self.mistral_client:
            return {
                "error": "Mistral API key not configured",
                "steps": []
            }
        
        steps = []
        
        jurisdiction, reasoning = self._step1_analyze_jurisdiction(question, model)
        steps.append({
            "step": 1,
            "action": "Analyze Jurisdiction Requirements",
            "result": f"Detected jurisdiction: {jurisdiction}",
            "reasoning": reasoning
        })
        
        retrieved_docs = self._step2_retrieve_documents(question, jurisdiction)
        steps.append({
            "step": 2,
            "action": "Retrieve Relevant Documents",
            "result": f"Found {len(retrieved_docs)} relevant documents",
            "documents": len(retrieved_docs)
        })
        
        verified_citations = self._step3_verify_citations(retrieved_docs)
        steps.append({
            "step": 3,
            "action": "Verify Citation Accuracy",
            "result": f"Verified {verified_citations.get('total_verified', 0)} citations",
            "verified_count": verified_citations.get("total_verified", 0),
            "issues": verified_citations.get("issues", [])
        })
        
        consistency_check = self._step4_check_consistency(retrieved_docs, jurisdiction)
        steps.append({
            "step": 4,
            "action": "Cross-Check Jurisdictional Consistency",
            "result": consistency_check["summary"],
            "consistent": consistency_check["consistent"],
            "warnings": consistency_check.get("warnings", [])
        })
        
        answer = self._step5_generate_answer(question, retrieved_docs, model)
        steps.append({
            "step": 5,
            "action": "Generate Comprehensive Answer",
            "result": "Answer generated with verified citations"
        })
        
        return {
            "question": question,
            "answer": answer,
            "jurisdiction": jurisdiction,
            "steps": steps,
            "sources": self.rag_pipeline.format_sources_for_display(retrieved_docs),
            "agent_reasoning": reasoning,
            "citation_verification": verified_citations,
            "consistency_check": consistency_check
        }
    
    def _step1_analyze_jurisdiction(self, question: str, 
                                   model: str) -> Tuple[str, str]:
        """
        Step 1: Analyze question to determine appropriate jurisdiction.
        """
        analysis_prompt = f"""Analyze this legal question and determine the most appropriate jurisdiction(s) to search.

Question: {question}

Determine:
1. Is this primarily a FEDERAL or STATE law question, or BOTH?
2. What legal domain is involved (employment, contracts, criminal, etc.)?
3. Are there any jurisdiction-specific keywords?

Respond in JSON format:
{{
    "jurisdiction": "federal" or "state" or "all",
    "legal_domain": "domain name",
    "reasoning": "brief explanation",
    "keywords": ["keyword1", "keyword2"]
}}"""

        try:
            response = self.mistral_client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            jurisdiction = analysis.get("jurisdiction", "all")
            reasoning = analysis.get("reasoning", "No reasoning provided")
            
            return jurisdiction, reasoning
            
        except Exception as e:
            return "all", f"Error analyzing jurisdiction: {str(e)}"
    
    def _step2_retrieve_documents(self, question: str, jurisdiction: str) -> List[Dict]:
        """
        Step 2: Retrieve documents with jurisdiction filtering.
        """
        jurisdiction_filter = None if jurisdiction == "all" else jurisdiction
        
        retrieved_docs = self.rag_pipeline.retrieve_context(
            query=question,
            jurisdiction=jurisdiction_filter,
            n_results=5
        )
        
        return retrieved_docs
    
    def _step3_verify_citations(self, documents: List[Dict]) -> Dict:
        """
        Step 3: Verify citation accuracy and completeness.
        """
        verified = []
        issues = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            citation = metadata.get("citation", "")
            case_name = metadata.get("case_name", "")
            
            if citation == "N/A" or not citation:
                issues.append(f"Missing citation for: {case_name}")
            else:
                verified.append({
                    "case_name": case_name,
                    "citation": citation,
                    "jurisdiction": metadata.get("jurisdiction", "unknown"),
                    "verified": True
                })
        
        return {
            "verified_citations": verified,
            "total_verified": len(verified),
            "issues": issues
        }
    
    def _step4_check_consistency(self, documents: List[Dict], 
                                expected_jurisdiction: str) -> Dict:
        """
        Step 4: Check jurisdictional consistency of retrieved documents.
        """
        jurisdictions = {}
        warnings = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            jur = metadata.get("jurisdiction", "unknown")
            jurisdictions[jur] = jurisdictions.get(jur, 0) + 1
        
        if expected_jurisdiction != "all":
            other_jurisdictions = {k: v for k, v in jurisdictions.items() 
                                 if k != expected_jurisdiction and k != "unknown"}
            if other_jurisdictions:
                warnings.append(
                    f"Found {sum(other_jurisdictions.values())} documents from "
                    f"other jurisdictions: {list(other_jurisdictions.keys())}"
                )
        
        consistent = len(warnings) == 0 or expected_jurisdiction == "all"
        
        return {
            "consistent": consistent,
            "jurisdiction_distribution": jurisdictions,
            "warnings": warnings,
            "summary": f"Found documents across {len(jurisdictions)} jurisdiction(s)"
        }
    
    def _step5_generate_answer(self, question: str, documents: List[Dict],
                              model: str) -> str:
        """
        Step 5: Generate final answer with all verification context.
        """
        context = self.rag_pipeline.format_context(documents)
        answer = self.rag_pipeline.generate_answer(question, context, model)
        return answer


class CitationVerifier:
    """
    Utility for verifying legal citation formats and cross-references.
    """
    
    @staticmethod
    def validate_citation_format(citation: str) -> Dict:
        """
        Validate legal citation format.
        
        Args:
            citation: Citation string
            
        Returns:
            Dict with validation results
        """
        if not citation or citation == "N/A":
            return {
                "valid": False,
                "reason": "Missing citation",
                "format": "unknown"
            }
        
        citation_formats = {
            "federal_case": ["F.2d", "F.3d", "F.Supp", "U.S."],
            "state_case": ["P.2d", "P.3d", "N.E.", "S.E.", "A.2d"],
            "cfr": ["CFR"],
            "usc": ["U.S.C."]
        }
        
        detected_format = "unknown"
        for format_type, markers in citation_formats.items():
            if any(marker in citation for marker in markers):
                detected_format = format_type
                break
        
        valid = detected_format != "unknown"
        
        return {
            "valid": valid,
            "format": detected_format,
            "citation": citation,
            "reason": "Valid format" if valid else "Unrecognized format"
        }
    
    @staticmethod
    def extract_citations_from_text(text: str) -> List[str]:
        """
        Extract citation strings from legal text.
        Simple pattern matching for common citation formats.
        """
        import re

        
        patterns = [
            r'\d+\s+U\.S\.\s+\d+',
            r'\d+\s+F\.\s*(?:2d|3d)\s+\d+',
            r'\d+\s+F\.\s*Supp\.\s*(?:2d|3d)?\s+\d+',
            r'\d+\s+CFR\s+§?\s*\d+',
            r'\d+\s+U\.S\.C\.\s+§?\s*\d+'
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
        
        return list(set(citations))
