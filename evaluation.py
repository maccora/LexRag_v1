from mistralai import Mistral
from typing import Dict, List, Optional
import os
import json


class AIJudgeEvaluator:
    """
    AI-as-judge evaluation framework using Mistral to score answer quality.
    Evaluates factual accuracy, citation validity, and jurisdictional alignment.
    """
    
    def __init__(self, mistral_api_key: Optional[str] = None):
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        
        if self.mistral_api_key:
            self.mistral_client = Mistral(api_key=self.mistral_api_key)
        else:
            self.mistral_client = None
    
    def evaluate_answer(self, question: str, answer: str, sources: List[Dict],
                       model: str = "mistral-small-latest") -> Dict:
        """
        Evaluate legal answer quality using AI judge.
        
        Args:
            question: User's legal question
            answer: Generated answer
            sources: Retrieved source documents
            model: Mistral model for evaluation
            
        Returns:
            Dictionary with scores and feedback
        """
        if not self.mistral_client:
            return {
                "error": "Mistral API key not configured",
                "overall_score": 0.0
            }
        
        sources_text = self._format_sources_for_eval(sources)
        
        eval_prompt = f"""You are an expert legal research evaluator. Assess the quality of this legal answer.

QUESTION: {question}

GENERATED ANSWER:
{answer}

AVAILABLE SOURCES:
{sources_text}

Evaluate the answer on these criteria (score 0-10 for each):

1. FACTUAL ACCURACY: Are all claims supported by the provided sources?
2. CITATION VALIDITY: Are citations correctly attributed and formatted?
3. JURISDICTIONAL ALIGNMENT: Does the answer properly distinguish between federal/state law when relevant?
4. COMPLETENESS: Does the answer fully address the question?
5. CLARITY: Is the answer clear and well-organized?

Respond in this EXACT JSON format:
{{
    "factual_accuracy": <score 0-10>,
    "citation_validity": <score 0-10>,
    "jurisdictional_alignment": <score 0-10>,
    "completeness": <score 0-10>,
    "clarity": <score 0-10>,
    "overall_score": <average of above>,
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "hallucination_detected": <true/false>,
    "feedback": "Brief constructive feedback in 2-3 sentences"
}}"""

        try:
            response = self.mistral_client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            evaluation = json.loads(result_text)
            
            evaluation["evaluator_model"] = model
            evaluation["timestamp"] = str(os.times())
            
            return evaluation
            
        except Exception as e:
            return {
                "error": f"Evaluation failed: {str(e)}",
                "overall_score": 0.0
            }
    
    def _format_sources_for_eval(self, sources: List[Dict]) -> str:
        """Format sources for evaluation prompt."""
        if not sources:
            return "No sources provided."
        
        formatted = []
        for i, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            text = source.get("text", "")[:300]
            formatted.append(
                f"[{i}] {metadata.get('case_name', 'Unknown')}, "
                f"{metadata.get('citation', 'N/A')}\n{text}...\n"
            )
        return "\n".join(formatted)
    
    def batch_evaluate(self, results: List[Dict], 
                      model: str = "mistral-small-latest") -> List[Dict]:
        """
        Evaluate multiple question-answer pairs.
        
        Args:
            results: List of {question, answer, sources} dicts
            model: Mistral model for evaluation
            
        Returns:
            List of evaluation results
        """
        evaluations = []
        for result in results:
            eval_result = self.evaluate_answer(
                question=result.get("question", ""),
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                model=model
            )
            evaluations.append({
                **result,
                "evaluation": eval_result
            })
        return evaluations
    
    def calculate_aggregate_metrics(self, evaluations: List[Dict]) -> Dict:
        """
        Calculate aggregate metrics across multiple evaluations.
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Aggregate statistics
        """
        if not evaluations:
            return {}
        
        metrics = {
            "total_evaluations": len(evaluations),
            "factual_accuracy": [],
            "citation_validity": [],
            "jurisdictional_alignment": [],
            "completeness": [],
            "clarity": [],
            "overall_scores": [],
            "hallucinations_detected": 0
        }
        
        for eval_result in evaluations:
            eval_data = eval_result.get("evaluation", {})
            
            if "error" not in eval_data:
                metrics["factual_accuracy"].append(eval_data.get("factual_accuracy", 0))
                metrics["citation_validity"].append(eval_data.get("citation_validity", 0))
                metrics["jurisdictional_alignment"].append(eval_data.get("jurisdictional_alignment", 0))
                metrics["completeness"].append(eval_data.get("completeness", 0))
                metrics["clarity"].append(eval_data.get("clarity", 0))
                metrics["overall_scores"].append(eval_data.get("overall_score", 0))
                
                if eval_data.get("hallucination_detected", False):
                    metrics["hallucinations_detected"] += 1
        
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        return {
            "total_evaluations": metrics["total_evaluations"],
            "avg_factual_accuracy": avg(metrics["factual_accuracy"]),
            "avg_citation_validity": avg(metrics["citation_validity"]),
            "avg_jurisdictional_alignment": avg(metrics["jurisdictional_alignment"]),
            "avg_completeness": avg(metrics["completeness"]),
            "avg_clarity": avg(metrics["clarity"]),
            "avg_overall_score": avg(metrics["overall_scores"]),
            "hallucination_rate": metrics["hallucinations_detected"] / metrics["total_evaluations"] if metrics["total_evaluations"] > 0 else 0
        }
