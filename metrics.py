from typing import List, Dict, Set, Optional
import numpy as np


class RetrievalMetrics:
    """
    Advanced retrieval metrics for RAG system evaluation.
    Implements Recall@K, Mean Reciprocal Rank, and other IR metrics.
    """
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Calculate Recall@K: fraction of relevant items in top-K results.
        
        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: Set of truly relevant document IDs
            k: Cutoff position
            
        Returns:
            Recall score (0-1)
        """
        if not relevant_ids:
            return 0.0
        
        top_k = set(retrieved_ids[:k])
        relevant_retrieved = top_k.intersection(relevant_ids)
        
        return len(relevant_retrieved) / len(relevant_ids)
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Calculate Precision@K: fraction of top-K results that are relevant.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k: Cutoff position
            
        Returns:
            Precision score (0-1)
        """
        if k == 0:
            return 0.0
        
        top_k = set(retrieved_ids[:k])
        relevant_retrieved = top_k.intersection(relevant_ids)
        
        return len(relevant_retrieved) / k
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        Rewards relevant documents appearing earlier in ranking.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            
        Returns:
            MRR score (0-1)
        """
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        Calculate Average Precision (AP).
        Average of precision values at positions where relevant docs appear.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            
        Returns:
            AP score (0-1)
        """
        if not relevant_ids:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                relevant_count += 1
                precision = relevant_count / i
                precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_ids)
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        Accounts for graded relevance scores.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevance_scores: Dict mapping doc_id to relevance score (0-1 or higher)
            k: Cutoff position
            
        Returns:
            NDCG score (0-1)
        """
        def dcg(ids: List[str], scores: Dict[str, float], k: int) -> float:
            """Calculate DCG@K."""
            dcg_sum = 0.0
            for i, doc_id in enumerate(ids[:k], 1):
                rel = scores.get(doc_id, 0.0)
                dcg_sum += rel / np.log2(i + 1)
            return dcg_sum
        
        if not relevance_scores:
            return 0.0
        
        ideal_ids = sorted(relevance_scores.keys(), 
                          key=lambda x: relevance_scores[x], 
                          reverse=True)
        
        ideal_dcg = dcg(ideal_ids, relevance_scores, k)
        
        if ideal_dcg == 0:
            return 0.0
        
        actual_dcg = dcg(retrieved_ids, relevance_scores, k)
        
        return actual_dcg / ideal_dcg
    
    @staticmethod
    def calculate_all_metrics(retrieved_docs: List[Dict], 
                             relevant_doc_ids: Optional[Set[str]] = None,
                             k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Calculate all retrieval metrics for a query result.
        
        Args:
            retrieved_docs: List of retrieved documents with metadata
            relevant_doc_ids: Set of known relevant document IDs (for ground truth)
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary with all metric scores
        """
        retrieved_ids = [doc.get("metadata", {}).get("case_name", str(i)) 
                        for i, doc in enumerate(retrieved_docs)]
        
        distances = [doc.get("distance", 1.0) for doc in retrieved_docs]
        relevance_scores = {doc_id: 1.0 - dist 
                           for doc_id, dist in zip(retrieved_ids, distances)}
        
        metrics = {
            "total_retrieved": len(retrieved_docs),
            "avg_relevance_score": np.mean(list(relevance_scores.values())) if relevance_scores else 0.0,
            "min_distance": min(distances) if distances else 1.0,
            "max_distance": max(distances) if distances else 1.0,
        }
        
        if relevant_doc_ids:
            metrics["mrr"] = RetrievalMetrics.mean_reciprocal_rank(retrieved_ids, relevant_doc_ids)
            
            for k in k_values:
                metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_doc_ids, k)
                metrics[f"precision@{k}"] = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_doc_ids, k)
                metrics[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(retrieved_ids, relevance_scores, k)
        else:
            for k in k_values:
                metrics[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(retrieved_ids, relevance_scores, k)
        
        return metrics


class QueryAnalytics:
    """
    Track and analyze query patterns and system performance over time.
    """
    
    def __init__(self):
        self.query_history = []
    
    def log_query(self, query: str, num_results: int, jurisdiction: str,
                  response_time: float, metrics: Dict):
        """Log a query and its metrics."""
        self.query_history.append({
            "query": query,
            "num_results": num_results,
            "jurisdiction": jurisdiction,
            "response_time": response_time,
            "metrics": metrics,
            "timestamp": str(np.datetime64('now'))
        })
    
    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics across all queries."""
        if not self.query_history:
            return {}
        
        response_times = [q["response_time"] for q in self.query_history]
        
        return {
            "total_queries": len(self.query_history),
            "avg_response_time": np.mean(response_times),
            "median_response_time": np.median(response_times),
            "p95_response_time": np.percentile(response_times, 95),
            "jurisdiction_distribution": self._count_jurisdictions(),
            "avg_results_per_query": np.mean([q["num_results"] for q in self.query_history])
        }
    
    def _count_jurisdictions(self) -> Dict[str, int]:
        """Count queries by jurisdiction."""
        counts = {}
        for q in self.query_history:
            jur = q.get("jurisdiction", "unknown")
            counts[jur] = counts.get(jur, 0) + 1
        return counts
