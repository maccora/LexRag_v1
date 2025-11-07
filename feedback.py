from typing import Dict, List, Optional
import json
import os
from datetime import datetime


class UserFeedbackSystem:
    """
    Collect and analyze user feedback on answer quality.
    Enables continuous improvement through user ratings and comments.
    """
    
    def __init__(self, feedback_file: str = "user_feedback.jsonl"):
        self.feedback_file = feedback_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create feedback file if it doesn't exist."""
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                pass
    
    def submit_feedback(self, question: str, answer: str, 
                       rating: int, comments: str = "",
                       sources: List[Dict] = None,
                       jurisdiction: str = "all") -> Dict:
        """
        Submit user feedback for a question-answer pair.
        
        Args:
            question: User's question
            answer: Generated answer
            rating: User rating (1-5 stars)
            comments: Optional user comments
            sources: Retrieved sources
            jurisdiction: Jurisdiction filter used
            
        Returns:
            Feedback record
        """
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        feedback_record = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer[:500],
            "rating": rating,
            "comments": comments,
            "jurisdiction": jurisdiction,
            "num_sources": len(sources) if sources else 0,
            "source_citations": [s.get("metadata", {}).get("citation", "N/A") 
                                for s in (sources or [])]
        }
        
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback_record) + '\n')
        
        return feedback_record
    
    def load_all_feedback(self) -> List[Dict]:
        """Load all feedback records."""
        feedback = []
        
        if not os.path.exists(self.feedback_file):
            return feedback
        
        with open(self.feedback_file, 'r') as f:
            for line in f:
                if line.strip():
                    feedback.append(json.loads(line))
        
        return feedback
    
    def get_statistics(self) -> Dict:
        """
        Calculate feedback statistics.
        
        Returns:
            Dict with aggregated feedback metrics
        """
        all_feedback = self.load_all_feedback()
        
        if not all_feedback:
            return {
                "total_feedback": 0,
                "avg_rating": 0.0,
                "rating_distribution": {i: 0 for i in range(1, 6)},
                "total_comments": 0
            }
        
        ratings = [f["rating"] for f in all_feedback]
        rating_dist = {i: ratings.count(i) for i in range(1, 6)}
        comments_count = sum(1 for f in all_feedback if f.get("comments", "").strip())
        
        return {
            "total_feedback": len(all_feedback),
            "avg_rating": sum(ratings) / len(ratings),
            "median_rating": sorted(ratings)[len(ratings) // 2],
            "rating_distribution": rating_dist,
            "total_comments": comments_count,
            "positive_feedback": sum(1 for r in ratings if r >= 4),
            "negative_feedback": sum(1 for r in ratings if r <= 2),
            "neutral_feedback": sum(1 for r in ratings if r == 3)
        }
    
    def get_low_rated_questions(self, threshold: int = 2) -> List[Dict]:
        """
        Get questions with low ratings for analysis.
        
        Args:
            threshold: Maximum rating to include (default: 2 stars or less)
            
        Returns:
            List of low-rated feedback records
        """
        all_feedback = self.load_all_feedback()
        low_rated = [f for f in all_feedback if f["rating"] <= threshold]
        
        return sorted(low_rated, key=lambda x: x["rating"])
    
    def get_top_issues(self, min_rating: int = 2) -> List[str]:
        """
        Extract common issues from low-rated feedback comments.
        
        Args:
            min_rating: Maximum rating to analyze
            
        Returns:
            List of common issue descriptions
        """
        low_rated = self.get_low_rated_questions(min_rating)
        
        issues = []
        for feedback in low_rated:
            if feedback.get("comments"):
                issues.append(feedback["comments"])
        
        return issues
    
    def get_feedback_by_jurisdiction(self) -> Dict[str, List[Dict]]:
        """
        Group feedback by jurisdiction.
        
        Returns:
            Dict mapping jurisdiction to feedback records
        """
        all_feedback = self.load_all_feedback()
        
        by_jurisdiction = {}
        for f in all_feedback:
            jur = f.get("jurisdiction", "unknown")
            if jur not in by_jurisdiction:
                by_jurisdiction[jur] = []
            by_jurisdiction[jur].append(f)
        
        return by_jurisdiction
    
    def get_average_rating_by_jurisdiction(self) -> Dict[str, float]:
        """
        Calculate average rating for each jurisdiction.
        
        Returns:
            Dict mapping jurisdiction to average rating
        """
        by_jurisdiction = self.get_feedback_by_jurisdiction()
        
        avg_ratings = {}
        for jur, feedback_list in by_jurisdiction.items():
            ratings = [f["rating"] for f in feedback_list]
            avg_ratings[jur] = sum(ratings) / len(ratings) if ratings else 0.0
        
        return avg_ratings
    
    def export_for_analysis(self, output_file: str = "feedback_export.json"):
        """
        Export feedback data for external analysis.
        
        Args:
            output_file: Path to output JSON file
        """
        all_feedback = self.load_all_feedback()
        stats = self.get_statistics()
        
        export_data = {
            "export_date": datetime.now().isoformat(),
            "statistics": stats,
            "feedback_records": all_feedback,
            "by_jurisdiction": self.get_average_rating_by_jurisdiction(),
            "low_rated_questions": self.get_low_rated_questions()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return output_file


class FeedbackAnalyzer:
    """
    Advanced analysis of user feedback patterns.
    """
    
    @staticmethod
    def identify_improvement_areas(feedback_data: List[Dict]) -> Dict:
        """
        Identify areas for improvement based on feedback patterns.
        
        Args:
            feedback_data: List of feedback records
            
        Returns:
            Analysis of improvement opportunities
        """
        if not feedback_data:
            return {"message": "No feedback data available"}
        
        low_rated = [f for f in feedback_data if f["rating"] <= 2]
        high_rated = [f for f in feedback_data if f["rating"] >= 4]
        
        improvements = {
            "total_analyzed": len(feedback_data),
            "low_rated_count": len(low_rated),
            "high_rated_count": len(high_rated),
            "needs_improvement": len(low_rated) > len(feedback_data) * 0.2,
            "improvement_rate": (len(high_rated) / len(feedback_data)) if feedback_data else 0
        }
        
        if low_rated:
            low_rated_jurisdictions = [f.get("jurisdiction", "unknown") for f in low_rated]
            from collections import Counter
            improvements["problematic_jurisdictions"] = dict(Counter(low_rated_jurisdictions).most_common(3))
        
        return improvements
    
    @staticmethod
    def generate_recommendations(stats: Dict) -> List[str]:
        """
        Generate actionable recommendations based on feedback statistics.
        
        Args:
            stats: Feedback statistics dict
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if stats.get("total_feedback", 0) == 0:
            recommendations.append("Encourage users to provide feedback on answers")
            return recommendations
        
        avg_rating = stats.get("avg_rating", 0)
        
        if avg_rating < 3.0:
            recommendations.append("CRITICAL: Average rating below 3 stars - review answer quality and citation accuracy")
        elif avg_rating < 3.5:
            recommendations.append("Average rating could be improved - consider tuning retrieval parameters")
        elif avg_rating >= 4.5:
            recommendations.append("Excellent performance! Consider promoting user testimonials")
        
        negative_ratio = stats.get("negative_feedback", 0) / max(stats.get("total_feedback", 1), 1)
        if negative_ratio > 0.3:
            recommendations.append("High negative feedback ratio - analyze low-rated questions for patterns")
        
        if stats.get("total_comments", 0) < stats.get("total_feedback", 0) * 0.2:
            recommendations.append("Low comment rate - encourage detailed feedback for improvement")
        
        return recommendations
