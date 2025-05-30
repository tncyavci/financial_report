"""
Accuracy Tracker for Turkish Financial RAG System
Tracks success/failure rates, basic accuracy, and manual evaluation metrics
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import os

class ResponseQuality(Enum):
    """Response quality categories"""
    EXCELLENT = 5  # Mükemmel - Tam doğru ve kapsamlı
    GOOD = 4      # İyi - Doğru ama eksik detay olabilir
    FAIR = 3      # Orta - Kısmen doğru
    POOR = 2      # Zayıf - Büyük ölçüde yanlış
    FAILED = 1    # Başarısız - Tamamen yanlış veya cevap yok

class QueryCategory(Enum):
    """Query categories for financial domain"""
    BALANCE_SHEET = "balance_sheet"      # Bilanço soruları
    INCOME_STATEMENT = "income_statement" # Gelir tablosu
    CASH_FLOW = "cash_flow"              # Nakit akışı
    RATIO_ANALYSIS = "ratio_analysis"     # Oran analizi
    GENERAL_INFO = "general_info"        # Genel bilgi
    COMPARISON = "comparison"            # Karşılaştırma
    TREND_ANALYSIS = "trend_analysis"    # Trend analizi

@dataclass
class QueryEvaluation:
    """Single query evaluation result"""
    query_id: str
    query_text: str
    category: QueryCategory
    expected_answer: Optional[str]
    actual_response: str
    quality_score: ResponseQuality
    response_time: float
    success: bool
    manual_notes: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class AccuracyMetrics:
    """Accuracy and success rate metrics"""
    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate: float
    average_quality_score: float
    quality_distribution: Dict[str, int]
    category_performance: Dict[str, Dict[str, float]]
    evaluation_date: str
    
class AccuracyTracker:
    """
    Tracks query success/failure rates and accuracy metrics
    Supports manual evaluation of 20-30 queries for academic validation
    """
    
    def __init__(self, save_path: str = "evaluation/reports/accuracy_metrics.json"):
        """
        Args:
            save_path: Path to save accuracy metrics
        """
        self.save_path = save_path
        self.evaluations: List[QueryEvaluation] = []
        self.test_queries: List[Dict] = []
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Load existing data
        self._load_existing_data()
        
        print(f"🎯 AccuracyTracker initialized - {len(self.evaluations)} evaluations loaded")
    
    def add_test_query(self, query: str, category: QueryCategory, expected_answer: str = None):
        """Add a test query to the evaluation set"""
        query_data = {
            'id': f"q_{len(self.test_queries) + 1:03d}",
            'query': query,
            'category': category.value,
            'expected_answer': expected_answer,
            'added_at': datetime.now().isoformat()
        }
        self.test_queries.append(query_data)
        self._save_test_queries()
        print(f"✅ Test query added: {query_data['id']}")
    
    def evaluate_response(self, 
                         query_id: str,
                         query_text: str, 
                         actual_response: str,
                         quality_score: ResponseQuality,
                         category: QueryCategory,
                         response_time: float,
                         expected_answer: str = None,
                         manual_notes: str = None) -> QueryEvaluation:
        """
        Evaluate a single query response
        
        Args:
            query_id: Unique identifier for query
            query_text: The original query
            actual_response: System's response
            quality_score: Manual quality assessment
            category: Query category
            response_time: Response time in seconds
            expected_answer: Expected correct answer
            manual_notes: Additional evaluation notes
        """
        
        # Determine success based on quality score
        success = quality_score.value >= 3  # Fair or better is considered success
        
        evaluation = QueryEvaluation(
            query_id=query_id,
            query_text=query_text,
            category=category,
            expected_answer=expected_answer,
            actual_response=actual_response,
            quality_score=quality_score,
            response_time=response_time,
            success=success,
            manual_notes=manual_notes
        )
        
        self.evaluations.append(evaluation)
        self._save_metrics()
        
        print(f"📊 Evaluation recorded: {query_id} - {quality_score.name} ({'✅ Success' if success else '❌ Failed'})")
        return evaluation
    
    def quick_evaluate(self, query: str, response: str, score: int, response_time: float = 0.0) -> QueryEvaluation:
        """Quick evaluation method for manual testing"""
        quality = ResponseQuality(score)
        category = QueryCategory.GENERAL_INFO  # Default category
        query_id = f"manual_{len(self.evaluations) + 1:03d}"
        
        return self.evaluate_response(
            query_id=query_id,
            query_text=query,
            actual_response=response,
            quality_score=quality,
            category=category,
            response_time=response_time
        )
    
    def get_accuracy_metrics(self) -> AccuracyMetrics:
        """Calculate comprehensive accuracy metrics"""
        if not self.evaluations:
            return AccuracyMetrics(
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                success_rate=0.0,
                average_quality_score=0.0,
                quality_distribution={},
                category_performance={},
                evaluation_date=datetime.now().isoformat()
            )
        
        total = len(self.evaluations)
        successful = sum(1 for eval in self.evaluations if eval.success)
        failed = total - successful
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        # Average quality score
        avg_quality = sum(eval.quality_score.value for eval in self.evaluations) / total
        
        # Quality distribution
        quality_dist = {}
        for quality in ResponseQuality:
            count = sum(1 for eval in self.evaluations if eval.quality_score == quality)
            quality_dist[quality.name] = count
        
        # Category performance
        category_perf = {}
        for category in QueryCategory:
            cat_evals = [eval for eval in self.evaluations if eval.category == category]
            if cat_evals:
                cat_success = sum(1 for eval in cat_evals if eval.success)
                cat_total = len(cat_evals)
                cat_avg_quality = sum(eval.quality_score.value for eval in cat_evals) / cat_total
                
                category_perf[category.value] = {
                    'total_queries': cat_total,
                    'success_rate': (cat_success / cat_total) * 100,
                    'average_quality': cat_avg_quality,
                    'avg_response_time': sum(eval.response_time for eval in cat_evals) / cat_total
                }
        
        return AccuracyMetrics(
            total_queries=total,
            successful_queries=successful,
            failed_queries=failed,
            success_rate=success_rate,
            average_quality_score=avg_quality,
            quality_distribution=quality_dist,
            category_performance=category_perf,
            evaluation_date=datetime.now().isoformat()
        )
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a comprehensive evaluation summary"""
        metrics = self.get_accuracy_metrics()
        
        return {
            'overview': {
                'total_evaluations': metrics.total_queries,
                'success_rate': f"{metrics.success_rate:.1f}%",
                'average_quality': f"{metrics.average_quality_score:.2f}/5",
                'evaluation_period': {
                    'start': self.evaluations[0].timestamp if self.evaluations else None,
                    'end': self.evaluations[-1].timestamp if self.evaluations else None
                }
            },
            'quality_breakdown': metrics.quality_distribution,
            'category_performance': metrics.category_performance,
            'recent_evaluations': [asdict(eval) for eval in self.evaluations[-5:]]
        }
    
    def generate_test_report(self) -> str:
        """Generate a formatted test report"""
        metrics = self.get_accuracy_metrics()
        
        report = f"""
# 📊 Turkish Financial RAG - Accuracy Test Report

## 📈 Summary Statistics
- **Total Queries Evaluated**: {metrics.total_queries}
- **Success Rate**: {metrics.success_rate:.1f}%
- **Average Quality Score**: {metrics.average_quality_score:.2f}/5.0
- **Evaluation Date**: {metrics.evaluation_date[:10]}

## 🎯 Quality Distribution
"""
        
        for quality, count in metrics.quality_distribution.items():
            percentage = (count / metrics.total_queries * 100) if metrics.total_queries > 0 else 0
            report += f"- **{quality}**: {count} queries ({percentage:.1f}%)\n"
        
        report += "\n## 📂 Category Performance\n"
        for category, perf in metrics.category_performance.items():
            report += f"- **{category.replace('_', ' ').title()}**: "
            report += f"{perf['success_rate']:.1f}% success, "
            report += f"{perf['average_quality']:.2f}/5 quality, "
            report += f"{perf['avg_response_time']:.2f}s avg time\n"
        
        # Recent evaluations
        if self.evaluations:
            report += "\n## 🕐 Recent Evaluations\n"
            for eval in self.evaluations[-3:]:
                report += f"- **{eval.query_id}**: {eval.quality_score.name} "
                report += f"({'✅' if eval.success else '❌'})\n"
                report += f"  Query: {eval.query_text[:60]}...\n"
        
        return report
    
    def clear_evaluations(self):
        """Clear all evaluations"""
        self.evaluations.clear()
        self._save_metrics()
        print("🗑️ All evaluations cleared")
    
    def _load_existing_data(self):
        """Load existing evaluation data"""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Load evaluations
                if 'evaluations' in data:
                    for eval_data in data['evaluations']:
                        eval_obj = QueryEvaluation(
                            query_id=eval_data['query_id'],
                            query_text=eval_data['query_text'],
                            category=QueryCategory(eval_data['category']),
                            expected_answer=eval_data.get('expected_answer'),
                            actual_response=eval_data['actual_response'],
                            quality_score=ResponseQuality(eval_data['quality_score']),
                            response_time=eval_data['response_time'],
                            success=eval_data['success'],
                            manual_notes=eval_data.get('manual_notes'),
                            timestamp=eval_data['timestamp']
                        )
                        self.evaluations.append(eval_obj)
                        
            # Load test queries
            test_queries_path = "evaluation/data/test_queries.json"
            if os.path.exists(test_queries_path):
                with open(test_queries_path, 'r', encoding='utf-8') as f:
                    self.test_queries = json.load(f)
                    
        except Exception as e:
            print(f"⚠️ Could not load existing data: {e}")
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            metrics = self.get_accuracy_metrics()
            
            data = {
                'system_info': {
                    'evaluation_system_version': '1.0.0',
                    'last_updated': datetime.now().isoformat(),
                    'total_evaluations': len(self.evaluations)
                },
                'metrics': asdict(metrics),
                'evaluations': [asdict(eval) for eval in self.evaluations]
            }
            
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            print(f"❌ Failed to save accuracy metrics: {e}")
    
    def _save_test_queries(self):
        """Save test queries to separate file"""
        try:
            queries_path = "evaluation/data/test_queries.json"
            os.makedirs(os.path.dirname(queries_path), exist_ok=True)
            
            with open(queries_path, 'w', encoding='utf-8') as f:
                json.dump(self.test_queries, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ Failed to save test queries: {e}")

# Global accuracy tracker instance
_global_accuracy_tracker = None

def get_global_accuracy_tracker() -> AccuracyTracker:
    """Get or create global accuracy tracker"""
    global _global_accuracy_tracker
    if _global_accuracy_tracker is None:
        _global_accuracy_tracker = AccuracyTracker()
    return _global_accuracy_tracker 