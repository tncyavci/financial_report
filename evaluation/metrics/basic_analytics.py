"""
Basic Analytics for Performance Metrics
Simple analysis and reporting functions
"""

import json
from typing import Dict, List, Any
from datetime import datetime, timedelta

class BasicAnalytics:
    """
    Basic analytics for performance metrics
    Provides simple analysis and reporting capabilities
    """
    
    def __init__(self, metrics_file: str = "evaluation/reports/performance_metrics.json"):
        """
        Args:
            metrics_file: Path to performance metrics file
        """
        self.metrics_file = metrics_file
        self.data = None
        self.load_metrics()
    
    def load_metrics(self):
        """Load metrics from file"""
        try:
            with open(self.metrics_file, 'r') as f:
                self.data = json.load(f)
            print(f"📊 Loaded metrics from {self.metrics_file}")
        except FileNotFoundError:
            print(f"⚠️ No metrics file found at {self.metrics_file}")
            self.data = None
        except Exception as e:
            print(f"❌ Error loading metrics: {e}")
            self.data = None
    
    def get_quick_stats(self) -> Dict[str, Any]:
        """Get quick performance statistics"""
        if not self.data or not self.data.get('metrics'):
            return {"error": "No metrics available"}
        
        metrics = self.data['metrics']
        
        # Basic stats
        durations = [m['duration'] for m in metrics]
        
        return {
            'total_operations': len(metrics),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'last_updated': self.data.get('system_info', {}).get('last_updated'),
            'gpu_available': self.data.get('system_info', {}).get('gpu_available', False)
        }
    
    def print_summary(self):
        """Print a quick summary of performance metrics"""
        if not self.data:
            print("❌ No metrics data available")
            return
        
        stats = self.get_quick_stats()
        if 'error' in stats:
            print(f"❌ {stats['error']}")
            return
        
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 40)
        print(f"Total Operations: {stats['total_operations']}")
        print(f"Average Duration: {stats['avg_duration']:.2f}s")
        print(f"Min Duration: {stats['min_duration']:.2f}s")
        print(f"Max Duration: {stats['max_duration']:.2f}s")
        print(f"GPU Available: {stats['gpu_available']}")
        print(f"Last Updated: {stats['last_updated']}")
        print("=" * 40) 