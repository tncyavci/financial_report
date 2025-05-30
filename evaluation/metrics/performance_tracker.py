"""
Performance Tracker for Turkish Financial RAG System
Monitors response times, memory usage, GPU utilization, and processing speeds
"""

import time
import psutil
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    timestamp: str
    operation: str
    duration: float
    memory_usage: Dict[str, float]
    gpu_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SystemStats:
    """System resource statistics"""
    cpu_percent: float
    memory_percent: float
    memory_available: float
    disk_usage: float
    gpu_available: bool = False
    gpu_memory_used: Optional[float] = None
    gpu_utilization: Optional[float] = None

class PerformanceTracker:
    """
    Real-time performance tracking for RAG operations
    Monitors: response times, memory, GPU, processing speeds
    """
    
    def __init__(self, save_path: str = "evaluation/reports/performance_metrics.json"):
        """
        Args:
            save_path: Path to save performance metrics
        """
        self.save_path = save_path
        self.metrics: List[PerformanceMetric] = []
        self.start_time = None
        self.operation_name = None
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        logger.info(f"🔍 PerformanceTracker initialized - save path: {save_path}")
    
    @contextmanager
    def track_operation(self, operation_name: str, metadata: Dict = None):
        """
        Context manager for tracking operation performance
        
        Usage:
            with tracker.track_operation("pdf_processing"):
                # Your code here
                process_pdf()
        """
        start_time = time.time()
        start_memory = self._get_memory_info()
        start_gpu = self._get_gpu_info() if TORCH_AVAILABLE else None
        
        logger.debug(f"🚀 Starting tracking: {operation_name}")
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            end_memory = self._get_memory_info()
            end_gpu = self._get_gpu_info() if TORCH_AVAILABLE else None
            
            # Calculate memory delta
            memory_delta = {
                'start_mb': start_memory['memory_percent'],
                'end_mb': end_memory['memory_percent'],
                'delta_mb': end_memory['memory_percent'] - start_memory['memory_percent'],
                'peak_mb': end_memory['memory_percent']  # Simplified
            }
            
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                operation=operation_name,
                duration=duration,
                memory_usage=memory_delta,
                gpu_info=end_gpu,
                metadata=metadata or {}
            )
            
            self.metrics.append(metric)
            self._save_metrics()
            
            logger.info(f"✅ {operation_name} completed in {duration:.2f}s")
    
    def track_document_processing(self, doc_type: str, doc_size_mb: float, page_count: int = None):
        """Track document processing performance"""
        metadata = {
            'document_type': doc_type,
            'size_mb': doc_size_mb,
            'page_count': page_count
        }
        return self.track_operation(f"document_processing_{doc_type}", metadata)
    
    def track_query_response(self, query_length: int, context_length: int, top_k: int = None):
        """Track query response performance"""
        metadata = {
            'query_length': query_length,
            'context_length': context_length,
            'top_k': top_k
        }
        return self.track_operation("query_response", metadata)
    
    def track_embedding_generation(self, text_count: int, batch_size: int = None):
        """Track embedding generation performance"""
        metadata = {
            'text_count': text_count,
            'batch_size': batch_size
        }
        return self.track_operation("embedding_generation", metadata)
    
    def track_vector_search(self, query_count: int, database_size: int):
        """Track vector search performance"""
        metadata = {
            'query_count': query_count,
            'database_size': database_size
        }
        return self.track_operation("vector_search", metadata)
    
    def get_system_stats(self) -> SystemStats:
        """Get current system resource statistics"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        stats = SystemStats(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=memory.percent,
            memory_available=memory.available / (1024**3),  # GB
            disk_usage=disk.percent
        )
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            stats.gpu_available = True
            stats.gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
            # Note: GPU utilization requires nvidia-ml-py, simplified here
            stats.gpu_utilization = None
        
        return stats
    
    def get_performance_summary(self, operation_filter: str = None) -> Dict[str, Any]:
        """
        Get performance summary statistics
        
        Args:
            operation_filter: Filter by operation type
        """
        if not self.metrics:
            return {"error": "No metrics available"}
        
        # Filter metrics if specified
        filtered_metrics = self.metrics
        if operation_filter:
            filtered_metrics = [m for m in self.metrics if operation_filter in m.operation]
        
        if not filtered_metrics:
            return {"error": f"No metrics found for operation: {operation_filter}"}
        
        durations = [m.duration for m in filtered_metrics]
        memory_deltas = [m.memory_usage.get('delta_mb', 0) for m in filtered_metrics]
        
        summary = {
            'total_operations': len(filtered_metrics),
            'time_range': {
                'start': filtered_metrics[0].timestamp,
                'end': filtered_metrics[-1].timestamp
            },
            'duration_stats': {
                'min': min(durations),
                'max': max(durations),
                'avg': sum(durations) / len(durations),
                'total': sum(durations)
            },
            'memory_stats': {
                'min_delta_mb': min(memory_deltas),
                'max_delta_mb': max(memory_deltas),
                'avg_delta_mb': sum(memory_deltas) / len(memory_deltas)
            },
            'operations_by_type': {}
        }
        
        # Group by operation type
        for metric in filtered_metrics:
            op_type = metric.operation
            if op_type not in summary['operations_by_type']:
                summary['operations_by_type'][op_type] = {
                    'count': 0,
                    'total_duration': 0,
                    'avg_duration': 0
                }
            
            summary['operations_by_type'][op_type]['count'] += 1
            summary['operations_by_type'][op_type]['total_duration'] += metric.duration
        
        # Calculate averages
        for op_type in summary['operations_by_type']:
            op_data = summary['operations_by_type'][op_type]
            op_data['avg_duration'] = op_data['total_duration'] / op_data['count']
        
        return summary
    
    def get_recent_metrics(self, count: int = 10) -> List[Dict]:
        """Get most recent metrics"""
        recent = self.metrics[-count:] if len(self.metrics) >= count else self.metrics
        return [asdict(metric) for metric in recent]
    
    def clear_metrics(self):
        """Clear all stored metrics"""
        self.metrics.clear()
        logger.info("🗑️ Performance metrics cleared")
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory information"""
        memory = psutil.virtual_memory()
        return {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3)
        }
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information if available"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
        
        return {
            'gpu_available': True,
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0),
            'memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            # Convert metrics to dict format for JSON serialization
            metrics_data = {
                'system_info': {
                    'torch_available': TORCH_AVAILABLE,
                    'gpu_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
                    'last_updated': datetime.now().isoformat()
                },
                'metrics': [asdict(metric) for metric in self.metrics],
                'summary': self.get_performance_summary()
            }
            
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"❌ Failed to save metrics: {e}")

# Convenience functions for quick tracking
def track_performance(operation_name: str, metadata: Dict = None):
    """Decorator for tracking function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = PerformanceTracker()
            with tracker.track_operation(operation_name, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Global tracker instance
_global_tracker = None

def get_global_tracker() -> PerformanceTracker:
    """Get or create global performance tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker 