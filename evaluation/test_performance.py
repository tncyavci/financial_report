"""
Test script for Performance Tracker
Demonstrates how to use performance tracking in the Turkish Financial RAG system
"""

import time
import random
from evaluation.metrics.performance_tracker import PerformanceTracker, get_global_tracker

def simulate_pdf_processing():
    """Simulate PDF processing operation"""
    # Simulate processing time (random 1-3 seconds)
    time.sleep(random.uniform(1, 3))
    return f"Processed PDF with {random.randint(10, 50)} pages"

def simulate_query_response():
    """Simulate query response operation"""
    # Simulate thinking time (random 0.5-2 seconds)
    time.sleep(random.uniform(0.5, 2))
    return "Generated response to financial query"

def simulate_embedding_generation():
    """Simulate embedding generation"""
    # Simulate embedding creation (random 0.2-1 seconds)
    time.sleep(random.uniform(0.2, 1))
    return f"Generated embeddings for {random.randint(50, 200)} text chunks"

def test_performance_tracker():
    """Test the performance tracker with various operations"""
    print("🔍 Testing Performance Tracker...")
    
    # Create tracker instance
    tracker = PerformanceTracker()
    
    # Test 1: PDF Processing
    print("\n📄 Test 1: PDF Processing")
    with tracker.track_document_processing("pdf", 2.5, 25):
        result = simulate_pdf_processing()
        print(f"   Result: {result}")
    
    # Test 2: Query Response
    print("\n💬 Test 2: Query Response")
    with tracker.track_query_response(150, 2500, 5):
        result = simulate_query_response()
        print(f"   Result: {result}")
    
    # Test 3: Embedding Generation
    print("\n🧠 Test 3: Embedding Generation")
    with tracker.track_embedding_generation(100, 32):
        result = simulate_embedding_generation()
        print(f"   Result: {result}")
    
    # Test 4: Custom Operation
    print("\n⚙️ Test 4: Custom Operation")
    with tracker.track_operation("custom_financial_analysis", {"analysis_type": "ratio_analysis"}):
        time.sleep(1.5)
        print("   Result: Completed financial ratio analysis")
    
    # Get system stats
    print("\n📊 System Statistics:")
    stats = tracker.get_system_stats()
    print(f"   CPU: {stats.cpu_percent:.1f}%")
    print(f"   Memory: {stats.memory_percent:.1f}%")
    print(f"   Available Memory: {stats.memory_available:.2f} GB")
    print(f"   GPU Available: {stats.gpu_available}")
    if stats.gpu_available:
        print(f"   GPU Memory Used: {stats.gpu_memory_used:.2f} GB")
    
    # Get performance summary
    print("\n📈 Performance Summary:")
    summary = tracker.get_performance_summary()
    if 'error' not in summary:
        print(f"   Total Operations: {summary['total_operations']}")
        print(f"   Average Duration: {summary['duration_stats']['avg']:.2f}s")
        print(f"   Total Time: {summary['duration_stats']['total']:.2f}s")
        print("\n   Operations by Type:")
        for op_type, data in summary['operations_by_type'].items():
            print(f"     {op_type}: {data['count']} ops, avg {data['avg_duration']:.2f}s")
    
    # Get recent metrics
    print("\n📋 Recent Metrics:")
    recent = tracker.get_recent_metrics(3)
    for metric in recent:
        print(f"   {metric['operation']}: {metric['duration']:.2f}s")
    
    print(f"\n💾 Metrics saved to: {tracker.save_path}")
    print("✅ Performance tracking test completed!")

def test_global_tracker():
    """Test the global tracker functionality"""
    print("\n🌐 Testing Global Tracker...")
    
    # Get global tracker
    tracker = get_global_tracker()
    
    # Run a simple operation
    with tracker.track_operation("global_test"):
        time.sleep(0.5)
        print("   Global tracker operation completed")
    
    print("✅ Global tracker test completed!")

if __name__ == "__main__":
    # Run tests
    test_performance_tracker()
    test_global_tracker()
    
    print("\n🎯 Next Steps:")
    print("   1. Integrate tracker into your existing RAG system")
    print("   2. Add tracking to PDF processing functions") 
    print("   3. Add tracking to query response functions")
    print("   4. Monitor performance metrics regularly")
    print("   5. Use data to optimize system performance") 