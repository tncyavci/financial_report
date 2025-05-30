"""
Performance Tracker Test Application
Standalone Streamlit app for testing performance monitoring
"""

import streamlit as st
import time
import random
import sys
import os
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics.performance_tracker import get_global_tracker
from evaluation.metrics.basic_analytics import BasicAnalytics

# Configure page
st.set_page_config(
    page_title="Performance Tracker Test",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize global tracker
tracker = get_global_tracker()

def simulate_pdf_processing(pages: int = None, processing_time: float = None):
    """Simulate PDF processing with configurable parameters"""
    if pages is None:
        pages = random.randint(10, 50)
    if processing_time is None:
        processing_time = random.uniform(1, 3)
    
    time.sleep(processing_time)
    return f"Processed PDF with {pages} pages in {processing_time:.2f}s"

def simulate_query_processing(query_length: int = None, response_time: float = None):
    """Simulate query processing"""
    if query_length is None:
        query_length = random.randint(50, 200)
    if response_time is None:
        response_time = random.uniform(0.5, 2.0)
    
    time.sleep(response_time)
    return f"Processed query ({query_length} chars) in {response_time:.2f}s"

def simulate_embedding_generation(chunk_count: int = None, processing_time: float = None):
    """Simulate embedding generation"""
    if chunk_count is None:
        chunk_count = random.randint(50, 200)
    if processing_time is None:
        processing_time = random.uniform(0.2, 1.5)
    
    time.sleep(processing_time)
    return f"Generated embeddings for {chunk_count} chunks in {processing_time:.2f}s"

def main():
    """Main application"""
    
    st.title("⚡ Performance Tracker Test Dashboard")
    st.markdown("**Turkish Financial RAG System - Performance Monitoring**")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎮 Test Controls")
        
        st.subheader("📄 PDF Processing Test")
        pdf_pages = st.slider("PDF Pages", 5, 100, 25)
        pdf_time = st.slider("Processing Time (s)", 0.5, 5.0, 2.0)
        
        if st.button("🚀 Run PDF Test"):
            with st.spinner("Processing PDF..."):
                with tracker.track_document_processing("pdf", pdf_pages/10, pdf_pages):
                    result = simulate_pdf_processing(pdf_pages, pdf_time)
                st.success(result)
        
        st.subheader("💬 Query Processing Test")
        query_len = st.slider("Query Length", 10, 500, 150)
        query_time = st.slider("Response Time (s)", 0.1, 3.0, 1.0)
        
        if st.button("🧠 Run Query Test"):
            with st.spinner("Processing query..."):
                with tracker.track_query_response(query_len, query_len*15):
                    result = simulate_query_processing(query_len, query_time)
                st.success(result)
        
        st.subheader("🔍 Embedding Test")
        chunk_count = st.slider("Chunk Count", 10, 500, 100)
        embed_time = st.slider("Embedding Time (s)", 0.1, 2.0, 0.8)
        
        if st.button("🧬 Run Embedding Test"):
            with st.spinner("Generating embeddings..."):
                with tracker.track_embedding_generation(chunk_count, 32):
                    result = simulate_embedding_generation(chunk_count, embed_time)
                st.success(result)
        
        st.divider()
        
        # Auto-run tests
        st.subheader("🔄 Auto Test")
        if st.button("🎯 Run All Tests"):
            tests = [
                ("PDF Processing", lambda: tracker.track_document_processing("pdf", 2.5, 25)),
                ("Query Response", lambda: tracker.track_query_response(150, 2500)),
                ("Embedding Generation", lambda: tracker.track_embedding_generation(100, 32))
            ]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (test_name, test_context) in enumerate(tests):
                status_text.text(f"Running {test_name}...")
                with test_context():
                    time.sleep(random.uniform(0.5, 2.0))
                progress_bar.progress((i + 1) / len(tests))
            
            status_text.text("✅ All tests completed!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Performance Metrics")
        
        # Real-time system stats
        if st.button("🔄 Refresh System Stats"):
            sys_stats = tracker.get_system_stats()
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("CPU Usage", f"{sys_stats.cpu_percent:.1f}%")
            with metrics_col2:
                st.metric("Memory Usage", f"{sys_stats.memory_percent:.1f}%")
            with metrics_col3:
                st.metric("Available RAM", f"{sys_stats.memory_available:.1f} GB")
            with metrics_col4:
                gpu_status = "✅ Available" if sys_stats.gpu_available else "❌ Not Available"
                st.metric("GPU", gpu_status)
        
        # Performance summary
        st.subheader("📈 Performance Summary")
        summary = tracker.get_performance_summary()
        
        if 'error' not in summary:
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Total Operations", summary['total_operations'])
            with summary_col2:
                st.metric("Average Duration", f"{summary['duration_stats']['avg']:.2f}s")
            with summary_col3:
                st.metric("Total Time", f"{summary['duration_stats']['total']:.2f}s")
            
            # Operations breakdown
            if summary.get('operations_by_type'):
                st.subheader("🔍 Operations Breakdown")
                
                ops_data = []
                for op_type, data in summary['operations_by_type'].items():
                    ops_data.append({
                        'Operation': op_type.replace('_', ' ').title(),
                        'Count': data['count'],
                        'Total Duration (s)': f"{data['total_duration']:.2f}",
                        'Average Duration (s)': f"{data['avg_duration']:.2f}"
                    })
                
                if ops_data:
                    df = pd.DataFrame(ops_data)
                    st.dataframe(df, use_container_width=True)
        else:
            st.info("No performance data available yet. Run some tests to see metrics!")
    
    with col2:
        st.header("📋 Recent Activity")
        
        # Recent metrics
        recent_metrics = tracker.get_recent_metrics(10)
        
        if recent_metrics:
            st.subheader("🕐 Last 10 Operations")
            for metric in recent_metrics[-5:]:  # Show last 5
                with st.container():
                    st.text(f"🔸 {metric['operation'].replace('_', ' ').title()}")
                    st.text(f"   Duration: {metric['duration']:.2f}s")
                    st.text(f"   Time: {metric['timestamp'][-8:]}")  # Last 8 chars (time)
                    st.divider()
        else:
            st.info("No recent activity")
        
        # Analytics
        st.subheader("📊 Analytics")
        try:
            analytics = BasicAnalytics()
            stats = analytics.get_quick_stats()
            
            if 'error' not in stats:
                st.json({
                    "Total Operations": stats['total_operations'],
                    "Avg Duration": f"{stats['avg_duration']:.2f}s",
                    "Min Duration": f"{stats['min_duration']:.2f}s",
                    "Max Duration": f"{stats['max_duration']:.2f}s",
                    "GPU Available": stats['gpu_available']
                })
            else:
                st.info("No analytics data available")
        except:
            st.info("Analytics not available")
    
    # Footer
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Metrics"):
            tracker.clear_metrics()
            st.success("Metrics cleared!")
            st.rerun()
    
    with col2:
        if st.button("💾 Download Metrics"):
            try:
                with open("evaluation/reports/performance_metrics.json", "r") as f:
                    metrics_data = f.read()
                st.download_button(
                    label="📥 Download JSON",
                    data=metrics_data,
                    file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except:
                st.error("No metrics file available")
    
    with col3:
        st.markdown("**Status:** System Ready ✅")

if __name__ == "__main__":
    main() 