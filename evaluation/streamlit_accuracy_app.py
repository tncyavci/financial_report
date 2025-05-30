"""
Enhanced Performance & Accuracy Tracker Test Application
Streamlit app with both performance tracking and manual accuracy evaluation
"""

import streamlit as st
import time
import random
import sys
import os
import json
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics.performance_tracker import get_global_tracker
from evaluation.metrics.accuracy_tracker import get_global_accuracy_tracker, ResponseQuality, QueryCategory
from evaluation.metrics.basic_analytics import BasicAnalytics

# Configure page
st.set_page_config(
    page_title="Performance & Accuracy Tracker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize trackers
perf_tracker = get_global_tracker()
acc_tracker = get_global_accuracy_tracker()

def simulate_rag_response(query: str, processing_time: float = None) -> str:
    """Simulate RAG system response"""
    if processing_time is None:
        processing_time = random.uniform(0.5, 3.0)
    
    time.sleep(processing_time)
    
    # Generate realistic financial responses based on query content
    if "varlık" in query.lower() or "aktif" in query.lower():
        return f"Şirketin toplam varlıkları {random.randint(800, 1500)} milyon TL'dir. Bu rakam bir önceki yıla göre %{random.randint(5, 15)} artış göstermiştir."
    elif "kar" in query.lower() and "net" in query.lower():
        return f"Net kar {random.randint(50, 200)} milyon TL olarak gerçekleşmiştir. Bu, geçen yılın aynı dönemine göre %{random.randint(-10, 25)} değişim anlamına gelmektedir."
    elif "satış" in query.lower() or "hasılat" in query.lower():
        return f"Net satış hasılatı {random.randint(1000, 3000)} milyon TL'dir. Şirket bu dönemde %{random.randint(8, 20)} büyüme kaydetmiştir."
    elif "borç" in query.lower():
        return f"Toplam borçlar {random.randint(300, 800)} milyon TL seviyesindedir. Borç/özkaynak oranı {random.uniform(0.3, 0.8):.2f}'dir."
    elif "oran" in query.lower():
        if "cari" in query.lower():
            return f"Cari oran {random.uniform(1.2, 2.5):.2f}'dir, bu da şirketin kısa vadeli yükümlülüklerini karşılayabilme kabiliyetini gösterir."
        elif "roe" in query.lower() or "özkaynak" in query.lower():
            return f"Özkaynak kârlılığı (ROE) %{random.uniform(8, 25):.1f} seviyesindedir."
        elif "roa" in query.lower() or "aktif" in query.lower():
            return f"Aktif kârlılığı (ROA) %{random.uniform(3, 15):.1f} olarak hesaplanmıştır."
    else:
        return f"Bu konuda detaylı analiz için finansal tablolara bakılması gerekmektedir. Şirketin genel performansı sektör ortalamasının üzerindedir."

def main():
    """Main application"""
    
    st.title("🎯 Performance & Accuracy Tracker Dashboard")
    st.markdown("**Turkish Financial RAG System - Comprehensive Evaluation**")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        
        page = st.selectbox(
            "Select Page",
            ["🚀 Performance Testing", "📊 Accuracy Evaluation", "📈 Analytics Dashboard", "🧪 Test Queries"]
        )
    
    if page == "🚀 Performance Testing":
        performance_testing_page()
    elif page == "📊 Accuracy Evaluation":
        accuracy_evaluation_page()
    elif page == "📈 Analytics Dashboard":
        analytics_dashboard_page()
    elif page == "🧪 Test Queries":
        test_queries_page()

def performance_testing_page():
    """Performance testing page"""
    st.header("⚡ Performance Testing")
    
    # Sidebar controls for performance testing
    with st.sidebar:
        st.subheader("🎮 Performance Controls")
        
        st.write("**📄 PDF Processing Test**")
        pdf_pages = st.slider("PDF Pages", 5, 100, 25, key="perf_pdf_pages")
        pdf_time = st.slider("Processing Time (s)", 0.5, 5.0, 2.0, key="perf_pdf_time")
        
        if st.button("🚀 Run PDF Test", key="perf_pdf_btn"):
            with st.spinner("Processing PDF..."):
                with perf_tracker.track_document_processing("pdf", pdf_pages/10, pdf_pages):
                    time.sleep(pdf_time)
                st.success(f"Processed PDF with {pdf_pages} pages in {pdf_time:.2f}s")
        
        st.write("**💬 Query Processing Test**")
        query_len = st.slider("Query Length", 10, 500, 150, key="perf_query_len")
        query_time = st.slider("Response Time (s)", 0.1, 3.0, 1.0, key="perf_query_time")
        
        if st.button("🧠 Run Query Test", key="perf_query_btn"):
            with st.spinner("Processing query..."):
                with perf_tracker.track_query_response(query_len, query_len*15):
                    time.sleep(query_time)
                st.success(f"Processed query ({query_len} chars) in {query_time:.2f}s")
    
    # Main performance metrics display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Performance Metrics")
        
        # System stats
        if st.button("🔄 Refresh System Stats"):
            sys_stats = perf_tracker.get_system_stats()
            
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
        summary = perf_tracker.get_performance_summary()
        if 'error' not in summary:
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Total Operations", summary['total_operations'])
            with summary_col2:
                st.metric("Average Duration", f"{summary['duration_stats']['avg']:.2f}s")
            with summary_col3:
                st.metric("Total Time", f"{summary['duration_stats']['total']:.2f}s")
    
    with col2:
        st.subheader("📋 Recent Activity")
        recent_metrics = perf_tracker.get_recent_metrics(5)
        
        if recent_metrics:
            for metric in recent_metrics:
                with st.container():
                    st.text(f"🔸 {metric['operation'].replace('_', ' ').title()}")
                    st.text(f"   Duration: {metric['duration']:.2f}s")
                    st.divider()

def accuracy_evaluation_page():
    """Accuracy evaluation page"""
    st.header("📊 Manual Accuracy Evaluation")
    
    # Load test queries
    test_queries_path = "evaluation/data/test_queries.json"
    test_queries = []
    
    try:
        if os.path.exists(test_queries_path):
            with open(test_queries_path, 'r', encoding='utf-8') as f:
                test_queries = json.load(f)
    except:
        st.warning("No test queries found. Generate them first in the Test Queries page.")
    
    if test_queries:
        st.info(f"📊 {len(test_queries)} test queries available for evaluation")
        
        # Query selection
        query_options = [f"{q['id']}: {q['query'][:50]}..." for q in test_queries]
        selected_idx = st.selectbox("Select Query to Evaluate", range(len(query_options)), 
                                  format_func=lambda x: query_options[x])
        
        if selected_idx is not None:
            selected_query = test_queries[selected_idx]
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("🔍 Query Evaluation")
                
                # Display query info
                st.text_area("Query", selected_query['query'], height=80, disabled=True)
                st.text_input("Category", selected_query['category'].replace('_', ' ').title(), disabled=True)
                st.text_area("Expected Answer", selected_query.get('expected_answer', 'N/A'), height=60, disabled=True)
                
                # Simulate RAG response
                if st.button("🤖 Generate RAG Response"):
                    with st.spinner("Generating response..."):
                        start_time = time.time()
                        response = simulate_rag_response(selected_query['query'])
                        response_time = time.time() - start_time
                        
                        st.session_state.current_response = response
                        st.session_state.current_response_time = response_time
                
                # Show generated response
                if 'current_response' in st.session_state:
                    st.text_area("System Response", st.session_state.current_response, height=100)
                    st.metric("Response Time", f"{st.session_state.current_response_time:.2f}s")
            
            with col2:
                st.subheader("✅ Manual Evaluation")
                
                # Quality scoring
                quality_score = st.selectbox(
                    "Quality Score",
                    [1, 2, 3, 4, 5],
                    index=2,
                    format_func=lambda x: f"{x} - {ResponseQuality(x).name}"
                )
                
                # Manual notes
                manual_notes = st.text_area("Evaluation Notes", height=100)
                
                # Submit evaluation
                if st.button("💾 Submit Evaluation") and 'current_response' in st.session_state:
                    
                    # Determine category
                    category_map = {
                        'balance_sheet': QueryCategory.BALANCE_SHEET,
                        'income_statement': QueryCategory.INCOME_STATEMENT,
                        'cash_flow': QueryCategory.CASH_FLOW,
                        'ratio_analysis': QueryCategory.RATIO_ANALYSIS,
                        'general_info': QueryCategory.GENERAL_INFO,
                        'comparison': QueryCategory.COMPARISON,
                        'trend_analysis': QueryCategory.TREND_ANALYSIS
                    }
                    
                    category = category_map.get(selected_query['category'], QueryCategory.GENERAL_INFO)
                    
                    # Record evaluation
                    evaluation = acc_tracker.evaluate_response(
                        query_id=selected_query['id'],
                        query_text=selected_query['query'],
                        actual_response=st.session_state.current_response,
                        quality_score=ResponseQuality(quality_score),
                        category=category,
                        response_time=st.session_state.current_response_time,
                        expected_answer=selected_query.get('expected_answer'),
                        manual_notes=manual_notes
                    )
                    
                    st.success(f"✅ Evaluation recorded! Score: {quality_score}/5")
                    
                    # Clear session state
                    if 'current_response' in st.session_state:
                        del st.session_state.current_response
                    if 'current_response_time' in st.session_state:
                        del st.session_state.current_response_time
    
    else:
        st.info("📝 No test queries available. Go to 'Test Queries' page to generate them.")

def analytics_dashboard_page():
    """Analytics dashboard page"""
    st.header("📈 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Performance Analytics")
        
        # Performance metrics
        perf_summary = perf_tracker.get_performance_summary()
        if 'error' not in perf_summary:
            st.metric("Total Operations", perf_summary['total_operations'])
            st.metric("Average Duration", f"{perf_summary['duration_stats']['avg']:.2f}s")
            
            # Operations breakdown
            if perf_summary.get('operations_by_type'):
                ops_data = []
                for op_type, data in perf_summary['operations_by_type'].items():
                    ops_data.append({
                        'Operation': op_type.replace('_', ' ').title(),
                        'Count': data['count'],
                        'Avg Duration': f"{data['avg_duration']:.2f}s"
                    })
                
                if ops_data:
                    df = pd.DataFrame(ops_data)
                    st.dataframe(df, use_container_width=True)
        else:
            st.info("No performance data available")
    
    with col2:
        st.subheader("🎯 Accuracy Analytics")
        
        # Accuracy metrics
        acc_metrics = acc_tracker.get_accuracy_metrics()
        
        if acc_metrics.total_queries > 0:
            st.metric("Success Rate", f"{acc_metrics.success_rate:.1f}%")
            st.metric("Average Quality", f"{acc_metrics.average_quality_score:.2f}/5")
            st.metric("Total Evaluations", acc_metrics.total_queries)
            
            # Quality distribution
            if acc_metrics.quality_distribution:
                quality_data = []
                for quality, count in acc_metrics.quality_distribution.items():
                    if count > 0:
                        quality_data.append({
                            'Quality': quality,
                            'Count': count,
                            'Percentage': f"{(count/acc_metrics.total_queries*100):.1f}%"
                        })
                
                if quality_data:
                    df = pd.DataFrame(quality_data)
                    st.dataframe(df, use_container_width=True)
        else:
            st.info("No accuracy evaluations available")
    
    # Combined report
    st.subheader("📋 Combined Report")
    
    if st.button("📄 Generate Report"):
        report = acc_tracker.generate_test_report()
        st.markdown(report)
        
        # Download button for report
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

def test_queries_page():
    """Test queries management page"""
    st.header("🧪 Test Queries Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Generate Test Queries")
        
        if st.button("🎯 Generate 30 Test Queries"):
            with st.spinner("Generating test queries..."):
                from evaluation.test_query_generator import generate_test_queries
                queries = generate_test_queries()
                st.success(f"✅ Generated {len(queries)} test queries!")
                st.rerun()
        
        # Show existing queries
        test_queries_path = "evaluation/data/test_queries.json"
        
        if os.path.exists(test_queries_path):
            try:
                with open(test_queries_path, 'r', encoding='utf-8') as f:
                    test_queries = json.load(f)
                
                st.subheader(f"📊 Existing Test Queries ({len(test_queries)} total)")
                
                # Category breakdown
                categories = {}
                for query in test_queries:
                    cat = query['category']
                    categories[cat] = categories.get(cat, 0) + 1
                
                category_data = []
                for cat, count in categories.items():
                    category_data.append({
                        'Category': cat.replace('_', ' ').title(),
                        'Count': count
                    })
                
                df = pd.DataFrame(category_data)
                st.dataframe(df, use_container_width=True)
                
                # Show sample queries
                st.subheader("📄 Sample Queries")
                for i, query in enumerate(test_queries[:5]):
                    with st.expander(f"Query {i+1}: {query['query'][:50]}..."):
                        st.write(f"**Category:** {query['category'].replace('_', ' ').title()}")
                        st.write(f"**Query:** {query['query']}")
                        st.write(f"**Expected:** {query.get('expected_answer', 'N/A')}")
                
            except Exception as e:
                st.error(f"Error loading test queries: {e}")
        else:
            st.info("No test queries found. Generate them using the button above.")
    
    with col2:
        st.subheader("🎯 Quick Evaluation")
        
        st.write("For testing individual queries:")
        
        test_query = st.text_area("Test Query", height=80)
        test_response = st.text_area("System Response", height=80)
        quick_score = st.selectbox("Quality Score", [1, 2, 3, 4, 5], index=2)
        
        if st.button("⚡ Quick Evaluate") and test_query and test_response:
            evaluation = acc_tracker.quick_evaluate(
                query=test_query,
                response=test_response,
                score=quick_score
            )
            st.success("✅ Quick evaluation recorded!")

# Footer functions
def render_footer():
    """Render footer with actions"""
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear Performance"):
            perf_tracker.clear_metrics()
            st.success("Performance metrics cleared!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Accuracy"):
            acc_tracker.clear_evaluations()
            st.success("Accuracy evaluations cleared!")
            st.rerun()
    
    with col3:
        if st.button("💾 Download Performance"):
            try:
                with open("evaluation/reports/performance_metrics.json", "r") as f:
                    data = f.read()
                st.download_button(
                    label="📥 Download",
                    data=data,
                    file_name=f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except:
                st.error("No performance data available")
    
    with col4:
        if st.button("💾 Download Accuracy"):
            try:
                with open("evaluation/reports/accuracy_metrics.json", "r") as f:
                    data = f.read()
                st.download_button(
                    label="📥 Download",
                    data=data,
                    file_name=f"accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except:
                st.error("No accuracy data available")

if __name__ == "__main__":
    main()
    render_footer() 