# 🔧 Performance Tracker Integration Guide

Bu kılavuz Performance Tracker'ı mevcut Turkish Financial RAG sisteminize nasıl entegre edeceğinizi açıklar.

## 🎯 **Entegrasyon Adımları**

### **1. Streamlit App'e Entegrasyon**

`streamlit_app.py` dosyasında şu değişiklikleri yapın:

```python
# Import performance tracker
from evaluation.metrics.performance_tracker import get_global_tracker

# Global tracker instance
tracker = get_global_tracker()

# PDF processing fonksiyonunda:
def process_uploaded_pdfs(uploaded_files):
    for file in uploaded_files:
        # Track PDF processing
        with tracker.track_document_processing("pdf", file.size/1024/1024, None):
            # Mevcut PDF processing kodunuz
            result = your_existing_pdf_process(file)
            
# Query response fonksiyonunda:
def handle_query(query, context):
    with tracker.track_query_response(len(query), len(context)):
        # Mevcut query processing kodunuz
        response = your_existing_query_handler(query, context)
        return response
```

### **2. Text Processor'a Entegrasyon**

`src/text_processor.py` dosyasında:

```python
from evaluation.metrics.performance_tracker import get_global_tracker

class TextProcessor:
    def __init__(self):
        self.tracker = get_global_tracker()
        
    def _create_embeddings(self, chunks, batch_size=32):
        with self.tracker.track_embedding_generation(len(chunks), batch_size):
            # Mevcut embedding generation kodunuz
            return existing_embedding_code(chunks)
```

### **3. Vector Store'a Entegrasyon**

`src/vector_store.py` dosyasında:

```python
from evaluation.metrics.performance_tracker import get_global_tracker

class VectorStore:
    def __init__(self):
        self.tracker = get_global_tracker()
        
    def search_similar(self, query_embedding, top_k=5):
        with self.tracker.track_vector_search(1, self.collection.count()):
            # Mevcut vector search kodunuz
            return existing_search_code(query_embedding, top_k)
```

## 📊 **Real-time Monitoring Dashboard**

Streamlit sidebar'ına performance monitoring ekleyin:

```python
# Sidebar'da performance section
with st.sidebar:
    st.subheader("⚡ Performance Monitoring")
    
    # Quick stats
    if st.button("📊 Show Performance Stats"):
        from evaluation.metrics.basic_analytics import BasicAnalytics
        analytics = BasicAnalytics()
        stats = analytics.get_quick_stats()
        
        if 'error' not in stats:
            st.metric("Total Operations", stats['total_operations'])
            st.metric("Avg Duration", f"{stats['avg_duration']:.2f}s")
            st.metric("GPU Available", stats['gpu_available'])
        
    # System stats
    if st.button("🖥️ System Resources"):
        tracker = get_global_tracker()
        sys_stats = tracker.get_system_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU", f"{sys_stats.cpu_percent:.1f}%")
            st.metric("Memory", f"{sys_stats.memory_percent:.1f}%")
        with col2:
            st.metric("Available RAM", f"{sys_stats.memory_available:.1f}GB")
            st.metric("GPU", "✅" if sys_stats.gpu_available else "❌")
```

## 🚀 **Automatic Performance Reporting**

Otomatik performance raporu oluşturun:

```python
# evaluation/generate_report.py
from evaluation.metrics.basic_analytics import BasicAnalytics
from evaluation.metrics.performance_tracker import get_global_tracker

def generate_daily_report():
    """Generate daily performance report"""
    analytics = BasicAnalytics()
    tracker = get_global_tracker()
    
    # Get comprehensive stats
    stats = analytics.get_quick_stats()
    summary = tracker.get_performance_summary()
    
    # Generate report
    report = f"""
# Daily Performance Report - {datetime.now().strftime('%Y-%m-%d')}

## Overview
- Total Operations: {stats.get('total_operations', 0)}
- Average Duration: {stats.get('avg_duration', 0):.2f}s
- GPU Available: {stats.get('gpu_available', False)}

## Performance by Operation Type
"""
    
    if 'operations_by_type' in summary:
        for op_type, data in summary['operations_by_type'].items():
            report += f"- {op_type}: {data['count']} ops, avg {data['avg_duration']:.2f}s\n"
    
    # Save report
    with open(f"evaluation/reports/daily_report_{datetime.now().strftime('%Y%m%d')}.md", 'w') as f:
        f.write(report)
    
    return report
```

## 🎯 **Integration Checklist**

### ✅ **Completed:**
- [x] Performance Tracker oluşturuldu
- [x] Basic Analytics eklendi
- [x] Test edildi ve çalışıyor
- [x] JSON rapor sistemi hazır

### 📋 **Next Steps:**
1. [ ] Streamlit app'e entegre et
2. [ ] PDF processing'e tracking ekle
3. [ ] Query response'a tracking ekle
4. [ ] Embedding generation'a tracking ekle
5. [ ] Vector search'e tracking ekle
6. [ ] Real-time dashboard ekle
7. [ ] Otomatik rapor sistemi kur

## 🔍 **Monitoring Best Practices**

### **1. Critical Metrics to Track:**
- Document processing time (target: <5s per page)
- Query response time (target: <3s)
- Memory usage (watch for leaks)
- GPU utilization (optimize for A100)

### **2. Alert Thresholds:**
```python
PERFORMANCE_THRESHOLDS = {
    'pdf_processing_per_page': 5.0,  # seconds
    'query_response': 3.0,           # seconds
    'embedding_generation': 2.0,     # seconds
    'memory_usage': 85.0             # percentage
}
```

### **3. Regular Monitoring:**
- Daily performance reports
- Weekly trend analysis
- Monthly optimization reviews

## 📈 **Academic Benefits**

Bu tracking sistemi akademik projeniz için:

1. **Measurable Results**: Objective performance data
2. **Before/After Comparison**: Optimization impact
3. **Reproducible Research**: Consistent benchmarking
4. **System Reliability**: Error rate monitoring
5. **Resource Efficiency**: Hardware utilization analysis

**✅ Performance Tracker sisteminiz akademik standartlarda monitoring capability sağlıyor!** 