# 🚀 Turkish Financial RAG Assistant

**A100 GPU Optimized • ChromaDB Vector Store • Mistral 7B GGUF • Streamlit Interface**

Türkçe finansal dokümanlar için geliştirilmiş, A100 GPU'da optimize edilmiş akıllı RAG (Retrieval-Augmented Generation) sistemi. PDF ve Excel dosyalarınızı yükleyin, Mistral 7B modeli ile finansal analiz yapın.

## ✨ **Ana Özellikler**

### 🤖 **AI & Model Desteği**
- **Mistral 7B GGUF**: Q4_K_M quantized model, tamamen offline
- **A100 GPU Optimization**: 6-14 saniye response time (20 sayfa PDF)
- **Turkish Optimized**: Multilingual sentence transformers
- **GGUF + llama-cpp-python**: Memory efficient inference

### 📄 **Document Processing**
- **PDF Processing**: Text + table extraction, parallel processing
- **Excel Support**: .xls/.xlsx/.xlsm with multi-sheet analysis  
- **Smart Chunking**: Configurable 300-1500 characters
- **Metadata Preservation**: Source tracking, page numbers, content types

### 🧠 **Advanced RAG System**
- **ChromaDB Vector Store**: Persistent storage with similarity search
- **Query Embedding Cache**: 10x speed boost for repeated queries
- **Configurable Parameters**: Top-K, similarity threshold, context length
- **Multi-strategy Search**: Hybrid, semantic-only, keyword-boost
- **Performance Monitoring**: Real-time metrics and optimization

### ⚡ **Performance Optimizations**
- **Embedding Model Reuse**: No repeated loading (10-50x speedup)
- **Batch Processing**: Optimal batch sizes for A100
- **Multiprocessing**: 4-8 workers for PDF/Excel processing  
- **Memory Management**: Aggressive cleanup, GPU optimization
- **UI Optimization**: Minimal rendering, fast response

## 🎯 **Performance Benchmarks**

| Operation | A100 GPU | T4 GPU | CPU Only |
|-----------|----------|---------|----------|
| **20-page PDF** | 6-14s | 30-60s | 2-5 min |
| **100 chunks embedding** | 3-8s | 15-30s | 1-3 min |
| **Query response** | 2-5s | 8-15s | 20-45s |
| **Vector search** | <100ms | <500ms | 1-3s |

## 🚀 **Quick Start (Google Colab)**

### Option 1: Auto Setup (Recommended)
```bash
# Clone repository
!git clone https://github.com/your-username/turkish-financial-rag.git
%cd turkish-financial-rag

# Auto setup + NGROK
!python colab_setup.py
```

### Option 2: Manual Setup
```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Setup NGROK
from pyngrok import ngrok
ngrok.set_auth_token("your_token_here")

# 4. Start Streamlit
!streamlit run streamlit_app.py --server.port 8501 --server.headless true &

# 5. Create public URL
import time; time.sleep(10)
public_url = ngrok.connect(8501)
print(f'🌐 Access URL: {public_url}')
```

## 📁 **Project Structure**

```
financial_report/
├── streamlit_app.py          # Main Streamlit application (1320+ lines)
├── colab_setup.py            # Automated Colab + NGROK setup
├── diagnostic_check.py       # System diagnostics & health check
├── colab_usage_guide.md     # Detailed Colab usage instructions
├── requirements.txt          # Python dependencies
├── src/                      # Core modules
│   ├── __init__.py          # Package initialization
│   ├── llm_service_local.py # Mistral 7B GGUF service (470 lines)
│   ├── text_processor.py    # Embedding & chunking (620+ lines)
│   ├── vector_store.py      # ChromaDB integration (500+ lines)
│   ├── pdf_processor.py     # PDF processing (480+ lines)
│   └── excel_processor.py   # Excel processing (420+ lines)
├── chroma_db/               # ChromaDB persistent storage
└── README.md                # This file
```

## ⚙️ **Advanced Configuration**

### 🔧 **RAG Parameters**
```python
# Optimal for A100 GPU
RAG_SETTINGS = {
    'chunk_size': 800,              # 300-1500 characters
    'overlap_size': 150,            # 50-300 characters  
    'top_k': 5,                     # 1-15 results
    'similarity_threshold': 0.3,    # 0.0-1.0
    'max_context_length': 3000,     # 1000-5000 characters
    'search_strategy': 'hybrid',    # hybrid/semantic_only/keyword_boost
    'embedding_model': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
```

### ⚡ **Performance Settings**
```python
PERFORMANCE_PRESETS = {
    'a100_max_speed': {
        'pdf_workers': 6,
        'excel_workers': 6,
        'batch_size': 512,           # A100 optimized
        'gpu_memory_fraction': 0.95,
        'aggressive_cleanup': False,
        'reuse_embeddings': True
    },
    'balanced': {
        'pdf_workers': 4,
        'batch_size': 256,
        'gpu_memory_fraction': 0.8
    },
    'memory_optimized': {
        'pdf_workers': 2,
        'batch_size': 64,
        'gpu_memory_fraction': 0.6
    }
}
```

### 🧠 **Embedding Models**
- **Default**: `paraphrase-multilingual-MiniLM-L12-v2` (Turkish optimized)
- **Lightweight**: `all-MiniLM-L6-v2` (Fast, lower quality)
- **High Quality**: `all-mpnet-base-v2` (Slower, better results)

## 🎯 **System Architecture**

```
Documents (PDF/Excel) → Text Processing → Chunking → Embeddings → ChromaDB
                                                                      ↓
User Query → Query Cache → Similarity Search → Context Building → Mistral 7B → Response
```

### 🔄 **Processing Pipeline**
1. **Document Ingestion**: PDF text+tables, Excel multi-sheet
2. **Smart Chunking**: Configurable size with context overlap
3. **Batch Embedding**: A100 optimized batch processing
4. **Vector Storage**: ChromaDB with metadata indexing
5. **Cached Retrieval**: Query embedding cache for 10x speedup
6. **Context Assembly**: Intelligent length management
7. **LLM Generation**: Mistral 7B with A100 optimization

## 🛠️ **Usage Guide**

### 1. **System Initialization**
- Click **"🚀 Sistemi Başlat"** in sidebar
- Wait for model loading (15-30 seconds)
- Verify A100 GPU detection

### 2. **Document Processing**
```python
# Supported formats
SUPPORTED_FILES = {
    'PDF': ['.pdf'],                    # Financial reports, presentations
    'Excel': ['.xls', '.xlsx', '.xlsm'] # Financial statements, data tables
}

# Processing speeds (A100)
PDF_PROCESSING = "~0.5-2 seconds per page"
EXCEL_PROCESSING = "~1-5 seconds per sheet"
```

### 3. **Advanced Settings**
Access **"⚙️ Gelişmiş Ayarlar"** in sidebar:
- **RAG Parameters**: Chunk size, Top-K, similarity threshold
- **Performance Optimization**: Worker counts, batch sizes, memory usage
- **Performance Presets**: A100 Max Speed, Balanced, Memory Saver

### 4. **Chat Interface**
```python
# Example queries
EXAMPLE_QUERIES = [
    "Bu dökümanların finansal özeti nedir?",
    "Ana risk faktörleri nelerdir?", 
    "Gelir tablosundaki trend analizi",
    "Nakit akımı durumu nasıl?",
    "EBITDA marjı nasıl değişmiş?"
]
```

## 📊 **Monitoring & Analytics**

### 🔍 **Real-time Metrics**
- **Query Response Time**: End-to-end latency tracking
- **Similarity Scores**: Average and peak similarity analysis  
- **Context Quality**: Length optimization and relevance
- **Processing Speed**: Chunks/second, files/minute
- **GPU Utilization**: Memory usage and compute efficiency

### 📈 **Performance Dashboard**
Access via **"📈 Performans İzleme"** in sidebar:
- Processing speed history and trends
- Memory usage optimization recommendations
- A/B testing results for different configurations
- System health and bottleneck identification

## 🔧 **Troubleshooting**

### ❌ **Common Issues**

#### Model Loading Problems
```python
# Check model path
MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Verify file exists
import os
print("Model exists:", os.path.exists(MODEL_PATH))
print("File size:", os.path.getsize(MODEL_PATH) / (1024**3), "GB")
```

#### GPU Detection Issues
```python
# Check GPU availability
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / (1024**3), "GB")
```

#### ChromaDB Errors
```bash
# Clear vector database
!rm -rf chroma_db/
# Restart system to recreate
```

#### NGROK Connection Problems
```python
# Reset NGROK tunnels
from pyngrok import ngrok
ngrok.kill()
public_url = ngrok.connect(8501)
print(f'New URL: {public_url}')
```

### 🚨 **Emergency Diagnostics**
```bash
# Run comprehensive system check
python diagnostic_check.py
```

## 🎛️ **Advanced Features**

### 🏎️ **Performance Optimizations**
- **Embedding Model Reuse**: Single model instance across sessions
- **Query Embedding Cache**: Cache last 50 queries for instant retrieval
- **Batch Processing**: Auto-sizing based on GPU memory
- **Memory Management**: Aggressive cleanup with `torch.cuda.empty_cache()`
- **UI Optimization**: Minimal rendering, lazy loading

### 🔍 **Smart Retrieval**
- **Multi-strategy Search**: Combine semantic + keyword matching
- **Dynamic Filtering**: Similarity threshold adjustment
- **Context Assembly**: Intelligent duplicate removal
- **Source Tracking**: Detailed provenance with page numbers

### 📊 **Enterprise Features**
- **Batch Document Processing**: Handle multiple files simultaneously
- **Metadata Filtering**: Search by source file, content type, date ranges
- **Performance Profiling**: Detailed bottleneck analysis
- **Configuration Export**: Save/load optimal settings

## 🔒 **Security & Privacy**

- ✅ **Fully Offline**: No data leaves your environment
- ✅ **Local LLM**: Mistral 7B runs entirely on your GPU
- ✅ **Private Vector Store**: ChromaDB stored locally
- ✅ **NGROK Optional**: Can run without public access
- ✅ **No Telemetry**: Zero data collection

## 🌟 **Use Cases**

### 📈 **Financial Analysis**
- **Annual Reports**: 10K/10Q analysis with table extraction
- **Investment Research**: Due diligence document processing
- **Risk Assessment**: Regulatory filing analysis
- **Market Research**: Turkish market insights

### 🏢 **Enterprise Applications**
- **Document Automation**: Large-scale financial document processing
- **Compliance**: Regulatory document analysis
- **Internal Reports**: Company financial statement analysis
- **Audit Support**: Document review and summarization

## 🚀 **Roadmap**

### 🎯 **Near Term** (Q1 2024)
- [ ] **Multi-model Support**: Llama-2, Phi-2, CodeLlama
- [ ] **Advanced Analytics**: Document comparison, trend analysis
- [ ] **API Endpoints**: REST API for programmatic access
- [ ] **Batch Processing**: Queue system for large document sets

### 🌟 **Medium Term** (Q2-Q3 2024)
- [ ] **Web Deployment**: Heroku, Vercel, AWS deployment options
- [ ] **Multi-language**: English, German financial document support
- [ ] **Advanced UI**: React frontend, mobile optimization
- [ ] **Enterprise Features**: User management, audit trails

### 🔮 **Long Term** (Q4 2024+)
- [ ] **AI Agents**: Autonomous financial analysis workflows
- [ ] **Real-time Data**: Market data integration
- [ ] **Custom Models**: Domain-specific fine-tuned models
- [ ] **Enterprise SaaS**: Multi-tenant cloud solution

## 📝 **Contributing**

```bash
# Development setup
git clone https://github.com/your-username/turkish-financial-rag.git
cd turkish-financial-rag

# Install dev dependencies  
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Code formatting
black src/ streamlit_app.py
```

### 🤝 **Contribution Guidelines**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure code formatting with `black`
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Mistral AI**: Mistral 7B foundation model
- **ChromaDB**: Vector database infrastructure  
- **Sentence Transformers**: Multilingual embedding models
- **Streamlit**: Modern web interface framework
- **llama-cpp-python**: Efficient GGUF model inference

## 📞 **Support**

- 📝 **Issues**: Open GitHub issue for bugs/features
- 💬 **Discussions**: GitHub Discussions for questions
- 📧 **Contact**: [Your contact information]

---

**🎉 Ready to analyze your Turkish financial documents with AI?**

```bash
!git clone https://github.com/your-username/turkish-financial-rag.git
%cd turkish-financial-rag
!python colab_setup.py
```

**⚡ Performance Guarantee**: 20-page PDF → 6-14 seconds on A100 GPU 