# 📄💬 Turkish Financial RAG ChatBot

Türkçe finansal dokümanlar için geliştirilmiş akıllı sohbet robotu. PDF ve Excel dosyalarınızı yükleyin, Local LLM ile finansal analiz yapın.

## ✨ Özellikler

- 🤖 **Local LLM Desteği**: Mistral 7B GGUF modeli ile tamamen offline çalışma
- 📄 **PDF İşleme**: Finansal raporlar, tablolar ve metinleri otomatik analiz
- 📊 **Excel Desteği**: XLS/XLSX dosyalarını okuma ve analiz etme
- 🧠 **Akıllı RAG**: ChromaDB vector database ile hızlı ve doğru bilgi erişimi
- 🌍 **Türkçe Optimizasyon**: Multilingual sentence transformers
- ⚡ **A100 GPU Optimizasyonu**: Colab Pro Plus için optimize edilmiş
- 🌐 **NGROK Desteği**: Public URL ile kolay erişim
- 🔒 **Güvenlik**: Tamamen local, internet gerektirmez

## 🚀 Colab Pro Plus + NGROK Hızlı Kurulum

### Tek Komutla Setup (Önerilen) 🎯

```bash
# 1. Repository clone
!git clone https://github.com/your-username/financial-rag-chatbot.git
%cd financial-rag-chatbot

# 2. Otomatik setup + NGROK başlatma (Tek komut!)
!python colab_setup.py
```

**🎉 Bu komut çalıştıktan sonra otomatik olarak:**
- Google Drive bağlanacak
- Tüm gereksinimler yüklenecek
- NGROK token ayarlanacak
- Streamlit başlatılacak
- Public URL oluşturulacak

### Manuel Setup (Gelişmiş)

```bash
# 1. Repository clone
!git clone https://github.com/your-username/financial-rag-chatbot.git
%cd financial-rag-chatbot

# 2. Drive bağla
from google.colab import drive
drive.mount('/content/drive')

# 3. Gereksinimleri yükle
!pip install -r requirements.txt

# 4. Model dosyasını yerleştir
# Path: /content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# 5. NGROK setup
from pyngrok import ngrok
ngrok.set_auth_token("2xmENf6pFX37FGhDuGBuWpBSRHG_2TfzVLgN9LiFCL2Zdi1Wf")

# 6. Streamlit + NGROK başlat
import threading
import time

def run_streamlit():
    !streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
streamlit_thread.start()

time.sleep(10)  # Streamlit başlangıcını bekle
public_url = ngrok.connect(8501)
print(f'🌐 Public URL: {public_url}')
```

## 📁 Proje Yapısı

```
financial_report/
├── streamlit_app.py       # Ana Streamlit uygulaması
├── colab_setup.py         # Colab + NGROK otomatik kurulum
├── src/
│   ├── __init__.py        # Package initialization
│   ├── pdf_processor.py   # PDF işleme modülü
│   ├── text_processor.py  # Metin ve embedding işleme
│   ├── vector_store.py    # ChromaDB vector database
│   ├── excel_processor.py # Excel dosya işleme
│   └── llm_service_local.py # Local LLM servisi
├── requirements.txt       # Python gereksinimleri (pyngrok dahil)
├── .gitignore            # Git ignore dosyası
└── README.md             # Bu dosya
```

## 🌐 NGROK Kullanımı

### Avantajları
- ✅ **Public URL**: Herhangi bir yerden erişim
- ✅ **Paylaşım**: URL'yi paylaşarak başkalarının erişimi
- ✅ **HTTPS**: Güvenli bağlantı
- ✅ **Kolay Setup**: Tek komutla hazır

### Önemli Notlar
- 🔑 NGROK token projeye dahil edilmiştir
- ⏰ Colab session kapandığında URL devre dışı kalır
- 🔄 Yeni session'da yeni URL oluşur
- 📱 Mobil cihazlardan da erişim mümkün

## 📊 Desteklenen Dosya Formatları

- **PDF**: Finansal raporlar, tablolar, metinler (20-70 sayfa optimum)
- **Excel**: .xls, .xlsx, .xlsm formatları
- **İçerik**: Türkçe finansal dokümanlar

## 🔧 Kullanım

1. **Sistem Başlatma**: Sol sidebar'dan "🚀 Sistemi Başlat" butonuna tıklayın
2. **Dosya Yükleme**: PDF veya Excel dosyalarını yükleyin ve işleyin
3. **Sohbet**: Ana ekranda finansal sorularınızı sorun
4. **İstatistikler**: Sol menüden sistem durumunu izleyin

### Örnek Sorular

- "Bu dökümanların özeti nedir?"
- "Finansal tablolardaki ana göstergeler nelerdir?"
- "Risk analizinde öne çıkan faktörler neler?"
- "Gelir tablosundaki trend nasıl?"
- "Nakit akımı durumu nasıl?"

## ⚙️ Teknik Detaylar

### GPU Optimizasyonları (A100)
- Context window: 8192 token
- Batch size: 4096
- GPU layers: Full offload
- Memory mapping: Enabled
- CUDA optimizations: A100 specific

### Vector Database (ChromaDB)
- Embedding Model: Multilingual sentence transformers
- Database: ChromaDB PersistentClient
- Chunk size: 800 karakter
- Overlap: 150 karakter
- Metadata filtering: Built-in support

### LLM Ayarları
- Model: Mistral 7B Instruct v0.2 (Q4_K_M)
- Temperature: 0.7
- Max tokens: 1024
- Top-p: 0.95
- Chat format: Mistral official

### Multiprocessing
- PDF işleme: Parallel page processing
- Excel işleme: Multi-sheet support
- Embedding: Batch processing

### NGROK Konfigürasyonu
- Port: 8501
- Protocol: HTTP/HTTPS
- Token: Embedded (güvenli)
- Tunnel type: Public

## 🛠️ Geliştirme

```bash
# Local geliştirme (NGROK olmadan)
git clone https://github.com/your-username/financial-rag-chatbot.git
cd financial-rag-chatbot
pip install -r requirements.txt

# Local test
streamlit run streamlit_app.py
```

## 🎯 Performans

- **A100 GPU**: ~5-10 saniye yanıt süresi
- **PDF Processing**: ~2-5 sayfa/saniye
- **Vector Search**: <100ms
- **Memory Usage**: ~8-12GB (A100)
- **NGROK Latency**: +50-100ms (tunnel overhead)

## 🔍 Troubleshooting

### Model bulunamadı hatası
```bash
# Model path'ini kontrol edin:
/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### GPU kullanılamıyor
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### ChromaDB hatası
```bash
# ChromaDB klasörünü temizle
!rm -rf chroma_db/
```

### NGROK bağlantı sorunları
```python
# NGROK tunnel'ları kontrol et
from pyngrok import ngrok
print(ngrok.get_tunnels())

# Tunnel'ları kapat ve yeniden başlat
ngrok.kill()
public_url = ngrok.connect(8501)
print(f'🌐 Yeni URL: {public_url}')
```

### Streamlit çalışmıyor
```bash
# Port kullanımını kontrol et
!netstat -tulpn | grep 8501

# Streamlit'i yeniden başlat
!pkill -f streamlit
!streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

## 📝 Lisans

MIT License

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📞 İletişim

Herhangi bir sorunuz için issue açabilirsiniz.

## 🚀 Roadmap

- [ ] Web deployment desteği (Heroku, Vercel)
- [ ] Daha fazla dosya formatı (.docx, .txt)
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] API endpoint desteği
- [ ] NGROK Pro features integration

## 🎉 Quick Start Guide

### 1-Dakika Setup:
```bash
!git clone https://github.com/your-username/financial-rag-chatbot.git
%cd financial-rag-chatbot
!python colab_setup.py
```

### URL Alın ve Kullanın:
1. Setup tamamlandığında public URL'yi kopyalayın
2. Tarayıcınızda açın
3. Finansal dokümanlarınızı yükleyin
4. Sohbet etmeye başlayın!

---

**Not**: Bu proje Colab Pro Plus A100 GPU + NGROK için optimize edilmiştir. ChromaDB kullanarak modern vector database desteği sağlar. Public URL ile herhangi bir yerden erişilebilir. 

## ⚙️ **Advanced RAG Settings**

The system now includes comprehensive configurable parameters for optimal RAG performance:

### 📏 **Chunk Configuration**
- **Chunk Size**: 300-1500 characters (default: 800)
- **Overlap Size**: 50-300 characters (default: 150)
- Real-time adjustment in Streamlit interface

### 🔍 **Retrieval Parameters**
- **Top-K Results**: 1-15 results (default: 5)
- **Similarity Threshold**: 0.0-1.0 (default: 0.3)
- **Max Context Length**: 1000-5000 characters (default: 3000)
- **Search Strategy**: hybrid, semantic_only, keyword_boost

### 🧠 **Embedding Models**
Choose from multiple embedding models:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (default, Turkish optimized)
- `sentence-transformers/all-MiniLM-L6-v2` (lightweight)
- `sentence-transformers/all-mpnet-base-v2` (high quality)

### 📊 **Performance Monitoring**
- Query response time tracking
- Similarity score analysis
- Context length optimization
- Chunk statistics with detailed metrics

## 🎯 **Features**

### Core RAG System
- **ChromaDB Vector Database**: Fast similarity search with cosine similarity
- **Mistral 7B GGUF Model**: A100 GPU optimized local inference
- **Multilingual Embeddings**: Turkish financial text optimization
- **Advanced Chunking**: Configurable size and overlap strategies

### Document Processing
- **PDF Processing**: Text + table extraction with multiprocessing
- **Excel Processing**: Multi-sheet analysis with metadata preservation
- **Smart Chunking**: Context-aware text segmentation
- **Metadata Tracking**: Source files, page numbers, content types

### User Interface
- **Modern Streamlit Interface**: Professional design with gradients
- **Advanced Settings Panel**: Real-time parameter adjustment
- **Performance Dashboard**: Query metrics and similarity tracking
- **RAG Process Transparency**: Detailed retrieval and context building insights

## 🔧 **Configuration Options**

### Basic Setup
```python
# Default settings
RAG_SETTINGS = {
    'chunk_size': 800,
    'overlap_size': 150,
    'top_k': 5,
    'similarity_threshold': 0.3,
    'max_context_length': 3000,
    'search_strategy': 'hybrid',
    'embedding_model': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
```

### Advanced Tuning

#### For Financial Reports (Large Context)
```python
settings = {
    'chunk_size': 1200,      # Larger chunks for financial contexts
    'overlap_size': 200,     # Higher overlap for continuity
    'top_k': 8,             # More results for comprehensive analysis
    'similarity_threshold': 0.25,  # Lower threshold for broader matches
    'max_context_length': 4000     # Larger context for detailed analysis
}
```

#### For Quick Q&A (Performance Optimized)
```python
settings = {
    'chunk_size': 600,       # Smaller chunks for faster processing
    'overlap_size': 100,     # Minimal overlap
    'top_k': 3,             # Fewer results for speed
    'similarity_threshold': 0.4,   # Higher threshold for precision
    'max_context_length': 2000     # Compact context
}
```

## 📈 **Performance Metrics**

The system tracks and displays:
- **Query Response Time**: End-to-end latency
- **Similarity Scores**: Average and peak similarity metrics
- **Context Quality**: Length and relevance analysis
- **Chunk Statistics**: Size distribution and processing efficiency
- **Memory Usage**: A100 GPU utilization monitoring

## 🛠️ **Manual Setup**

### Prerequisites
```bash
# Python 3.8+ recommended
pip install -r requirements.txt
```

### Local Installation
```bash
git clone <repository>
cd financial_report

# Install dependencies
pip install streamlit chromadb sentence-transformers
pip install PyPDF2 openpyxl pandas numpy
pip install llama-cpp-python pyngrok

# Start application
streamlit run streamlit_app.py
```

### NGROK Setup
1. Get your token from [ngrok.com](https://ngrok.com)
2. Add to `colab_setup.py` or set as environment variable:
```python
NGROK_TOKEN = "your_token_here"
```

## 📊 **System Architecture**

```
Documents (PDF/Excel) → Text Processing → Chunking → Embeddings → ChromaDB
                                                                      ↓
User Query → Embedding → Similarity Search → Context Building → Mistral 7B → Response
```

### Processing Pipeline
1. **Document Ingestion**: PDF/Excel parsing with table extraction
2. **Smart Chunking**: Configurable size with overlap for context preservation
3. **Embedding Generation**: Turkish-optimized multilingual embeddings
4. **Vector Storage**: ChromaDB with metadata indexing
5. **Retrieval**: Similarity search with configurable parameters
6. **Context Building**: Intelligent context assembly with length limits
7. **Generation**: Mistral 7B inference with A100 optimization

## 🎯 **Use Cases**

### Financial Analysis
- **Annual Reports**: Comprehensive financial statement analysis
- **Investment Research**: Due diligence and risk assessment
- **Regulatory Documents**: Compliance and regulatory analysis
- **Market Research**: Turkish market insights and trends

### Configurable for Different Scenarios
- **Executive Summaries**: High-level overview (large chunks, broad similarity)
- **Detail Extraction**: Specific data points (small chunks, high precision)
- **Comparative Analysis**: Cross-document insights (high top-k, moderate threshold)

## 🔧 **Troubleshooting**

### NGROK Session Conflicts
```python
# If you see "limited to 1 simultaneous ngrok agent sessions"
pyngrok.ngrok.disconnect_all()
pyngrok.ngrok.kill()
```

### Performance Optimization
- **A100 GPU**: Optimal chunk_size: 800-1200
- **T4 GPU**: Recommended chunk_size: 400-800  
- **CPU Only**: Use smaller chunks: 300-600

### Memory Management
- Monitor context length for large documents
- Adjust top_k based on available memory
- Use similarity threshold to filter irrelevant results

## 🚀 **Advanced Features**

### Real-time Parameter Adjustment
- Modify RAG settings without restarting
- A/B test different configurations
- Performance impact visualization

### Detailed Analytics
- Query performance profiling
- Similarity score distribution analysis
- Context quality metrics
- Processing time breakdown

### Enterprise Features
- Batch document processing
- Custom embedding model support
- Advanced search strategies
- Comprehensive logging and monitoring

## 📞 **Support**

For technical issues or feature requests, please check:
1. System requirements (A100 GPU recommended)
2. NGROK token configuration
3. Model file paths
4. Parameter tuning guidelines

---
**Built for Turkish Financial Market Analysis | A100 GPU Optimized | ChromaDB + Mistral 7B** 