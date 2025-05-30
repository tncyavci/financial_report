# 📚 Dependencies Guide - Turkish Financial RAG System

Bu dokümanda projenin `requirements.txt` dosyasındaki tüm kütüphanelerin kullanım amaçları ve projemizdeki rolleri detaylı olarak açıklanmaktadır.

---

## 🔧 Core Dependencies (Ana Bağımlılıklar)

### `streamlit>=1.28.0`
- **Amaç**: Web arayüzü ve kullanıcı etkileşimi
- **Kullanım**: Ana uygulama framework'ü
- **Proje İçi Rol**: 
  - PDF/Excel dosya yükleme arayüzü
  - RAG sorgulama paneli
  - Gelişmiş ayarlar ve yapılandırma
  - Real-time response streaming
  - Interactive chat interface

### `pandas>=2.0.0`
- **Amaç**: Veri manipülasyonu ve analizi
- **Kullanım**: Excel dosyalarını okuma ve işleme
- **Proje İçi Rol**:
  - Excel çalışma sayfalarını DataFrame'e dönüştürme
  - Finansal tablo verilerini parse etme
  - Veri temizleme ve formatı düzenleme
  - Chunk metadata yönetimi

### `numpy>=1.24.0`
- **Amaç**: Sayısal hesaplamalar ve array işlemleri
- **Kullanım**: Embedding vektörleri ve matematik işlemleri
- **Proje İçı Rol**:
  - Embedding vektörlerinin numpy array formatında saklanması
  - Benzerlik skorları hesaplama
  - Vector similarity calculations
  - Efficient array operations for RAG

---

## 📄 PDF Processing (PDF İşleme)

### `PyPDF2>=3.0.0`
- **Amaç**: Temel PDF metin çıkarma
- **Kullanım**: PDF dosyalarından text extraction
- **Proje İçi Rol**:
  - Primary PDF text extractor
  - Sayfa bazlı metin çıkarma
  - PDF metadata okuma
  - Fallback extraction method

### `pdfplumber>=0.9.0`
- **Amaç**: Gelişmiş PDF analizi ve tablo çıkarma
- **Kullanım**: PDF'lerden yapılandırılmış veri çıkarma
- **Proje İçi Rol**:
  - Tablo detection ve extraction
  - Financial tables parsing
  - Layout-aware text extraction
  - Koordinat bazlı veri çıkarma

### `pymupdf>=1.23.0`
- **Amaç**: Yüksek performanslı PDF işleme
- **Kullanım**: Hızlı PDF rendering ve text extraction
- **Proje İçi Rol**:
  - High-performance PDF processing
  - Multi-format document support
  - Image ve diagram detection
  - Advanced layout analysis

---

## 📊 Excel Processing (Excel İşleme)

### `openpyxl>=3.1.0`
- **Amaç**: Modern Excel dosyalarını okuma (.xlsx)
- **Kullanım**: Excel 2007+ formatları için
- **Proje İçi Rol**:
  - .xlsx dosyalarını okuma
  - Çoklu çalışma sayfası desteği
  - Finansal spreadsheet parsing
  - Cell metadata preserving

### `xlrd>=2.0.0`
- **Amaç**: Eski Excel dosyalarını okuma (.xls)
- **Kullanım**: Excel 97-2003 formatları için
- **Proje İçi Rol**:
  - Legacy .xls file support
  - Backward compatibility
  - Old financial reports processing

---

## 🧠 NLP and Embeddings (Doğal Dil İşleme ve Vektör Temsilleri)

### `sentence-transformers>=2.2.0`
- **Amaç**: Metin embedding modelleri
- **Kullanım**: Türkçe/çok dilli sentence embeddings
- **Proje İçi Rol**:
  - **Primary embedding model**: `paraphrase-multilingual-MiniLM-L12-v2`
  - Query ve document embeddings
  - Semantic similarity hesaplama
  - Vector representations for RAG

### `transformers>=4.30.0`
- **Amaç**: Hugging Face transformer modelleri
- **Kullanım**: LLM modelleri ve tokenization
- **Proje İçi Rol**:
  - **Mistral 7B** model support
  - **Llama 3.1 8B** model support
  - AutoTokenizer ve AutoModel
  - HuggingFace ecosystem integration

### `torch>=2.0.0`
- **Amaç**: Deep learning framework
- **Kullanım**: Neural network backend
- **Proje İçi Rol**:
  - **GPU/CUDA detection**: `torch.cuda.is_available()`
  - **Device management**: GPU vs CPU optimization
  - **Memory management**: `torch.cuda.empty_cache()`
  - **Model loading**: tensor operations ve inference

---

## 🗄️ Vector Database (Vektör Veritabanı)

### `chromadb>=0.4.15`
- **Amaç**: Vektör veritabanı ve similarity search
- **Kullanım**: Embedding storage ve retrieval
- **Proje İçi Rol**:
  - **Vector storage**: Dokümanlarden embedding'leri saklama
  - **Similarity search**: Query'ye en yakın chunk'ları bulma
  - **Persistent storage**: Embedding'leri disk'te saklama
  - **RAG retrieval**: Top-K document retrieval
  - **Collection management**: Farklı doküman setleri

---

## 🦙 GGUF Model Support (GGUF Model Desteği)

### `llama-cpp-python>=0.2.0`
- **Amaç**: GGUF formatındaki LLM'leri çalıştırma
- **Kullanım**: Quantized modeller için
- **Proje İçi Rol**:
  - **Mistral 7B GGUF** support
  - **Memory efficient inference**: Quantized modeller
  - **Local model execution**: GPU/CPU optimization
  - **Fast response generation**: GGUF performance benefits
  - **Llama.cpp backend**: C++ optimizations

---

## 📝 Text Processing (Metin İşleme)

### `langdetect>=1.0.9`
- **Amaç**: Otomatik dil tespiti
- **Kullanım**: Türkçe/İngilizce doküman ayrımı
- **Proje İçi Rol**:
  - **Language detection**: `detect(text)` → 'tr', 'en', vs.
  - **Multilingual RAG**: Dil bazlı optimizasyon
  - **Turkish focus**: Türkçe finansal dokümanları tanımlama
  - **Content routing**: Dile göre işleme stratejisi

### `nltk>=3.8.0`
- **Amaç**: Natural Language Processing toolkit
- **Kullanım**: Tokenization ve sentence splitting
- **Proje İçi Rol**:
  - **Sentence tokenization**: `sent_tokenize(text, language='turkish')`
  - **Intelligent chunking**: Cümle sınırlarına göre bölme
  - **Turkish language support**: Türkçe'ye özel rules
  - **Text preprocessing**: Smart text segmentation

---

## 🛠️ Utilities (Yardımcı Araçlar)

### `python-dotenv>=1.0.0`
- **Amaç**: Environment variables yönetimi
- **Kullanım**: Konfigürasyon ve API keys
- **Proje İçi Rol**:
  - **Environment config**: `.env` dosyası desteği
  - **API key management**: Güvenli credential storage
  - **Development settings**: Local vs production configs

### `tqdm>=4.65.0`
- **Amaç**: Progress bar gösterimi
- **Kullanım**: Uzun işlemler için görsel feedback
- **Proje İçi Rol**:
  - **PDF processing progress**: Batch processing feedback
  - **Embedding generation**: Chunk processing progress
  - **File upload tracking**: Upload progress indicator
  - **User experience**: Visual processing status

### `requests>=2.31.0`
- **Amaç**: HTTP istekleri
- **Kullanım**: API çağrıları ve dosya indirme
- **Proje İçi Rol**:
  - **Model downloading**: HuggingFace model fetch
  - **External API calls**: Potential future integrations
  - **Resource fetching**: External dependency downloads

---

## 🚀 Colab & Ngrok (Google Colab ve Tunnel Desteği)

### `pyngrok>=6.0.0`
- **Amaç**: ngrok tunnel oluşturma
- **Kullanım**: Google Colab'da public URL
- **Proje İçi Rol**:
  - **Colab deployment**: Public access tunnel
  - **Remote access**: Local Streamlit → Public URL
  - **Development tunneling**: Secure HTTPS endpoints
  - **Demo sharing**: Easy application sharing

---

## 🔧 Development (Geliştirme)

### `typing-extensions>=4.7.0`
- **Amaç**: Gelişmiş type hinting
- **Kullanım**: Modern Python typing features
- **Proje İçi Rol**:
  - **Type safety**: Better code quality
  - **IDE support**: Enhanced autocomplete
  - **Code documentation**: Self-documenting types
  - **Error prevention**: Compile-time type checking

---

## 📊 Dependency Statistics

| Kategori | Kütüphane Sayısı | Toplam Boyut (tahmini) |
|----------|------------------|------------------------|
| Core | 3 | ~50MB |
| PDF Processing | 3 | ~30MB |
| Excel Processing | 2 | ~10MB |
| NLP & Embeddings | 3 | ~2GB |
| Vector Database | 1 | ~100MB |
| GGUF Support | 1 | ~150MB |
| Text Processing | 2 | ~50MB |
| Utilities | 3 | ~20MB |
| Colab & Development | 2 | ~10MB |
| **TOPLAM** | **20** | **~2.4GB** |

---

## ⚡ Performance Impact

### 🚀 **Kritik Performans Kütüphaneleri:**
1. **torch**: GPU acceleration
2. **sentence-transformers**: Fast embeddings
3. **chromadb**: Optimized vector search
4. **llama-cpp-python**: GGUF efficiency

### 🐌 **Potansiyel Bottleneck'lar:**
1. **transformers**: Large model loading
2. **pyngrok**: Network dependency
3. **nltk**: First-run data download

### 💡 **Optimization Tips:**
- GPU kullanımı için **torch CUDA** support
- **Embedding cache** için ChromaDB persistence
- **Batch processing** için optimized chunk sizes
- **Memory management** için regular cache cleanup

---

## 🔄 Update Strategy

### **Kritik Updates:**
- **transformers**: Yeni model desteği için
- **chromadb**: Performance iyileştirmeleri için
- **torch**: CUDA compatibility için

### **Güvenli Updates:**
- **streamlit**: UI iyileştirmeleri
- **pandas**: Veri işleme optimizasyonları
- **tqdm**: Progress bar enhancements

Bu kılavuz sayesinde projenin her dependency'si ve bunların RAG sistemindeki rolleri net olarak anlaşılabilir. 🎯 