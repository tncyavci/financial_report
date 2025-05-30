# 🎛️ Gelişmiş Ayarlar ve Performans Optimizasyonları Rehberi

**Turkish Financial RAG Assistant - Advanced Configuration Guide**

Bu dokümantasyon sistemdeki tüm gelişmiş ayarların ne için kullanıldığını, nasıl optimize edildiğini ve hangi durumlarda hangi değerlerin seçilmesi gerektiğini açıklar.

## 📊 **RAG Parametreleri**

### 🔧 **Chunk Size (Chunk Boyutu)**
**Değer Aralığı:** 50-1500 karakter  
**Varsayılan:** 800 karakter

#### 🎯 **Ne İçin Kullanılır:**
- Dökümanları küçük parçalara böler (chunking)
- Her chunk ayrı bir embedding vektörü oluşturur
- RAG sisteminin temel işlem birimi

#### ⚙️ **Optimize Edilme Nedenleri:**

**Çok Küçük Chunk Size (50-200):**
- ✅ **Avantajlar:**
  - Sayısal veriler ve KPI'lar için ideal
  - Tablolardaki tek satır bilgiler
  - Finansal metrikler (EBITDA, ROE vb.)
  - Başlık ve önemli kısa bilgiler
  - Ultra hızlı embedding
- ❌ **Dezavantajlar:**
  - Context tamamen kaybolur
  - Çok fazla chunk oluşur
  - Bilgi parçalanması riski yüksek

**Küçük Chunk Size (200-600):**
- ✅ **Avantajlar:**
  - Daha detaylı ve spesifik bilgi
  - Hızlı embedding oluşturma
  - Düşük memory kullanımı
  - Tablo satırları ve kısa paragraflar
- ❌ **Dezavantajlar:**
  - Context kaybı riski
  - Parçalanmış bilgi
  - Daha fazla chunk sayısı

**Büyük Chunk Size (1000-1500):**
- ✅ **Avantajlar:**
  - Daha kapsamlı context
  - İlişkisel bilgi korunması
  - Daha az chunk sayısı
- ❌ **Dezavantajlar:**
  - Yavaş embedding
  - Yüksek memory kullanımı
  - Bilgi karışması riski

#### 🎯 **Kullanım Senaryoları:**
```python
# Sayısal veriler ve KPI'lar için
chunk_size = 100  # Kısa finansal metrikler

# Tablo satırları için
chunk_size = 200  # Tek satır bilgiler

# Hızlı soru-cevap için
chunk_size = 600  # Spesifik bilgiler

# Kapsamlı analiz için  
chunk_size = 1200  # Geniş context

# Genel kullanım için
chunk_size = 800   # Optimal denge
```

#### 📊 **Finansal Veri Türleri ve Optimal Chunk Sizes:**
```python
financial_data_chunks = {
    50-100:   "KPI'lar, oranlar, tek metrikler",
    100-200:  "Tablo satırları, kısa tanımlar", 
    200-400:  "Paragraf başına analiz",
    400-800:  "Bölüm bazlı bilgiler",
    800-1200: "Sayfa bazlı kapsamlı analiz",
    1200+:    "Multi-sayfa karşılaştırmalı analiz"
}
```

---

### 🔗 **Overlap Size (Örtüşme Boyutu)**
**Değer Aralığı:** 50-300 karakter  
**Varsayılan:** 150 karakter

#### 🎯 **Ne İçin Kullanılır:**
- Chunk'lar arası bilgi kontinüitesi sağlar
- Cümle ve paragraf bütünlüğünü korur
- Context kaybını önler

#### ⚙️ **Optimize Edilme Nedenleri:**

**Düşük Overlap (50-100):**
- Hızlı işleme
- Az tekrar bilgi
- Memory tasarrufu

**Yüksek Overlap (200-300):**
- Daha iyi context kontinüitesi
- Güçlü bilgi bağlantıları
- Kayıp bilgi riski düşük

#### 📊 **Optimal Oranlar:**
```python
# Overlap/Chunk oranı
overlap_ratio = overlap_size / chunk_size

# Önerilen oranlar:
optimal_ratio = 0.15-0.25  # %15-25 arası ideal
```

---

### 🔍 **Top-K Results**
**Değer Aralığı:** 1-15 sonuç  
**Varsayılan:** 5 sonuç

#### 🎯 **Ne İçin Kullanılır:**
- Vector search'te kaç benzer chunk döndürülecek
- LLM'e gönderilecek context miktarını belirler
- Kalite vs hız dengesini kontrol eder

#### ⚙️ **Optimize Edilme Nedenleri:**

**Düşük Top-K (1-3):**
- ✅ Ultra hızlı response
- ✅ Net ve odaklı cevaplar
- ❌ Sınırlı bilgi kapsamı

**Yüksek Top-K (8-15):**
- ✅ Kapsamlı bilgi analizi
- ✅ Çoklu perspektif
- ❌ Yavaş işleme
- ❌ Bilgi kirliliği riski

#### 🎯 **Kullanım Senaryoları:**
```python
# Hızlı lookup için
top_k = 3

# Detaylı analiz için
top_k = 8

# Araştırma ve karşılaştırma için
top_k = 12
```

---

### 📊 **Similarity Threshold (Benzerlik Eşiği)**
**Değer Aralığı:** 0.0-1.0  
**Varsayılan:** 0.3

#### 🎯 **Ne İçin Kullanılır:**
- Minimum benzerlik skorunu belirler
- İlgisiz sonuçları filtreler
- Cevap kalitesini kontrol eder

#### ⚙️ **Similarity Score Anlamları:**
```python
similarity_scores = {
    0.8-1.0: "Çok yüksek benzerlik - Neredeyse aynı",
    0.6-0.8: "Yüksek benzerlik - Güçlü ilişki",
    0.4-0.6: "Orta benzerlik - İlgili içerik", 
    0.2-0.4: "Düşük benzerlik - Zayıf ilişki",
    0.0-0.2: "Çok düşük benzerlik - İlgisiz"
}
```

#### 🎯 **Threshold Optimizasyonu:**
```python
# Kesin bilgi için
threshold = 0.5  # Sadece çok ilgili sonuçlar

# Genel sorular için  
threshold = 0.3  # Dengeli filtreleme

# Araştırma için
threshold = 0.2  # Geniş kapsam
```

---

### 📄 **Max Context Length**
**Değer Aralığı:** 1000-5000 karakter  
**Varsayılan:** 3000 karakter

#### 🎯 **Ne İçin Kullanılır:**
- LLM'e gönderilecek maksimum context boyutu
- Token limitlerini kontrol eder
- Generation kalitesini optimize eder

#### ⚙️ **Context Length vs Performance:**
```python
context_performance = {
    1000-2000: "Hızlı, basit sorular",
    2000-3000: "Optimal denge",
    3000-4000: "Detaylı analiz", 
    4000-5000: "Kapsamlı araştırma"
}
```

---

## ⚡ **Performance Optimizasyonları**

### 🚀 **Worker Sayıları**

#### 📄 **PDF Workers**
**Değer Aralığı:** 1-8 worker  
**Varsayılan:** 4 worker

**Ne İçin Kullanılır:**
- PDF sayfalarını paralel işler
- CPU çekirdeklerini etkin kullanır
- İşleme hızını artırır

**Optimizasyon Stratejileri:**
```python
pdf_workers = {
    "macOS": 2-4,      # Multiprocessing sorunları
    "Linux A100": 6-8, # Maximum performance
    "Windows": 4-6,     # Dengeli performans
    "CPU Only": 2-3    # Memory sınırları
}
```

#### 📊 **Excel Workers**
**Değer Aralığı:** 1-8 worker  
**Varsayılan:** 4 worker

**Ne İçin Kullanılır:**
- Excel sheet'lerini paralel işler
- Multi-sheet dosyalar için kritik
- I/O bottleneck'leri azaltır

---

### 🧠 **Batch Processing**

#### 📦 **Batch Size**
**Değer Aralığı:** 16-512  
**Auto/Manual:** Otomatik önerilir

**Ne İçin Kullanılır:**
- Embedding modelinin kaç chunk'ı aynı anda işleyeceği
- GPU memory kullanımını optimize eder
- Throughput'u maksimize eder

#### 🎯 **GPU-Specific Batch Sizes:**
```python
optimal_batch_sizes = {
    "A100 (40GB)": {
        "speed_optimized": 512,
        "balanced": 256, 
        "memory_safe": 128
    },
    "T4 (16GB)": {
        "speed_optimized": 128,
        "balanced": 64,
        "memory_safe": 32
    },
    "CPU": {
        "all_modes": 16
    }
}
```

#### 🔧 **Auto Batch Logic:**
```python
def calculate_optimal_batch(chunk_count, gpu_type, performance_mode):
    if gpu_type == "A100":
        if chunk_count <= 50:
            return chunk_count  # Process all at once
        elif performance_mode == "speed_optimized":
            return 512
        elif performance_mode == "memory_optimized": 
            return 64
        else:
            return 256
```

---

### 💾 **Memory Management**

#### 🗑️ **Aggressive Cleanup**
**Değer:** True/False  
**Varsayılan:** True

**Ne İçin Kullanılır:**
- İşlem sonrası memory temizliği
- GPU cache'ini boşaltır
- Memory leak'leri önler

**Kullanım:**
```python
if aggressive_cleanup:
    import gc
    gc.collect()  # Python memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # GPU cache cleanup
```

#### ♻️ **Embedding Model Reuse**
**Değer:** True/False  
**Varsayılan:** True

**Ne İçin Kullanılır:**
- Model'i bellekte tutar
- Tekrar yükleme süresini elimine eder
- 10-50x hız artışı sağlar

**Performance Impact:**
```python
reuse_impact = {
    "First Load": "15-30 seconds",
    "With Reuse": "0.1-0.5 seconds",
    "Speedup": "30x-150x"
}
```

#### 🎮 **GPU Memory Fraction**
**Değer Aralığı:** 0.5-1.0  
**Varsayılan:** 0.9

**Ne İçin Kullanılır:**
- GPU memory'nin ne kadarının kullanılacağı
- OOM (Out of Memory) hatalarını önler
- Sistem stabilitesi sağlar

**Kullanım Senaryoları:**
```python
gpu_memory_usage = {
    0.95: "A100 Max Performance",
    0.8:  "Dengeli kullanım", 
    0.6:  "Conservative mode",
    0.5:  "Minimum safe mode"
}
```

---

## 🏁 **Performance Modes**

### 🚀 **Speed Optimized**
**Target:** A100 GPU maximum performance

**Ayarlar:**
```python
speed_optimized = {
    'pdf_workers': 6,
    'excel_workers': 6,
    'batch_size': 512,
    'gpu_memory_fraction': 0.95,
    'aggressive_cleanup': False,  # No time for cleanup
    'reuse_embeddings': True
}
```

**Kullanım Durumu:**
- A100 GPU ile maximum hız gerekli
- Büyük dosya setleri
- Production environment

### ⚖️ **Balanced Mode**
**Target:** Hız ve memory dengesi

**Ayarlar:**
```python
balanced = {
    'pdf_workers': 4,
    'excel_workers': 4, 
    'batch_size': 256,
    'gpu_memory_fraction': 0.8,
    'aggressive_cleanup': True,
    'reuse_embeddings': True
}
```

**Kullanım Durumu:**
- T4 GPU veya shared environment
- Orta boyut dosyalar
- Development environment

### 💾 **Memory Optimized**
**Target:** Minimum memory kullanımı

**Ayarlar:**
```python
memory_optimized = {
    'pdf_workers': 2,
    'excel_workers': 2,
    'batch_size': 64,
    'gpu_memory_fraction': 0.6,
    'aggressive_cleanup': True,
    'reuse_embeddings': False  # Save memory
}
```

**Kullanım Durumu:**
- Düşük memory GPU'lar
- CPU-only sistemler
- Memory-constrained environment

---

## 🧠 **Embedding Model Seçimi**

### 🎯 **Model Karşılaştırması**

#### **paraphrase-multilingual-MiniLM-L12-v2** (Varsayılan)
```python
model_specs = {
    "name": "paraphrase-multilingual-MiniLM-L12-v2",
    "size": "418 MB",
    "dimensions": 384,
    "languages": "50+ dil (Türkçe dahil)",
    "speed": "Orta",
    "quality": "Yüksek (Türkçe için optimize)",
    "use_case": "Türkçe finansal dokümanlar"
}
```

#### **all-MiniLM-L6-v2** (Hızlı)
```python
model_specs = {
    "name": "all-MiniLM-L6-v2", 
    "size": "90 MB",
    "dimensions": 384,
    "languages": "İngilizce ağırlıklı",
    "speed": "Çok hızlı",
    "quality": "Orta",
    "use_case": "Hız kritik durumlarda"
}
```

#### **all-mpnet-base-v2** (Kaliteli)
```python
model_specs = {
    "name": "all-mpnet-base-v2",
    "size": "438 MB", 
    "dimensions": 768,
    "languages": "İngilizce",
    "speed": "Yavaş",
    "quality": "Çok yüksek",
    "use_case": "En yüksek kalite gerekli"
}
```

---

## 🔍 **Search Strategy (Arama Stratejisi)**

### 🎯 **Hybrid Mode** (Varsayılan)
**Ne İçin Kullanılır:**
- Semantic + keyword matching kombinasyonu
- En dengeli sonuçlar
- Türkçe finansal terimler için optimize

**Çalışma Mantığı:**
```python
def hybrid_search(query, documents):
    semantic_scores = get_semantic_similarity(query, documents)
    keyword_scores = get_keyword_matches(query, documents) 
    
    # Weighted combination
    final_scores = (0.7 * semantic_scores) + (0.3 * keyword_scores)
    return rank_by_scores(final_scores)
```

### 🧠 **Semantic Only**
**Ne İçin Kullanılır:**
- Tamamen anlamsal benzerlik
- Concept-based arama
- Sıkı embedding dependency

### 🔤 **Keyword Boost**
**Ne İçin Kullanılır:**
- Özel terimler ve kısaltmalar
- Exact match gereken durumlar
- Technical financial terms

---

## 📊 **Cache Sistemi**

### ⚡ **Query Embedding Cache**
**Kapasite:** 50 query  
**TTL:** Session boyunca

**Ne İçin Kullanılır:**
- Aynı soruların tekrar embedding'ini önler
- 10x-100x hız artışı sağlar
- Memory efficient LRU cache

**Cache Logic:**
```python
def get_cached_embedding(query, n_results):
    cache_key = f"{query}_{n_results}"
    
    if cache_key in query_cache:
        return query_cache[cache_key]  # 0.1s
    else:
        embedding = generate_embedding(query)  # 2-5s
        query_cache[cache_key] = embedding
        return embedding
```

### 📈 **Performance Impact:**
```python
cache_performance = {
    "First Query": "2-5 seconds (embedding generation)",
    "Cached Query": "0.1-0.3 seconds", 
    "Speed Improvement": "10x-50x",
    "Memory Cost": "~50MB (50 queries)"
}
```

---

## 🎯 **Kullanım Senaryoları ve Optimum Ayarlar**

### 📊 **Finansal Rapor Analizi**
```python
financial_analysis = {
    'chunk_size': 1200,        # Geniş context
    'overlap_size': 200,       # Yüksek kontinüite  
    'top_k': 8,               # Kapsamlı analiz
    'similarity_threshold': 0.25,  # Geniş kapsam
    'max_context_length': 4000,    # Detaylı analiz
    'performance_mode': 'speed_optimized'
}
```

### ⚡ **Hızlı Soru-Cevap**
```python
quick_qa = {
    'chunk_size': 600,         # Hızlı işleme
    'overlap_size': 100,       # Minimal overlap
    'top_k': 3,               # Odaklı sonuçlar
    'similarity_threshold': 0.4,   # Kesin eşleşme
    'max_context_length': 2000,    # Kompakt context
    'performance_mode': 'speed_optimized'
}
```

### 🔍 **Araştırma ve Keşif**
```python
research_mode = {
    'chunk_size': 800,         # Dengeli boyut
    'overlap_size': 150,       # Orta kontinüite
    'top_k': 12,              # Geniş araştırma
    'similarity_threshold': 0.2,   # Esnek filtreleme
    'max_context_length': 4500,    # Geniş context
    'performance_mode': 'balanced'
}
```

### 💾 **Memory-Constrained Ortam**
```python
memory_safe = {
    'chunk_size': 600,         # Küçük chunk'lar
    'overlap_size': 100,       # Minimal overlap
    'top_k': 3,               # Az sonuç
    'similarity_threshold': 0.4,   # Filtreleme
    'max_context_length': 2000,    # Küçük context
    'performance_mode': 'memory_optimized'
}
```

---

## 📈 **Performance Monitoring Verileri**

### 🔍 **Tracked Metrics**

#### **Response Time Breakdown:**
```python
timing_metrics = {
    "embedding_generation": "Query → Vector (0.1-3s)",
    "vector_search": "Similarity search (<100ms)", 
    "context_building": "Chunk assembly (50-200ms)",
    "llm_generation": "Mistral inference (1-4s)",
    "total_response": "End-to-end (2-8s)"
}
```

#### **Quality Metrics:**
```python
quality_metrics = {
    "similarity_scores": "Retrieval relevance (0.0-1.0)",
    "context_length": "Information density",
    "chunk_diversity": "Source coverage",
    "response_completeness": "Answer quality"
}
```

#### **Resource Utilization:**
```python
resource_metrics = {
    "gpu_memory_usage": "Peak GPU RAM (GB)",
    "cpu_utilization": "Processing load (%)",
    "embedding_cache_hit_rate": "Cache efficiency (%)",
    "chunks_per_second": "Processing throughput"
}
```

---

## 🚨 **Troubleshooting Kılavuzu**

### ❌ **Yaygın Sorunlar ve Çözümleri**

#### **Yavaş Response Time**
```python
speed_optimization_checklist = [
    "✓ Embedding model reuse aktif mi?",
    "✓ Query cache çalışıyor mu?", 
    "✓ Batch size GPU'ya uygun mu?",
    "✓ Top-K çok yüksek değil mi?",
    "✓ Context length optimum mu?",
    "✓ Performance mode doğru mu?"
]
```

#### **Memory Issues**
```python
memory_optimization_checklist = [
    "✓ GPU memory fraction düşürün (0.8→0.6)",
    "✓ Batch size küçültün (256→128→64)",
    "✓ Aggressive cleanup aktif",
    "✓ Chunk size küçültün (800→600)", 
    "✓ Top-K azaltın (5→3)",
    "✓ Memory optimized mode kullanın"
]
```

#### **Düşük Kalite Sonuçlar**
```python
quality_improvement_checklist = [
    "✓ Similarity threshold düşürün (0.3→0.2)",
    "✓ Top-K artırın (5→8)",
    "✓ Chunk size artırın (800→1000)",
    "✓ Overlap size artırın (150→200)",
    "✓ Context length artırın (3000→4000)",
    "✓ Embedding model değiştirin"
]
```

---

## 💡 **Best Practices**

### 🎯 **Optimal Configuration Workflow**

1. **Baseline ile başlayın** (varsayılan ayarlar)
2. **Use case'e göre mode seçin** (speed/balanced/memory)
3. **Küçük test dosyalarıyla iterate edin**
4. **Performance metrics'leri izleyin**
5. **Kalite vs hız dengesini bulun**
6. **Production'da A/B test yapın**

### 📊 **Performance Tuning Strategy**
```python
tuning_strategy = {
    "Step 1": "Hardware capability assessment",
    "Step 2": "Use case requirements analysis", 
    "Step 3": "Baseline performance measurement",
    "Step 4": "Systematic parameter optimization",
    "Step 5": "Performance validation",
    "Step 6": "Production deployment"
}
```

---

**🎉 Bu kılavuz ile Turkish Financial RAG Assistant'ın tüm gelişmiş ayarlarını optimize edebilir, maximum performance ve kalite elde edebilirsiniz!** 