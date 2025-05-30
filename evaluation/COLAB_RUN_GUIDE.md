# 🚀 Google Colab Evaluation Guide

Turkish Financial RAG System - Evaluation modülünü Google Colab'da çalıştırma kılavuzu

## 📋 **Hazırlık (5 dakika)**

### **1. NGROK Hesabı Oluşturun**
1. https://ngrok.com/ adresine gidin
2. Ücretsiz hesap oluşturun
3. Dashboard'dan **auth token**'inizi kopyalayın

## 🚀 **Colab'da Çalıştırma**

### **Yöntem 1: Hızlı Test (Sadece Performance)**

```python
# Cell 1: Repository klonla
!git clone https://github.com/your-username/financial_report.git
%cd financial_report
```

```python
# Cell 2: Hızlı performance test
!PYTHONPATH=/content/financial_report python evaluation/test_performance.py
```

### **Yöntem 2: Full Evaluation (Performance + Accuracy)**

#### **Cell 1: Setup**
```python
# Repository klonla ve setup
!git clone https://github.com/your-username/financial_report.git
%cd financial_report

# System bilgilerini kontrol et
import sys
print(f"Python: {sys.version}")
print(f"Working Directory: {os.getcwd()}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except:
    print("PyTorch not available")
```

#### **Cell 2: Dependencies**
```python
# Requirements install
!pip install streamlit pandas psutil pyngrok -q

# Python path setup
import sys
import os
sys.path.append('/content/financial_report')

print("✅ Dependencies installed!")
```

#### **Cell 3: Test Queries Generate**
```python
# Test queries oluştur
!PYTHONPATH=/content/financial_report python evaluation/test_query_generator.py

# Dosyaları kontrol et
!ls -la evaluation/data/
!ls -la evaluation/reports/
```

#### **Cell 4: NGROK Setup**
```python
# NGROK token'inizi buraya girin
from pyngrok import ngrok

# Token'inizi buraya yazın
NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"  # https://ngrok.com/

ngrok.set_auth_token(NGROK_TOKEN)
print("✅ NGROK configured!")
```

#### **Cell 5: Streamlit App Çalıştır**
```python
# Enhanced Streamlit app başlat
import subprocess
import threading
import time
from pyngrok import ngrok

def run_streamlit():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "evaluation/streamlit_accuracy_app.py",  # Enhanced app
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ])

# Streamlit'i background'da başlat
thread = threading.Thread(target=run_streamlit)
thread.daemon = True
thread.start()

# Başlamasını bekle
print("⏳ Starting Streamlit...")
time.sleep(15)

# NGROK tunnel oluştur
print("🌐 Creating public tunnel...")
public_url = ngrok.connect(8501)

print(f"\n🎉 SUCCESS! Your app is ready:")
print(f"🔗 {public_url}")
print(f"\n📊 Available Features:")
print(f"   ⚡ Performance Testing")
print(f"   🎯 Accuracy Evaluation (31 test queries)")
print(f"   📈 Analytics Dashboard")
print(f"   🧪 Test Query Management")
```

### **Yöntem 3: Otomatik Setup**

#### **Single Cell Run:**
```python
# Tek hücrede tüm setup
!git clone https://github.com/your-username/financial_report.git
%cd financial_report

# Hızlı setup çalıştır
!python evaluation/colab_launcher.py

# Token girin
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN_HERE")

# Enhanced app çalıştır
!PYTHONPATH=/content/financial_report python evaluation/colab_test_setup.py
```

## 🎯 **App Kullanımı**

### **📊 4 Ana Sayfa:**

#### **1. 🚀 Performance Testing**
- PDF processing simulation
- Query response timing
- System resource monitoring
- Real-time performance metrics

#### **2. 📊 Accuracy Evaluation**
- 31 test query'den birini seç
- "Generate RAG Response" tıkla
- Manuel evaluation (1-5 scale)
- Success/failure tracking

#### **3. 📈 Analytics Dashboard**
- Combined performance + accuracy metrics
- Category-based analysis
- Downloadable reports

#### **4. 🧪 Test Queries**
- 31 test query görüntüle
- Yeni query'ler ekle
- Quick evaluation

### **🔄 Manuel Evaluation Workflow:**

1. **Test Queries sayfasına git**
2. **"Generate 30 Test Queries" tıkla**
3. **Accuracy Evaluation sayfasına geç**
4. **Query seç ve "Generate RAG Response" tıkla**
5. **1-5 scale ile score ver:**
   - 5 = EXCELLENT (Mükemmel)
   - 4 = GOOD (İyi)
   - 3 = FAIR (Orta) - Success threshold
   - 2 = POOR (Zayıf)
   - 1 = FAILED (Başarısız)
6. **"Submit Evaluation" tıkla**
7. **Analytics'de sonuçları gör**

## 📊 **Expected Results**

### **Performance Metrics:**
- PDF Processing: 1-3 seconds
- Query Response: 0.5-2 seconds
- Embedding Generation: 0.2-1.5 seconds

### **Accuracy Metrics:**
- Success Rate: 70-85%
- Average Quality: 3.5-4.2/5
- Category breakdown by financial domain

## 🛠️ **Troubleshooting**

### **❌ NGROK Error:**
```python
# Token'i yeniden set edin
from pyngrok import ngrok
ngrok.kill()
ngrok.set_auth_token("YOUR_TOKEN")
```

### **❌ Import Error:**
```python
# Path'i manuel ekleyin
import sys
sys.path.append('/content/financial_report')
```

### **❌ Port Conflict:**
```python
# Mevcut process'leri kill edin
!pkill -f streamlit
!pkill -f ngrok
```

### **❌ Streamlit Not Starting:**
```python
# Manuel başlatma
!PYTHONPATH=/content/financial_report streamlit run evaluation/streamlit_accuracy_app.py --server.port 8502
```

## 📈 **Academic Usage**

### **Performance Analysis:**
```python
# Performance data al
from evaluation.metrics.performance_tracker import get_global_tracker
tracker = get_global_tracker()
summary = tracker.get_performance_summary()
print(f"Average response time: {summary['duration_stats']['avg']:.2f}s")
```

### **Accuracy Analysis:**
```python
# Accuracy data al
from evaluation.metrics.accuracy_tracker import get_global_accuracy_tracker
acc_tracker = get_global_accuracy_tracker()
metrics = acc_tracker.get_accuracy_metrics()
print(f"Success rate: {metrics.success_rate:.1f}%")
print(f"Average quality: {metrics.average_quality_score:.2f}/5")
```

### **Combined Report:**
```python
# Akademik rapor oluştur
report = acc_tracker.generate_test_report()
print(report)

# Download için
with open('evaluation_report.md', 'w') as f:
    f.write(report)
```

## 💾 **Data Export**

### **JSON Download:**
- Performance metrics: `performance_metrics.json`
- Accuracy metrics: `accuracy_metrics.json`
- Test queries: `test_queries.json`

### **Academic Papers:**
```python
# Publication-ready data
data = {
    "performance": tracker.get_performance_summary(),
    "accuracy": acc_tracker.get_accuracy_metrics(),
    "methodology": "Manual evaluation of 31 financial queries",
    "scoring_system": "5-point Likert scale",
    "success_threshold": "Score >= 3"
}

import json
with open('academic_data.json', 'w') as f:
    json.dump(data, f, indent=2, default=str)
```

## 🎯 **Next Steps**

1. **Test 10-15 query** ile başlayın
2. **Success rate patterns** analiz edin
3. **Category performance** inceleyin
4. **Academic paper** için data toplayın

**✅ Bu guide ile Colab'da full evaluation sisteminizi çalıştırabilir ve 31 test query ile comprehensive accuracy analysis yapabilirsiniz!** 