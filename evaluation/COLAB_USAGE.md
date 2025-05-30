# 🚀 Google Colab Usage Guide - Performance Tracker Test

Bu kılavuz Performance Tracker test uygulamasını Google Colab'da nasıl çalıştıracağınızı açıklar.

## 📋 **Prerequisites**

1. Google Colab hesabı
2. NGROK hesabı (ücretsiz) - https://ngrok.com/
3. NGROK auth token

## 🚀 **Quick Start (5 Dakikada Çalıştır)**

### **Adım 1: Colab Notebook Oluştur**
1. https://colab.research.google.com/ adresine git
2. "New notebook" oluştur
3. Runtime → Change runtime type → T4 GPU seç (opsiyonel)

### **Adım 2: Projeyi Klonla**
```python
# Clone the repository
!git clone https://github.com/your-username/financial_report.git
%cd financial_report
```

### **Adım 3: NGROK Setup**
```python
# Install and setup NGROK
!pip install pyngrok

from pyngrok import ngrok
# NGROK auth token'inizi buraya girin
ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")
```

### **Adım 4: Performance Tracker Test'i Çalıştır**
```python
# Run the performance tracker test
!python evaluation/colab_test_setup.py
```

## 🎯 **Adım Adım Detaylı Kurulum**

### **1. Yeni Colab Notebook**

Colab'da yeni bir notebook açın ve aşağıdaki hücreleri sırayla çalıştırın:

```python
# Cell 1: Repository clone
!git clone https://github.com/your-username/financial_report.git
%cd financial_report
!ls -la
```

```python
# Cell 2: Check system info
import sys
import torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

```python
# Cell 3: Install requirements
!pip install streamlit pandas psutil pyngrok -q
print("✅ Requirements installed!")
```

```python
# Cell 4: Setup NGROK (Replace with your token)
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")
print("✅ NGROK configured!")
```

```python
# Cell 5: Run performance test
!python evaluation/colab_test_setup.py
```

## 🔧 **Manual Setup (Advanced)**

Eğer otomatik setup çalışmazsa:

```python
# Manual setup
import subprocess
import sys
import os

# Create directories
os.makedirs("evaluation/reports", exist_ok=True)
os.makedirs("evaluation/metrics", exist_ok=True)

# Start Streamlit manually
import threading
from pyngrok import ngrok

def run_streamlit():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "evaluation/streamlit_test_app.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ])

# Start Streamlit in background
thread = threading.Thread(target=run_streamlit)
thread.daemon = True
thread.start()

# Wait and create tunnel
import time
time.sleep(10)
public_url = ngrok.connect(8501)
print(f"Access your app at: {public_url}")
```

## 🎮 **Test Uygulaması Kullanımı**

### **📱 Interface Overview**
- **Sidebar**: Test kontrolleri
- **Main Area**: Performance metrics ve analytics
- **Footer**: Yönetim butonları

### **🧪 Test Tipleri**

#### **1. PDF Processing Test**
- Slider ile sayfa sayısı ayarla (5-100)
- Processing time simule et (0.5-5s)
- "🚀 Run PDF Test" butonuna tıkla

#### **2. Query Processing Test** 
- Query length ayarla (10-500 karakter)
- Response time simule et (0.1-3s)
- "🧠 Run Query Test" butonuna tıkla

#### **3. Embedding Generation Test**
- Chunk count ayarla (10-500)
- Generation time simule et (0.1-2s)
- "🧬 Run Embedding Test" butonuna tıkla

#### **4. Auto Test**
- "🎯 Run All Tests" ile hepsini otomatik çalıştır
- Progress bar ile ilerleme takibi

### **📊 Monitoring Features**

#### **System Stats**
- CPU usage monitoring
- Memory usage tracking
- GPU availability detection
- Real-time resource monitoring

#### **Performance Analytics**
- Total operations count
- Average duration calculation
- Operations breakdown by type
- Recent activity log

#### **Data Export**
- JSON format performance data
- Download button ile export
- Timestamped file names

## 🔍 **Troubleshooting**

### **Common Issues:**

#### **❌ NGROK Auth Error**
```
Solution: Get free token from https://ngrok.com/
ngrok.set_auth_token("your_token_here")
```

#### **❌ Module Import Error**
```python
# Add path manually
import sys
sys.path.append('/content/financial_report')
```

#### **❌ Streamlit Port Conflict**
```python
# Kill existing processes
!pkill -f streamlit
# Try different port
ngrok.connect(8502)
```

#### **❌ GPU Not Detected**
```python
# Check GPU assignment
!nvidia-smi
# If no GPU, tests will run on CPU (slower but works)
```

### **Performance Tips:**

1. **Use T4 GPU** for better performance simulation
2. **Clear metrics** between test sessions
3. **Download data** before session expires
4. **Monitor memory** usage during tests

## 📈 **Expected Results**

### **Typical Performance (T4 GPU):**
- PDF Processing: 1-3 seconds
- Query Response: 0.5-2 seconds  
- Embedding Generation: 0.2-1.5 seconds
- System Memory: 20-40% usage

### **Metrics Generated:**
- Duration statistics (min/max/avg)
- Memory usage deltas
- GPU utilization (if available)
- Operations breakdown
- Timestamp tracking

## 🎯 **Academic Usage**

Bu test uygulaması akademik projeniz için:

1. **Performance Baseline**: Mevcut sistem performansı
2. **Optimization Tracking**: İyileştirme öncesi/sonrası
3. **Resource Analysis**: Hardware kullanım analizi
4. **Reproducible Results**: Tutarlı test environment
5. **Data Export**: Akademik rapor için JSON data

## 📝 **Next Steps**

Test sonrasında:

1. **Performance Data'yı analiz edin**
2. **Bottleneck'leri tespit edin** 
3. **Optimization strategy'leri geliştirin**
4. **Real sisteme entegre edin**
5. **Academic paper'a dahil edin**

**✅ Bu test uygulaması ile performance monitoring sisteminizin tüm özelliklerini Colab'da test edebilirsiniz!** 