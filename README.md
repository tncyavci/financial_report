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