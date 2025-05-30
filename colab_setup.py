"""
Google Colab Pro Plus A100 Setup Script
Turkish Financial RAG Chatbot için özel kurulum - NGROK desteği ile
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import time

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NGROK Token
NGROK_TOKEN = "2xmENf6pFX37FGhDuGBuWpBSRHG_2TfzVLgN9LiFCL2Zdi1Wf"

def check_gpu_availability():
    """GPU durumunu kontrol et"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 GPU Bulundu: {gpu_name}")
            logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            
            # A100 kontrolü
            if "A100" in gpu_name:
                logger.info("✅ A100 GPU tespit edildi - Optimal performans bekleniyor!")
                return True, "A100"
            else:
                logger.warning(f"⚠️ A100 değil: {gpu_name} - Performans etkilenebilir")
                return True, gpu_name
        else:
            logger.error("❌ GPU bulunamadı!")
            return False, None
    except Exception as e:
        logger.error(f"❌ GPU kontrol hatası: {e}")
        return False, None

def mount_google_drive():
    """Google Drive'ı bağla"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        logger.info("✅ Google Drive başarıyla bağlandı")
        
        # Model path kontrolü
        model_path = "/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / 1024**3
            logger.info(f"✅ Model dosyası bulundu: {model_size:.2f} GB")
            return True, model_path
        else:
            logger.warning(f"⚠️ Model dosyası bulunamadı: {model_path}")
            logger.info("📋 Model dosyasını şu path'e yüklediğinizden emin olun:")
            logger.info(model_path)
            return False, model_path
            
    except Exception as e:
        logger.error(f"❌ Drive bağlama hatası: {e}")
        return False, None

def install_system_dependencies():
    """Sistem bağımlılıklarını yükle"""
    try:
        logger.info("📦 Sistem bağımlılıkları yükleniyor...")
        
        # C++ compiler ve build tools
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "build-essential"], check=True)
        
        logger.info("✅ Sistem bağımlılıkları yüklendi")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sistem bağımlılık hatası: {e}")
        return False

def install_python_dependencies():
    """Python paketlerini yükle"""
    try:
        logger.info("🐍 Python bağımlılıkları yükleniyor...")
        
        # Requirements dosyasından yükle
        if os.path.exists("requirements.txt"):
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            logger.info("✅ requirements.txt'den paketler yüklendi")
        else:
            # Manuel yükleme
            packages = [
                "streamlit>=1.28.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "PyPDF2>=3.0.0",
                "pdfplumber>=0.9.0",
                "pymupdf>=1.23.0",
                "openpyxl>=3.1.0",
                "xlrd>=2.0.0",
                "sentence-transformers>=2.2.0",
                "transformers>=4.30.0",
                "torch>=2.0.0",
                "chromadb>=0.4.15",
                "llama-cpp-python>=0.2.0",
                "langdetect>=1.0.9",
                "nltk>=3.8.0",
                "python-dotenv>=1.0.0",
                "tqdm>=4.65.0",
                "requests>=2.31.0"
            ]
            
            for package in packages:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        
        # NGROK kurulumu
        logger.info("🌐 NGROK yükleniyor...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyngrok"], check=True)
                
        logger.info("✅ Python bağımlılıkları yüklendi")
        return True
        
    except Exception as e:
        logger.error(f"❌ Python bağımlılık hatası: {e}")
        return False

def setup_ngrok():
    """NGROK'u kurulum ve token ayarla"""
    try:
        logger.info("🌐 NGROK kurulumu yapılıyor...")
        
        from pyngrok import ngrok
        
        # NGROK token'ı ayarla
        ngrok.set_auth_token(NGROK_TOKEN)
        logger.info("✅ NGROK token ayarlandı")
        
        # Mevcut tunnel'ları kapat
        ngrok.kill()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ NGROK kurulum hatası: {e}")
        return False

def setup_environment():
    """Çevre değişkenlerini ayarla"""
    try:
        # CUDA optimizasyonları
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  # A100 için
        
        # Memory optimizasyonları
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # ChromaDB için
        os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
        
        # Streamlit için
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["STREAMLIT_SERVER_PORT"] = "8501"
        os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
        
        logger.info("✅ Çevre değişkenleri ayarlandı")
        return True
        
    except Exception as e:
        logger.error(f"❌ Çevre değişkeni hatası: {e}")
        return False

def download_nltk_data():
    """NLTK verilerini indir"""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("✅ NLTK verileri indirildi")
        return True
    except Exception as e:
        logger.error(f"❌ NLTK veri hatası: {e}")
        return False

def create_directory_structure():
    """Dizin yapısını oluştur"""
    try:
        directories = [
            "src",
            "chroma_db",
            "temp_uploads",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
        logger.info("✅ Dizin yapısı oluşturuldu")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dizin oluşturma hatası: {e}")
        return False

def verify_installation():
    """Kurulumu doğrula"""
    try:
        logger.info("🔍 Kurulum doğrulanıyor...")
        
        # Critical imports
        import streamlit
        import torch
        import chromadb
        import sentence_transformers
        import transformers
        from pyngrok import ngrok
        
        logger.info(f"✅ Streamlit: {streamlit.__version__}")
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ ChromaDB: {chromadb.__version__}")
        logger.info(f"✅ Sentence Transformers: {sentence_transformers.__version__}")
        logger.info(f"✅ Transformers: {transformers.__version__}")
        logger.info(f"✅ PyNgrok: {ngrok.__version__}")
        
        # CUDA kontrolü
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA: {torch.version.cuda}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Doğrulama hatası: {e}")
        return False

def start_streamlit_with_ngrok():
    """Streamlit'i başlat ve NGROK tunnel oluştur"""
    try:
        logger.info("🚀 Streamlit + NGROK başlatılıyor...")
        
        from pyngrok import ngrok
        import threading
        import subprocess
        import time
        
        # Streamlit'i arka planda başlat
        def run_streamlit():
            cmd = [
                sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--server.enableXsrfProtection", "false",
                "--server.enableCORS", "false"
            ]
            subprocess.run(cmd)
        
        # Streamlit thread'ini başlat
        streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
        streamlit_thread.start()
        
        # Streamlit'in başlamasını bekle
        logger.info("⏳ Streamlit başlangıcı bekleniyor...")
        time.sleep(10)
        
        # NGROK tunnel'ı oluştur
        public_url = ngrok.connect(8501)
        
        logger.info("🎉 Setup tamamlandı!")
        logger.info("=" * 60)
        logger.info(f"🌐 Public URL: {public_url}")
        logger.info("=" * 60)
        logger.info("📱 Bu URL'yi tarayıcınızda açabilirsiniz")
        logger.info("🔗 URL'yi paylaşarak başkalarının da erişimini sağlayabilirsiniz")
        
        return True, public_url
        
    except Exception as e:
        logger.error(f"❌ Streamlit + NGROK başlatma hatası: {e}")
        return False, None

def main():
    """Ana setup fonksiyonu"""
    logger.info("🚀 Turkish Financial RAG Chatbot - Colab + NGROK Setup Başlıyor...")
    logger.info("=" * 60)
    
    setup_steps = [
        ("GPU Kontrolü", check_gpu_availability),
        ("Google Drive Bağlama", mount_google_drive),
        ("Sistem Bağımlılıkları", install_system_dependencies),
        ("Python Bağımlılıkları", install_python_dependencies),
        ("NGROK Kurulumu", setup_ngrok),
        ("Çevre Değişkenleri", setup_environment),
        ("NLTK Verileri", download_nltk_data),
        ("Dizin Yapısı", create_directory_structure),
        ("Kurulum Doğrulama", verify_installation)
    ]
    
    success_count = 0
    total_steps = len(setup_steps)
    
    for step_name, step_function in setup_steps:
        logger.info(f"📋 {step_name}...")
        try:
            if step_function():
                success_count += 1
                logger.info(f"✅ {step_name} tamamlandı")
            else:
                logger.error(f"❌ {step_name} başarısız")
        except Exception as e:
            logger.error(f"❌ {step_name} hatası: {e}")
        
        logger.info("-" * 40)
    
    # Sonuç
    logger.info("=" * 60)
    logger.info(f"📊 Setup Sonucu: {success_count}/{total_steps} adım başarılı")
    
    if success_count == total_steps:
        logger.info("🎉 Setup başarıyla tamamlandı!")
        logger.info("🚀 Şimdi Streamlit + NGROK başlatılacak...")
        logger.info("-" * 40)
        
        # Streamlit + NGROK başlat
        success, public_url = start_streamlit_with_ngrok()
        return success, public_url
    else:
        logger.warning("⚠️  Setup kısmen tamamlandı. Bazı adımlar başarısız.")
        return False, None

def get_manual_startup_commands():
    """Manuel başlatma komutları"""
    commands = [
        "# Manuel başlatma komutları:",
        "from pyngrok import ngrok",
        "import subprocess",
        "import threading",
        "",
        "# 1. NGROK token ayarla",
        f'ngrok.set_auth_token("{NGROK_TOKEN}")',
        "",
        "# 2. Streamlit'i arka planda başlat",
        "def run_streamlit():",
        "    !streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true",
        "",
        "import threading",
        "streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)",
        "streamlit_thread.start()",
        "",
        "# 3. NGROK tunnel oluştur",
        "import time",
        "time.sleep(10)  # Streamlit'in başlamasını bekle",
        "public_url = ngrok.connect(8501)",
        "print(f'🌐 Public URL: {public_url}')"
    ]
    return "\n".join(commands)

if __name__ == "__main__":
    success, public_url = main()
    
    if success and public_url:
        print("\n" + "="*60)
        print("🎉 BAŞARILI! Uygulamanız çalışıyor!")
        print(f"🌐 Public URL: {public_url}")
        print("="*60)
        print("📱 Bu URL'yi tarayıcınızda açın")
        print("🔗 URL'yi paylaşarak başkalarının erişimini sağlayın")
        print("⚠️  Notebook'u kapatırsanız URL devre dışı kalır")
    else:
        print("\n❌ Setup başarısız. Manuel başlatma komutları:")
        print(get_manual_startup_commands()) 