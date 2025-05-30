"""
Local LLM Servisi
GGUF ve HuggingFace model desteği
A100 GPU optimizasyonları dahil
"""

import logging
import os
import torch
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# GGUF desteği için llama-cpp-python import
try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
    logger.info("✅ llama-cpp-python available for GGUF models")
except ImportError:
    GGUF_AVAILABLE = False
    logger.warning("⚠️ llama-cpp-python not available for GGUF models")

# HuggingFace transformers desteği
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HF_AVAILABLE = True
    logger.info("✅ HuggingFace transformers available")
except ImportError:
    HF_AVAILABLE = False
    logger.warning("⚠️ HuggingFace transformers not available")

@dataclass
class ChatMessage:
    """Chat mesajı container'ı"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime

# Önerilen modeller
RECOMMENDED_MODELS = {
    "mistral_7b_gguf": {
        "name": "Mistral 7B GGUF",
        "type": "gguf",
        "description": "Mistral 7B Instruct - GGUF format (Önerilen)",
        "memory_requirement": "8GB",
        "performance": "Yüksek"
    },
    "llama_3_1_8b": {
        "name": "Llama 3.1 8B",
        "type": "huggingface",
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "description": "Meta Llama 3.1 8B Instruct",
        "memory_requirement": "16GB",
        "performance": "Çok Yüksek"
    },
    "custom_drive_model": {
        "name": "Custom Drive Model",
        "type": "custom",
        "description": "Google Drive'daki özel model",
        "memory_requirement": "Değişken",
        "performance": "Değişken"
    }
}

class GGUFModelService:
    """
    GGUF modelleri için servis sınıfı
    A100 GPU optimizasyonları dahil
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: GGUF model dosyasının yolu
        """
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """GGUF modelini yükle ve A100 için optimize et"""
        try:
            logger.info(f"🚀 GGUF model yükleniyor: {self.model_name}")
            
            # GPU ve sistem kontrolü
            gpu_info = self._detect_gpu()
            logger.info(f"🔧 GPU Info: {gpu_info}")
            
            # A100 optimizasyonları
            model_params = self._get_optimized_params(gpu_info)
            logger.info(f"⚙️ Model parametreleri: {model_params}")
            
            # Modeli yükle
            self.llm = Llama(
                model_path=self.model_path,
                **model_params
            )
            
            logger.info("✅ GGUF model başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"❌ GGUF model yüklenemedi: {e}")
            self.llm = None
    
    def _detect_gpu(self) -> Dict:
        """GPU bilgilerini topla"""
        gpu_info = {
            "has_cuda": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": "",
            "gpu_memory": 0,
            "is_a100": False
        }
        
        if gpu_info["has_cuda"] and gpu_info["gpu_count"] > 0:
            try:
                gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
                gpu_properties = torch.cuda.get_device_properties(0)
                gpu_info["gpu_memory"] = gpu_properties.total_memory / 1e9  # GB
                gpu_info["is_a100"] = "A100" in gpu_info["gpu_name"]
                
                logger.info(f"🎯 GPU: {gpu_info['gpu_name']}")
                logger.info(f"💾 GPU Memory: {gpu_info['gpu_memory']:.1f}GB")
                
            except Exception as e:
                logger.warning(f"⚠️ GPU bilgisi alınamadı: {e}")
        
        return gpu_info
    
    def _get_optimized_params(self, gpu_info: Dict) -> Dict:
        """GPU'ya göre optimize edilmiş parametreleri döndür"""
        params = {
            "verbose": True,
            "use_mmap": True,
            "use_mlock": False,
            "numa": False,
        }
        
        if gpu_info["has_cuda"]:
            if gpu_info["is_a100"]:
                # A100 için maksimum performans
                params.update({
                    "n_ctx": 8192,          # Büyük context window
                    "n_batch": 4096,        # Büyük batch size
                    "n_threads": 8,         # Daha fazla CPU thread
                    "n_gpu_layers": -1,     # Tüm layer'ları GPU'ya
                    "low_vram": False,      # A100'de VRAM bol
                    "f16_kv": True,         # Half precision
                    "logits_all": False,    # Memory optimization
                    "n_threads_batch": 8,   # Batch threading
                })
                logger.info("🎯 A100 optimization enabled")
                
            elif gpu_info["gpu_memory"] > 20:
                # Yüksek VRAM GPU'lar
                params.update({
                    "n_ctx": 6144,
                    "n_batch": 3072,
                    "n_threads": 6,
                    "n_gpu_layers": -1,
                    "low_vram": False,
                    "f16_kv": True,
                    "logits_all": False,
                    "n_threads_batch": 6,
                })
                logger.info("🚀 High-end GPU optimization enabled")
                
            elif gpu_info["gpu_memory"] > 8:
                # Orta seviye GPU'lar
                params.update({
                    "n_ctx": 4096,
                    "n_batch": 2048,
                    "n_threads": 4,
                    "n_gpu_layers": 32,     # Kısmi GPU offload
                    "low_vram": True,
                    "f16_kv": True,
                    "logits_all": False,
                    "n_threads_batch": 4,
                })
                logger.info("📊 Mid-range GPU optimization enabled")
                
            else:
                # Düşük VRAM GPU'lar
                params.update({
                    "n_ctx": 2048,
                    "n_batch": 512,
                    "n_threads": 4,
                    "n_gpu_layers": 10,     # Az layer GPU'da
                    "low_vram": True,
                    "f16_kv": True,
                    "logits_all": False,
                    "n_threads_batch": 2,
                })
                logger.info("⚡ Low VRAM GPU optimization enabled")
        else:
            # CPU-only mod
            params.update({
                "n_ctx": 2048,
                "n_batch": 512,
                "n_threads": min(8, os.cpu_count()),
                "n_gpu_layers": 0,
                "n_threads_batch": min(4, os.cpu_count()),
            })
            logger.info("💻 CPU-only mode enabled")
        
        return params
    
    def generate_response(self, query: str, context: str, chat_history: Optional[List] = None) -> Tuple[str, float]:
        """
        Response oluştur
        
        Returns:
            Tuple[str, float]: (response, duration_seconds)
        """
        start_time = datetime.now()
        
        if not self.llm:
            fallback_response = self._generate_fallback_response(query, context)
            duration = (datetime.now() - start_time).total_seconds()
            return fallback_response, duration
        
        try:
            # Prompt oluştur
            prompt = self._create_prompt(query, context, chat_history)
            
            # A100 için optimize edilmiş generation parametreleri
            generation_params = self._get_generation_params()
            
            logger.debug(f"🔤 Prompt length: {len(prompt)} chars")
            
            # Response oluştur
            response = self.llm(
                prompt,
                **generation_params
            )
            
            # Response'u temizle
            response_text = self._extract_response_text(response)
            response_text = self._clean_response(response_text)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ GGUF response generated in {duration:.2f}s")
            
            return response_text, duration
            
        except Exception as e:
            logger.error(f"❌ GGUF generation failed: {e}")
            fallback_response = self._generate_fallback_response(query, context)
            duration = (datetime.now() - start_time).total_seconds()
            return fallback_response, duration
    
    def _create_prompt(self, query: str, context: str, chat_history: Optional[List] = None) -> str:
        """Türkçe finansal analiz için prompt oluştur"""
        
        system_prompt = """Sen finansal dokümanları analiz eden uzman bir asistansın. Verilen bağlam bilgilerini kullanarak Türkçe cevap ver.

Kurallar:
- Sadece bağlam bilgilerini kullan
- Eğer bağlamda cevap yoksa "Bu bilgi dokümanlarda bulunmuyor" de
- Finansal terimleri doğru kullan
- Sayısal verileri dikkatli kontrol et
- Kısa ve net cevaplar ver
- Kaynakları belirt"""

        # Mistral chat format
        if "mistral" in self.model_name.lower():
            prompt = f"<s>[INST] {system_prompt}\n\nBağlam Bilgileri:\n{context}\n\nSoru: {query} [/INST]"
        else:
            # Genel format
            prompt = f"System: {system_prompt}\n\nContext: {context}\n\nUser: {query}\nAssistant:"
        
        return prompt
    
    def _get_generation_params(self) -> Dict:
        """Generation parametrelerini döndür"""
        return {
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stop": ["</s>", "[INST]", "[/INST]", "System:", "User:"],
            "stream": False,
            "echo": False,
        }
    
    def _extract_response_text(self, response) -> str:
        """Response'dan metni çıkar"""
        if isinstance(response, dict):
            if 'choices' in response:
                return response['choices'][0]['text'].strip()
            elif 'content' in response:
                return response['content'].strip()
        
        return str(response).strip()
    
    def _clean_response(self, response_text: str) -> str:
        """Response'u temizle"""
        # Unwanted token'ları kaldır
        unwanted_tokens = ["</s>", "[INST]", "[/INST]", "<s>", "System:", "User:", "Assistant:"]
        for token in unwanted_tokens:
            response_text = response_text.replace(token, "")
        
        # Fazla boşlukları temizle
        response_text = " ".join(response_text.split())
        
        # Tekrar eden satırları kaldır
        lines = response_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in cleaned_lines[-3:]:  # Son 3 satırda tekrar varsa atla
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """GGUF model çalışmadığında fallback response"""
        if context:
            return f"""
**[GGUF Model Hatası]**

Sorunuz: {query}

Bulunan İlgili Bilgiler:
{context[:500]}...

💡 GGUF model düzgün yüklendiğinde bu bilgileri analiz ederek detaylı cevap verebilirim.
"""
        else:
            return "Bu konuda dokümanlarda ilgili bilgi bulunamadı."
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.llm is not None,
            "device": "gpu" if self.llm else "unknown",
            "model_path": self.model_path,
            "service_type": "gguf"
        }

class HuggingFaceModelService:
    """
    HuggingFace modelleri için servis sınıfı
    """
    
    def __init__(self, model_id: str):
        """
        Args:
            model_id: HuggingFace model ID
        """
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """HuggingFace modelini yükle"""
        try:
            logger.info(f"🚀 HuggingFace model yükleniyor: {self.model_id}")
            
            # Device seçimi
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Model ve tokenizer yükle
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            # Pipeline oluştur
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if device == "cuda" else None
            )
            
            logger.info(f"✅ HuggingFace model yüklendi - {device}")
            
        except Exception as e:
            logger.error(f"❌ HuggingFace model yüklenemedi: {e}")
    
    def generate_response(self, query: str, context: str, chat_history: Optional[List] = None) -> Tuple[str, float]:
        """Response oluştur"""
        start_time = datetime.now()
        
        if not self.pipeline:
            fallback_response = "HuggingFace model yüklenmedi."
            duration = (datetime.now() - start_time).total_seconds()
            return fallback_response, duration
        
        try:
            # Prompt oluştur
            prompt = self._create_prompt(query, context)
            
            # Generate
            outputs = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Response çıkar
            response_text = outputs[0]['generated_text'][len(prompt):].strip()
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ HuggingFace response generated in {duration:.2f}s")
            
            return response_text, duration
            
        except Exception as e:
            logger.error(f"❌ HuggingFace generation failed: {e}")
            fallback_response = f"Generation hatası: {str(e)}"
            duration = (datetime.now() - start_time).total_seconds()
            return fallback_response, duration
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Prompt oluştur"""
        return f"""Sen finansal dokümanları analiz eden bir asistansın.

Bağlam: {context}

Soru: {query}

Cevap:"""
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        return {
            "model_name": self.model_id,
            "model_loaded": self.model is not None,
            "device": "gpu" if torch.cuda.is_available() else "cpu",
            "service_type": "huggingface"
        }

def create_optimized_llm_service(model_choice: str, model_path: Optional[str] = None):
    """
    Optimize edilmiş LLM servisi oluştur
    
    Args:
        model_choice: Model seçimi
        model_path: GGUF model path'i (eğer gerekirse)
    """
    
    if model_choice == "mistral_7b_gguf" and model_path and GGUF_AVAILABLE:
        return GGUFModelService(model_path)
    
    elif model_choice == "llama_3_1_8b" and HF_AVAILABLE:
        return HuggingFaceModelService("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    elif model_choice == "custom_drive_model" and model_path:
        # Path'e göre format belirle
        if model_path.endswith('.gguf') and GGUF_AVAILABLE:
            return GGUFModelService(model_path)
        elif HF_AVAILABLE:
            return HuggingFaceModelService(model_path)
    
    # Fallback
    logger.warning(f"⚠️ Model oluşturulamadı: {model_choice}")
    return None 