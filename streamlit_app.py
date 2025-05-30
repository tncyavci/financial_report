"""
Turkish Financial RAG Chatbot
Streamlit Arayüzü - A100 GPU Optimized
"""

import streamlit as st
import os
import logging
import asyncio
from pathlib import Path
import tempfile
import time
from typing import List, Dict, Optional

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Turkish Financial RAG Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subheader {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .success-message {
        background: linear-gradient(90deg, #00b09b, #96c93d);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 2rem;
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Başlık
st.markdown('<div class="main-header">💰 Turkish Financial RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem;">A100 GPU ile Optimize Edilmiş Türkçe Finans Döküman Analizi</p>', unsafe_allow_html=True)

# Session state initialization
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retrieval_service' not in st.session_state:
    st.session_state.retrieval_service = None
if 'llm_service' not in st.session_state:
    st.session_state.llm_service = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

def initialize_system():
    """Sistemi başlat"""
    try:
        with st.spinner("🚀 Sistem başlatılıyor..."):
            # Import statements
            from src.text_processor import EmbeddingService
            from src.vector_store import VectorStore, RetrievalService
            from src.llm_service_local import GGUFModelService
            
            # Embedding servisini başlat
            if 'embedding_service' not in st.session_state:
                st.session_state.embedding_service = EmbeddingService()
                logger.info("✅ Embedding service başlatıldı")
            
            # Vector store başlat
            if st.session_state.vector_store is None:
                st.session_state.vector_store = VectorStore(persist_directory="./chroma_db")
                logger.info("✅ Vector store başlatıldı")
            
            # Retrieval service başlat
            if st.session_state.retrieval_service is None:
                st.session_state.retrieval_service = RetrievalService(
                    st.session_state.vector_store, 
                    st.session_state.embedding_service
                )
                logger.info("✅ Retrieval service başlatıldı")
            
            # LLM service başlat
            if st.session_state.llm_service is None:
                model_path = "/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
                
                # Colab ortamı kontrol
                if os.path.exists(model_path):
                    st.session_state.llm_service = GGUFModelService(model_path=model_path)
                    logger.info("✅ GGUF LLM service başlatıldı")
                else:
                    st.warning("⚠️ Model dosyası bulunamadı. HuggingFace modeli kullanılacak.")
                    from src.llm_service_local import HuggingFaceModelService
                    st.session_state.llm_service = HuggingFaceModelService()
                    logger.info("✅ HuggingFace LLM service başlatıldı")
            
            st.session_state.system_initialized = True
            return True
            
    except Exception as e:
        st.error(f"❌ Sistem başlatma hatası: {e}")
        logger.error(f"System initialization error: {e}")
        return False

def process_uploaded_files(uploaded_files):
    """Yüklenen dosyaları işle"""
    try:
        from src.pdf_processor import PDFProcessor
        from src.excel_processor import ExcelProcessor
        from src.text_processor import TextProcessor
        
        pdf_processor = PDFProcessor()
        excel_processor = ExcelProcessor()
        text_processor = TextProcessor()
        
        all_chunks = []
        processed_info = []
        
        for uploaded_file in uploaded_files:
            with st.spinner(f"📄 {uploaded_file.name} işleniyor..."):
                # Geçici dosya oluştur
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    file_chunks = []
                    
                    # Dosya türüne göre işle
                    if uploaded_file.name.lower().endswith('.pdf'):
                        # PDF işleme
                        pdf_content = pdf_processor.extract_content(tmp_path)
                        
                        for page_num, page_data in pdf_content.items():
                            # Text chunks
                            if page_data.get('text'):
                                chunks = text_processor.create_chunks(
                                    page_data['text'],
                                    source_file=uploaded_file.name,
                                    page_number=page_num,
                                    content_type='text'
                                )
                                file_chunks.extend(chunks)
                            
                            # Table chunks
                            if page_data.get('tables'):
                                for table_idx, table in enumerate(page_data['tables']):
                                    table_text = table.get('text', '')
                                    if table_text:
                                        chunks = text_processor.create_chunks(
                                            table_text,
                                            source_file=uploaded_file.name,
                                            page_number=page_num,
                                            content_type='table',
                                            metadata={'table_index': table_idx}
                                        )
                                        file_chunks.extend(chunks)
                    
                    elif uploaded_file.name.lower().endswith(('.xlsx', '.xls', '.xlsm')):
                        # Excel işleme
                        excel_content = excel_processor.extract_content(tmp_path)
                        
                        for sheet_name, sheet_data in excel_content.items():
                            if sheet_data.get('text'):
                                chunks = text_processor.create_chunks(
                                    sheet_data['text'],
                                    source_file=uploaded_file.name,
                                    page_number=1,  # Excel için sheet index
                                    content_type='table',
                                    metadata={'sheet_name': sheet_name}
                                )
                                file_chunks.extend(chunks)
                    
                    # Embeddings oluştur
                    if file_chunks:
                        st.session_state.embedding_service.embed_chunks(file_chunks)
                        all_chunks.extend(file_chunks)
                        
                        processed_info.append({
                            'filename': uploaded_file.name,
                            'chunks': len(file_chunks),
                            'size': len(uploaded_file.getvalue()),
                            'type': uploaded_file.name.split('.')[-1].upper()
                        })
                
                finally:
                    # Geçici dosyayı sil
                    os.unlink(tmp_path)
        
        # Vector store'a ekle
        if all_chunks:
            st.session_state.vector_store.add_documents(all_chunks)
            st.session_state.processed_files.extend(processed_info)
            
            return len(all_chunks), processed_info
        else:
            return 0, []
            
    except Exception as e:
        st.error(f"❌ Dosya işleme hatası: {e}")
        logger.error(f"File processing error: {e}")
        return 0, []

def generate_response(query: str) -> str:
    """Query için cevap üret"""
    try:
        # Context retrieve et
        retrieval_result = st.session_state.retrieval_service.retrieve_context(query, n_results=5)
        
        if not retrieval_result.results:
            return "⚠️ İlgili bilgi bulunamadı. Lütfen daha spesifik bir soru sorun veya ilgili dökümanları yüklediğinizden emin olun."
        
        # LLM ile cevap üret
        response = st.session_state.llm_service.generate_response(
            query=query,
            context=retrieval_result.combined_context
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return f"❌ Cevap üretme hatası: {str(e)}"

# Sidebar
with st.sidebar:
    st.markdown('<div class="subheader">🛠️ Sistem Kontrolleri</div>', unsafe_allow_html=True)
    
    # Sistem başlatma
    if not st.session_state.system_initialized:
        if st.button("🚀 Sistemi Başlat", type="primary"):
            if initialize_system():
                st.success("✅ Sistem başarıyla başlatıldı!")
                st.rerun()
            else:
                st.error("❌ Sistem başlatılamadı!")
    else:
        st.success("✅ Sistem Aktif")
    
    # Dosya yükleme
    if st.session_state.system_initialized:
        st.markdown('<div class="subheader">📁 Dosya Yükleme</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "PDF veya Excel dosyalarını yükleyin",
            type=['pdf', 'xlsx', 'xls', 'xlsm'],
            accept_multiple_files=True,
            help="Türkçe finans dökümanlarını yükleyebilirsiniz"
        )
        
        if uploaded_files and st.button("📄 Dosyaları İşle"):
            chunk_count, processed_info = process_uploaded_files(uploaded_files)
            
            if chunk_count > 0:
                st.markdown(f'<div class="success-message">✅ {chunk_count} chunk başarıyla işlendi!</div>', unsafe_allow_html=True)
                
                for info in processed_info:
                    st.write(f"📄 **{info['filename']}** ({info['type']}) - {info['chunks']} chunk")
            else:
                st.markdown('<div class="error-message">❌ Dosya işlenemedi!</div>', unsafe_allow_html=True)
    
    # Sistem istatistikleri
    if st.session_state.system_initialized and st.session_state.retrieval_service:
        st.markdown('<div class="subheader">📊 Sistem İstatistikleri</div>', unsafe_allow_html=True)
        
        try:
            stats = st.session_state.retrieval_service.get_retrieval_stats()
            vector_stats = stats.get('vector_store_stats', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("📄 Toplam Doküman", vector_stats.get('total_documents', 0))
                st.metric("🗂️ Kaynak Sayısı", vector_stats.get('unique_sources', 0))
            
            with col2:
                st.metric("💬 Chat Geçmişi", len(st.session_state.chat_history))
                st.metric("📁 İşlenen Dosya", len(st.session_state.processed_files))
            
            # İçerik dağılımı
            content_dist = vector_stats.get('content_type_distribution', {})
            if content_dist:
                st.write("**İçerik Türü Dağılımı:**")
                for content_type, count in content_dist.items():
                    st.write(f"• {content_type}: {count}")
        
        except Exception as e:
            st.error(f"İstatistik hatası: {e}")
    
    # Sistem temizleme
    if st.session_state.system_initialized:
        st.markdown('<div class="subheader">🗑️ Sistem Temizleme</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Vector Store Temizle", type="secondary"):
            try:
                st.session_state.vector_store.clear()
                st.session_state.processed_files = []
                st.success("✅ Vector store temizlendi!")
            except Exception as e:
                st.error(f"❌ Temizleme hatası: {e}")
        
        if st.button("💬 Chat Geçmişini Temizle", type="secondary"):
            st.session_state.chat_history = []
            st.success("✅ Chat geçmişi temizlendi!")

# Ana içerik
if not st.session_state.system_initialized:
    st.info("👈 Lütfen önce sistemi başlatın")
else:
    # Chat arayüzü
    st.markdown('<div class="subheader">💬 Chat Arayüzü</div>', unsafe_allow_html=True)
    
    # Chat geçmişini göster
    chat_container = st.container()
    
    with chat_container:
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>👤 Siz:</strong><br>{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 Asistan:</strong><br>{message}</div>', unsafe_allow_html=True)
    
    # Yeni mesaj girişi
    user_input = st.chat_input("Türkçe finans sorularınızı sorun...")
    
    if user_input:
        # Kullanıcı mesajını ekle
        st.session_state.chat_history.append(("user", user_input))
        
        # Cevap üret
        with st.spinner("🤔 Düşünüyor..."):
            response = generate_response(user_input)
        
        # Cevabı ekle
        st.session_state.chat_history.append(("assistant", response))
        
        # Sayfayı yenile
        st.rerun()
    
    # Örnek sorular
    if not st.session_state.chat_history:
        st.markdown('<div class="subheader">💡 Örnek Sorular</div>', unsafe_allow_html=True)
        
        example_questions = [
            "Bu dökümanların özeti nedir?",
            "Finansal tablolardaki ana göstergeler nelerdir?",
            "Risk analizinde öne çıkan faktörler neler?",
            "Gelir tablosundaki trend nasıl?",
            "Nakit akımı durumu nasıl?"
        ]
        
        cols = st.columns(3)
        for i, question in enumerate(example_questions):
            with cols[i % 3]:
                if st.button(question, key=f"example_{i}"):
                    st.session_state.chat_history.append(("user", question))
                    with st.spinner("🤔 Düşünüyor..."):
                        response = generate_response(question)
                    st.session_state.chat_history.append(("assistant", response))
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #95a5a6;">Turkish Financial RAG Assistant | A100 GPU Optimized | ChromaDB + Mistral 7B</p>',
        unsafe_allow_html=True
    ) 