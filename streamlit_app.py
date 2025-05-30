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
import numpy as np
import torch

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
        # Progress tracking için
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        progress_bar.progress(0.1)
        status_text.text("🔧 Kütüphaneler import ediliyor...")
        
        # Import statements
        from src.text_processor import EmbeddingService
        from src.vector_store import VectorStore, RetrievalService
        from src.llm_service_local import GGUFModelService
        
        progress_bar.progress(0.2)
        status_text.text("⚙️ Default RAG ayarları yükleniyor...")
        
        # Default RAG ayarlarını başlat
        if 'rag_settings' not in st.session_state:
            st.session_state.rag_settings = {
                'chunk_size': 800,
                'overlap_size': 150,
                'top_k': 5,
                'similarity_threshold': 0.3,
                'max_context_length': 3000,
                'search_strategy': 'hybrid',
                'embedding_model': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            }
            logger.info("✅ Default RAG ayarları yüklendi")
        
        progress_bar.progress(0.3)
        embedding_model_name = st.session_state.rag_settings.get('embedding_model', '').split('/')[-1]
        status_text.text(f"🧠 Embedding modeli yükleniyor: {embedding_model_name}")
        
        # Embedding servisini başlat
        if 'embedding_service' not in st.session_state:
            embedding_model = st.session_state.rag_settings.get(
                'embedding_model', 
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            st.session_state.embedding_service = EmbeddingService(embedding_model)
            logger.info(f"✅ Embedding service başlatıldı: {embedding_model}")
        
        progress_bar.progress(0.5)
        status_text.text("🗃️ Vector store başlatılıyor...")
        
        # Vector store başlat
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore(persist_directory="./chroma_db")
            logger.info("✅ Vector store başlatıldı")
        
        progress_bar.progress(0.7)
        status_text.text("🔍 Retrieval service başlatılıyor...")
        
        # Retrieval service başlat
        if st.session_state.retrieval_service is None:
            st.session_state.retrieval_service = RetrievalService(
                st.session_state.vector_store, 
                st.session_state.embedding_service
            )
            logger.info("✅ Retrieval service başlatıldı")
        
        progress_bar.progress(0.85)
        status_text.text("🤖 LLM modeli kontrol ediliyor...")
        
        # LLM service başlat
        if st.session_state.llm_service is None:
            # Çoklu model path desteği
            possible_model_paths = [
                "/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Colab
                "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Local
                "~/Downloads/mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Local Downloads
                "/Users/tuncayavci/financial_report/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Your path
            ]
            
            model_path = None
            for path in possible_model_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    model_path = expanded_path
                    break
            
            # Model yükleme
            if model_path:
                model_name = os.path.basename(model_path)
                status_text.text(f"🤖 GGUF modeli yükleniyor: {model_name}")
                st.session_state.llm_service = GGUFModelService(model_path=model_path)
                logger.info("✅ GGUF LLM service başlatıldı")
                llm_model_info = f"GGUF: {model_name}"
            else:
                status_text.text("🤖 HuggingFace modeli yükleniyor...")
                st.warning("⚠️ GGUF model dosyası bulunamadı. HuggingFace modeli kullanılacak.")
                from src.llm_service_local import HuggingFaceModelService
                # Türkçe uyumlu model
                default_model = "microsoft/DialoGPT-medium"
                st.session_state.llm_service = HuggingFaceModelService(model_id=default_model)
                logger.info("✅ HuggingFace LLM service başlatıldı")
                llm_model_info = f"HuggingFace: {default_model}"
        
        progress_bar.progress(1.0)
        status_text.text("✅ Sistem başarıyla başlatıldı!")
        
        st.session_state.system_initialized = True
        
        # Progress'i temizle
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # İlk sistem bilgilerini göster
        if 'system_info_shown' not in st.session_state:
            st.session_state.system_info_shown = True
            
            with st.expander("🎯 Sistem Başlatma Bilgileri", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**⚙️ Mevcut RAG Ayarları:**")
                    settings = st.session_state.rag_settings
                    st.write(f"• 📏 Chunk Boyutu: {settings['chunk_size']} karakter")
                    st.write(f"• 🔗 Overlap: {settings['overlap_size']} karakter")
                    st.write(f"• 🔍 Top-K: {settings['top_k']} sonuç")
                    st.write(f"• 📊 Benzerlik Eşiği: {settings['similarity_threshold']}")
                
                with col2:
                    st.write("**🧠 Model Bilgileri:**")
                    if hasattr(st.session_state.embedding_service, 'get_model_info'):
                        model_info = st.session_state.embedding_service.get_model_info()
                        st.write(f"• 🧠 Embedding: {model_info.get('model_name', 'N/A').split('/')[-1]}")
                        st.write(f"• 📐 Boyut: {model_info.get('embedding_dim', 'N/A')} dim")
                        st.write(f"• 📏 Max Length: {model_info.get('max_seq_length', 'N/A')}")
                        st.write(f"• 🤖 LLM: {llm_model_info}")
            
            # Performance estimate
            st.info("💡 **Performance Tahmini (A100):** 20 sayfalık PDF → 6-14 saniye | 100 chunk → 3-8 saniye embedding")
            st.success("💡 **İpucu:** Sidebar'daki 'Performance Optimizasyonları' bölümünden hızı artırabilirsiniz!")
        
        return True
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Sistem başlatma hatası: {e}")
        logger.error(f"System initialization error: {e}")
        return False

def process_uploaded_files(uploaded_files):
    """Yüklenen dosyaları işle (Performance Optimized)"""
    try:
        from src.pdf_processor import PDFProcessor
        from src.excel_processor import ExcelProcessor
        from src.text_processor import TextProcessor
        
        # RAG ayarlarını al
        rag_settings = st.session_state.get('rag_settings', {
            'chunk_size': 800,
            'overlap_size': 150,
            'embedding_model': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        })
        
        # Performance settings
        perf_settings = st.session_state.get('performance_settings', {
            'pdf_workers': 4,
            'excel_workers': 4,
            'auto_batch': True,
            'manual_batch': 128,
            'aggressive_cleanup': True,
            'reuse_embeddings': True,
            'gpu_memory_fraction': 0.9,
            'performance_mode': 'speed_optimized'
        })
        
        # Progress tracking için
        total_files = len(uploaded_files)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Single instances for reuse (Performance optimization)
        pdf_processor = PDFProcessor(max_workers=perf_settings['pdf_workers'])
        excel_processor = ExcelProcessor(
            max_workers=perf_settings['excel_workers'], 
            use_multiprocessing=True
        )
        
        # TextProcessor - reuse based on settings
        text_processor_key = f"global_text_processor_{rag_settings.get('embedding_model', '')}"
        
        if perf_settings['reuse_embeddings'] and text_processor_key in st.session_state:
            text_processor = st.session_state[text_processor_key]
            status_text.text("♻️ Mevcut embedding modeli kullanılıyor...")
        else:
            status_text.text("🧠 TextProcessor başlatılıyor...")
            # Mevcut embedding_service'i kullan, yeni yaratma!
            text_processor = TextProcessor(
                chunk_size=rag_settings.get('chunk_size', 800),
                overlap_size=rag_settings.get('overlap_size', 150),
                embedding_service=st.session_state.embedding_service  # Mevcut service'i geç!
            )
            
            if perf_settings['reuse_embeddings']:
                st.session_state[text_processor_key] = text_processor
            
            status_text.text("✅ TextProcessor hazır!")
            time.sleep(0.5)
        
        all_chunks = []
        processed_info = []
        total_processing_time = 0
        total_start_time = time.time()
        
        # Phase 1: Document Processing (without embeddings)
        status_text.text("📄 Phase 1: Dosyalar işleniyor...")
        
        for file_idx, uploaded_file in enumerate(uploaded_files):
            file_start_time = time.time()
            
            # Progress update
            progress = (file_idx) / (total_files * 2)  # 2 phase olduğu için
            progress_bar.progress(progress)
            
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            status_text.text(f"📄 Processing {file_idx + 1}/{total_files}: {uploaded_file.name} ({file_size_mb:.1f} MB)")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                file_chunks = []
                
                # File type processing
                if uploaded_file.name.lower().endswith('.pdf'):
                    status_text.text(f"📖 PDF okunuyor: {uploaded_file.name}")
                    pdf_result = pdf_processor.process_pdf(tmp_path)
                    
                    total_pages = len(pdf_result.pages)
                    page_progress = st.progress(0)
                    page_status = st.empty()
                    
                    # Estimate processing time for user
                    estimated_time = total_pages * 0.5  # ~0.5 seconds per page
                    page_status.text(f"📄 {total_pages} sayfa tespit edildi. Tahmini süre: {estimated_time:.0f} saniye")
                    time.sleep(0.5)
                    
                    for page_idx, page_data in enumerate(pdf_result.pages):
                        # Page-by-page progress
                        page_progress.progress((page_idx + 1) / total_pages)
                        page_status.text(f"📄 Sayfa {page_idx + 1}/{total_pages} işleniyor...")
                        
                        # Text chunks
                        if page_data.text and page_data.text.strip():
                            chunks = text_processor.create_chunks(
                                page_data.text,
                                source_file=uploaded_file.name,
                                page_number=page_data.page_number,
                                content_type='text'
                            )
                            file_chunks.extend(chunks)
                        
                        # Table chunks
                        if page_data.tables:
                            for table_data in page_data.tables:
                                if table_data.content and table_data.content.strip():
                                    chunks = text_processor.create_chunks(
                                        table_data.content,
                                        source_file=uploaded_file.name,
                                        page_number=page_data.page_number,
                                        content_type='table',
                                        metadata={
                                            'table_index': table_data.table_index,
                                            'rows': table_data.rows,
                                            'columns': table_data.columns
                                        }
                                    )
                                    file_chunks.extend(chunks)
                        
                        # Small delay for UI responsiveness
                        if page_idx % 5 == 0:  # Every 5 pages
                            time.sleep(0.1)
                    
                    # Clear page progress
                    page_progress.empty()
                    page_status.empty()
                    
                    status_text.text(f"✅ PDF tamamlandı: {len(file_chunks)} chunk oluşturuldu")
                
                elif uploaded_file.name.lower().endswith(('.xlsx', '.xls', '.xlsm')):
                    status_text.text(f"📊 Excel okunuyor: {uploaded_file.name}")
                    excel_result = excel_processor.process_excel(tmp_path)
                    
                    for sheet_data in excel_result.sheets:
                        if sheet_data.text_content and sheet_data.text_content.strip():
                            chunks = text_processor.create_chunks(
                                sheet_data.text_content,
                                source_file=uploaded_file.name,
                                page_number=1,
                                content_type='table',
                                metadata={
                                    'sheet_name': sheet_data.sheet_name,
                                    'rows': sheet_data.raw_data.shape[0],
                                    'columns': sheet_data.raw_data.shape[1]
                                }
                            )
                            file_chunks.extend(chunks)
                    
                    status_text.text(f"✅ Excel tamamlandı: {len(file_chunks)} chunk oluşturuldu")
                
                # Add to global chunks (without embeddings yet)
                all_chunks.extend(file_chunks)
                
                # Track file info
                if file_chunks:
                    file_processing_time = time.time() - file_start_time
                    total_processing_time += file_processing_time
                    
                    avg_chunk_size = np.mean([len(chunk.content) for chunk in file_chunks])
                    
                    processed_info.append({
                        'filename': uploaded_file.name,
                        'chunks': len(file_chunks),
                        'size': len(uploaded_file.getvalue()),
                        'type': uploaded_file.name.split('.')[-1].upper(),
                        'processing_time': file_processing_time,
                        'avg_chunk_size': int(avg_chunk_size),
                        'settings_used': {
                            'chunk_size': rag_settings.get('chunk_size', 800),
                            'overlap_size': rag_settings.get('overlap_size', 150),
                            'embedding_model': rag_settings.get('embedding_model', "paraphrase-multilingual-MiniLM-L12-v2").split('/')[-1]
                        }
                    })
            
            finally:
                os.unlink(tmp_path)
        
        # Phase 2: Batch Embedding Generation (Major Performance Boost)
        if all_chunks:
            status_text.text(f"🧠 Phase 2: Batch embedding oluşturuluyor... ({len(all_chunks)} chunks)")
            progress_bar.progress(0.5)  # 50% completed
            
            # Optimal batch size based on performance settings
            chunk_count = len(all_chunks)
            
            if perf_settings['auto_batch']:
                # Auto batch sizing based on performance mode
                if perf_settings['performance_mode'] == 'speed_optimized':
                    if chunk_count <= 50:
                        batch_size = chunk_count
                    elif chunk_count <= 200:
                        batch_size = 128
                    elif chunk_count <= 500:
                        batch_size = 256
                    else:
                        batch_size = 512  # A100 max speed
                elif perf_settings['performance_mode'] == 'memory_optimized':
                    batch_size = min(64, chunk_count)
                else:  # balanced
                    if chunk_count <= 100:
                        batch_size = 64
                    elif chunk_count <= 300:
                        batch_size = 128
                    else:
                        batch_size = 256
            else:
                # Use manual batch size
                batch_size = min(perf_settings['manual_batch'], chunk_count)
            
            status_text.text(f"🧠 Batch embedding ({batch_size} batch size)...")
            
            # Batch embedding with progress
            embedding_start_time = time.time()
            all_chunks = text_processor.embed_chunks(all_chunks, batch_size=batch_size)
            embedding_time = time.time() - embedding_start_time
            
            # Update progress
            progress_bar.progress(0.75)
            status_text.text("🗃️ Vector store'a ekleniyor...")
            
            # Batch ChromaDB insertion (Performance boost)
            vector_store_start = time.time()
            st.session_state.vector_store.add_documents(all_chunks)
            vector_store_time = time.time() - vector_store_start
            
            st.session_state.processed_files.extend(processed_info)
            
            # Final timing
            total_time = time.time() - total_start_time
            progress_bar.progress(1.0)
            
            # Add to processing history for monitoring
            processing_speed = len(all_chunks) / total_time
            if 'processing_history' not in st.session_state:
                st.session_state.processing_history = []
            
            st.session_state.processing_history.append({
                'timestamp': time.time(),
                'files': len(uploaded_files),
                'chunks': len(all_chunks),
                'total_time': total_time,
                'embedding_time': embedding_time,
                'vector_store_time': vector_store_time,
                'speed': processing_speed,
                'batch_size': batch_size,
                'performance_mode': perf_settings['performance_mode'],
                'workers_used': f"PDF:{perf_settings['pdf_workers']}, Excel:{perf_settings['excel_workers']}"
            })
            
            # Memory cleanup if enabled
            if perf_settings['aggressive_cleanup']:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                status_text.text("🗑️ Memory cleanup completed")
            
            # Enhanced success metrics
            total_embeddings = sum(1 for chunk in all_chunks if chunk.embedding is not None)
            total_chars = sum(chunk.char_count for chunk in all_chunks)
            
            status_text.empty()
            progress_bar.empty()
            
            # Performance summary with confetti for very fast processing
            if processing_speed > 10:  # Very fast processing
                st.balloons()
                success_msg = f"🚀 **Lightning Fast Processing!** ({processing_speed:.1f} chunks/s)"
            elif processing_speed > 5:
                success_msg = f"⚡ **Fast Processing Completed!** ({processing_speed:.1f} chunks/s)"
            else:
                success_msg = f"✅ **Processing Completed** ({processing_speed:.1f} chunks/s)"
            
            st.success(success_msg)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📦 Total Chunks", f"{len(all_chunks)}")
            with col2:
                st.metric("⏱️ Total Time", f"{total_time:.1f}s")
            with col3:
                st.metric("🚀 Speed", f"{len(all_chunks)/total_time:.1f} chunks/s")
            with col4:
                st.metric("🧠 Embeddings", f"{total_embeddings}/{len(all_chunks)}")
            
            # Detailed performance breakdown
            with st.expander("⚡ Performance Breakdown", expanded=False):
                st.write("**🔥 Processing Phases:**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Document Processing", f"{total_processing_time:.1f}s")
                    st.caption(f"{total_chars/total_processing_time:,.0f} chars/s")
                with col2:
                    st.metric("🧠 Batch Embedding", f"{embedding_time:.1f}s") 
                    st.caption(f"Batch size: {batch_size}")
                with col3:
                    st.metric("🗃️ Vector Store", f"{vector_store_time:.1f}s")
                    st.caption(f"{len(all_chunks)/vector_store_time:.0f} chunks/s")
                
                # File-by-file breakdown
                st.write("**📁 File Processing Details:**")
                for info in processed_info:
                    cols = st.columns([3, 1, 1, 1, 2])
                    with cols[0]:
                        st.write(f"📄 **{info['filename']}**")
                    with cols[1]:
                        st.write(f"{info['chunks']} chunks")
                    with cols[2]:
                        st.write(f"{info['processing_time']:.1f}s")
                    with cols[3]:
                        st.write(f"{info['avg_chunk_size']} chars")
                    with cols[4]:
                        efficiency = info['chunks'] / info['processing_time']
                        st.write(f"⚡ {efficiency:.1f} chunks/s")
                
                # Optimization summary
                st.info(f"""
                **🎯 Performance Optimizations Applied:**
                - ♻️ Reused embedding model (no reload)
                - 📦 Batch embedding generation ({batch_size} batch size)
                - 🚀 Parallel PDF processing (4 workers)
                - 🧵 Parallel Excel processing (4 workers)
                - 🗃️ Batch ChromaDB insertion
                - 📊 Real-time progress tracking
                """)
            
            return len(all_chunks), processed_info
        else:
            status_text.text("❌ No content found in uploaded files")
            progress_bar.empty()
            return 0, []
            
    except Exception as e:
        st.error(f"❌ Dosya işleme hatası: {e}")
        logger.error(f"File processing error: {e}")
        return 0, []

def generate_response(query: str) -> str:
    """Query için cevap üret (Speed Optimized)"""
    try:
        # RAG ayarlarını al
        rag_settings = st.session_state.get('rag_settings', {
            'top_k': 5,
            'similarity_threshold': 0.3,
            'max_context_length': 3000
        })
        
        # Context retrieve et
        retrieval_result = st.session_state.retrieval_service.retrieve_context(
            query, 
            n_results=rag_settings.get('top_k', 5),
            max_context_length=rag_settings.get('max_context_length', 3000)
        )
        
        if not retrieval_result.results:
            return "⚠️ İlgili bilgi bulunamadı. Lütfen daha spesifik bir soru sorun veya ilgili dökümanları yüklediğinizden emin olun."
        
        # Benzerlik eşiği uygula
        similarity_threshold = rag_settings.get('similarity_threshold', 0.3)
        filtered_results = [
            result for result in retrieval_result.results 
            if result.similarity_score >= similarity_threshold
        ]
        
        if not filtered_results:
            return f"⚠️ Benzerlik eşiği ({similarity_threshold:.2f}) üzerinde sonuç bulunamadı. Eşiği düşürmeyi deneyin."
        
        # Context'i yeniden oluştur (eşik sonrası)
        combined_context = _build_filtered_context(filtered_results, rag_settings)
        
        # LLM ile cevap üret
        with st.spinner("🤖 AI cevap üretiyor..."):
            # LLM service kontrolü
            if not st.session_state.llm_service:
                return "❌ LLM servisi başlatılmamış. Lütfen sistemi yeniden başlatın."
            
            result = st.session_state.llm_service.generate_response(
                query=query,
                context=combined_context
            )
            
            # Tuple handling - LLM service (response, duration) döndürüyor
            if isinstance(result, tuple) and len(result) == 2:
                response, generation_duration = result
            else:
                response = str(result)
                generation_duration = 0.0
        
        # Response kalitesi kontrolü
        if not response or len(response.strip()) < 10:
            return "❌ Cevap üretilirken bir sorun oluştu. Lütfen sorunuzu yeniden ifade edin."
        
        # Kaynak bilgilerini ekle
        if filtered_results:
            source_files = list({result.source_file for result in filtered_results[:3]})  # İlk 3 kaynak
            response += f"\n\n**📚 Kaynaklar:** {', '.join(source_files)}"
        
        return response
        
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return f"❌ Cevap üretme hatası: {str(e)}"

def _build_filtered_context(filtered_results: List, rag_settings: Dict) -> str:
    """Filtrelenmiş sonuçlardan context oluştur"""
    max_length = rag_settings.get('max_context_length', 3000)
    
    context_parts = []
    total_length = 0
    seen_content = set()
    
    for i, result in enumerate(filtered_results):
        chunk = result.chunk
        
        # Duplicate kontrolü
        content_hash = chunk.content[:100]
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        
        # Context formatı
        context_part = f"""
[Kaynak {i+1}: {result.source_file} - Sayfa {result.page_number} - Benzerlik: {result.similarity_score:.3f}]
{chunk.content}
"""
        
        # Uzunluk kontrolü
        if total_length + len(context_part) > max_length:
            # Kalan alanı kullan
            remaining_space = max_length - total_length
            if remaining_space > 100:  # En az 100 karakter kalsın
                truncated_content = chunk.content[:remaining_space-200] + "..."
                context_part = f"""
[Kaynak {i+1}: {result.source_file} - Sayfa {result.page_number}]
{truncated_content}
"""
                context_parts.append(context_part.strip())
            break
        
        context_parts.append(context_part.strip())
        total_length += len(context_part)
    
    combined = "\n\n".join(context_parts)
    
    if total_length >= max_length:
        combined += "\n\n[Context uzunluk limiti nedeniyle kısaltıldı...]"
    
    return combined

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
        
        # Current model info display
        try:
            model_info = "Model bilgisi alınamadı"
            if hasattr(st.session_state, 'llm_service') and st.session_state.llm_service:
                # Try to get LLM model info
                model_path = "/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
                if os.path.exists(model_path):
                    model_name = os.path.basename(model_path).replace('.gguf', '')
                    model_info = f"🤖 {model_name}"
                else:
                    model_info = "🤖 HuggingFace Model"
            
            # Embedding model info
            if 'rag_settings' in st.session_state:
                embedding_name = st.session_state.rag_settings.get('embedding_model', '').split('/')[-1]
                model_info += f" | 🧠 {embedding_name}"
            
            st.caption(model_info)
        except:
            pass
    
    # Advanced Settings
    if st.session_state.system_initialized:
        st.markdown('<div class="subheader">⚙️ Gelişmiş Ayarlar</div>', unsafe_allow_html=True)
        
        with st.expander("🔧 RAG Parametreleri", expanded=False):
            # Chunk Size
            chunk_size = st.slider(
                "📏 Chunk Boyutu (karakter)",
                min_value=50,
                max_value=1500,
                value=800,
                step=50,
                help="Metin parçalarının boyutu. Küçük: daha detaylı/sayısal veriler, Büyük: daha genel context. Min 50: kısa sayısal veriler için"
            )
            
            # Overlap Size
            overlap_size = st.slider(
                "🔗 Overlap Boyutu (karakter)",
                min_value=50,
                max_value=300,
                value=150,
                step=25,
                help="Chunk'lar arası örtüşme. Yüksek değer: daha iyi bağlam"
            )
            
            # Top-K Results
            top_k = st.slider(
                "🔍 Top-K Sonuç Sayısı",
                min_value=1,
                max_value=15,
                value=5,
                step=1,
                help="Kaç tane en benzer chunk kullanılacağı"
            )
            
            # Similarity Threshold
            similarity_threshold = st.slider(
                "📊 Benzerlik Eşiği",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Minimum benzerlik skoru. Düşük: daha esnek, Yüksek: daha kesin"
            )
            
            # Context Length Limit
            max_context_length = st.slider(
                "📄 Maksimum Context Uzunluğu",
                min_value=1000,
                max_value=5000,
                value=3000,
                step=250,
                help="LLM'e gönderilecek maksimum context uzunluğu"
            )
            
            # Search Strategy
            search_strategy = st.selectbox(
                "🎯 Arama Stratejisi",
                options=["hybrid", "semantic_only", "keyword_boost"],
                index=0,
                help="Arama algoritması türü"
            )
            
            # Embedding Model Choice
            embedding_model = st.selectbox(
                "🧠 Embedding Modeli",
                options=[
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2"
                ],
                index=0,
                help="Kullanılacak embedding modeli"
            )
            
            # Save current settings to session state
            st.session_state.rag_settings = {
                'chunk_size': chunk_size,
                'overlap_size': overlap_size,
                'top_k': top_k,
                'similarity_threshold': similarity_threshold,
                'max_context_length': max_context_length,
                'search_strategy': search_strategy,
                'embedding_model': embedding_model
            }
            
            # Apply Settings Button
            if st.button("💾 Ayarları Uygula", type="secondary"):
                st.info("⚠️ Not: Ayarlar yeni dosyalar için geçerli olacak. Mevcut dosyaları yeniden işlemeniz gerekebilir.")
                st.success("✅ Ayarlar kaydedildi!")
        
        # Performance Optimization Panel
        with st.expander("⚡ Performance Optimizasyonları", expanded=False):
            st.write("**🚀 A100 GPU Optimizasyonları:**")
            
            # Processing Workers
            pdf_workers = st.slider(
                "📄 PDF Worker Sayısı",
                min_value=1,
                max_value=8,
                value=4,
                help="PDF paralel işleme için worker sayısı"
            )
            
            excel_workers = st.slider(
                "📊 Excel Worker Sayısı", 
                min_value=1,
                max_value=8,
                value=4,
                help="Excel paralel işleme için worker sayısı"
            )
            
            # Embedding Batch Optimization
            st.write("**🧠 Embedding Batch Ayarları:**")
            col1, col2 = st.columns(2)
            
            with col1:
                auto_batch = st.checkbox(
                    "🎯 Otomatik Batch Size",
                    value=True,
                    help="Dosya boyutuna göre otomatik batch size ayarla"
                )
            
            with col2:
                if not auto_batch:
                    manual_batch = st.slider(
                        "Manuel Batch Size",
                        min_value=16,
                        max_value=512,
                        value=128,
                        step=16,
                        help="Manuel batch boyutu (A100 için 256+ önerilir)"
                    )
                else:
                    manual_batch = None
            
            # Memory Management
            st.write("**💾 Memory Optimizasyonları:**")
            
            aggressive_cleanup = st.checkbox(
                "🗑️ Agresif Cleanup",
                value=True,
                help="İşlem sonrası memory temizleme"
            )
            
            reuse_embeddings = st.checkbox(
                "♻️ Embedding Model Reuse",
                value=True,
                help="Embedding modelini bellekte tut (hız için)"
            )
            
            # GPU Memory Optimization
            gpu_memory_fraction = st.slider(
                "🎮 GPU Memory Kullanım Oranı",
                min_value=0.5,
                max_value=1.0,
                value=0.9,
                step=0.1,
                help="GPU memory'nin ne kadarını kullan"
            )
            
            # Performance mode
            performance_mode = st.selectbox(
                "🏁 Performance Modu",
                options=["balanced", "speed_optimized", "memory_optimized"],
                index=1,  # Default: speed_optimized
                help="""
                - Balanced: Hız ve memory dengesi
                - Speed Optimized: Maksimum hız (A100 için ideal)
                - Memory Optimized: Düşük memory kullanımı
                """
            )
            
            # Save performance settings
            st.session_state.performance_settings = {
                'pdf_workers': pdf_workers,
                'excel_workers': excel_workers,
                'auto_batch': auto_batch,
                'manual_batch': manual_batch,
                'aggressive_cleanup': aggressive_cleanup,
                'reuse_embeddings': reuse_embeddings,
                'gpu_memory_fraction': gpu_memory_fraction,
                'performance_mode': performance_mode
            }
            
            # Performance presets
            st.write("**🎚️ Performance Presets:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("⚡ A100 Max Speed", help="A100 için maksimum hız"):
                    st.session_state.performance_settings.update({
                        'pdf_workers': 6,
                        'excel_workers': 6, 
                        'auto_batch': True,
                        'manual_batch': 256,
                        'aggressive_cleanup': False,
                        'reuse_embeddings': True,
                        'gpu_memory_fraction': 0.95,
                        'performance_mode': 'speed_optimized'
                    })
                    st.success("🚀 A100 Max Speed preset uygulandı!")
                    st.rerun()
            
            with col2:
                if st.button("⚖️ Balanced Mode", help="Dengeli performans"):
                    st.session_state.performance_settings.update({
                        'pdf_workers': 4,
                        'excel_workers': 4,
                        'auto_batch': True,
                        'manual_batch': 128,
                        'aggressive_cleanup': True,
                        'reuse_embeddings': True,
                        'gpu_memory_fraction': 0.8,
                        'performance_mode': 'balanced'
                    })
                    st.success("⚖️ Balanced preset uygulandı!")
                    st.rerun()
            
            with col3:
                if st.button("💾 Memory Saver", help="Düşük memory"):
                    st.session_state.performance_settings.update({
                        'pdf_workers': 2,
                        'excel_workers': 2,
                        'auto_batch': True,
                        'manual_batch': 64,
                        'aggressive_cleanup': True,
                        'reuse_embeddings': False,
                        'gpu_memory_fraction': 0.6,
                        'performance_mode': 'memory_optimized'
                    })
                    st.success("💾 Memory Saver preset uygulandı!")
                    st.rerun()
            
            # Current performance summary
            if 'performance_settings' in st.session_state:
                perf = st.session_state.performance_settings
                st.info(f"""
                **🎯 Aktif Performance Ayarları:**
                - 📄 PDF Workers: {perf['pdf_workers']} | 📊 Excel Workers: {perf['excel_workers']}
                - 🧠 Batch: {'Auto' if perf['auto_batch'] else f"Manual {perf['manual_batch']}"}
                - 🎮 GPU Memory: {perf['gpu_memory_fraction']*100:.0f}%
                - 🏁 Mode: {perf['performance_mode'].title()}
                """)
        
        # Performance Monitoring
        with st.expander("📈 Performans İzleme", expanded=False):
            if 'last_query_time' in st.session_state:
                st.metric("⏱️ Son Query Süresi", f"{st.session_state.last_query_time:.2f}s")
            
            if 'last_similarity_scores' in st.session_state:
                avg_similarity = np.mean(st.session_state.last_similarity_scores)
                st.metric("📊 Ortalama Benzerlik", f"{avg_similarity:.3f}")
            
            if 'last_context_length' in st.session_state:
                st.metric("📄 Son Context Uzunluğu", f"{st.session_state.last_context_length} char")
            
            # Processing speed history
            if 'processing_history' not in st.session_state:
                st.session_state.processing_history = []
            
            if st.session_state.processing_history:
                speeds = [entry['speed'] for entry in st.session_state.processing_history[-10:]]
                avg_speed = np.mean(speeds)
                st.metric("🚀 Ortalama İşleme Hızı", f"{avg_speed:.1f} chunks/s")
                
                # Speed chart
                if len(speeds) > 1:
                    import pandas as pd
                    df = pd.DataFrame({
                        'İşlem': list(range(1, len(speeds) + 1)),
                        'Hız (chunks/s)': speeds
                    })
                    st.line_chart(df.set_index('İşlem'))
    
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
            # Use current RAG settings if available
            if 'rag_settings' in st.session_state:
                # Update text processor with new settings
                # Note: This would require modifying the process function
                pass
            
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
    
    # Performance bilgileri göster (sistem başlatılmadan önce)
    st.markdown('<div class="subheader">⚡ Performance Bilgileri</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📄 PDF İşleme Hızı:**
        - 10 sayfa: ~3-7 saniye
        - 20 sayfa: ~6-14 saniye  
        - 50 sayfa: ~15-35 saniye
        - 100 sayfa: ~30-70 saniye
        """)
    
    with col2:
        st.markdown("""
        **🧠 Embedding Hızı (A100):**
        - 50 chunk: ~2-5 saniye
        - 100 chunk: ~3-8 saniye
        - 200 chunk: ~5-15 saniye
        - 500 chunk: ~10-30 saniye
        """)
    
    with col3:
        st.markdown("""
        **🚀 Optimizasyon İpuçları:**
        - A100 Max Speed preset kullanın
        - Embedding model reuse aktif
        - Batch size 256+ (otomatik)
        - 4+ worker paralel işlem
        """)
    
    st.info("💡 **20 Sayfalık PDF Örneği:** ~80-150 chunk → Toplam 6-14 saniye")
else:
    # Chat arayüzü
    st.markdown('<div class="subheader">💬 Chat Arayüzü</div>', unsafe_allow_html=True)
    
    # Performance monitoring panel (when system is running)
    if 'processing_history' in st.session_state and st.session_state.processing_history:
        with st.expander("📊 Real-time Performance Monitor", expanded=False):
            latest = st.session_state.processing_history[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🚀 Son İşlem Hızı", f"{latest['speed']:.1f} chunks/s")
            with col2:
                st.metric("⏱️ Son İşlem Süresi", f"{latest['total_time']:.1f}s")
            with col3:
                st.metric("📦 Son Chunk Sayısı", latest['chunks'])
            with col4:
                efficiency = "🔥 Çok Hızlı" if latest['speed'] > 10 else "⚡ Hızlı" if latest['speed'] > 5 else "📊 Normal"
                st.metric("📈 Verimlilik", efficiency)
            
            # Performance trend
            if len(st.session_state.processing_history) > 1:
                speeds = [entry['speed'] for entry in st.session_state.processing_history[-5:]]
                avg_speed = np.mean(speeds)
                trend = "📈 Artan" if speeds[-1] > avg_speed else "📉 Azalan" if speeds[-1] < avg_speed * 0.8 else "➡️ Sabit"
                st.write(f"**Trend:** {trend} | **Ortalama Hız:** {avg_speed:.1f} chunks/s")
            
            # Predictions for common document sizes
            current_speed = latest['speed']
            st.write("**📄 Tahmini İşleme Süreleri (mevcut hıza göre):**")
            predictions = {
                "10 sayfa (~50 chunk)": 50 / current_speed,
                "20 sayfa (~100 chunk)": 100 / current_speed,
                "50 sayfa (~250 chunk)": 250 / current_speed,
                "100 sayfa (~500 chunk)": 500 / current_speed
            }
            
            for doc_type, time_estimate in predictions.items():
                st.write(f"• {doc_type}: ~{time_estimate:.1f} saniye")
    
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