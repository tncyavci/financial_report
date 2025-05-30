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
            
            # Embedding servisini başlat
            if 'embedding_service' not in st.session_state:
                embedding_model = st.session_state.rag_settings.get(
                    'embedding_model', 
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                st.session_state.embedding_service = EmbeddingService(embedding_model)
                logger.info(f"✅ Embedding service başlatıldı: {embedding_model}")
            
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
                            st.write(f"• Model: {model_info.get('model_name', 'N/A').split('/')[-1]}")
                            st.write(f"• Boyut: {model_info.get('embedding_dim', 'N/A')} dim")
                            st.write(f"• Max Length: {model_info.get('max_seq_length', 'N/A')}")
                
                st.info("💡 **İpucu:** Sidebar'daki 'Gelişmiş Ayarlar' bölümünden RAG parametrelerini özelleştirebilirsiniz!")
            
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
        
        # RAG ayarlarını al
        rag_settings = st.session_state.get('rag_settings', {
            'chunk_size': 800,
            'overlap_size': 150,
            'embedding_model': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        })
        
        pdf_processor = PDFProcessor()
        excel_processor = ExcelProcessor()
        
        # TextProcessor'ı güncel ayarlarla oluştur
        text_processor = TextProcessor(
            chunk_size=rag_settings.get('chunk_size', 800),
            overlap_size=rag_settings.get('overlap_size', 150),
            embedding_model=rag_settings.get('embedding_model', "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        )
        
        all_chunks = []
        processed_info = []
        total_processing_time = 0
        
        for uploaded_file in uploaded_files:
            file_start_time = time.time()
            
            with st.spinner(f"📄 {uploaded_file.name} işleniyor... (Chunk: {rag_settings.get('chunk_size', 800)}, Overlap: {rag_settings.get('overlap_size', 150)})"):
                # Geçici dosya oluştur
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    file_chunks = []
                    
                    # Dosya türüne göre işle
                    if uploaded_file.name.lower().endswith('.pdf'):
                        # PDF işleme - Doğru metod adını kullan
                        pdf_result = pdf_processor.process_pdf(tmp_path)
                        
                        for page_data in pdf_result.pages:
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
                    
                    elif uploaded_file.name.lower().endswith(('.xlsx', '.xls', '.xlsm')):
                        # Excel işleme - Doğru metod adını kullan
                        excel_result = excel_processor.process_excel(tmp_path)
                        
                        for sheet_data in excel_result.sheets:
                            if sheet_data.text_content and sheet_data.text_content.strip():
                                chunks = text_processor.create_chunks(
                                    sheet_data.text_content,
                                    source_file=uploaded_file.name,
                                    page_number=1,  # Excel için sheet index
                                    content_type='table',
                                    metadata={
                                        'sheet_name': sheet_data.sheet_name,
                                        'rows': sheet_data.raw_data.shape[0],
                                        'columns': sheet_data.raw_data.shape[1]
                                    }
                                )
                                file_chunks.extend(chunks)
                    
                    # Embeddings oluştur
                    if file_chunks:
                        st.session_state.embedding_service.embed_chunks(file_chunks)
                        all_chunks.extend(file_chunks)
                        
                        # Dosya işleme süresi
                        file_processing_time = time.time() - file_start_time
                        total_processing_time += file_processing_time
                        
                        # Chunk istatistikleri hesapla
                        avg_chunk_size = np.mean([len(chunk.content) for chunk in file_chunks])
                        min_chunk_size = min([len(chunk.content) for chunk in file_chunks])
                        max_chunk_size = max([len(chunk.content) for chunk in file_chunks])
                        
                        processed_info.append({
                            'filename': uploaded_file.name,
                            'chunks': len(file_chunks),
                            'size': len(uploaded_file.getvalue()),
                            'type': uploaded_file.name.split('.')[-1].upper(),
                            'processing_time': file_processing_time,
                            'avg_chunk_size': int(avg_chunk_size),
                            'min_chunk_size': min_chunk_size,
                            'max_chunk_size': max_chunk_size,
                            'settings_used': {
                                'chunk_size': rag_settings.get('chunk_size', 800),
                                'overlap_size': rag_settings.get('overlap_size', 150),
                                'embedding_model': rag_settings.get('embedding_model', "paraphrase-multilingual-MiniLM-L12-v2").split('/')[-1]
                            }
                        })
                
                finally:
                    # Geçici dosyayı sil
                    os.unlink(tmp_path)
        
        # Vector store'a ekle
        if all_chunks:
            with st.spinner("🗃️ Vector store'a ekleniyor..."):
                st.session_state.vector_store.add_documents(all_chunks)
                st.session_state.processed_files.extend(processed_info)
            
            # İşleme özeti göster
            st.success(f"✅ Toplam {len(all_chunks)} chunk işlendi! ({total_processing_time:.2f}s)")
            
            # Detaylı istatistikler
            with st.expander("📊 İşleme Detayları", expanded=True):
                st.write("**📁 Dosya Bazında İstatistikler:**")
                
                for info in processed_info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"📄 {info['filename']}", f"{info['chunks']} chunk")
                        st.caption(f"⏱️ {info['processing_time']:.1f}s")
                    with col2:
                        st.metric("📏 Ortalama Chunk", f"{info['avg_chunk_size']} char")
                        st.caption(f"Min: {info['min_chunk_size']} | Max: {info['max_chunk_size']}")
                    with col3:
                        st.metric("⚙️ Chunk Ayarları", f"{info['settings_used']['chunk_size']}")
                        st.caption(f"Overlap: {info['settings_used']['overlap_size']}")
                
                # Genel istatistikler
                st.write("**🎯 Kullanılan RAG Ayarları:**")
                st.json(rag_settings)
            
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
        # Performance tracking başlat
        start_time = time.time()
        
        # RAG ayarlarını al
        rag_settings = st.session_state.get('rag_settings', {
            'top_k': 5,
            'similarity_threshold': 0.3,
            'max_context_length': 3000
        })
        
        # Context retrieve et
        retrieval_result = st.session_state.retrieval_service.retrieve_context(
            query, 
            n_results=rag_settings.get('top_k', 5)
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
        
        # Performance metrikleri kaydet
        similarity_scores = [result.similarity_score for result in filtered_results]
        st.session_state.last_similarity_scores = similarity_scores
        st.session_state.last_context_length = len(combined_context)
        
        # Debug bilgileri göster
        with st.expander("🔍 RAG İşlemi Detayları", expanded=False):
            st.write("**⚙️ Kullanılan Ayarlar:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔍 Top-K", rag_settings.get('top_k', 5))
                st.metric("📊 Benzerlik Eşiği", f"{similarity_threshold:.2f}")
            with col2:
                st.metric("✅ Filtrelenmiş Sonuç", len(filtered_results))
                st.metric("📄 Context Uzunluğu", len(combined_context))
            with col3:
                st.metric("📊 Ortalama Benzerlik", f"{np.mean(similarity_scores):.3f}")
                st.metric("📊 En Yüksek Benzerlik", f"{max(similarity_scores):.3f}")
            
            st.write("**🔎 Bulunan Sonuçlar:**")
            for i, result in enumerate(filtered_results):
                similarity_percent = result.similarity_score * 100
                st.write(f"""
                **Sonuç {i+1}:** `{result.source_file}` (Sayfa {result.page_number})
                - 🏷️ Tip: {result.content_type} | 📊 Benzerlik: {similarity_percent:.1f}%
                """)
                
                # İçerik önizlemesi
                preview = result.chunk.content[:150] + "..." if len(result.chunk.content) > 150 else result.chunk.content
                st.code(preview)
            
            # Context önizlemesi
            with st.expander("📋 Oluşturulan Context", expanded=False):
                st.text_area("Context", combined_context, height=200)
        
        # LLM ile cevap üret
        with st.spinner("🤖 AI cevap üretiyor..."):
            response = st.session_state.llm_service.generate_response(
                query=query,
                context=combined_context
            )
        
        # Query süresini kaydet
        query_time = time.time() - start_time
        st.session_state.last_query_time = query_time
        
        # Response kalitesi kontrolü
        if len(response.strip()) < 10:
            return "❌ Cevap üretilirken bir sorun oluştu. Lütfen sorunuzu yeniden ifade edin."
        
        # Kaynak bilgilerini ekle
        source_info = []
        seen_sources = set()
        
        for result in filtered_results:
            source_key = f"{result.source_file}_{result.page_number}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                source_info.append(f"📄 {result.source_file} (Sayfa {result.page_number})")
        
        if source_info:
            response += f"\n\n**📚 Kaynaklar:**\n" + "\n".join(source_info)
        
        # Performance özeti ekle
        response += f"\n\n**⚡ Performans:** {query_time:.2f}s | {len(filtered_results)} sonuç | Ort. benzerlik: {np.mean(similarity_scores):.3f}"
        
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
    
    # Advanced Settings
    if st.session_state.system_initialized:
        st.markdown('<div class="subheader">⚙️ Gelişmiş Ayarlar</div>', unsafe_allow_html=True)
        
        with st.expander("🔧 RAG Parametreleri", expanded=False):
            # Chunk Size
            chunk_size = st.slider(
                "📏 Chunk Boyutu (karakter)",
                min_value=300,
                max_value=1500,
                value=800,
                step=50,
                help="Metin parçalarının boyutu. Küçük: daha detaylı, Büyük: daha genel"
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
        
        # Performance Monitoring
        with st.expander("📈 Performans İzleme", expanded=False):
            if 'last_query_time' in st.session_state:
                st.metric("⏱️ Son Query Süresi", f"{st.session_state.last_query_time:.2f}s")
            
            if 'last_similarity_scores' in st.session_state:
                avg_similarity = np.mean(st.session_state.last_similarity_scores)
                st.metric("📊 Ortalama Benzerlik", f"{avg_similarity:.3f}")
            
            if 'last_context_length' in st.session_state:
                st.metric("📄 Son Context Uzunluğu", f"{st.session_state.last_context_length} char")
    
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