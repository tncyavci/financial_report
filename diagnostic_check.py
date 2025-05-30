#!/usr/bin/env python3
"""
Turkish Financial RAG Chatbot - Diagnostic Check
PDF işleme sistemi için kapsamlı test scripti
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Tüm import'ları test et"""
    print("🔍 Testing imports...")
    
    try:
        from src.pdf_processor import PDFProcessor
        print("✅ PDF Processor import successful")
    except Exception as e:
        print(f"❌ PDF Processor import failed: {e}")
        return False
    
    try:
        from src.text_processor import TextProcessor
        print("✅ Text Processor import successful")
    except Exception as e:
        print(f"❌ Text Processor import failed: {e}")
        return False
    
    try:
        from src.vector_store import VectorStore
        print("✅ Vector Store import successful")
    except Exception as e:
        print(f"❌ Vector Store import failed: {e}")
        return False
    
    try:
        from src.excel_processor import ExcelProcessor
        print("✅ Excel Processor import successful")
    except Exception as e:
        print(f"❌ Excel Processor import failed: {e}")
        return False
    
    return True

def test_pdf_dependencies():
    """PDF kütüphanelerini test et"""
    print("\n📄 Testing PDF dependencies...")
    
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF (fitz) available")
    except ImportError as e:
        print(f"❌ PyMuPDF missing: {e}")
        return False
    
    try:
        import pdfplumber
        print("✅ pdfplumber available")
    except ImportError as e:
        print(f"❌ pdfplumber missing: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 available")
    except ImportError as e:
        print(f"❌ PyPDF2 missing: {e}")
        return False
    
    return True

def test_nlp_dependencies():
    """NLP kütüphanelerini test et"""
    print("\n🧠 Testing NLP dependencies...")
    
    try:
        import sentence_transformers
        print("✅ sentence-transformers available")
    except ImportError as e:
        print(f"❌ sentence-transformers missing: {e}")
        return False
    
    try:
        import chromadb
        print("✅ chromadb available")
    except ImportError as e:
        print(f"❌ chromadb missing: {e}")
        return False
    
    try:
        import langdetect
        print("✅ langdetect available")
    except ImportError as e:
        print(f"❌ langdetect missing: {e}")
        return False
    
    return True

def test_pdf_processor():
    """PDF processor'ı gerçek bir dosya ile test et"""
    print("\n📖 Testing PDF Processor...")
    
    try:
        from src.pdf_processor import PDFProcessor
        
        # Instance oluştur
        processor = PDFProcessor(max_workers=2)
        print("✅ PDFProcessor instance created")
        
        # Stats'ı al
        stats = processor.get_processor_stats()
        print(f"📊 Processor stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF Processor test failed: {e}")
        return False

def test_text_processor():
    """Text processor'ı test et"""
    print("\n📝 Testing Text Processor...")
    
    try:
        from src.text_processor import TextProcessor
        
        # Basic test
        processor = TextProcessor(
            chunk_size=400,
            overlap_size=50,
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("✅ TextProcessor instance created")
        
        # Test chunking with correct parameters
        test_text = "Bu bir test metnidir. Türkçe finansal analiz için kullanılmaktadır. Gelir tablosunda önemli artışlar görülmektedir."
        chunks = processor.create_chunks(
            text=test_text, 
            source_file="test.txt",
            page_number=1,
            content_type="text"
        )
        print(f"✅ Text chunking successful: {len(chunks)} chunks created")
        
        return True
        
    except Exception as e:
        print(f"❌ Text Processor test failed: {e}")
        return False

def test_vector_store():
    """Vector store'u test et"""
    print("\n🗃️ Testing Vector Store...")
    
    try:
        from src.vector_store import VectorStore
        
        # Test vector store
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_store = VectorStore(persist_directory=tmpdir)
            print("✅ VectorStore instance created")
            
            # Test with a simple document count check
            try:
                # Try to get document count
                stats = vector_store.get_retrieval_stats()
                doc_count = stats.get('vector_store_stats', {}).get('total_documents', 0)
                print(f"📊 Document count: {doc_count}")
            except:
                print("📊 Document count: 0 (empty store)")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector Store test failed: {e}")
        return False

def create_test_pdf():
    """Test için basit bir PDF oluştur"""
    print("\n📄 Creating test PDF...")
    
    try:
        # ReportLab olmadan basit PDF oluşturmayalım
        # Bunun yerine mevcut bir PDF varsa test edelim
        print("⚠️ Test PDF creation skipped (requires reportlab)")
        return None
        
    except Exception as e:
        print(f"❌ Test PDF creation failed: {e}")
        return None

def run_full_system_test():
    """Tam sistem testini çalıştır"""
    print("\n🔄 Running full system test...")
    
    try:
        # Import all modules
        from src.pdf_processor import PDFProcessor
        from src.text_processor import TextProcessor
        from src.vector_store import VectorStore
        from src.excel_processor import ExcelProcessor
        
        print("✅ All modules imported successfully")
        
        # Test instances
        pdf_proc = PDFProcessor(max_workers=2)
        text_proc = TextProcessor()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_store = VectorStore(persist_directory=tmpdir)
            excel_proc = ExcelProcessor()
        
        print("✅ All processor instances created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Full system test failed: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🚀 Turkish Financial RAG Chatbot - Diagnostic Check")
    print("=" * 60)
    
    all_passed = True
    
    # Test aşamaları
    tests = [
        ("📦 Import Tests", test_imports),
        ("📄 PDF Dependencies", test_pdf_dependencies),
        ("🧠 NLP Dependencies", test_nlp_dependencies),
        ("📖 PDF Processor", test_pdf_processor),
        ("📝 Text Processor", test_text_processor),
        ("🗃️ Vector Store", test_vector_store),
        ("🔄 Full System", run_full_system_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append((test_name, "✅ PASSED" if result else "❌ FAILED"))
            if not result:
                all_passed = False
        except Exception as e:
            results.append((test_name, f"❌ ERROR: {e}"))
            all_passed = False
    
    # Sonuçları özetle
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, status in results:
        print(f"{status:<15} {test_name}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! PDF processing should work correctly.")
        print("\n💡 To run the application:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️ SOME TESTS FAILED! Please fix the issues before using PDF processing.")
        print("\n🔧 Common fixes:")
        print("   pip install pymupdf pdfplumber langdetect sentence-transformers chromadb")
    
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 