"""
Enhanced Google Colab Setup for Performance & Accuracy Evaluation
Complete setup for Turkish Financial RAG evaluation system
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def print_header():
    """Print setup header"""
    print("=" * 70)
    print("🎯 Turkish Financial RAG - Enhanced Evaluation Setup")
    print("Performance Tracking + Accuracy Evaluation + 31 Test Queries")
    print("=" * 70)
    print(f"📅 Setup started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def install_requirements():
    """Install all required packages"""
    print("\n📦 Installing requirements...")
    
    requirements = [
        "streamlit>=1.28.0",
        "pandas>=2.0.0", 
        "psutil>=5.9.0",
        "pyngrok>=6.0.0"
    ]
    
    for req in requirements:
        try:
            print(f"   Installing {req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req, "-q"])
            print(f"   ✅ {req}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed: {req} - {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def setup_python_path():
    """Setup Python path for imports"""
    print("\n🐍 Setting up Python path...")
    
    # Detect environment
    if "/content" in os.getcwd():
        base_path = "/content/financial_report"
        print("   🌐 Google Colab environment detected")
    else:
        base_path = os.getcwd()
        print("   💻 Local environment detected")
    
    # Add to Python path
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
    
    print(f"   📁 Base path: {base_path}")
    print(f"   🐍 Python path updated")
    return base_path

def create_directories():
    """Create necessary directory structure"""
    print("\n📁 Creating directory structure...")
    
    dirs = [
        "evaluation/reports",
        "evaluation/data",
        "evaluation/metrics"
    ]
    
    for dir_path in dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"   ✅ {dir_path}")
        except Exception as e:
            print(f"   ❌ Failed to create {dir_path}: {e}")
    
    return True

def generate_test_queries():
    """Generate test queries for evaluation"""
    print("\n📝 Generating test queries...")
    
    try:
        # Import and run test query generator
        from evaluation.test_query_generator import generate_test_queries
        queries = generate_test_queries()
        
        print(f"   ✅ Generated {len(queries)} test queries")
        print(f"   📁 Saved to: evaluation/data/test_queries.json")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to generate test queries: {e}")
        print("   💡 You can generate them manually in the Streamlit app")
        return False

def test_system():
    """Test the evaluation system"""
    print("\n🧪 Testing evaluation system...")
    
    try:
        # Test performance tracker
        from evaluation.metrics.performance_tracker import get_global_tracker
        perf_tracker = get_global_tracker()
        
        with perf_tracker.track_operation("colab_setup_test"):
            time.sleep(0.5)
        
        print("   ✅ Performance tracker working")
        
        # Test accuracy tracker
        from evaluation.metrics.accuracy_tracker import get_global_accuracy_tracker, ResponseQuality
        acc_tracker = get_global_accuracy_tracker()
        
        # Quick test evaluation
        acc_tracker.quick_evaluate(
            query="Test query for setup",
            response="Test response",
            score=4
        )
        
        print("   ✅ Accuracy tracker working")
        
        # Get system stats
        sys_stats = perf_tracker.get_system_stats()
        print(f"   📊 CPU: {sys_stats.cpu_percent:.1f}%")
        print(f"   🧠 Memory: {sys_stats.memory_percent:.1f}%")
        print(f"   🎮 GPU: {'Available' if sys_stats.gpu_available else 'Not Available'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ System test failed: {e}")
        return False

def setup_ngrok():
    """Setup NGROK for Streamlit access"""
    print("\n🌐 Setting up NGROK...")
    
    try:
        from pyngrok import ngrok
        
        # Kill any existing tunnels
        ngrok.kill()
        
        print("   ✅ NGROK ready (token required)")
        print("   💡 Get your token from: https://ngrok.com/")
        return True
        
    except ImportError:
        print("   ❌ NGROK import failed")
        return False

def run_streamlit_app(app_type="enhanced"):
    """Run the Streamlit application"""
    print(f"\n🚀 Starting Streamlit application ({app_type})...")
    
    try:
        from pyngrok import ngrok
        import threading
        
        # Choose app based on type
        if app_type == "enhanced":
            app_file = "evaluation/streamlit_accuracy_app.py"
            print("   📊 Starting Enhanced App (Performance + Accuracy)")
        else:
            app_file = "evaluation/streamlit_test_app.py" 
            print("   ⚡ Starting Basic App (Performance Only)")
        
        # Start Streamlit in background
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            app_file,
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for Streamlit to start
        print("   ⏳ Waiting for Streamlit to start...")
        time.sleep(12)
        
        # Create NGROK tunnel
        print("   🌐 Creating public tunnel...")
        public_url = ngrok.connect(8501)
        
        print(f"\n🎉 SUCCESS! Your evaluation app is ready:")
        print(f"   🔗 {public_url}")
        
        if app_type == "enhanced":
            print(f"\n📊 Enhanced Features Available:")
            print(f"   ⚡ Performance Testing")
            print(f"   🎯 Accuracy Evaluation (31 test queries)")
            print(f"   📈 Combined Analytics Dashboard")
            print(f"   🧪 Test Query Management")
        else:
            print(f"\n⚡ Basic Features Available:")
            print(f"   📄 PDF Processing Simulation")
            print(f"   💬 Query Response Testing")
            print(f"   📊 Performance Metrics")
        
        print(f"\n🎮 Usage Instructions:")
        print(f"   1. Click the link above to access your app")
        print(f"   2. Use the sidebar to navigate between pages")
        print(f"   3. Run tests and evaluate responses")
        print(f"   4. Download results as JSON/MD files")
        
        # Keep running
        try:
            streamlit_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping application...")
            streamlit_process.terminate()
            ngrok.kill()
            
    except Exception as e:
        print(f"   ❌ Failed to start application: {e}")
        print(f"   💡 Try manual setup:")
        print(f"   !streamlit run {app_file} --server.port 8502")
        return False
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    print(f"\n🎯 Next Steps:")
    print(f"   1. 📝 Generate test queries (if not done)")
    print(f"   2. 🧪 Run manual evaluations")
    print(f"   3. 📊 Analyze results")
    print(f"   4. 📄 Generate academic reports")
    
    print(f"\n📋 Evaluation Workflow:")
    print(f"   • Go to 'Test Queries' page")
    print(f"   • Generate 31 financial test queries")
    print(f"   • Switch to 'Accuracy Evaluation' page")
    print(f"   • Select queries and evaluate responses")
    print(f"   • Score 1-5 (3+ = success)")
    print(f"   • Check 'Analytics Dashboard' for results")
    
    print(f"\n💾 Data Export:")
    print(f"   • Performance metrics: JSON format")
    print(f"   • Accuracy evaluations: JSON format")
    print(f"   • Combined reports: Markdown format")

def main():
    """Main setup function"""
    print_header()
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return
    
    # Step 2: Setup Python path
    base_path = setup_python_path()
    
    # Step 3: Create directories
    if not create_directories():
        print("❌ Setup failed at directory creation")
        return
    
    # Step 4: Generate test queries
    generate_test_queries()  # Non-blocking
    
    # Step 5: Test system
    if not test_system():
        print("⚠️ System test failed, but continuing...")
    
    # Step 6: Setup NGROK
    if not setup_ngrok():
        print("❌ NGROK setup failed")
        print("💡 You can still run locally without NGROK")
        return
    
    print("\n" + "=" * 70)
    print("✅ Setup completed successfully!")
    print("=" * 70)
    
    # Ask user for app type
    print("\n🎯 Choose app type:")
    print("   1. Enhanced App (Performance + Accuracy) - Recommended")
    print("   2. Basic App (Performance only)")
    
    # For Colab, auto-select enhanced
    if "/content" in os.getcwd():
        app_type = "enhanced"
        print("   🌐 Auto-selecting Enhanced App for Colab")
    else:
        choice = input("   Enter choice (1 or 2): ").strip()
        app_type = "enhanced" if choice != "2" else "basic"
    
    print(f"\n⚠️ IMPORTANT: Set your NGROK token before continuing!")
    print(f"   Run this code:")
    print(f"   from pyngrok import ngrok")
    print(f"   ngrok.set_auth_token('YOUR_TOKEN_FROM_NGROK.COM')")
    
    input("\n   Press Enter when token is set...")
    
    # Step 7: Run Streamlit app
    run_streamlit_app(app_type)
    
    # Step 8: Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 