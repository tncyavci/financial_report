"""
Google Colab Setup Script for Performance Tracker Test
Sets up environment and runs Streamlit test app with NGROK tunnel
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def install_requirements():
    """Install required packages for the test"""
    print("📦 Installing requirements...")
    
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
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {req}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def setup_ngrok():
    """Setup NGROK for public access"""
    print("🌐 Setting up NGROK...")
    
    try:
        from pyngrok import ngrok
        
        # Kill any existing tunnels
        ngrok.kill()
        
        print("✅ NGROK setup complete!")
        return True
    except ImportError:
        print("❌ NGROK import failed")
        return False

def create_directory_structure():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    dirs = [
        "evaluation/reports",
        "evaluation/metrics",
        "evaluation/data"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   ✅ Created {dir_path}")
    
    return True

def run_streamlit_app():
    """Run the Streamlit test application"""
    print("🚀 Starting Streamlit application...")
    
    try:
        from pyngrok import ngrok
        import threading
        
        # Start Streamlit in background
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "evaluation/streamlit_test_app.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
        
        # Wait for Streamlit to start
        print("   ⏳ Waiting for Streamlit to start...")
        time.sleep(10)
        
        # Create NGROK tunnel
        print("   🌐 Creating NGROK tunnel...")
        public_url = ngrok.connect(8501)
        
        print(f"\n🎉 SUCCESS! Access your Performance Tracker Test at:")
        print(f"   🔗 {public_url}")
        print(f"\n⚡ Features available:")
        print(f"   📄 PDF Processing Simulation")
        print(f"   💬 Query Processing Simulation") 
        print(f"   🧬 Embedding Generation Simulation")
        print(f"   📊 Real-time Performance Metrics")
        print(f"   🖥️ System Resource Monitoring")
        print(f"   📈 Performance Analytics")
        print(f"\n🎮 Test Instructions:")
        print(f"   1. Use sidebar controls to run individual tests")
        print(f"   2. Click 'Run All Tests' for comprehensive testing")
        print(f"   3. Monitor real-time metrics in main dashboard")
        print(f"   4. Download performance data as JSON")
        
        # Keep running
        try:
            streamlit_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping application...")
            streamlit_process.terminate()
            ngrok.kill()
            
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("🔧 Performance Tracker Test Setup for Google Colab")
    print("=" * 60)
    print(f"📅 Setup started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return
    
    # Step 2: Create directories
    if not create_directory_structure():
        print("❌ Setup failed at directory creation")
        return
    
    # Step 3: Setup NGROK
    if not setup_ngrok():
        print("❌ Setup failed at NGROK setup")
        return
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)
    
    # Step 4: Run application
    run_streamlit_app()

if __name__ == "__main__":
    main() 