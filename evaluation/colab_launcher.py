"""
🚀 Simple Colab Launcher for Performance Tracker Test
One-command setup for Google Colab testing
"""

import os
import sys
import subprocess

def colab_quick_start():
    """Quick setup and run for Google Colab"""
    print("🚀 Turkish Financial RAG - Performance Tracker Test")
    print("=" * 50)
    
    # Add current directory to Python path
    current_dir = "/content" if "/content" in os.getcwd() else "."
    sys.path.append(current_dir)
    
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[-1]}")
    
    # Install minimal requirements
    requirements = [
        "streamlit",
        "psutil", 
        "pyngrok"
    ]
    
    print("\n📦 Installing requirements...")
    for req in requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req, "-q"], 
                         check=True, capture_output=True)
            print(f"   ✅ {req}")
        except:
            print(f"   ❌ Failed: {req}")
    
    # Create directories
    os.makedirs("evaluation/reports", exist_ok=True)
    print("📁 Directories created")
    
    print("\n🎯 Ready for testing!")
    print("\nNext steps:")
    print("1. Get NGROK token from https://ngrok.com/")
    print("2. Run: from pyngrok import ngrok; ngrok.set_auth_token('YOUR_TOKEN')")
    print("3. Run: !python evaluation/colab_test_setup.py")
    
    return "Setup complete!"

def test_quick():
    """Quick test without Streamlit"""
    print("🧪 Running quick performance test...")
    
    # Add to path
    sys.path.append(".")
    
    try:
        from evaluation.metrics.performance_tracker import get_global_tracker
        import time
        import random
        
        tracker = get_global_tracker()
        
        # Quick tests
        with tracker.track_operation("colab_test"):
            time.sleep(random.uniform(0.5, 1.5))
        
        # Get stats
        stats = tracker.get_system_stats()
        summary = tracker.get_performance_summary()
        
        print(f"✅ Test completed!")
        print(f"📊 CPU: {stats.cpu_percent:.1f}%")
        print(f"🧠 Memory: {stats.memory_percent:.1f}%")
        print(f"⚡ Operations: {summary.get('total_operations', 0)}")
        
        return "Test successful!"
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    # Auto-detect environment and run appropriate setup
    if "/content" in os.getcwd():
        print("🌐 Google Colab detected")
        colab_quick_start()
    else:
        print("💻 Local environment detected")
        test_quick() 