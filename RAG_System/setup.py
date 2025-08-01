#!/usr/bin/env python3
# setup.py - Automated setup script for the RAG system

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Ensure Python 3.8+ is being used."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_postgresql():
    """Check if PostgreSQL is accessible."""
    try:
        import psycopg2
        print("✅ PostgreSQL driver (psycopg2) is installed")
        return True
    except ImportError:
        print("❌ PostgreSQL driver not found. Will be installed with requirements.")
        return False

def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            print("   Available models:")
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    print(f"     - {line.split()[0]}")
            return True
        else:
            print("⚠️  Ollama is installed but not running")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install from https://ollama.ai")
        return False

def create_directories():
    """Create necessary directories."""
    dirs = ['data_cache', 'logs', 'config', 'scripts']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ Created necessary directories")

def setup_env_file():
    """Create .env file from template if it doesn't exist."""
    if not Path('.env').exists():
        if Path('.env.template').exists():
            shutil.copy('.env.template', '.env')
            print("✅ Created .env file from template")
            print("   ⚠️  Please edit .env with your database credentials!")
        else:
            print("❌ .env.template not found!")
            return False
    else:
        print("✅ .env file already exists")
    return True

def install_requirements():
    """Install Python requirements."""
    print("\n📦 Installing Python requirements...")
    requirements = [
        'pandas>=2.0.0',
        'sqlalchemy>=2.0.0',
        'psycopg2-binary>=2.9.0',
        'pyarrow>=12.0.0',
        'transformers>=4.30.0',
        'torch>=2.0.0',
        'langchain>=0.0.200',
        'langchain-community>=0.0.10',
        'sentence-transformers>=2.2.0',
        'python-dotenv>=1.0.0',
        'scikit-learn>=1.3.0',
        'numpy>=1.24.0',
        'gradio>=4.0.0',
        'prometheus-client>=0.17.0',
        'psutil>=5.9.0',
        'pgvector>=0.2.0'
    ]
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + requirements)
        print("✅ All requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install some requirements")
        return False

def setup_postgresql_extension():
    """Provide instructions for setting up pgvector extension."""
    print("\n🗄️  PostgreSQL Setup Instructions:")
    print("   Run these commands in PostgreSQL as superuser:")
    print("   ```sql")
    print("   CREATE EXTENSION IF NOT EXISTS vector;")
    print("   ```")
    print("\n   On Ubuntu/Debian:")
    print("   ```bash")
    print("   sudo apt install postgresql-15-pgvector")
    print("   ```")
    print("\n   On macOS with Homebrew:")
    print("   ```bash")
    print("   brew install pgvector")
    print("   ```")

def download_ollama_model():
    """Provide instructions for downloading Ollama models."""
    print("\n🤖 Ollama Model Setup:")
    print("   To download the recommended model, run:")
    print("   ```bash")
    print("   ollama pull qwen:7b")
    print("   ```")
    print("\n   Other recommended models:")
    print("   - ollama pull llama3:8b")
    print("   - ollama pull mistral:7b")

def create_systemd_service():
    """Create systemd service file for the RAG system."""
    service_content = """[Unit]
Description=RAG System Pipeline
After=network.target postgresql.service

[Service]
Type=simple
User={user}
WorkingDirectory={cwd}
Environment="PATH={path}"
ExecStart={python} {cwd}/pipeline_orchestrator.py --mode auto
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    cwd = os.getcwd()
    user = os.getenv('USER')
    python = sys.executable
    path = os.environ.get('PATH')
    
    service_file = service_content.format(
        user=user,
        cwd=cwd,
        python=python,
        path=path
    )
    
    print("\n🔧 Systemd Service Setup (Optional):")
    print("   To run the pipeline as a service, create:")
    print(f"   /etc/systemd/system/rag-pipeline.service")
    print("\n   Content:")
    print("   " + "\n   ".join(service_file.split('\n')))

def main():
    """Main setup function."""
    print("🚀 RAG System Setup Script")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check dependencies
    postgresql_ok = check_postgresql()
    ollama_ok = check_ollama()
    
    # Create directories
    create_directories()
    
    # Setup environment file
    env_ok = setup_env_file()
    
    # Install requirements
    if input("\n📦 Install Python requirements? (y/n): ").lower() == 'y':
        install_requirements()
    
    # Show additional setup instructions
    setup_postgresql_extension()
    download_ollama_model()
    
    if input("\n🔧 Show systemd service setup? (y/n): ").lower() == 'y':
        create_systemd_service()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print(f"   Python: ✅")
    print(f"   PostgreSQL Driver: {'✅' if postgresql_ok else '⚠️  Needs installation'}")
    print(f"   Ollama: {'✅' if ollama_ok else '❌ Needs installation'}")
    print(f"   Environment File: {'✅' if env_ok else '❌ Needs configuration'}")
    
    print("\n🎯 Next Steps:")
    if not env_ok:
        print("   1. Edit .env file with your database credentials")
    if not ollama_ok:
        print("   2. Install Ollama from https://ollama.ai")
        print("   3. Run: ollama pull qwen:7b")
    print("   4. Ensure PostgreSQL has the 'vector' extension")
    print("   5. Run: python pipeline_orchestrator.py")
    print("\n✨ Setup complete!")

if __name__ == '__main__':
    main()
