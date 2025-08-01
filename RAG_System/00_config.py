# 00_config.py
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Logging setup für die gesamte Anwendung
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'logs/rag_system_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables from the .env file
load_dotenv()

# --- Database Configuration ---
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- AI Model Configuration ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

# Vector dimension mapping für verschiedene Modelle
EMBEDDING_MODEL_DIMS = {
    'all-MiniLM-L6-v2': 384,
    'all-mpnet-base-v2': 768,
    'all-MiniLM-L12-v2': 384,
    'paraphrase-multilingual-MiniLM-L12-v2': 384,
    'distiluse-base-multilingual-cased-v1': 512
}

# Automatische Dimension basierend auf Modell
EMBEDDING_DIM = EMBEDDING_MODEL_DIMS.get(EMBEDDING_MODEL_NAME, 384)

# --- Pipeline Configuration ---
PCA_N_COMPONENTS = int(os.getenv("PCA_N_COMPONENTS", "15"))
VECTOR_TABLE_NAME = "rag_documents_final"
SCHEMA_VERSION_TABLE = "rag_schema_versions"
PROCESSING_LOG_TABLE = "rag_processing_log"

# --- Incremental Processing ---
ENABLE_INCREMENTAL = os.getenv("ENABLE_INCREMENTAL", "true").lower() == "true"
INCREMENTAL_BATCH_SIZE = int(os.getenv("INCREMENTAL_BATCH_SIZE", "1000"))
PCA_BATCH_SIZE = int(os.getenv("PCA_BATCH_SIZE", "50000"))

# --- Monitoring Configuration ---
ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8000"))
MONITORING_NAMESPACE = os.getenv("MONITORING_NAMESPACE", "rag_system")

# --- File Paths ---
RAW_DATA_PATH = "data_cache/raw_data.parquet"
CHUNKS_PATH = "data_cache/processed_chunks.pkl"
EMBEDDINGS_PATH = "data_cache/embeddings.npy"
PCA_MODEL_PATH = "data_cache/pca_model.pkl"
INCREMENTAL_STATE_PATH = "data_cache/incremental_state.json"

# Create necessary directories
os.makedirs("data_cache", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configuration check on startup
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME, OLLAMA_MODEL, EMBEDDING_MODEL_NAME]):
    logging.error("Some environment variables are not set! Please check your .env file.")
    exit()

# Log configuration summary
logging.info(f"Configuration loaded successfully:")
logging.info(f"  - Database: {DB_NAME} @ {DB_HOST}")
logging.info(f"  - Embedding Model: {EMBEDDING_MODEL_NAME} (dim={EMBEDDING_DIM})")
logging.info(f"  - Incremental Processing: {ENABLE_INCREMENTAL}")
logging.info(f"  - Monitoring: {ENABLE_MONITORING}")
