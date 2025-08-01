# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import time
import psutil
import threading
from config import ENABLE_MONITORING, PROMETHEUS_PORT, MONITORING_NAMESPACE, logging

class MetricsCollector:
    """
    Collects and exposes metrics for the RAG system via Prometheus.
    Provides insights into performance, errors, and system health.
    """
    
    def __init__(self):
        if not ENABLE_MONITORING:
            # Create dummy metrics that do nothing
            self._create_dummy_metrics()
            return
        
        self._create_metrics()
        self._start_metrics_server()
        self._start_system_metrics_collector()
    
    def _create_metrics(self):
        """Create Prometheus metrics."""
        # Counters
        self.tables_processed = Counter(
            f'{MONITORING_NAMESPACE}_tables_processed_total',
            'Total number of tables processed'
        )
        
        self.rows_extracted = Counter(
            f'{MONITORING_NAMESPACE}_rows_extracted_total',
            'Total number of rows extracted from database'
        )
        
        self.chunks_created = Counter(
            f'{MONITORING_NAMESPACE}_chunks_created_total',
            'Total number of text chunks created'
        )
        
        self.embeddings_generated = Counter(
            f'{MONITORING_NAMESPACE}_embeddings_generated_total',
            'Total number of embeddings generated'
        )
        
        self.vectors_loaded = Counter(
            f'{MONITORING_NAMESPACE}_vectors_loaded_total',
            'Total number of vectors loaded into database'
        )
        
        self.vectors_updated = Counter(
            f'{MONITORING_NAMESPACE}_vectors_updated_total',
            'Total number of vectors updated in database'
        )
        
        self.extraction_errors = Counter(
            f'{MONITORING_NAMESPACE}_extraction_errors_total',
            'Total number of extraction errors',
            ['table_name']
        )
        
        self.queries_processed = Counter(
            f'{MONITORING_NAMESPACE}_queries_processed_total',
            'Total number of RAG queries processed'
        )
        
        # Histograms
        self.extraction_duration = Histogram(
            f'{MONITORING_NAMESPACE}_extraction_duration_seconds',
            'Time spent extracting data',
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.embedding_duration = Histogram(
            f'{MONITORING_NAMESPACE}_embedding_duration_seconds',
            'Time spent generating embeddings',
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.pca_duration = Histogram(
            f'{MONITORING_NAMESPACE}_pca_duration_seconds',
            'Time spent on PCA operations',
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
        )
        
        self.load_duration = Histogram(
            f'{MONITORING_NAMESPACE}_load_duration_seconds',
            'Time spent loading vectors to database',
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.index_creation_duration = Histogram(
            f'{MONITORING_NAMESPACE}_index_creation_duration_seconds',
            'Time spent creating/updating vector index',
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0)
        )
        
        self.query_duration = Histogram(
            f'{MONITORING_NAMESPACE}_query_duration_seconds',
            'Time spent processing RAG queries',
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        # Gauges
        self.total_vectors_in_db = Gauge(
            f'{MONITORING_NAMESPACE}_total_vectors_in_database',
            'Current total number of vectors in database'
        )
        
        self.memory_usage_mb = Gauge(
            f'{MONITORING_NAMESPACE}_memory_usage_megabytes',
            'Current memory usage in MB'
        )
        
        self.cpu_usage_percent = Gauge(
            f'{MONITORING_NAMESPACE}_cpu_usage_percent',
            'Current CPU usage percentage'
        )
        
        self.disk_usage_percent = Gauge(
            f'{MONITORING_NAMESPACE}_disk_usage_percent',
            'Current disk usage percentage'
        )
        
        # Info
        self.system_info = Info(
            f'{MONITORING_NAMESPACE}_system',
            'System information'
        )
        
        self.system_info.info({
            'python_version': str(psutil.Process().exe()),
            'cpu_count': str(psutil.cpu_count()),
            'total_memory_gb': str(round(psutil.virtual_memory().total / (1024**3), 2))
        })
    
    def _create_dummy_metrics(self):
        """Create dummy metrics when monitoring is disabled."""
        class DummyMetric:
            def inc(self, *args, **kwargs): pass
            def dec(self, *args, **kwargs): pass
            def set(self, *args, **kwargs): pass
            def observe(self, *args, **kwargs): pass
            def time(self): return DummyTimer()
            def info(self, *args, **kwargs): pass
        
        class DummyTimer:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        
        # Set all metrics to dummy instances
        for attr in ['tables_processed', 'rows_extracted', 'chunks_created',
                    'embeddings_generated', 'vectors_loaded', 'vectors_updated',
                    'extraction_errors', 'queries_processed', 'extraction_duration',
                    'embedding_duration', 'pca_duration', 'load_duration',
                    'index_creation_duration', 'query_duration', 'total_vectors_in_db',
                    'memory_usage_mb', 'cpu_usage_percent', 'disk_usage_percent',
                    'system_info']:
            setattr(self, attr, DummyMetric())
    
    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server."""
        try:
            start_http_server(PROMETHEUS_PORT)
            logging.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
        except Exception as e:
            logging.error(f"Failed to start metrics server: {e}")
    
    def _start_system_metrics_collector(self):
        """Start background thread to collect system metrics."""
        def collect_system_metrics():
            while True:
                try:
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.memory_usage_mb.set(memory.used / (1024 * 1024))
                    
                    # CPU usage
                    self.cpu_usage_percent.set(psutil.cpu_percent(interval=1))
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.disk_usage_percent.set(disk.percent)
                    
                except Exception as e:
                    logging.error(f"Error collecting system metrics: {e}")
                
                time.sleep(30)  # Collect every 30 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def update_db_metrics(self, engine):
        """Update database-related metrics."""
        if not ENABLE_MONITORING:
            return
        
        try:
            with engine.connect() as conn:
                from config import VECTOR_TABLE_NAME
                result = conn.execute(
                    f"SELECT COUNT(*) FROM {VECTOR_TABLE_NAME}"
                ).scalar()
                self.total_vectors_in_db.set(result)
        except Exception as e:
            logging.error(f"Error updating DB metrics: {e}")

# Singleton instance
_metrics_collector = None

def get_metrics():
    """Get or create the singleton metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

# Convenience export
MetricsCollector = get_metrics
