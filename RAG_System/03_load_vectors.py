# 03_load_vectors.py
import pickle
import numpy as np
import hashlib
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values
from monitoring import MetricsCollector
from config import (
    DB_URI, CHUNKS_PATH, EMBEDDINGS_PATH, 
    VECTOR_TABLE_NAME, EMBEDDING_DIM, logging
)

class IncrementalVectorLoader:
    """
    Handles efficient loading of vectors into PostgreSQL with support for
    incremental updates, deduplication, and optimized indexing.
    """
    
    def __init__(self):
        self.engine = create_engine(DB_URI)
        self.metrics = MetricsCollector()
        self._init_database()
    
    def _init_database(self):
        """Initialize database with vector extension and optimized table."""
        with self.engine.connect() as conn:
            logging.info("Initializing vector database...")
            
            # Enable vector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Create table if not exists with content hash for deduplication
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {VECTOR_TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_hash VARCHAR(64) UNIQUE NOT NULL,
                    source_table VARCHAR(255) NOT NULL,
                    source_id BIGINT NOT NULL,
                    semantic_topic VARCHAR(50),
                    embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER DEFAULT 1,
                    INDEX idx_source (source_table, source_id),
                    INDEX idx_topic (semantic_topic),
                    INDEX idx_updated (updated_at)
                );
            """))
            
            # Create update trigger
            conn.execute(text(f"""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    NEW.version = OLD.version + 1;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
                
                DROP TRIGGER IF EXISTS update_{VECTOR_TABLE_NAME}_updated_at ON {VECTOR_TABLE_NAME};
                
                CREATE TRIGGER update_{VECTOR_TABLE_NAME}_updated_at 
                BEFORE UPDATE ON {VECTOR_TABLE_NAME} 
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            """))
            
            conn.commit()
            logging.info("Database initialization complete")
    
    def load_vectors(self, mode='incremental'):
        """
        Load vectors into database with specified mode.
        Args:
            mode: 'incremental' (default), 'full_reload', or 'update_only'
        """
        logging.info(f"Phase 3: Loading vectors in {mode} mode...")
        
        # Load chunks and embeddings
        with open(CHUNKS_PATH, 'rb') as f:
            chunks = pickle.load(f)
        embeddings = np.load(EMBEDDINGS_PATH)
        
        if mode == 'full_reload':
            self._full_reload(chunks, embeddings)
        elif mode == 'update_only':
            self._update_existing(chunks, embeddings)
        else:  # incremental
            self._incremental_load(chunks, embeddings)
        
        # Update index
        self._update_vector_index()
        
        # Log statistics
        self._log_statistics()
    
    def _full_reload(self, chunks, embeddings):
        """Perform full reload by truncating and reloading all data."""
        logging.info("Performing full data reload...")
        
        with self.engine.connect() as conn:
            # Truncate table
            conn.execute(text(f"TRUNCATE TABLE {VECTOR_TABLE_NAME} RESTART IDENTITY;"))
            conn.commit()
        
        # Load all data
        self._batch_upsert(chunks, embeddings, update_existing=False)
    
    def _incremental_load(self, chunks, embeddings):
        """Load new vectors and update existing ones if changed."""
        logging.info("Performing incremental load...")
        
        # Get existing content hashes
        existing_hashes = self._get_existing_hashes()
        
        # Separate new and potentially updated chunks
        new_data = []
        update_data = []
        
        for chunk, embedding in zip(chunks, embeddings):
            content_hash = self._calculate_content_hash(chunk.page_content)
            
            if content_hash not in existing_hashes:
                new_data.append((chunk, embedding, content_hash))
            else:
                # Check if metadata changed
                existing_meta = existing_hashes[content_hash]
                if (chunk.metadata.get('semantic_topic') != existing_meta.get('semantic_topic') or
                    chunk.metadata.get('source_id') != existing_meta.get('source_id')):
                    update_data.append((chunk, embedding, content_hash))
        
        logging.info(f"Found {len(new_data)} new vectors and {len(update_data)} to update")
        
        # Insert new data
        if new_data:
            self._batch_insert(new_data)
        
        # Update existing data
        if update_data:
            self._batch_update(update_data)
    
    def _update_existing(self, chunks, embeddings):
        """Update only existing vectors (useful for PCA re-computation)."""
        logging.info("Updating existing vectors only...")
        
        update_data = []
        for chunk, embedding in zip(chunks, embeddings):
            content_hash = self._calculate_content_hash(chunk.page_content)
            update_data.append((chunk, embedding, content_hash))
        
        self._batch_update(update_data)
    
    def _calculate_content_hash(self, content):
        """Calculate SHA256 hash of content for deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _get_existing_hashes(self):
        """Get existing content hashes with metadata."""
        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT content_hash, source_table, source_id, semantic_topic
                FROM {VECTOR_TABLE_NAME}
            """))
            
            return {
                row[0]: {
                    'source_table': row[1],
                    'source_id': row[2],
                    'semantic_topic': row[3]
                }
                for row in result
            }
    
    def _batch_insert(self, data):
        """Efficiently insert new vectors in batches."""
        logging.info(f"Inserting {len(data)} new vectors...")
        
        insert_data = [
            (
                chunk.page_content,
                content_hash,
                chunk.metadata.get('source_table'),
                chunk.metadata.get('source_id'),
                chunk.metadata.get('semantic_topic'),
                embedding
            )
            for chunk, embedding, content_hash in data
        ]
        
        raw_conn = self.engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                with self.metrics.load_duration.time():
                    execute_values(
                        cur,
                        f"""INSERT INTO {VECTOR_TABLE_NAME} 
                            (content, content_hash, source_table, source_id, 
                             semantic_topic, embedding)
                            VALUES %s
                            ON CONFLICT (content_hash) DO NOTHING""",
                        insert_data,
                        page_size=500
                    )
                raw_conn.commit()
                self.metrics.vectors_loaded.inc(len(data))
        finally:
            raw_conn.close()
    
    def _batch_update(self, data):
        """Efficiently update existing vectors."""
        logging.info(f"Updating {len(data)} existing vectors...")
        
        raw_conn = self.engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                with self.metrics.load_duration.time():
                    for chunk, embedding, content_hash in data:
                        cur.execute(f"""
                            UPDATE {VECTOR_TABLE_NAME}
                            SET source_table = %s,
                                source_id = %s,
                                semantic_topic = %s,
                                embedding = %s
                            WHERE content_hash = %s
                        """, (
                            chunk.metadata.get('source_table'),
                            chunk.metadata.get('source_id'),
                            chunk.metadata.get('semantic_topic'),
                            embedding,
                            content_hash
                        ))
                raw_conn.commit()
                self.metrics.vectors_updated.inc(len(data))
        finally:
            raw_conn.close()
    
    def _batch_upsert(self, chunks, embeddings, update_existing=True):
        """Perform batch upsert operation."""
        data_to_upsert = []
        
        for chunk, embedding in zip(chunks, embeddings):
            content_hash = self._calculate_content_hash(chunk.page_content)
            data_to_upsert.append((
                chunk.page_content,
                content_hash,
                chunk.metadata.get('source_table'),
                chunk.metadata.get('source_id'),
                chunk.metadata.get('semantic_topic'),
                embedding
            ))
        
        raw_conn = self.engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                if update_existing:
                    conflict_action = """
                        ON CONFLICT (content_hash) 
                        DO UPDATE SET
                            source_table = EXCLUDED.source_table,
                            source_id = EXCLUDED.source_id,
                            semantic_topic = EXCLUDED.semantic_topic,
                            embedding = EXCLUDED.embedding
                    """
                else:
                    conflict_action = "ON CONFLICT (content_hash) DO NOTHING"
                
                with self.metrics.load_duration.time():
                    execute_values(
                        cur,
                        f"""INSERT INTO {VECTOR_TABLE_NAME} 
                            (content, content_hash, source_table, source_id, 
                             semantic_topic, embedding)
                            VALUES %s
                            {conflict_action}""",
                        data_to_upsert,
                        page_size=500
                    )
                raw_conn.commit()
                self.metrics.vectors_loaded.inc(len(data_to_upsert))
        finally:
            raw_conn.close()
    
    def _update_vector_index(self):
        """Create or update vector index with progress monitoring."""
        logging.info("Updating vector index...")
        
        with self.engine.connect() as conn:
            # Check if index exists
            result = conn.execute(text(f"""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = '{VECTOR_TABLE_NAME}' 
                AND indexname LIKE '%embedding%'
            """)).fetchone()
            
            if result:
                # Reindex if exists
                logging.info("Reindexing existing vector index...")
                conn.execute(text(f"REINDEX INDEX {result[0]};"))
            else:
                # Create new index
                logging.info("Creating new HNSW vector index...")
                with self.metrics.index_creation_duration.time():
                    conn.execute(text(f"""
                        CREATE INDEX idx_{VECTOR_TABLE_NAME}_embedding 
                        ON {VECTOR_TABLE_NAME} 
                        USING hnsw (embedding vector_l2_ops)
                        WITH (m = 16, ef_construction = 64);
                    """))
            
            conn.commit()
        
        logging.info("Vector index update complete")
    
    def _log_statistics(self):
        """Log database statistics."""
        with self.engine.connect() as conn:
            stats = conn.execute(text(f"""
                SELECT 
                    COUNT(*) as total_vectors,
                    COUNT(DISTINCT source_table) as unique_tables,
                    COUNT(DISTINCT semantic_topic) as unique_topics,
                    MAX(updated_at) as last_update
                FROM {VECTOR_TABLE_NAME}
            """)).fetchone()
            
            logging.info(f"Database statistics:")
            logging.info(f"  - Total vectors: {stats[0]:,}")
            logging.info(f"  - Unique source tables: {stats[1]}")
            logging.info(f"  - Unique semantic topics: {stats[2]}")
            logging.info(f"  - Last update: {stats[3]}")
    
    def cleanup_old_versions(self, keep_versions=3):
        """Remove old versions of updated vectors to save space."""
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                DELETE FROM {VECTOR_TABLE_NAME}
                WHERE (content_hash, version) NOT IN (
                    SELECT content_hash, version
                    FROM (
                        SELECT content_hash, version,
                               ROW_NUMBER() OVER (PARTITION BY content_hash 
                                                ORDER BY version DESC) as rn
                        FROM {VECTOR_TABLE_NAME}
                    ) t
                    WHERE rn <= :keep_versions
                )
            """), {'keep_versions': keep_versions})
            conn.commit()

if __name__ == '__main__':
    loader = IncrementalVectorLoader()
    loader.load_vectors(mode='incremental')
