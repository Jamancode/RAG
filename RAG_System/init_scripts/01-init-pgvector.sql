-- init_scripts/01-init-pgvector.sql
-- PostgreSQL initialization script for RAG system

-- Create vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create additional useful extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search optimization

-- Create dedicated schema for RAG system
CREATE SCHEMA IF NOT EXISTS rag_system;

-- Set search path
SET search_path TO rag_system, public;

-- Create optimized configuration for vector operations
ALTER SYSTEM SET shared_buffers = '512MB';
ALTER SYSTEM SET effective_cache_size = '2GB';
ALTER SYSTEM SET work_mem = '32MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET random_page_cost = 1.1;  -- For SSD storage
ALTER SYSTEM SET effective_io_concurrency = 200;  -- For SSD storage

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_gin_content ON rag_documents_final USING gin(to_tsvector('english', content));

-- Create materialized view for statistics (optional)
CREATE MATERIALIZED VIEW IF NOT EXISTS rag_system.vector_statistics AS
SELECT 
    COUNT(*) as total_vectors,
    COUNT(DISTINCT source_table) as unique_sources,
    COUNT(DISTINCT semantic_topic) as unique_topics,
    AVG(LENGTH(content)) as avg_content_length,
    MAX(created_at) as last_created,
    MAX(updated_at) as last_updated
FROM rag_documents_final;

-- Create function to refresh statistics
CREATE OR REPLACE FUNCTION rag_system.refresh_statistics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY rag_system.vector_statistics;
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions
GRANT ALL ON SCHEMA rag_system TO raguser;
GRANT ALL ON ALL TABLES IN SCHEMA rag_system TO raguser;
GRANT ALL ON ALL SEQUENCES IN SCHEMA rag_system TO raguser;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA rag_system TO raguser;

-- Create performance monitoring table
CREATE TABLE IF NOT EXISTS rag_system.query_performance (
    id SERIAL PRIMARY KEY,
    query_text TEXT,
    embedding_time_ms FLOAT,
    search_time_ms FLOAT,
    total_time_ms FLOAT,
    result_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on performance table
CREATE INDEX idx_query_performance_created ON rag_system.query_performance(created_at DESC);

-- Notification for completion
DO $$
BEGIN
    RAISE NOTICE 'RAG System database initialization completed successfully';
END $$;