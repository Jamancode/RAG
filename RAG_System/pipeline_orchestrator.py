# pipeline_orchestrator.py
"""
Main orchestrator for the RAG pipeline.
Handles full and incremental processing with proper error handling and monitoring.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Import all pipeline components
from config import logging, ENABLE_INCREMENTAL
from schema_tracker import SchemaTracker
from extract_data import IncrementalDataExtractor
from transform_data import IncrementalTransformer
from load_vectors import IncrementalVectorLoader
from monitoring import MetricsCollector

class PipelineOrchestrator:
    """
    Orchestrates the entire RAG pipeline with support for different modes:
    - full: Complete rebuild of all data
    - incremental: Process only new/changed data
    - update-embeddings: Re-generate embeddings with existing chunks
    - update-topics: Re-calculate PCA topics only
    """
    
    def __init__(self):
        self.schema_tracker = SchemaTracker()
        self.extractor = IncrementalDataExtractor()
        self.transformer = IncrementalTransformer()
        self.loader = IncrementalVectorLoader()
        self.metrics = MetricsCollector()
        
    def run_pipeline(self, mode='auto', force=False):
        """
        Run the complete pipeline in specified mode.
        
        Args:
            mode: 'auto', 'full', 'incremental', 'update-embeddings', 'update-topics'
            force: Force operation even if no changes detected
        """
        start_time = time.time()
        logging.info(f"{'='*60}")
        logging.info(f"Starting RAG Pipeline - Mode: {mode}")
        logging.info(f"Timestamp: {datetime.now().isoformat()}")
        logging.info(f"{'='*60}")
        
        try:
            # Determine actual mode
            if mode == 'auto':
                mode = self._determine_mode(force)
                logging.info(f"Auto-detected mode: {mode}")
            
            # Execute pipeline based on mode
            if mode == 'full':
                self._run_full_pipeline()
            elif mode == 'incremental':
                self._run_incremental_pipeline()
            elif mode == 'update-embeddings':
                self._update_embeddings_only()
            elif mode == 'update-topics':
                self._update_topics_only()
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Update metrics
            self.metrics.update_db_metrics(self.loader.engine)
            
            # Log success
            duration = time.time() - start_time
            logging.info(f"{'='*60}")
            logging.info(f"Pipeline completed successfully in {duration:.2f} seconds")
            logging.info(f"{'='*60}")
            
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def _determine_mode(self, force):
        """Automatically determine the best mode based on current state."""
        # Check for schema changes
        schema_changed, changes = self.schema_tracker.check_and_update_schema()
        
        if schema_changed and any([
            changes.get('removed_tables'),
            any(m.get('removed_columns') for m in changes.get('modified_tables', {}).values())
        ]):
            logging.info("Major schema changes detected - full rebuild required")
            return 'full'
        
        # Check if this is first run
        if not Path(self.transformer.CHUNKS_PATH).exists():
            logging.info("No existing data found - full pipeline required")
            return 'full'
        
        # Check if incremental is disabled
        if not ENABLE_INCREMENTAL:
            logging.info("Incremental processing disabled - using full mode")
            return 'full'
        
        # Default to incremental
        return 'incremental'
    
    def _run_full_pipeline(self):
        """Run complete pipeline from scratch."""
        logging.info("Running FULL pipeline...")
        
        # Step 1: Extract all data
        logging.info("\n--- Step 1/3: Data Extraction ---")
        self.extractor.extract_and_save(force_full_reload=True)
        
        # Step 2: Transform data
        logging.info("\n--- Step 2/3: Data Transformation ---")
        self.transformer.transform_data(incremental=False)
        
        # Step 3: Load vectors
        logging.info("\n--- Step 3/3: Vector Loading ---")
        self.loader.load_vectors(mode='full_reload')
    
    def _run_incremental_pipeline(self):
        """Run incremental updates only."""
        logging.info("Running INCREMENTAL pipeline...")
        
        # Step 1: Extract new/changed data
        logging.info("\n--- Step 1/3: Incremental Data Extraction ---")
        self.extractor.extract_and_save(force_full_reload=False)
        
        # Step 2: Transform new data
        logging.info("\n--- Step 2/3: Incremental Transformation ---")
        self.transformer.transform_data(incremental=True)
        
        # Step 3: Load new vectors
        logging.info("\n--- Step 3/3: Incremental Vector Loading ---")
        self.loader.load_vectors(mode='incremental')
    
    def _update_embeddings_only(self):
        """Re-generate embeddings for existing chunks."""
        logging.info("Re-generating embeddings only...")
        
        # Skip extraction, just transform
        logging.info("\n--- Regenerating Embeddings ---")
        self.transformer.transform_data(incremental=False)
        
        # Update existing vectors
        logging.info("\n--- Updating Vectors ---")
        self.loader.load_vectors(mode='update_only')
    
    def _update_topics_only(self):
        """Re-calculate PCA topics without changing embeddings."""
        logging.info("Re-calculating PCA topics only...")
        
        # This would require a specialized method in transformer
        # For now, we'll do a full transform
        logging.warning("Topic-only update not yet implemented, running full transform")
        self._update_embeddings_only()
    
    def cleanup_old_data(self, days=30):
        """Clean up old data and optimize database."""
        logging.info(f"Cleaning up data older than {days} days...")
        
        # Clean old vector versions
        self.loader.cleanup_old_versions(keep_versions=3)
        
        # Vacuum database
        with self.loader.engine.connect() as conn:
            conn.execute("VACUUM ANALYZE")
        
        logging.info("Cleanup complete")
    
    def get_pipeline_status(self):
        """Get current pipeline status and statistics."""
        status = {
            'last_extraction': None,
            'last_transformation': None,
            'total_vectors': 0,
            'schema_version': None,
            'incremental_enabled': ENABLE_INCREMENTAL
        }
        
        # Get extraction status
        try:
            with self.extractor.engine.connect() as conn:
                result = conn.execute("""
                    SELECT MAX(last_processed_at) 
                    FROM rag_processing_log 
                    WHERE status = 'success'
                """).scalar()
                status['last_extraction'] = result
        except:
            pass
        
        # Get vector count
        try:
            with self.loader.engine.connect() as conn:
                result = conn.execute(f"""
                    SELECT COUNT(*) FROM {self.loader.VECTOR_TABLE_NAME}
                """).scalar()
                status['total_vectors'] = result
        except:
            pass
        
        # Get schema version
        try:
            with self.schema_tracker.engine.connect() as conn:
                result = conn.execute("""
                    SELECT created_at 
                    FROM rag_schema_versions 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """).scalar()
                status['schema_version'] = result
        except:
            pass
        
        return status

def main():
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description='RAG Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in auto mode (detects what needs to be done)
  python pipeline_orchestrator.py
  
  # Force full rebuild
  python pipeline_orchestrator.py --mode full
  
  # Run incremental update
  python pipeline_orchestrator.py --mode incremental
  
  # Re-generate embeddings with new model
  python pipeline_orchestrator.py --mode update-embeddings
  
  # Show current status
  python pipeline_orchestrator.py --status
  
  # Clean up old data
  python pipeline_orchestrator.py --cleanup
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['auto', 'full', 'incremental', 'update-embeddings', 'update-topics'],
        default='auto',
        help='Pipeline execution mode (default: auto)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force operation even if no changes detected'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show pipeline status and exit'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up old data and optimize database'
    )
    
    parser.add_argument(
        '--cleanup-days',
        type=int,
        default=30,
        help='Days of data to keep during cleanup (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Handle different commands
    if args.status:
        status = orchestrator.get_pipeline_status()
        print("\nPipeline Status:")
        print(f"  Last Extraction: {status['last_extraction']}")
        print(f"  Total Vectors: {status['total_vectors']:,}")
        print(f"  Schema Version: {status['schema_version']}")
        print(f"  Incremental Mode: {'Enabled' if status['incremental_enabled'] else 'Disabled'}")
        
    elif args.cleanup:
        orchestrator.cleanup_old_data(args.cleanup_days)
        
    else:
        orchestrator.run_pipeline(mode=args.mode, force=args.force)

if __name__ == '__main__':
    main()
