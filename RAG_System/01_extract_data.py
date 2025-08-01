# 01_extract_data.py
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from schema_tracker import SchemaTracker
from monitoring import MetricsCollector
from config import (
    DB_URI, RAW_DATA_PATH, ENABLE_INCREMENTAL, 
    INCREMENTAL_BATCH_SIZE, logging
)

class IncrementalDataExtractor:
    """
    Handles both full and incremental data extraction from PostgreSQL.
    Tracks changes and only processes new/modified data when possible.
    """
    
    def __init__(self):
        self.engine = create_engine(DB_URI)
        self.schema_tracker = SchemaTracker()
        self.metrics = MetricsCollector()
        
    def extract_and_save(self, force_full_reload=False):
        """
        Main extraction method with incremental support.
        Args:
            force_full_reload: Force full data reload ignoring incremental state
        """
        logging.info("Phase 1: Starting data extraction from PostgreSQL...")
        
        # Check for schema changes
        schema_changed, changes = self.schema_tracker.check_and_update_schema()
        
        if schema_changed and not force_full_reload:
            logging.info("Schema changes detected. Evaluating impact...")
            force_full_reload = self._should_force_reload(changes)
        
        if force_full_reload or not ENABLE_INCREMENTAL:
            logging.info("Performing FULL data extraction...")
            self._full_extraction()
        else:
            logging.info("Performing INCREMENTAL data extraction...")
            self._incremental_extraction()
    
    def _should_force_reload(self, changes):
        """Determine if schema changes require full reload."""
        # Force reload if tables were removed or columns were modified
        if changes.get('removed_tables') or any(
            mods.get('modified_columns') or mods.get('removed_columns') 
            for mods in changes.get('modified_tables', {}).values()
        ):
            return True
        return False
    
    def _full_extraction(self):
        """Perform full data extraction from all tables."""
        inspector = inspect(self.engine)
        table_names = [t for t in inspector.get_table_names() if not t.startswith('rag_')]
        
        all_data_frames = []
        
        with self.engine.connect() as connection:
            for table_name in table_names:
                try:
                    self.metrics.tables_processed.inc()
                    logging.info(f"Processing table: '{table_name}'")
                    
                    # Get text columns
                    query = text(f"SELECT * FROM \"{table_name}\" LIMIT 1;")
                    df_info = pd.read_sql(query, connection)
                    text_cols = [col for col in df_info.columns 
                               if pd.api.types.is_string_dtype(df_info[col])]
                    
                    if not text_cols:
                        logging.warning(f"No text columns found in table '{table_name}'. Skipping.")
                        continue
                    
                    # Process in chunks
                    chunks = self._extract_table_chunks(
                        connection, table_name, text_cols, incremental=False
                    )
                    all_data_frames.extend(chunks)
                    
                except Exception as e:
                    self.metrics.extraction_errors.inc()
                    logging.error(f"Error loading table '{table_name}': {e}")
                    self.schema_tracker.log_processing(
                        table_name, 0, status='error', error_message=str(e)
                    )
        
        self._save_extracted_data(all_data_frames)
    
    def _incremental_extraction(self):
        """Perform incremental extraction only for new/modified data."""
        inspector = inspect(self.engine)
        table_names = [t for t in inspector.get_table_names() if not t.startswith('rag_')]
        
        # Load existing data
        existing_data = []
        try:
            existing_df = pd.read_parquet(RAW_DATA_PATH)
            existing_data = [existing_df]
            logging.info(f"Loaded {len(existing_df)} existing records")
        except FileNotFoundError:
            logging.info("No existing data found, performing full extraction")
            self._full_extraction()
            return
        
        new_data_frames = []
        
        with self.engine.connect() as connection:
            for table_name in table_names:
                try:
                    self.metrics.tables_processed.inc()
                    logging.info(f"Checking table '{table_name}' for updates...")
                    
                    # Get text columns
                    query = text(f"SELECT * FROM \"{table_name}\" LIMIT 1;")
                    df_info = pd.read_sql(query, connection)
                    text_cols = [col for col in df_info.columns 
                               if pd.api.types.is_string_dtype(df_info[col])]
                    
                    if not text_cols:
                        continue
                    
                    # Get incremental query
                    inc_query, params, meta = self.schema_tracker.get_incremental_query(
                        table_name, text_cols
                    )
                    
                    # Process incremental chunks
                    chunks = self._extract_table_chunks(
                        connection, table_name, text_cols, 
                        incremental=True, query=inc_query, params=params, meta=meta
                    )
                    
                    if chunks:
                        new_data_frames.extend(chunks)
                        logging.info(f"Found {sum(len(c) for c in chunks)} new records in '{table_name}'")
                    
                except Exception as e:
                    self.metrics.extraction_errors.inc()
                    logging.error(f"Error in incremental load for '{table_name}': {e}")
        
        # Combine existing and new data
        if new_data_frames:
            all_data = existing_data + new_data_frames
            self._save_extracted_data(all_data, is_update=True)
        else:
            logging.info("No new data found in incremental extraction")
    
    def _extract_table_chunks(self, connection, table_name, text_cols, 
                            incremental=False, query=None, params=None, meta=None):
        """Extract data from a table in chunks."""
        chunks = []
        
        if not incremental:
            # Full extraction query
            concatenated_cols = " || ' ' || ".join(
                [f"COALESCE(\"{col}\"::text, '')" for col in text_cols]
            )
            query = text(f"""
                SELECT {concatenated_cols} AS combined_text 
                FROM \"{table_name}\"
            """)
            params = {}
        else:
            query = text(query)
        
        # Stream data in chunks
        chunk_iterator = pd.read_sql_query(
            query, connection, params=params, 
            chunksize=INCREMENTAL_BATCH_SIZE
        )
        
        last_id = None
        total_rows = 0
        
        for i, chunk_df in enumerate(chunk_iterator):
            with self.metrics.extraction_duration.time():
                chunk_df.dropna(subset=['combined_text'], inplace=True)
                chunk_df['source_table'] = table_name
                chunk_df['source_id'] = range(
                    i * INCREMENTAL_BATCH_SIZE, 
                    (i * INCREMENTAL_BATCH_SIZE) + len(chunk_df)
                )
                
                if incremental and meta and meta.get('id_col') and 'row_id' in chunk_df.columns:
                    last_id = chunk_df['row_id'].max()
                
                chunks.append(chunk_df)
                total_rows += len(chunk_df)
                self.metrics.rows_extracted.inc(len(chunk_df))
                
                logging.info(f"  ... {len(chunk_df)} rows loaded from '{table_name}' (Block {i+1})")
        
        # Log successful processing
        if incremental and meta:
            self.schema_tracker.log_processing(
                table_name, total_rows, status='success',
                last_id=last_id, timestamp_col=meta.get('timestamp_col')
            )
        
        return chunks
    
    def _save_extracted_data(self, data_frames, is_update=False):
        """Save extracted data to Parquet file."""
        if not data_frames:
            logging.error("No data to save")
            return
        
        if isinstance(data_frames[0], pd.DataFrame):
            final_df = pd.concat(data_frames, ignore_index=True)
        else:
            # Flatten list of lists
            all_frames = []
            for frame_list in data_frames:
                if isinstance(frame_list, list):
                    all_frames.extend(frame_list)
                else:
                    all_frames.append(frame_list)
            final_df = pd.concat(all_frames, ignore_index=True)
        
        # Remove duplicates if updating
        if is_update:
            final_df = final_df.drop_duplicates(
                subset=['source_table', 'source_id'], 
                keep='last'
            )
        
        final_df.to_parquet(RAW_DATA_PATH, index=False)
        logging.info(
            f"Phase 1 complete. Data saved to '{RAW_DATA_PATH}'. "
            f"Total unique records: {len(final_df)}"
        )

if __name__ == '__main__':
    extractor = IncrementalDataExtractor()
    extractor.extract_and_save()
