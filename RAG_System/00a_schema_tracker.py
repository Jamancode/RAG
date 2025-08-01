# 00a_schema_tracker.py
import json
import hashlib
from datetime import datetime
from sqlalchemy import create_engine, text, inspect
from config import DB_URI, SCHEMA_VERSION_TABLE, PROCESSING_LOG_TABLE, logging

class SchemaTracker:
    """
    Tracks database schema changes and maintains version history.
    Enables incremental processing by detecting what has changed.
    """
    
    def __init__(self):
        self.engine = create_engine(DB_URI)
        self._init_tracking_tables()
        
    def _init_tracking_tables(self):
        """Create tracking tables if they don't exist."""
        with self.engine.connect() as conn:
            # Schema version tracking table
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA_VERSION_TABLE} (
                    id SERIAL PRIMARY KEY,
                    version_hash VARCHAR(64) UNIQUE NOT NULL,
                    schema_snapshot JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    changes JSONB
                );
            """))
            
            # Processing log for incremental updates
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {PROCESSING_LOG_TABLE} (
                    id SERIAL PRIMARY KEY,
                    table_name VARCHAR(255) NOT NULL,
                    last_processed_at TIMESTAMP,
                    last_max_id BIGINT,
                    last_modified_field VARCHAR(255),
                    row_count BIGINT,
                    status VARCHAR(50),
                    error_message TEXT,
                    UNIQUE(table_name)
                );
            """))
            conn.commit()
            
    def get_current_schema(self):
        """Get current database schema as a structured dict."""
        inspector = inspect(self.engine)
        schema = {}
        
        for table_name in inspector.get_table_names():
            if table_name.startswith('rag_'):  # Skip our own tables
                continue
                
            columns = {}
            for col in inspector.get_columns(table_name):
                columns[col['name']] = {
                    'type': str(col['type']),
                    'nullable': col['nullable'],
                    'default': str(col['default']) if col['default'] else None
                }
            
            # Get indexes
            indexes = []
            for idx in inspector.get_indexes(table_name):
                indexes.append({
                    'name': idx['name'],
                    'columns': idx['column_names'],
                    'unique': idx['unique']
                })
            
            # Get primary keys
            pk = inspector.get_pk_constraint(table_name)
            
            schema[table_name] = {
                'columns': columns,
                'indexes': indexes,
                'primary_key': pk['constrained_columns'] if pk else []
            }
            
        return schema
    
    def _calculate_schema_hash(self, schema):
        """Calculate a hash of the schema for change detection."""
        schema_json = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()
    
    def detect_changes(self, old_schema, new_schema):
        """Detect what has changed between two schema versions."""
        changes = {
            'added_tables': [],
            'removed_tables': [],
            'modified_tables': {}
        }
        
        old_tables = set(old_schema.keys())
        new_tables = set(new_schema.keys())
        
        # Find added/removed tables
        changes['added_tables'] = list(new_tables - old_tables)
        changes['removed_tables'] = list(old_tables - new_tables)
        
        # Find modified tables
        for table in old_tables & new_tables:
            table_changes = {
                'added_columns': [],
                'removed_columns': [],
                'modified_columns': []
            }
            
            old_cols = set(old_schema[table]['columns'].keys())
            new_cols = set(new_schema[table]['columns'].keys())
            
            table_changes['added_columns'] = list(new_cols - old_cols)
            table_changes['removed_columns'] = list(old_cols - new_cols)
            
            # Check for modified columns
            for col in old_cols & new_cols:
                if old_schema[table]['columns'][col] != new_schema[table]['columns'][col]:
                    table_changes['modified_columns'].append(col)
            
            if any(table_changes.values()):
                changes['modified_tables'][table] = table_changes
                
        return changes
    
    def check_and_update_schema(self):
        """
        Check if schema has changed and update version if needed.
        Returns: (has_changed, changes_dict)
        """
        current_schema = self.get_current_schema()
        current_hash = self._calculate_schema_hash(current_schema)
        
        with self.engine.connect() as conn:
            # Get last schema version
            result = conn.execute(text(f"""
                SELECT version_hash, schema_snapshot 
                FROM {SCHEMA_VERSION_TABLE} 
                ORDER BY created_at DESC 
                LIMIT 1
            """)).fetchone()
            
            if not result:
                # First run, save initial schema
                conn.execute(text(f"""
                    INSERT INTO {SCHEMA_VERSION_TABLE} (version_hash, schema_snapshot, changes)
                    VALUES (:hash, :snapshot, :changes)
                """), {
                    'hash': current_hash,
                    'snapshot': json.dumps(current_schema),
                    'changes': json.dumps({'initial': True})
                })
                conn.commit()
                logging.info("Initial schema version saved")
                return True, {'initial': True}
            
            last_hash, last_schema = result
            
            if last_hash == current_hash:
                logging.info("Schema unchanged")
                return False, {}
            
            # Schema has changed
            last_schema_dict = json.loads(last_schema)
            changes = self.detect_changes(last_schema_dict, current_schema)
            
            # Save new version
            conn.execute(text(f"""
                INSERT INTO {SCHEMA_VERSION_TABLE} (version_hash, schema_snapshot, changes)
                VALUES (:hash, :snapshot, :changes)
            """), {
                'hash': current_hash,
                'snapshot': json.dumps(current_schema),
                'changes': json.dumps(changes)
            })
            conn.commit()
            
            logging.info(f"Schema changes detected and saved: {changes}")
            return True, changes
    
    def get_incremental_query(self, table_name, text_cols):
        """
        Generate SQL query for incremental data extraction.
        Uses timestamp or ID-based incremental loading.
        """
        with self.engine.connect() as conn:
            # Get last processing info
            result = conn.execute(text(f"""
                SELECT last_processed_at, last_max_id, last_modified_field
                FROM {PROCESSING_LOG_TABLE}
                WHERE table_name = :table_name AND status = 'success'
            """), {'table_name': table_name}).fetchone()
            
            # Build base query with text concatenation
            concatenated_cols = " || ' ' || ".join([f"COALESCE(\"{col}\"::text, '')" for col in text_cols])
            base_query = f"SELECT {concatenated_cols} AS combined_text"
            
            # Try to find ID and timestamp columns
            inspector = inspect(self.engine)
            columns = {col['name']: col for col in inspector.get_columns(table_name)}
            
            # Look for ID column
            id_col = None
            for potential_id in ['id', 'ID', f'{table_name}_id', 'pk']:
                if potential_id in columns:
                    id_col = potential_id
                    break
            
            # Look for timestamp columns
            timestamp_col = None
            for potential_ts in ['updated_at', 'modified_at', 'last_modified', 'created_at']:
                if potential_ts in columns and 'timestamp' in str(columns[potential_ts]['type']).lower():
                    timestamp_col = potential_ts
                    break
            
            if id_col:
                base_query += f", \"{id_col}\" as row_id"
            
            base_query += f" FROM \"{table_name}\""
            
            # Build WHERE clause for incremental loading
            where_clauses = []
            params = {}
            
            if result:
                last_processed_at, last_max_id, last_modified_field = result
                
                if timestamp_col and last_processed_at:
                    where_clauses.append(f"\"{timestamp_col}\" > :last_timestamp")
                    params['last_timestamp'] = last_processed_at
                    
                elif id_col and last_max_id is not None:
                    where_clauses.append(f"\"{id_col}\" > :last_id")
                    params['last_id'] = last_max_id
            
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)
            
            # Add ordering for consistent processing
            if id_col:
                base_query += f" ORDER BY \"{id_col}\""
            
            return base_query, params, {'id_col': id_col, 'timestamp_col': timestamp_col}
    
    def log_processing(self, table_name, row_count, status='success', error_message=None, 
                      last_id=None, timestamp_col=None):
        """Log processing results for incremental tracking."""
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO {PROCESSING_LOG_TABLE} 
                (table_name, last_processed_at, last_max_id, last_modified_field, 
                 row_count, status, error_message)
                VALUES (:table_name, :timestamp, :last_id, :timestamp_col, 
                        :row_count, :status, :error_message)
                ON CONFLICT (table_name) 
                DO UPDATE SET 
                    last_processed_at = EXCLUDED.last_processed_at,
                    last_max_id = EXCLUDED.last_max_id,
                    last_modified_field = EXCLUDED.last_modified_field,
                    row_count = {PROCESSING_LOG_TABLE}.row_count + EXCLUDED.row_count,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message
            """), {
                'table_name': table_name,
                'timestamp': datetime.now(),
                'last_id': last_id,
                'timestamp_col': timestamp_col,
                'row_count': row_count,
                'status': status,
                'error_message': error_message
            })
            conn.commit()
