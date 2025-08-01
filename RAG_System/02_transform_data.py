# 02_transform_data.py
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.decomposition import IncrementalPCA
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from monitoring import MetricsCollector
from config import (
    RAW_DATA_PATH, CHUNKS_PATH, EMBEDDINGS_PATH, PCA_MODEL_PATH,
    INCREMENTAL_STATE_PATH, EMBEDDING_MODEL_NAME, PCA_N_COMPONENTS, 
    PCA_BATCH_SIZE, logging
)

class IncrementalTransformer:
    """
    Handles text chunking, embedding generation, and PCA topic extraction
    with support for incremental processing and memory-efficient batch operations.
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, 
            chunk_overlap=100
        )
        self.embedding_model = None
        self.pca_model = None
        self.metrics = MetricsCollector()
        self._load_state()
    
    def _load_state(self):
        """Load incremental processing state if exists."""
        try:
            with open(INCREMENTAL_STATE_PATH, 'r') as f:
                self.state = json.load(f)
        except FileNotFoundError:
            self.state = {
                'last_processed_index': 0,
                'pca_fitted': False,
                'total_chunks_processed': 0
            }
    
    def _save_state(self):
        """Save incremental processing state."""
        with open(INCREMENTAL_STATE_PATH, 'w') as f:
            json.dump(self.state, f)
    
    def transform_data(self, incremental=True):
        """
        Main transformation method with incremental support.
        Args:
            incremental: Whether to process only new data
        """
        logging.info("Phase 2: Starting transformation (Embedding & PCA Topic Extraction)...")
        
        # Load embedding model
        logging.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Load or initialize PCA model
        self._load_or_init_pca()
        
        # Process data
        if incremental and self.state['last_processed_index'] > 0:
            self._incremental_transform()
        else:
            self._full_transform()
    
    def _load_or_init_pca(self):
        """Load existing PCA model or create new one."""
        try:
            with open(PCA_MODEL_PATH, 'rb') as f:
                self.pca_model = pickle.load(f)
                logging.info("Loaded existing PCA model")
        except FileNotFoundError:
            self.pca_model = IncrementalPCA(n_components=PCA_N_COMPONENTS)
            logging.info("Initialized new IncrementalPCA model")
    
    def _full_transform(self):
        """Perform full transformation of all data."""
        logging.info("Performing FULL transformation...")
        
        # Load raw data
        df = pd.read_parquet(RAW_DATA_PATH)
        
        # Create chunks
        all_chunks = self._create_chunks(df)
        logging.info(f"Created {len(all_chunks)} chunks from {len(df)} documents")
        
        # Process embeddings in batches
        all_embeddings = []
        all_processed_chunks = []
        
        for batch_start in range(0, len(all_chunks), PCA_BATCH_SIZE):
            batch_end = min(batch_start + PCA_BATCH_SIZE, len(all_chunks))
            batch_chunks = all_chunks[batch_start:batch_end]
            
            logging.info(f"Processing batch {batch_start//PCA_BATCH_SIZE + 1} "
                        f"({batch_start}-{batch_end} of {len(all_chunks)})")
            
            # Generate embeddings for batch
            batch_embeddings = self._generate_embeddings(batch_chunks)
            
            # Fit PCA incrementally
            if not self.state['pca_fitted'] or batch_start == 0:
                self.pca_model.partial_fit(batch_embeddings)
            
            all_embeddings.append(batch_embeddings)
            all_processed_chunks.extend(batch_chunks)
            
            # Free memory
            del batch_chunks
            del batch_embeddings
        
        # Concatenate all embeddings
        all_embeddings_np = np.vstack(all_embeddings)
        
        # Transform all embeddings with fitted PCA
        logging.info("Applying PCA transformation to all embeddings...")
        pca_labels = self._apply_pca_topics(all_embeddings_np, all_processed_chunks)
        
        # Save results
        self._save_results(all_processed_chunks, all_embeddings_np)
        
        # Update state
        self.state['last_processed_index'] = len(df)
        self.state['pca_fitted'] = True
        self.state['total_chunks_processed'] = len(all_processed_chunks)
        self._save_state()
        
        # Save PCA model
        with open(PCA_MODEL_PATH, 'wb') as f:
            pickle.dump(self.pca_model, f)
        
        logging.info(f"Phase 2 complete. Processed {len(all_processed_chunks)} chunks")
    
    def _incremental_transform(self):
        """Process only new data incrementally."""
        logging.info("Performing INCREMENTAL transformation...")
        
        # Load existing chunks and embeddings
        with open(CHUNKS_PATH, 'rb') as f:
            existing_chunks = pickle.load(f)
        existing_embeddings = np.load(EMBEDDINGS_PATH)
        
        # Load raw data and find new records
        df = pd.read_parquet(RAW_DATA_PATH)
        new_df = df.iloc[self.state['last_processed_index']:]
        
        if len(new_df) == 0:
            logging.info("No new data to process")
            return
        
        logging.info(f"Found {len(new_df)} new records to process")
        
        # Create chunks for new data
        new_chunks = self._create_chunks(new_df)
        
        # Process new chunks in batches
        new_embeddings = []
        
        for batch_start in range(0, len(new_chunks), PCA_BATCH_SIZE):
            batch_end = min(batch_start + PCA_BATCH_SIZE, len(new_chunks))
            batch_chunks = new_chunks[batch_start:batch_end]
            
            logging.info(f"Processing new batch {batch_start//PCA_BATCH_SIZE + 1}")
            
            # Generate embeddings
            batch_embeddings = self._generate_embeddings(batch_chunks)
            
            # Optional: Update PCA with new data (can be disabled for stability)
            if self.state['total_chunks_processed'] < 1000000:  # Only update for smaller datasets
                self.pca_model.partial_fit(batch_embeddings)
            
            new_embeddings.append(batch_embeddings)
        
        # Combine embeddings
        new_embeddings_np = np.vstack(new_embeddings)
        
        # Apply PCA to new embeddings
        self._apply_pca_topics(new_embeddings_np, new_chunks)
        
        # Combine with existing data
        all_chunks = existing_chunks + new_chunks
        all_embeddings = np.vstack([existing_embeddings, new_embeddings_np])
        
        # Save combined results
        self._save_results(all_chunks, all_embeddings)
        
        # Update state
        self.state['last_processed_index'] = len(df)
        self.state['total_chunks_processed'] = len(all_chunks)
        self._save_state()
        
        # Save updated PCA model
        with open(PCA_MODEL_PATH, 'wb') as f:
            pickle.dump(self.pca_model, f)
        
        logging.info(f"Incremental update complete. Total chunks: {len(all_chunks)}")
    
    def _create_chunks(self, df):
        """Create text chunks from dataframe."""
        documents = []
        
        for _, row in df.iterrows():
            if row['combined_text'] and isinstance(row['combined_text'], str):
                doc = Document(
                    page_content=row['combined_text'],
                    metadata={
                        'source_table': row['source_table'],
                        'source_id': row['source_id']
                    }
                )
                documents.append(doc)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        self.metrics.chunks_created.inc(len(chunks))
        
        return chunks
    
    def _generate_embeddings(self, chunks):
        """Generate embeddings for chunks with monitoring."""
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        with self.metrics.embedding_duration.time():
            embeddings = self.embedding_model.encode(
                chunk_texts,
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        self.metrics.embeddings_generated.inc(len(embeddings))
        return embeddings
    
    def _apply_pca_topics(self, embeddings, chunks):
        """Apply PCA transformation and assign topics to chunks."""
        with self.metrics.pca_duration.time():
            # Transform embeddings
            transformed = self.pca_model.transform(embeddings)
            
            # Assign topics based on strongest component
            pca_labels = []
            for scores in transformed:
                topic_idx = np.argmax(np.abs(scores))
                topic_label = f"Topic_{topic_idx}"
                pca_labels.append(topic_label)
            
            # Update chunk metadata
            for chunk, label in zip(chunks, pca_labels):
                chunk.metadata['semantic_topic'] = label
        
        return pca_labels
    
    def _save_results(self, chunks, embeddings):
        """Save processed chunks and embeddings."""
        # Save chunks
        with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump(chunks, f)
        
        # Save embeddings
        np.save(EMBEDDINGS_PATH, embeddings)
        
        logging.info(f"Saved {len(chunks)} chunks and embeddings")
    
    def get_pca_explained_variance(self):
        """Get explained variance ratio for each PCA component."""
        if self.pca_model and hasattr(self.pca_model, 'explained_variance_ratio_'):
            return self.pca_model.explained_variance_ratio_
        return None

if __name__ == '__main__':
    transformer = IncrementalTransformer()
    transformer.transform_data()
