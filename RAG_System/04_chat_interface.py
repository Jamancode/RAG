# 04_chat_interface.py
import gradio as gr
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from monitoring import MetricsCollector
from config import (
    DB_URI, EMBEDDING_MODEL_NAME, OLLAMA_MODEL, 
    VECTOR_TABLE_NAME, logging
)
import time
import json

# --- Initialization ---
try:
    logging.info("Phase 4: Starting the Enhanced RAG Interface...")
    
    # Initialize monitoring
    metrics = MetricsCollector()
    
    # Load models
    logging.info(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    logging.info(f"Connecting to Ollama LLM: '{OLLAMA_MODEL}'...")
    ollama_llm = Ollama(model=OLLAMA_MODEL)
    
    # Database connection with connection pooling
    engine = create_engine(
        DB_URI, 
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )
    
    # Update DB metrics on startup
    metrics.update_db_metrics(engine)
    
    logging.info("All systems initialized successfully. Interface is ready.")
    
except Exception as e:
    logging.error(f"Critical error during initialization: {e}")
    logging.error("Please ensure PostgreSQL and Ollama are running and the .env file is correct.")
    exit()

# --- Query Functions ---

def query_rag_system(user_query: str, num_results: int = 5, topic_filter: str = None):
    """
    Execute RAG query with monitoring and optional filtering.
    
    Args:
        user_query: The user's question
        num_results: Number of similar documents to retrieve
        topic_filter: Optional semantic topic filter
    
    Returns:
        Tuple of (response, sources, metrics)
    """
    if not user_query:
        return "Please enter a question.", "", {}
    
    start_time = time.time()
    query_metrics = {}
    
    logging.info(f"Received query: '{user_query}' (results: {num_results}, filter: {topic_filter})")
    
    # Track query
    metrics.queries_processed.inc()
    
    try:
        # 1. Embed the user query
        with metrics.query_duration.time():
            query_embedding = embedding_model.encode(user_query, convert_to_numpy=True)
            query_metrics['embedding_time'] = time.time() - start_time
        
        # 2. Build and execute vector search query
        with engine.connect() as conn:
            search_start = time.time()
            
            # Base query
            query = f"""
                SELECT content, source_table, source_id, semantic_topic,
                       (embedding <-> :query_vec) AS distance
                FROM {VECTOR_TABLE_NAME}
            """
            
            # Add topic filter if specified
            params = {'query_vec': query_embedding}
            if topic_filter and topic_filter != "All Topics":
                query += " WHERE semantic_topic = :topic"
                params['topic'] = topic_filter
            
            query += f"""
                ORDER BY distance ASC
                LIMIT :limit
            """
            params['limit'] = num_results
            
            results = conn.execute(text(query), params).fetchall()
            query_metrics['search_time'] = time.time() - search_start
        
        if not results:
            logging.warning("No relevant documents found in the DB.")
            return (
                "I could not find any relevant information in the database for your query.",
                "No sources found.",
                query_metrics
            )
        
        # 3. Prepare context and sources
        context_parts = []
        source_info = []
        
        for i, (content, source_table, source_id, topic, distance) in enumerate(results):
            context_parts.append(f"[Document {i+1}]\n{content}")
            source_info.append(
                f"**Source {i+1}:** Table `{source_table}`, ID `{source_id}` "
                f"(Topic: `{topic}`, Distance: {distance:.4f})"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # 4. Generate response with LLM
        llm_start = time.time()
        prompt = f"""You are a precise and helpful assistant. Answer the user's question based exclusively on the following context.

Do not quote or repeat parts of the context verbatim, but formulate an independent, comprehensive answer.
If the answer is not contained in the context, state clearly: "The answer to this question is not contained in the provided information."

**Context:**
{context}

**Question:**
{user_query}

**Answer:**"""
        
        logging.info("Context found. Generating response with the LLM...")
        response = ollama_llm.invoke(prompt)
        query_metrics['llm_time'] = time.time() - llm_start
        
        # 5. Prepare final response
        query_metrics['total_time'] = time.time() - start_time
        query_metrics['num_sources'] = len(results)
        
        # Format sources
        sources_text = "\n".join(source_info)
        
        # Format metrics
        metrics_text = (
            f"\n\n### Query Metrics\n"
            f"- Embedding time: {query_metrics['embedding_time']:.3f}s\n"
            f"- Search time: {query_metrics['search_time']:.3f}s\n"
            f"- LLM generation time: {query_metrics['llm_time']:.3f}s\n"
            f"- Total time: {query_metrics['total_time']:.3f}s\n"
            f"- Sources retrieved: {query_metrics['num_sources']}"
        )
        
        return response, sources_text + metrics_text, query_metrics
        
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return f"An error occurred: {str(e)}", "", {}

def get_available_topics():
    """Get list of available semantic topics for filtering."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT DISTINCT semantic_topic 
                FROM {VECTOR_TABLE_NAME} 
                WHERE semantic_topic IS NOT NULL
                ORDER BY semantic_topic
            """)).fetchall()
            
            topics = ["All Topics"] + [row[0] for row in result]
            return topics
    except Exception as e:
        logging.error(f"Error fetching topics: {e}")
        return ["All Topics"]

def get_system_stats():
    """Get current system statistics."""
    try:
        with engine.connect() as conn:
            stats = conn.execute(text(f"""
                SELECT 
                    COUNT(*) as total_vectors,
                    COUNT(DISTINCT source_table) as unique_tables,
                    COUNT(DISTINCT semantic_topic) as unique_topics,
                    AVG(LENGTH(content)) as avg_content_length,
                    MAX(updated_at) as last_update
                FROM {VECTOR_TABLE_NAME}
            """)).fetchone()
            
            return f"""### System Statistics
- **Total Vectors:** {stats[0]:,}
- **Unique Tables:** {stats[1]}
- **Semantic Topics:** {stats[2]}
- **Avg Content Length:** {int(stats[3])} chars
- **Last Update:** {stats[4]}
"""
    except Exception as e:
        return f"Error loading statistics: {str(e)}"

# --- Gradio Interface ---

def create_interface():
    """Create the Gradio interface with all features."""
    
    with gr.Blocks(title="Enhanced RAG System") as interface:
        gr.Markdown("# 🧠 Enhanced RAG System for PostgreSQL")
        gr.Markdown(
            "An intelligent retrieval system with incremental updates, "
            "semantic topic filtering, and comprehensive monitoring."
        )
        
        with gr.Tab("Query"):
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        lines=3,
                        label="Your Question",
                        placeholder="e.g., 'What are the latest sales trends?' or 'Find information about customer complaints'"
                    )
                with gr.Column(scale=1):
                    num_results = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of Results"
                    )
                    topic_filter = gr.Dropdown(
                        choices=get_available_topics(),
                        value="All Topics",
                        label="Filter by Topic"
                    )
                    query_btn = gr.Button("🔍 Search", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    response_output = gr.Markdown(label="Answer")
                with gr.Column():
                    sources_output = gr.Markdown(label="Sources & Metrics")
        
        with gr.Tab("System Status"):
            with gr.Row():
                stats_output = gr.Markdown()
                refresh_btn = gr.Button("🔄 Refresh Statistics")
            
            # Load initial statistics
            stats_output.value = get_system_stats()
        
        with gr.Tab("Examples"):
            gr.Examples(
                examples=[
                    ["Which customers had the highest revenue last quarter?"],
                    ["What are the most common product issues reported?"],
                    ["Find all contracts expiring in the next 3 months"],
                    ["Show me employee performance metrics"],
                    ["What are the trending topics in customer feedback?"]
                ],
                inputs=query_input
            )
        
        # Event handlers
        def handle_query(query, num_results, topic):
            response, sources, _ = query_rag_system(query, num_results, topic)
            return response, sources
        
        query_btn.click(
            fn=handle_query,
            inputs=[query_input, num_results, topic_filter],
            outputs=[response_output, sources_output]
        )
        
        query_input.submit(
            fn=handle_query,
            inputs=[query_input, num_results, topic_filter],
            outputs=[response_output, sources_output]
        )
        
        refresh_btn.click(
            fn=lambda: get_system_stats(),
            outputs=stats_output
        )
        
        # Auto-refresh topics dropdown when tab changes
        def refresh_topics():
            return gr.Dropdown(choices=get_available_topics())
        
        interface.load(
            fn=refresh_topics,
            outputs=topic_filter
        )
    
    return interface

if __name__ == '__main__':
    interface = create_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
