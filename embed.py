import os
import pickle
import json
import numpy as np
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.schema import TextNode
import faiss
from dotenv import load_dotenv
from utils import get_document_id
import logging
from rank_bm25 import BM25Okapi
import re
from functools import partial
import pinecone

# Pinecone configuration (to be set via environment variables or config)
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'YOUR_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV', 'YOUR_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'your-index-name')

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Utility functions for Pinecone

def get_pinecone_index():
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX_NAME, dimension=768, metric='cosine')
    return pinecone.Index(PINECONE_INDEX_NAME)


def upsert_vectors_to_pinecone(vectors, ids):
    """Upsert a batch of vectors to Pinecone."""
    index = get_pinecone_index()
    # Pinecone expects list of (id, vector) tuples
    items = list(zip(ids, vectors))
    index.upsert(vectors=items)


def query_pinecone(query_vector, top_k=5):
    """Query Pinecone for similar vectors."""
    index = get_pinecone_index()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return result

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini embedding model
embed_model = GeminiEmbedding(
    model_name="models/embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Configure global settings
Settings.embed_model = embed_model

VECTOR_STORE_DIR = "vector_store"
INDEX_FILE = "document_index.faiss"
METADATA_FILE = "chunk_metadata.pkl"
BM25_FILE = "bm25_index.pkl"

# Performance settings
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)  # Optimal thread count
EMBEDDING_BATCH_SIZE = 10  # Process embeddings in batches
CHUNK_BATCH_SIZE = 50  # Process chunks in batches

class RAGIndex:
    """Standard RAG Index using dense retrieval with Pinecone and metadata."""
    
    def __init__(self):
        self.dense_index = get_pinecone_index()  # Pinecone index for semantic search
        self.metadata = {}
        self.document_chunks = {}  # Store chunks by document_id
    
    def create_or_load_index(self):
        """Create or load index metadata (Pinecone is always cloud-managed)"""
        start_time = time.time()
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        metadata_path = os.path.join(VECTOR_STORE_DIR, METADATA_FILE)

        # Load metadata if available
        if os.path.exists(metadata_path):
            load_start = time.time()
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get('metadata', {})
                self.document_chunks = data.get('document_chunks', {})
            load_time = time.time() - load_start
            logger.info(f"Loaded index metadata in {load_time:.2f}s")
        else:
            self.metadata = {}
            self.document_chunks = {}
            logger.info("Created new RAG index (Pinecone for dense vectors)")
        total_time = time.time() - start_time
        logger.info(f"Index initialization completed in {total_time:.2f}s")
        return self

def process_chunk_parallel(chunk_data: Tuple[Dict[str, Any], int]) -> Tuple[List[Dict], List[Dict]]:
    """Process a single chunk for multi-level chunking (for parallel execution)"""
    chunk, idx = chunk_data
    text = chunk['text']
    
    # Create short chunks (200-400 chars) for precise retrieval
    short_parts = split_into_short_chunks(text, max_chars=300)
    
    # Create long chunks (800-1200 chars) for context
    long_parts = split_into_long_chunks(text, max_chars=1000)
    
    short_chunks = []
    long_chunks = []
    
    # Process short chunks
    for i, short_text in enumerate(short_parts):
        short_chunk = {
            **chunk,
            'text': short_text,
            'chunk_id': f"{chunk['chunk_id']}_short_{i}",
            'chunk_type': 'short',
            'parent_id': chunk['chunk_id']
        }
        short_chunks.append(short_chunk)
    
    # Process long chunks  
    for i, long_text in enumerate(long_parts):
        long_chunk = {
            **chunk,
            'text': long_text,
            'chunk_id': f"{chunk['chunk_id']}_long_{i}",
            'chunk_type': 'long',
            'parent_id': chunk['chunk_id']
        }
        long_chunks.append(long_chunk)
    
    return short_chunks, long_chunks

def create_multi_level_chunks_parallel(chunks: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """Create both short (for precision) and long (for context) chunks using parallel processing"""
    start_time = time.time()
    logger.info(f"Starting parallel multi-level chunking for {len(chunks)} chunks")
    
    all_short_chunks = []
    all_long_chunks = []
    
    # Prepare data for parallel processing
    chunk_data = [(chunk, idx) for idx, chunk in enumerate(chunks)]
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        chunk_start = time.time()
        future_to_chunk = {
            executor.submit(process_chunk_parallel, data): data[1] 
            for data in chunk_data
        }
        
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                short_chunks, long_chunks = future.result()
                all_short_chunks.extend(short_chunks)
                all_long_chunks.extend(long_chunks)
            except Exception as exc:
                logger.error(f'Chunk {chunk_idx} generated an exception: {exc}')
        
        chunk_time = time.time() - chunk_start
        logger.info(f"Parallel chunking completed in {chunk_time:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(f"Multi-level chunking: {len(chunks)} → {len(all_short_chunks)} short + {len(all_long_chunks)} long chunks in {total_time:.2f}s")
    
    return all_short_chunks, all_long_chunks

def process_embedding_batch(texts_batch: List[str]) -> List[List[float]]:
    """Process a batch of texts for embeddings"""
    batch_start = time.time()
    embeddings = []
    
    for text in texts_batch:
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Use zero vector as fallback
            embeddings.append([0.0] * 768)
    
    batch_time = time.time() - batch_start
    logger.info(f"Generated {len(embeddings)} embeddings in {batch_time:.2f}s")
    return embeddings

async def generate_embeddings_parallel(texts: List[str]) -> List[List[float]]:
    """Generate embeddings in parallel batches"""
    start_time = time.time()
    logger.info(f"Starting parallel embedding generation for {len(texts)} texts")
    
    if not texts:
        return []
    
    # Split texts into batches
    batches = [texts[i:i + EMBEDDING_BATCH_SIZE] for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)]
    logger.info(f"Split into {len(batches)} batches of max {EMBEDDING_BATCH_SIZE} texts each")
    
    all_embeddings = []
    
    # Process batches with controlled concurrency
    with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
        batch_start = time.time()
        
        # Submit all batch jobs
        future_to_batch = {
            executor.submit(process_embedding_batch, batch): idx 
            for idx, batch in enumerate(batches)
        }
        
        # Collect results in order
        batch_results = [None] * len(batches)
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_embeddings = future.result()
                batch_results[batch_idx] = batch_embeddings
            except Exception as exc:
                logger.error(f'Batch {batch_idx} generated an exception: {exc}')
                # Fallback to zero embeddings for this batch
                batch_size = len(batches[batch_idx])
                batch_results[batch_idx] = [[0.0] * 768] * batch_size
        
        # Flatten results
        for batch_embeddings in batch_results:
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
        
        batch_time = time.time() - batch_start
        logger.info(f"All embedding batches completed in {batch_time:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(f"Generated {len(all_embeddings)} embeddings in {total_time:.2f}s (avg: {total_time/len(texts)*1000:.1f}ms per text)")
    
    return all_embeddings

def process_tokenization_batch(texts_batch: List[str]) -> List[List[str]]:
    """Process a batch of texts for tokenization"""
    return [tokenize_text(text) for text in texts_batch]

def tokenize_texts_parallel(texts: List[str]) -> List[List[str]]:
    """Tokenize texts in parallel"""
    start_time = time.time()
    logger.info(f"Starting parallel tokenization for {len(texts)} texts")
    
    if not texts:
        return []
    
    # Split texts into batches
    batches = [texts[i:i + CHUNK_BATCH_SIZE] for i in range(0, len(texts), CHUNK_BATCH_SIZE)]
    
    all_tokenized = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tokenize_start = time.time()
        
        future_to_batch = {
            executor.submit(process_tokenization_batch, batch): idx 
            for idx, batch in enumerate(batches)
        }
        
        batch_results = [None] * len(batches)
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_tokenized = future.result()
                batch_results[batch_idx] = batch_tokenized
            except Exception as exc:
                logger.error(f'Tokenization batch {batch_idx} generated an exception: {exc}')
                batch_results[batch_idx] = [[] for _ in batches[batch_idx]]
        
        # Flatten results
        for batch_tokenized in batch_results:
            if batch_tokenized:
                all_tokenized.extend(batch_tokenized)
        
        tokenize_time = time.time() - tokenize_start
        logger.info(f"Parallel tokenization completed in {tokenize_time:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(f"Tokenized {len(texts)} texts in {total_time:.2f}s")
    
    return all_tokenized

async def add_chunks_to_index(rag_index: RAGIndex, chunks: List[Dict[str, Any]], document_path: str = None):
    """Add chunks to index with parallel multi-level chunking"""
    total_start_time = time.time()
    logger.info(f"Starting index update for {len(chunks)} chunks")
    
    if not chunks:
        logger.warning("No chunks to add to index")
        return rag_index
    
    document_id = chunks[0]['document_id']
    
    # Remove existing chunks for this document
    remove_start = time.time()
    remove_document_from_index(rag_index, document_id)
    remove_time = time.time() - remove_start
    logger.info(f"Document removal completed in {remove_time:.2f}s")
    
    # Create multi-level chunks in parallel
    chunking_start = time.time()
    short_chunks, long_chunks = create_multi_level_chunks_parallel(chunks)
    all_chunks = short_chunks + long_chunks
    chunking_time = time.time() - chunking_start
    logger.info(f"Multi-level chunking completed in {chunking_time:.2f}s")
    
    # Store document chunks
    storage_start = time.time()
    rag_index.document_chunks[document_id] = {
        'original': chunks,
        'short': short_chunks,
        'long': long_chunks
    }
    storage_time = time.time() - storage_start
    logger.info(f"Document chunks storage completed in {storage_time:.2f}s")
    
    # Generate embeddings in parallel
    embedding_start = time.time()
    chunk_texts = [chunk['text'] for chunk in all_chunks]
    embeddings = await generate_embeddings_parallel(chunk_texts)
    embedding_time = time.time() - embedding_start
    logger.info(f"Embedding generation completed in {embedding_time:.2f}s")
    
    # Add to dense index (Pinecone)
    upsert_start = time.time()
    ids = [f"{chunk['document_id']}_{chunk['chunk_id']}" for chunk in all_chunks]
    upsert_vectors_to_pinecone(embeddings, ids)
    upsert_time = time.time() - upsert_start
    logger.info(f"Pinecone index update completed in {upsert_time:.2f}s")
    
    # Update metadata
    metadata_start = time.time()
    for i, chunk in enumerate(all_chunks):
        rag_index.metadata[ids[i]] = chunk
    metadata_time = time.time() - metadata_start
    logger.info(f"Metadata update completed in {metadata_time:.2f}s")
    
    # Save indexes
    save_start = time.time()
    save_index(rag_index)
    save_time = time.time() - save_start
    logger.info(f"Index saving completed in {save_time:.2f}s")
    
    total_time = time.time() - total_start_time
    
    # Summary log
    logger.info(f"""
=== INDEX UPDATE SUMMARY ===
Total chunks processed: {len(chunks)} → {len(all_chunks)} ({len(short_chunks)} short + {len(long_chunks)} long)
Document ID: {document_id}

Timing Breakdown:
- Document removal: {remove_time:.2f}s
- Multi-level chunking: {chunking_time:.2f}s
- Storage: {storage_time:.2f}s  
- Embedding generation: {embedding_time:.2f}s
- Pinecone index update: {upsert_time:.2f}s
- Metadata update: {metadata_time:.2f}s
- Index saving: {save_time:.2f}s

TOTAL TIME: {total_time:.2f}s
Performance: {len(all_chunks)/total_time:.1f} chunks/second
=====================================
    """)
    
    return rag_index

def split_into_short_chunks(text: str, max_chars: int = 300) -> List[str]:
    """Split text into short, precise chunks"""
    if len(text) <= max_chars:
        return [text]
    
    # Split by sentences for short chunks
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def split_into_long_chunks(text: str, max_chars: int = 1000) -> List[str]:
    """Split text into longer chunks for context"""
    if len(text) <= max_chars:
        return [text]
    
    # Split by paragraphs for long chunks
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def tokenize_text(text: str) -> List[str]:
    """Simple tokenization for BM25"""
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split and filter empty tokens
    tokens = [token for token in text.split() if token.strip()]
    return tokens

def search_index(rag_index: RAGIndex, query: str, document_id: str, top_k: int = 5) -> List[Dict]:
    """Perform dense search using Pinecone"""
    search_start = time.time()
    logger.info(f"Starting search for query: '{query[:50]}...' in document {document_id}")
    
    # Dense retrieval (semantic similarity)
    dense_start = time.time()
    dense_results = dense_search(rag_index, query, document_id, top_k * 2)
    dense_time = time.time() - dense_start
    
    # Expand short chunks to include their long context
    expand_start = time.time()
    expanded_results = expand_with_context(rag_index, dense_results)
    expand_time = time.time() - expand_start
    
    total_search_time = time.time() - search_start
    
    logger.info(f"""
=== SEARCH SUMMARY ===
Query: '{query[:50]}...'
Results: {len(expanded_results)} chunks
Timing:
- Dense search: {dense_time:.3f}s ({len(dense_results)} results)
- Context expansion: {expand_time:.3f}s
Total: {total_search_time:.3f}s
============================
    """)
    
    return expanded_results

def dense_search(rag_index: RAGIndex, query: str, document_id: str, top_k: int) -> List[Dict]:
    """Dense semantic search using Pinecone"""
    
    query_embedding = generate_embeddings([query])[0]
    search_k = min(pinecone.list_indexes()[PINECONE_INDEX_NAME].describe_index_stats()['total_vector_count'], top_k * 3)
    
    # Pinecone search expects a list of (id, vector) tuples
    items = [(f"{document_id}_{i}", query_embedding) for i in range(search_k)]
    
    results = []
    for item in pinecone.list_indexes()[PINECONE_INDEX_NAME].query(
        vector=query_embedding,
        top_k=search_k,
        include_metadata=True
    )['matches']:
        if item['id'] in rag_index.metadata:
            chunk_data = rag_index.metadata[item['id']]
            if chunk_data.get('document_id') == document_id:
                results.append({
                    'score': float(item['score']),
                    'chunk_data': chunk_data,
                    'retrieval_type': 'dense'
                })
        
        if len(results) >= top_k:
            break
    
    return results

def expand_with_context(rag_index: RAGIndex, results: List[Dict]) -> List[Dict]:
    """Expand short chunks with their corresponding long context"""
    
    expanded_results = []
    
    for result in results:
        chunk_data = result['chunk_data']
        chunk_type = chunk_data.get('chunk_type', 'original')
        
        if chunk_type == 'short':
            # Find corresponding long chunk for context
            parent_id = chunk_data.get('parent_id')
            document_id = chunk_data.get('document_id')
            
            # Look for long chunk with same parent
            long_context = None
            if document_id in rag_index.document_chunks:
                for long_chunk in rag_index.document_chunks[document_id].get('long', []):
                    if long_chunk.get('parent_id') == parent_id:
                        long_context = long_chunk['text']
                        break
            
            # Add context to result
            result['context'] = long_context or chunk_data['text']
            result['short_text'] = chunk_data['text']
        else:
            result['context'] = chunk_data['text']
            result['short_text'] = chunk_data['text']
        
        expanded_results.append(result)
    
    return expanded_results

def remove_document_from_index(rag_index: RAGIndex, document_id: str):
    """Remove document from index"""
    
    # Remove from metadata
    indices_to_remove = []
    for idx, chunk_data in rag_index.metadata.items():
        if chunk_data.get('document_id') == document_id:
            indices_to_remove.append(idx)
    
    for idx in indices_to_remove:
        del rag_index.metadata[idx]
    
    # Remove from document chunks
    if document_id in rag_index.document_chunks:
        del rag_index.document_chunks[document_id]
    
    # Note: For production, you'd want to rebuild FAISS and BM25 indexes
    # For now, we'll just mark them as removed in metadata
    
    if indices_to_remove:
        logger.info(f"Removed {len(indices_to_remove)} chunks for document {document_id}")

def save_index(rag_index: RAGIndex):
    """Save index to disk"""
    
    # Save metadata and other data
    metadata_path = os.path.join(VECTOR_STORE_DIR, METADATA_FILE)
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'metadata': rag_index.metadata,
            'document_chunks': rag_index.document_chunks
        }, f)

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Gemini (synchronous version for compatibility)"""
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result['embedding'])
    return embeddings

# Legacy functions for backward compatibility
def create_or_load_index():
    """Legacy function - creates index"""
    rag_index = RAGIndex()
    return rag_index.create_or_load_index(), {}

def add_chunks_to_index(index, chunks: List[Dict[str, Any]], document_path: str = None):
    """Legacy function - uses index"""
    if isinstance(index, RAGIndex):
        # Use async version
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(add_chunks_to_index(index, chunks, document_path))
    else:
        # Fallback to old implementation
        rag_index = RAGIndex()
        rag_index.dense_index = index
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(add_chunks_to_index(rag_index, chunks, document_path))

def search_document(index, metadata, query: str, document_id: str, top_k: int = 5):
    """Legacy function - uses search_index"""
    if isinstance(index, RAGIndex):
        return search_index(index, query, document_id, top_k)
    else:
        # Fallback to old dense search
        rag_index = RAGIndex()
        rag_index.dense_index = index
        rag_index.metadata = metadata
        return dense_search(rag_index, query, document_id, top_k)
