import os
import pickle
import logging
from typing import List, Dict, Any
import embed

logger = logging.getLogger(__name__)

def retrieve_chunks(index, question: str, document_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks using hybrid search"""
    
    # Load metadata to find document_id
    metadata_path = os.path.join(embed.VECTOR_STORE_DIR, embed.METADATA_FILE)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and 'metadata' in data:
                metadata = data['metadata']
                document_chunks = data.get('document_chunks', {})
            else:
                metadata = data  # Legacy format
                document_chunks = {}
    else:
        logger.error("No metadata file found")
        return []
    
    # Find document_id for the given document_path
    document_id = None
    for chunk_data in metadata.values():
        if chunk_data.get('file_path') == document_path:
            document_id = chunk_data.get('document_id')
            break
    
    if not document_id:
        logger.error(f"No document_id found for path: {document_path}")
        return []
    
    logger.info(f"Retrieving chunks for document {document_id}, question: {question[:50]}...")
    
    # Use hybrid search if available
    if isinstance(index, embed.RAGIndex):
        results = embed.search_index(index, question, document_id, top_k)
    else:
        # Fallback to dense search
        results = embed.search_document(index, metadata, question, document_id, top_k)
    
    # Format results for reasoning module
    relevant_chunks = []
    for result in results:
        chunk_data = result['chunk_data']
        
        # Use expanded context if available
        text_content = result.get('context', chunk_data['text'])
        short_text = result.get('short_text', chunk_data['text'][:200])
        
        relevant_chunks.append({
            'text': text_content,  # Use long context for reasoning
            'short_text': short_text,  # Keep short text for reference
            'title': chunk_data['title'],
            'page_number': chunk_data['page_number'],
            'type': chunk_data['type'],
            'score': result['score'],
            'chunk_id': chunk_data['chunk_id'],
            'document_id': chunk_data['document_id'],
            'retrieval_type': result.get('retrieval_type', 'hybrid'),
            'chunk_type': chunk_data.get('chunk_type', 'original')
        })
    
    logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks using hybrid search")
    return relevant_chunks
