import os
import re
import hashlib
import time
from typing import List, Dict, Any
from llama_parse import LlamaParse
from llama_index.core.schema import Document
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# Initialize LlamaParse
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",  # Can be "markdown" or "text"
    verbose=True,
    language="en",
    split_by_page=True
)

async def parse_document(file_path: str) -> List[Document]:
    """
    Parse document using LlamaParse to extract text and structure.
    Returns list of LlamaIndex Document objects.
    """
    logger.info(f"Parsing document: {file_path}")
    
    try:
        # Parse the document
        documents = await parser.aload_data(file_path)
        logger.info(f"Successfully parsed {len(documents)} pages/sections")
        return documents
    except Exception as e:
        logger.error(f"Error parsing document: {e}")
        raise

def generate_document_id(file_path: str) -> str:
    """Generate unique document ID based on file path and timestamp"""
    file_name = os.path.basename(file_path)
    timestamp = str(int(time.time()))
    content = f"{file_name}_{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()[:12]

def chunk_document(documents: List[Document], file_path: str = None) -> List[Dict[str, Any]]:
    """
    Chunk documents with automatic document ID assignment.
    """
    logger.info(f"Chunking {len(documents)} document sections...")
    
    # Generate unique document ID
    document_id = generate_document_id(file_path) if file_path else f"doc_{int(time.time())}"
    document_name = os.path.basename(file_path) if file_path else "unknown_document"
    
    logger.info(f"Assigned document ID: {document_id}")
    
    chunks = []
    
    for doc_idx, doc in enumerate(documents):
        text = doc.text
        page_num = doc_idx + 1
        
        logger.info(f"Processing page {page_num} for document {document_id}")
        
        # Split by sections
        sections = split_by_headers(text)
        
        for section_idx, section in enumerate(sections):
            base_metadata = {
                'document_id': document_id,
                'document_name': document_name,
                'file_path': file_path or '',
                'title': section['title'],
                'page_number': page_num,
                'section_index': section_idx,
                'chunk_id': f"{document_id}_p{page_num}_s{section_idx}",
                'created_at': int(time.time())
            }
            
            if is_tabular_section(section['content']):
                # Table chunk processing
                table_chunks = chunk_table_content(section['content'])
                for chunk_idx, table_chunk in enumerate(table_chunks):
                    chunks.append({
                        **base_metadata,
                        'chunk_id': f"{document_id}_p{page_num}_s{section_idx}_t{chunk_idx}",
                        'text': table_chunk,
                        'type': 'table',
                        'table_info': analyze_table_structure(table_chunk)
                    })
            else:
                # Text chunk processing
                text_chunks = chunk_text_content(section['content'])
                for chunk_idx, text_chunk in enumerate(text_chunks):
                    chunks.append({
                        **base_metadata,
                        'chunk_id': f"{document_id}_p{page_num}_s{section_idx}_c{chunk_idx}",
                        'text': text_chunk,
                        'type': 'text',
                        'text_info': analyze_text_structure(text_chunk)
                    })
    
    logger.info(f"Generated {len(chunks)} chunks for document {document_id}")
    return chunks

def split_by_headers(text: str) -> List[Dict[str, str]]:
    """Split text into sections based on headers"""
    
    # Header patterns (markdown, numbered, caps, etc.)
    header_patterns = [
        r'^(#{1,6}\s+.+)$',           # Markdown headers (# ## ###)
        r'^(\d+\.[\d\.]*\s+.+)$',     # Numbered (1.1, 1.1.1)
        r'^([A-Z][A-Z\s]{5,}:?)$',    # ALL CAPS headers
        r'^([A-Z][a-z][\w\s]{3,}:)$', # Title case with colon
        r'^(\*\*[^*]+\*\*)$',         # Bold markdown
        r'^(__[^_]+__)$'              # Bold underscores
    ]
    
    lines = text.split('\n')
    sections = []
    current_section = {'title': 'Introduction', 'content': ''}
    
    for line in lines:
        line = line.strip()
        if not line:
            current_section['content'] += '\n'
            continue
            
        # Check if line is a header
        is_header = False
        for pattern in header_patterns:
            if re.match(pattern, line, re.MULTILINE):
                # Save current section
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Start new section
                header_text = re.match(pattern, line).group(1)
                current_section = {
                    'title': header_text.strip('# *_:'),
                    'content': ''
                }
                is_header = True
                break
        
        if not is_header:
            current_section['content'] += line + '\n'
    
    # Add the last section
    if current_section['content'].strip():
        sections.append(current_section)
    
    # If no sections found, create one section
    if not sections:
        sections = [{'title': 'Document Content', 'content': text}]
    
    logger.info(f"Split text into {len(sections)} sections")
    return sections

def is_tabular_section(content: str) -> bool:
    """Determine if content contains tabular data"""
    
    lines = content.split('\n')
    if len(lines) < 2:
        return False
    
    # Count pipe separators
    pipe_lines = sum(1 for line in lines if '|' in line and line.count('|') >= 2)
    pipe_ratio = pipe_lines / len(lines)
    
    # Count tab separators
    tab_lines = sum(1 for line in lines if '\t' in line and line.count('\t') >= 1)
    tab_ratio = tab_lines / len(lines)
    
    # Check for consistent column structure
    column_counts = []
    for line in lines:
        if '|' in line:
            column_counts.append(line.count('|'))
        elif '\t' in line:
            column_counts.append(line.count('\t'))
    
    # Check consistency
    if column_counts:
        most_common_count = max(set(column_counts), key=column_counts.count)
        consistency = column_counts.count(most_common_count) / len(column_counts)
    else:
        consistency = 0
    
    # Table keywords
    table_keywords = ['table', 'column', 'row', 'premium', 'coverage', 'benefit', 'amount']
    keyword_present = any(keyword.lower() in content.lower() for keyword in table_keywords)
    
    # Decision logic
    is_table = (
        pipe_ratio > 0.4 or 
        tab_ratio > 0.3 or 
        (consistency > 0.6 and len(column_counts) > 1) or
        (keyword_present and (pipe_ratio > 0.2 or tab_ratio > 0.2))
    )
    
    return is_table

def chunk_table_content(content: str, max_rows: int = 10) -> List[str]:
    """Chunk table content into smaller tables"""
    
    lines = content.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    if len(non_empty_lines) <= max_rows:
        return [content]
    
    chunks = []
    header_line = ""
    
    # Try to identify header
    for i, line in enumerate(non_empty_lines[:3]):
        if '|' in line or '\t' in line:
            header_line = line
            start_idx = i + 1
            break
    else:
        start_idx = 0
    
    # Chunk the remaining lines
    for i in range(start_idx, len(non_empty_lines), max_rows):
        chunk_lines = non_empty_lines[i:i + max_rows]
        
        # Add header to each chunk (except first)
        if header_line and i > start_idx:
            chunk_content = header_line + '\n' + '\n'.join(chunk_lines)
        else:
            chunk_content = '\n'.join(chunk_lines)
        
        chunks.append(chunk_content)
    
    return chunks

def chunk_text_content(content: str, max_chars: int = 1000) -> List[str]:
    """Chunk text content into smaller pieces"""
    
    if len(content) <= max_chars:
        return [content]
    
    # Split by paragraphs first
    paragraphs = content.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If single paragraph is too long, split by sentences
        if len(paragraph) > max_chars:
            sentences = re.split(r'[.!?]+', paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk) + len(sentence) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
        else:
            # Normal paragraph handling
            if len(current_chunk) + len(paragraph) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def analyze_text_structure(text: str) -> Dict[str, Any]:
    """Analyze text structure for metadata"""
    
    paragraphs = text.split('\n\n')
    sentences = re.split(r'[.!?]+', text)
    words = text.split()
    
    return {
        'paragraph_count': len([p for p in paragraphs if p.strip()]),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'word_count': len(words)
    }

def analyze_table_structure(table_text: str) -> Dict[str, Any]:
    """Analyze table structure for metadata"""
    
    lines = table_text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Estimate columns
    column_counts = []
    for line in non_empty_lines:
        if '|' in line:
            column_counts.append(line.count('|') + 1)
        elif '\t' in line:
            column_counts.append(line.count('\t') + 1)
    
    estimated_columns = max(column_counts) if column_counts else 1
    
    # Check for headers (first few lines with consistent structure)
    has_header = False
    if len(non_empty_lines) >= 2:
        first_line_cols = non_empty_lines[0].count('|') or non_empty_lines[0].count('\t')
        second_line_cols = non_empty_lines[1].count('|') or non_empty_lines[1].count('\t')
        has_header = first_line_cols == second_line_cols and first_line_cols > 0
    
    return {
        'estimated_columns': estimated_columns,
        'row_count': len(non_empty_lines),
        'has_header': has_header
    }
