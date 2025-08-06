import os
import aiofiles
import requests
import uuid
from urllib.parse import urlparse

async def download_document(url: str) -> str:
    """
    Download a document from URL and save it locally.
    Returns the local file path.
    """
    save_dir = os.path.join(os.path.dirname(__file__), '../vector_store')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get file extension from URL
    parsed_url = urlparse(url)
    file_ext = os.path.splitext(parsed_url.path)[1]
    if not file_ext:
        file_ext = '.pdf'  # Default to PDF
    
    local_path = os.path.join(save_dir, f"{uuid.uuid4()}{file_ext}")
    
    response = requests.get(url)
    response.raise_for_status()
    
    async with aiofiles.open(local_path, 'wb') as f:
        await f.write(response.content)
    
    return local_path

def get_document_id(file_path: str) -> str:
    """Generate a unique document ID from file path."""
    return os.path.splitext(os.path.basename(file_path))[0]
