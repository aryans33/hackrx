import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv
from pathlib import Path
from models import DocumentRequest, RunResponse, HackRXRequest, HackRXResponse
import utils, parse, embed, retrieve, reason
import logging

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="LLM-Powered Document Query-Answering System with LlamaIndex")

# Add validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.body()
        body_str = body.decode('utf-8') if body else "Empty body"
    except:
        body_str = "Could not decode body"
    
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Request body: {body_str}")
    logger.error(f"Content-Type: {request.headers.get('content-type')}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(), 
            "message": "Invalid request format",
            "body": body_str,
            "content_type": request.headers.get('content-type')
        }
    )

# Security
security = HTTPBearer()
HACKRX_TOKEN = "3b3b7f8e0cb19ee38fcc3d4874a8df6dadcdbfec21b7bbe39a73407e2a7af8a0"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return credentials.credentials

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the LLM-Powered Document Q&A API. Visit /docs for usage."}

@app.post("/run", response_model=RunResponse)
async def run_query(request: DocumentRequest):
    """
    Process a document and answer questions about it.
    """
    logger.info(f"Received request: documents={request.documents}, questions={len(request.questions)}")
    
    try:
        # Validate required fields
        if not request.documents:
            raise HTTPException(status_code=400, detail="documents field is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Handle document path/URL
        document_path = request.documents
        
        if os.path.exists(document_path):
            doc_path = document_path
            logger.info(f"Using local file: {doc_path}")
        elif document_path.startswith(('http://', 'https://')):
            doc_path = await utils.download_document(document_path)
            logger.info(f"Downloaded file to: {doc_path}")
        else:
            raise HTTPException(status_code=400, detail=f"File not found: {document_path}")
        
        # Parse and chunk document with auto-generated document ID
        logger.info("Starting document parsing...")
        parsed_doc = await parse.parse_document(doc_path)
        chunks = parse.chunk_document(parsed_doc, doc_path)
        
        # Create hybrid index and add chunks
        logger.info("Creating hybrid embeddings and storing...")
        rag_index = embed.RAGIndex().create_or_load_index()
        embed.add_chunks_to_index(rag_index, chunks, doc_path)
        
        # Process each question and collect only answers
        answers = []
        for i, question in enumerate(request.questions, 1):
            logger.info(f"Processing question {i}/{len(request.questions)}: {question[:50]}...")
            
            # Retrieve relevant chunks using hybrid search
            relevant_chunks = retrieve.retrieve_chunks(rag_index, question, doc_path)
            
            # Generate answer using Mistral/GROQ
            answer_data = await reason.generate_answer(question, relevant_chunks)
            
            # Extract only the answer text
            answers.append(answer_data.get("answer", "Unable to determine answer"))
        
        logger.info(f"Successfully processed {len(answers)} questions using hybrid RAG")
        return RunResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest, token: str = Depends(verify_token)):
    """
    HackRX Competition endpoint with document isolation
    """
    logger.info(f"Received HackRX request: documents={request.documents}, questions={len(request.questions)}")
    
    try:
        # Validate required fields
        if not request.documents:
            raise HTTPException(status_code=400, detail="documents field is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Handle document path/URL
        document_path = request.documents
        
        if os.path.exists(document_path):
            doc_path = document_path
        elif document_path.startswith(('http://', 'https://')):
            doc_path = await utils.download_document(document_path)
        else:
            raise HTTPException(status_code=400, detail=f"File not found: {document_path}")
        
        # Parse and chunk document with auto-generated document ID
        parsed_doc = await parse.parse_document(doc_path)
        chunks = parse.chunk_document(parsed_doc, doc_path)
        
        # Create embeddings and store in FAISS
        index, metadata = embed.create_or_load_index()
        embed.add_chunks_to_index(index, chunks, doc_path)
        
        # Process each question
        answers = []
        for i, question in enumerate(request.questions):
            if i > 0:
                await asyncio.sleep(1)  # Rate limiting
                
            # Retrieve relevant chunks
            relevant_chunks = retrieve.retrieve_chunks(index, question, doc_path)
            
            # Generate answer using Mistral/GROQ
            answer_data = await reason.generate_answer(question, relevant_chunks)
            
            answers.append(answer_data.get("answer", "Unable to determine answer"))
        
        return HackRXResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error processing HackRX request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function for single document processing (used by test script)
async def process_document(document_path: str, question: str):
    """Process a single document and question - used internally"""
    try:
        # Parse and chunk document
        parsed_doc = await parse.parse_document(document_path)
        chunks = parse.chunk_document(parsed_doc, document_path)
        
        # Create embeddings and store in FAISS
        index, metadata = embed.create_or_load_index()
        embed.add_chunks_to_index(index, chunks, document_path)
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve.retrieve_chunks(index, question, document_path)
        
        # Generate answer using Mistral/GROQ
        answer_data = await reason.generate_answer(question, relevant_chunks)
        
        return answer_data
        
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
