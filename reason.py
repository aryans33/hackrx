import os
import json
import asyncio
from typing import List, Dict, Any
import httpx
from dotenv import load_dotenv
from models import ClauseReference
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# GROQ Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama3-8b-8192"  # Working model from test

# Fallback for testing without GROQ key
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found. Using local Mistral as fallback.")
    GROQ_API_BASE = os.getenv("MISTRAL_API_BASE", "http://localhost:11434/v1")
    GROQ_MODEL = os.getenv("MISTRAL_MODEL_NAME", "mistral-7b-instruct")

async def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate an answer using Mistral LLM based on retrieved chunks.
    """
    logger.info(f"Generating answer for question: {question[:50]}...")
    logger.info(f"Using {len(chunks)} chunks as context")
    
    # Prepare context from chunks with detailed formatting
    context_parts = []
    clause_refs = []
    
    for i, chunk in enumerate(chunks):
        # Add to context
        context_parts.append(f"[Source {i+1} - Page {chunk.get('page_number', 'N/A')}]\n{chunk['text']}\n")
        
        # Create clause reference
        clause_refs.append(ClauseReference(
            title=chunk.get('title', f"Section {i+1}"),
            page_number=chunk.get('page_number', 1),
            text_snippet=chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
        ))
    
    context = "\n".join(context_parts)
    
    # Create prompt
    prompt = f"""Based on the following document excerpts, answer the question comprehensively and accurately.

Document Context:
{context}

Question: {question}

Please provide:
1. A direct answer to the question
2. Your reasoning based on the document
3. Your confidence level (High/Medium/Low)

Answer:"""

    try:
        # Call LLM API
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add authorization header if API key exists
        if GROQ_API_KEY:
            headers["Authorization"] = f"Bearer {GROQ_API_KEY}"
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert insurance policy analyst. Provide accurate, detailed answers based strictly on the provided document excerpts."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GROQ_API_BASE}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result['choices'][0]['message']['content']
                
                # Parse the response to extract answer, rationale, and confidence
                answer_data = parse_llm_response(llm_response)
                
                # Add clause references
                answer_data['relevant_clauses'] = clause_refs
                
                return answer_data
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return create_fallback_response(question, chunks, clause_refs)
                
    except Exception as e:
        logger.error(f"Error calling LLM API: {str(e)}")
        return create_fallback_response(question, chunks, clause_refs)

def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response to extract structured information"""
    
    # Try to extract answer, rationale, and confidence
    lines = response.strip().split('\n')
    
    answer = ""
    rationale = ""
    confidence = "Medium"
    
    # Simple parsing logic
    current_section = "answer"
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for section markers
        if any(keyword in line.lower() for keyword in ['reasoning', 'rationale', 'because', 'explanation']):
            current_section = "rationale"
            if ':' in line:
                rationale += line.split(':', 1)[1].strip() + " "
            continue
        elif any(keyword in line.lower() for keyword in ['confidence', 'certainty']):
            current_section = "confidence"
            if ':' in line:
                conf_text = line.split(':', 1)[1].strip().lower()
                if 'high' in conf_text:
                    confidence = "High"
                elif 'low' in conf_text:
                    confidence = "Low"
                else:
                    confidence = "Medium"
            continue
        
        # Add content to current section
        if current_section == "answer":
            answer += line + " "
        elif current_section == "rationale":
            rationale += line + " "
    
    # If no clear structure, use the whole response as answer
    if not answer.strip():
        answer = response
        rationale = "Based on the provided document excerpts."
    
    return {
        "answer": answer.strip(),
        "rationale": rationale.strip() or "Based on the provided document excerpts.",
        "confidence": confidence
    }

def create_fallback_response(question: str, chunks: List[Dict[str, Any]], clause_refs: List[ClauseReference]) -> Dict[str, Any]:
    """Create fallback response when LLM fails"""
    
    if chunks:
        # Try to create a simple answer from the chunks
        combined_text = " ".join([chunk['text'][:200] for chunk in chunks[:2]])
        answer = f"Based on the available information: {combined_text}..."
        rationale = "This answer is derived from the most relevant sections of the document."
        confidence = "Low"
    else:
        answer = "I couldn't find sufficient information in the document to answer this question."
        rationale = "No relevant content was found for this query."
        confidence = "Low"
    
    return {
        "answer": answer,
        "rationale": rationale,
        "confidence": confidence,
        "relevant_clauses": clause_refs
    }
