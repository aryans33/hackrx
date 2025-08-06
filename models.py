from pydantic import BaseModel
from typing import List, Optional

class DocumentRequest(BaseModel):
    documents: str  # Changed from document_url to documents
    questions: List[str]

class HackRXRequest(BaseModel):
    documents: str
    questions: List[str]

class ClauseReference(BaseModel):
    title: str
    page_number: int
    text_snippet: str

class Response(BaseModel):
    question: str
    answer: str
    rationale: str
    confidence: str
    relevant_clauses: List[ClauseReference]

class RunResponse(BaseModel):
    answers: List[str]  # Changed from responses to answers
     # Optional field for processed document path
    

class HackRXResponse(BaseModel):
    answers: List[str]
