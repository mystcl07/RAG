# backend/schemas.py
# Defines Pydantic models for validating API requests/responses, 
# including documents, messages, conversation responses, query inputs, and URL scraping requests.

from pydantic import BaseModel
from typing import List, Dict, Any

class Document(BaseModel):
    content: str
    metadata: Dict[str, Any]

class Message(BaseModel):
    role: str
    content: str
    source_documents: List[Document] = []

class ConversationResponse(BaseModel):
    messages: List[Message]

class QueryRequest(BaseModel):
    question: str
    user_id: str
    search_mode: str = "Semantic"

class ScrapeUrlRequest(BaseModel):
    url: str