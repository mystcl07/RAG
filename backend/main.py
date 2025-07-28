# backend/main.py
# Implements a FastAPI backend with endpoints for uploading PDFs, scraping URLs, 
# answering queries (with summarization/translation), managing conversation history, and clearing documents/memory.

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from backend.document_processor import upload_pdf, load_pdf, scrape_url, split_text, index_docs, clear_vector_store
from backend.chains import answer_question, summarize_text, translate_text
from backend.retrievers import retrieve_docs, hybrid_retrieval
from backend.models import SessionLocal, Conversation
from backend.schemas import QueryRequest, ConversationResponse, Message, ScrapeUrlRequest
from backend.crud import save_message, get_conversations
from backend.config import PDFS_DIRECTORY
from langchain.memory import ConversationBufferWindowMemory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Global vector store and memory
vector_store = None
memory = ConversationBufferWindowMemory(k=3)

@app.post("/upload_pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    global vector_store
    try:
        file_path = upload_pdf(file)
        documents = load_pdf(file_path)
        chunked_documents = split_text(documents)
        vector_store = index_docs(chunked_documents, vector_store)
        logger.info(f"PDF {file.filename} processed successfully")
        return {"message": "PDF processed successfully"}
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process PDF")

@app.post("/scrape_url")
async def scrape_url_endpoint(request: ScrapeUrlRequest):
    global vector_store
    try:
        documents = scrape_url(request.url)
        if not documents:
            raise HTTPException(status_code=400, detail="Failed to scrape URL")
        chunked_documents = split_text(documents)
        vector_store = index_docs(chunked_documents, vector_store)
        logger.info(f"URL {request.url} processed successfully")
        return {"message": "URL processed successfully"}
    except Exception as e:
        logger.error(f"Error scraping URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to scrape URL: {str(e)}")

@app.post("/query", response_model=ConversationResponse)
async def query_endpoint(request: QueryRequest, db: Session = Depends(get_db)):
    global vector_store, memory
    question = request.question
    user_id = request.user_id
    search_mode = request.search_mode

    logger.info(f"Received query: {question} (user_id: {user_id}, search_mode: {search_mode})")

    # Save user message
    save_message(db, user_id, "user", question)

    try:
        source_documents = []
        if question.lower() == "summarize":
            if not vector_store:
                response = "No documents available to summarize."
            else:
                all_text = "\n\n".join([doc.page_content for doc in vector_store.docstore._dict.values()])
                response = summarize_text(all_text)
        elif question.lower().startswith("translate:"):
            target_lang = question.split(":", 1)[1].strip() if ":" in question else "French"
            if not vector_store:
                response = "No documents available to translate."
            else:
                all_text = "\n\n".join([doc.page_content for doc in vector_store.docstore._dict.values()])
                response = translate_text(all_text, target_lang)
        else:
            related_documents = hybrid_retrieval(question, vector_store) if search_mode == "Hybrid" else retrieve_docs(question, vector_store)
            source_documents = [
                {"content": doc.page_content[:300], "metadata": doc.metadata}
                for doc in related_documents
            ]
            if not related_documents:
                response = "I couldn't find relevant information in the provided sources."
            else:
                response = answer_question(question, related_documents, memory)
        logger.info(f"Generated response: {response[:100]}...")

        # Save assistant response
        save_message(db, user_id, "assistant", response)

        # Update memory
        memory.save_context({"input": question}, {"output": response})

        # Get recent conversations
        conversations = get_conversations(db, user_id)
        messages = [
            Message(
                role=conv.role,
                content=conv.content,
                source_documents=source_documents if conv.role == "assistant" and source_documents else []
            ) for conv in conversations
        ]
        logger.info(f"Returning {len(messages)} messages")
        return ConversationResponse(messages=messages)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.get("/history", response_model=ConversationResponse)
async def history_endpoint(user_id: str, db: Session = Depends(get_db)):
    """Fetch conversation history without generating a response."""
    try:
        conversations = db.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.id.asc()).limit(10).all()
        messages = [
            Message(
                role=conv.role,
                content=conv.content,
                source_documents=[]  # No source docs for history fetch
            ) for conv in conversations
        ]
        logger.info(f"Fetched {len(messages)} messages for user_id: {user_id}")
        return ConversationResponse(messages=messages)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.get("/clear_documents")
async def clear_documents_endpoint():
    global vector_store
    vector_store = clear_vector_store(vector_store)
    logger.info("Document cache cleared")
    return {"message": "Document cache cleared"}

@app.get("/clear_memory")
async def clear_memory_endpoint():
    global memory
    memory.clear()
    logger.info("Conversation history cleared")
    return {"message": "Conversation history cleared"}

@app.get("/clear_conversation")
async def clear_conversation_endpoint(db: Session = Depends(get_db)):
    global memory
    try:
        # Clear in-memory conversation
        memory.clear()
        # Clear database conversations for the user
        db.query(Conversation).filter(Conversation.user_id == "default_user").delete()
        db.commit()
        logger.info(f"Cleared conversation for user_id: default_user")
        return {"message": "Conversation cleared"}
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")