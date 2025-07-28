# backend/document_processor.py
# Handles PDF uploads, URL scraping, text splitting into chunks, 
# and indexing documents into a FAISS vector store for retrieval, with error logging and retries.

import os
from langchain_community.document_loaders import PDFPlumberLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS  # Updated import
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from backend.config import PDFS_DIRECTORY, EMBEDDINGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_pdf(file):
    file_path = os.path.join(PDFS_DIRECTORY, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path

def load_pdf(file_path):
    try:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        if not documents:
            logger.error(f"No content extracted from PDF: {file_path}")
            return []
        return documents
    except Exception as e:
        logger.error(f"Failed to process PDF {file_path}: {str(e)}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def scrape_url(url: str):
    try:
        loader = WebBaseLoader(url, requests_kwargs={"timeout": 10, "headers": {"User-Agent": os.getenv("USER_AGENT", "LangChain-WebBaseLoader")}})
        documents = loader.load()
        if documents:
            cleaned_text = " ".join(doc.page_content.strip() for doc in documents if doc.page_content)
            return [Document(page_content=cleaned_text, metadata={"source": url})]
        else:
            logger.error(f"No content retrieved from {url}")
            return []
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {str(e)}")
        return []

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_docs(documents, vector_store=None):
    if documents:
        if vector_store is None:
            vector_store = FAISS.from_documents(documents, EMBEDDINGS)
            logger.info(f"Indexed new vector store with {len(documents)} chunks")
        else:
            vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} chunks to existing vector store")
    else:
        logger.warning("No documents to index")
    return vector_store

def clear_vector_store(vector_store=None):
    logger.info("Vector store cleared")
    return None