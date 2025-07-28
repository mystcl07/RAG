# backend/retrievers.py
# Contains functions for semantic search using FAISS and hybrid search 
# combining BM25 (keyword-based) and FAISS (semantic) to retrieve relevant documents.
from langchain_community.retrievers import BM25Retriever  # Updated import
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrieve_docs(query, vector_store):
    """
    Perform semantic search using FAISS vector store.
    
    Args:
        query (str): The user's query.
        vector_store: FAISS vector store instance.
    
    Returns:
        list: List of relevant documents.
    """
    if vector_store is None:
        logger.warning("Vector store is None - no documents indexed")
        return []
    docs = vector_store.similarity_search(query, k=5)
    logger.info(f"Retrieved {len(docs)} documents for query: {query}")
    return docs

def hybrid_retrieval(query, vector_store):
    """
    Perform hybrid search combining BM25 and FAISS.
    
    Args:
        query (str): The user's query.
        vector_store: FAISS vector store instance.
    
    Returns:
        list: List of relevant documents.
    """
    if vector_store is None:
        logger.warning("Vector store is None - no documents indexed")
        return []
    bm25_retriever = BM25Retriever.from_documents(
        [Document(page_content=doc.page_content, metadata=doc.metadata)
         for doc in vector_store.docstore._dict.values()]
    )
    bm25_retriever.k = 2
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )
    docs = ensemble_retriever.get_relevant_documents(query)
    logger.info(f"Hybrid retrieved {len(docs)} documents for query: {query}")
    return docs