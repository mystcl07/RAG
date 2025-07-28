# app.py
# Creates a Streamlit frontend for uploading PDFs/URLs, selecting search modes, 
# displaying conversation history with source documents, and sending queries to the FastAPI backend.

import streamlit as st
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend API configuration
API_BASE_URL = "http://localhost:8000"
USER_ID = "default_user"

# Set up Streamlit page
st.set_page_config(page_title="Conversational AI with LangChain", layout="wide", page_icon="🤖")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to fetch conversation history
def fetch_conversation_history():
    """Fetch conversation history from the backend without generating a response."""
    try:
        response = requests.get(f"{API_BASE_URL}/history?user_id={USER_ID}")
        response.raise_for_status()
        data = response.json()
        messages = data.get("messages", [])
        st.session_state.messages = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "source_documents": msg.get("source_documents", [])
            }
            for msg in messages  # Already in chronological order from backend
        ]
        logger.info(f"Fetched {len(messages)} messages: {[m['content'][:50] for m in st.session_state.messages]}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch conversation history: {e}")
        st.error("Failed to connect to backend. Please ensure it’s running.")
        return False

# Load conversation history on startup if empty
if not st.session_state.messages:
    fetch_conversation_history()

# Sidebar for document upload and settings
with st.sidebar:
    st.title("📝 Document Upload & Settings")
    st.markdown("---")

    # Document upload
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post(f"{API_BASE_URL}/upload_pdf", files=files)
                response.raise_for_status()
                st.success("PDF processed successfully!")
            except requests.exceptions.RequestException as e:
                logger.error(f"PDF upload failed: {e}")
                st.error(f"Failed to upload PDF: {e}")

    # URL input
    url = st.text_input("Enter URL:")
    if url and st.button("Process URL"):
        with st.spinner("Processing URL..."):
            try:
                response = requests.post(f"{API_BASE_URL}/scrape_url", json={"url": url})
                response.raise_for_status()
                st.success("URL processed successfully!")
            except requests.exceptions.RequestException as e:
                logger.error(f"URL processing failed: {e}")
                st.error(f"Failed to process URL: {e}")

    # Settings
    st.markdown("---")
    st.subheader("⚙️ Options")
    search_mode = st.selectbox("Search Mode", ["Semantic", "Hybrid"], index=0)

    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Conversation"):
            try:
                response = requests.get(f"{API_BASE_URL}/clear_conversation")
                response.raise_for_status()
                st.session_state.messages = []
                st.success("Conversation cleared!")
                st.rerun()
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to clear conversation: {e}")
                st.error("Failed to clear conversation.")
    with col2:
        if st.button("📂 Clear Documents"):
            try:
                response = requests.get(f"{API_BASE_URL}/clear_documents")
                response.raise_for_status()
                st.success("Documents cleared!")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to clear documents: {e}")
                st.error("Failed to clear documents.")

# Main chat interface
st.title("💬 Conversational AI with LangChain")
st.markdown("---")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("source_documents"):
            with st.expander("📚 Source Documents"):
                for i, doc in enumerate(message["source_documents"], 1):
                    st.markdown(f"**Document {i}**")
                    st.caption(f"Source: {doc['metadata'].get('source', 'Unknown')}")
                    st.text(doc["content"] + "...")
                    st.markdown("---")

# Handle new user queries
if user_query := st.chat_input("Ask a question about your documents..."):
    # Display user question
    with st.chat_message("user"):
        st.markdown(user_query)
    # Append user message immediately
    st.session_state.messages.append({"role": "user", "content": user_query, "source_documents": []})

    # Send query to backend
    with st.spinner("Processing..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={"question": user_query, "user_id": USER_ID, "search_mode": search_mode}
            )
            response.raise_for_status()
            data = response.json()
            messages = data.get("messages", [])
            # Update session state with messages in chronological order
            st.session_state.messages = [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "source_documents": msg.get("source_documents", [])
                }
                for msg in reversed(messages)  # Reverse to show oldest first
            ]
            logger.info(f"Updated messages: {[m['content'][:50] for m in st.session_state.messages]}")
            # Display the latest assistant response
            if messages and messages[0]["role"] == "assistant":  # Backend returns desc, so first is newest
                with st.chat_message("assistant"):
                    st.markdown(messages[0]["content"])
                    if messages[0].get("source_documents"):
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(messages[0]["source_documents"], 1):
                                st.markdown(f"**Document {i}**")
                                st.caption(f"Source: {doc['metadata'].get('source', 'Unknown')}")
                                st.text(doc["content"] + "...")
                                st.markdown("---")
        except requests.exceptions.RequestException as e:
            logger.error(f"Query failed: {e}")
            st.error(f"Failed to process query: {e}")