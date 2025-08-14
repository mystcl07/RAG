**Hybrid Retrieval System with RAG**

This is a Retrieval-Augmented Generation (RAG) project built with LangChain, designed to answer questions from PDFs, URLs, or both using Semantic Search (FAISS) or Hybrid Search (FAISS + BM25). The system processes documents, indexes them for retrieval, and provides answers via a user-friendly Streamlit interface, with a FastAPI backend handling retrieval and generation.

**Features**

Document Processing: Upload PDFs or scrape URLs to extract and index text content.
Hybrid Retrieval: Combines FAISS (semantic search) and BM25 (keyword-based search) with weights 0.7 and 0.3, respectively, for balanced retrieval.
Query Support: Answers questions, summarizes content, or translates text based on user input.
Interactive Interface: Streamlit frontend for uploading documents, submitting queries, and viewing results with highlighted keywords.
Backend API: FastAPI handles document processing, retrieval, and conversation management.
Conversation History: Stores user interactions in a SQLite database for context-aware responses.

**Prerequisites**

Python 3.9+
Dependencies listed in requirements.txt
A Gemini API key (for the language model)

**Setup**

**Clone the Repository:**
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

**Install Dependencies:**
   pip install -r requirements.txt

**Set Up Environment Variables:**
Create a .env file in the project root.
Add your Gemini API key: plaintext GEMINI_API_KEY=your-api-key

**Configure Directories:**
Update the PDFS_DIRECTORY in backend/config.py to a valid path (e.g., ./pdfs) where uploaded PDFs will be stored.
Ensure the directory exists and is writable: bash mkdir pdfs

**Running the Project**
**Start the FastAPI Backend:** **Open a terminal and run:**
   uvicorn backend.main:app --host 0.0.0.0 --port 8000

**Start the Streamlit Frontend: Open a second terminal and run:**
   streamlit run frontend/app.py

**Access the Application:**
Open your browser and navigate to http://localhost:8501 (default Streamlit port).
Upload a PDF or provide a URL, then submit queries to retrieve answers or summaries.
Usage
**Upload Documents:** Use the Streamlit interface to upload PDFs or enter URLs for scraping.
Ask questions to get answers based on retrieved segments.
**Search Modes:** Choose between Semantic Search (FAISS) or Hybrid Search (FAISS + BM25).
**View Results:** Results include the top-k relevant segments with highlighted keywords and source metadata.
