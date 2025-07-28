# backend/config.py
# Loads environment variables, sets up directories, initializes Google Generative AI embeddings and chat model, 
# and defines prompt templates for QA, summarization, and translation.

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDFS_DIRECTORY = os.getenv("PDFS_DIRECTORY")
FAISS_INDEX_DIRECTORY = os.getenv("FAISS_INDEX_DIRECTORY")
DATABASE_URL = os.getenv("DATABASE_URL")

os.makedirs(PDFS_DIRECTORY, exist_ok=True)
os.makedirs(FAISS_INDEX_DIRECTORY, exist_ok=True)

EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
MODEL = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0.4)

QA_TEMPLATE = """
Conversation History:
{history}

Context:
{context}

Question: {question}
Answer:"""

SUMMARY_TEMPLATE = """
Summarize this in 3-5 bullet points:\n\n{text}
"""

TRANSLATION_TEMPLATE = """
Translate this to {target_language}:\n\n{text}
"""