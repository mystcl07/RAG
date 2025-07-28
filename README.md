This is a Langchain RAG project that answers questions from a pdf, url or both combined using Semantic Search or Hybrid RAG.

To run the project:

Change the directories wherever applicable and add an API key in the .env to make use of Gemini

Open 2 terminals and use the commands in each terminal

1. uvicorn backend.main:app --host 0.0.0.0 --port 8000
2. streamlit run frontend/app.py
