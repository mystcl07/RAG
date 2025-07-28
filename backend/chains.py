# chains.py 
# Defines functions to answer questions using a language model with document context and conversation history, 
# summarize text into bullet points, and translate text to a specified language.

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from backend.config import MODEL, QA_TEMPLATE, SUMMARY_TEMPLATE, TRANSLATION_TEMPLATE

def answer_question(question, documents, memory):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(QA_TEMPLATE)
    chain = prompt | MODEL
    
    response = chain.invoke({
        "question": question,
        "context": context,
        "history": memory.load_memory_variables({})['history']
    })
    
    memory.save_context({"input": question}, {"output": response.content})
    return response.content

def summarize_text(text):
    prompt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
    chain = prompt | MODEL
    result = chain.invoke({"text": text})
    return result.content

def translate_text(text, target_language="Hindi"):
    prompt = ChatPromptTemplate.from_template(TRANSLATION_TEMPLATE)
    chain = prompt | MODEL
    result = chain.invoke({"text": text, "target_language": target_language})
    return result.content