import os
from dotenv import load_dotenv

import streamlit as st

load_dotenv()


#  Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

LLM_MODEL = "gemini-3.5-flash"         
EMBEDDING_MODEL = "models/embedding-001" 
TEMPERATURE = 0.8             


def get_api_key():
    
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except:
        
        return os.getenv("GOOGLE_API_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



LLM_MODEL = "gemini-3.5-flash"          
EMBEDDING_MODEL = "models/embedding-001" 
TEMPERATURE = 0.2                     



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")



CHUNK_SIZE = 300      
CHUNK_SIZE = 300       

CHUNK_OVERLAP = 20     

CHUNK_SIZE = 300      
CHUNK_OVERLAP = 20    


TOP_K_RESULTS = 5      


FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True


STREAMLIT_TITLE = "RAG Chatbot"
STREAMLIT_ICON = "🤖"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")
