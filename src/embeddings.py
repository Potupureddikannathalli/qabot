import os
import google.generativeai as genai
from dotenv import load_dotenv

# Try importing OpenAI client. If not installed, handle gracefully.
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Load env variables
load_dotenv()

def init_gemini():
    """Initializes the Gemini API client using the environment variable key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please create a .env file and set it.")
    genai.configure(api_key=api_key)

def get_embeddings(texts, model=None, batch_size=100, is_query=False):
    """
    Generates embeddings for a list of texts in batches.
    Supports both Google Gemini API and OpenAI API based on configured keys.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    # If OpenAI key is set and Gemini key is missing, or OpenAI is explicitly preferred
    if openai_key and (not gemini_key or os.getenv("EMBEDDING_PROVIDER") == "openai"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library is not installed. Please run pip install openai.")
        
        client = OpenAI(api_key=openai_key)
        embedding_model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=embedding_model
                )
                embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                print(f"Error generating OpenAI embeddings: {e}")
                raise e
        return embeddings
        
    else:
        # Fallback to Google Gemini
        init_gemini()
        embedding_model = model or os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        task_type = "retrieval_query" if is_query else "retrieval_document"
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = genai.embed_content(
                    model=embedding_model,
                    content=batch,
                    task_type=task_type
                )
                embeddings.extend(response['embedding'])
            except Exception as e:
                print(f"Error generating Gemini embeddings: {e}")
                raise e
        return embeddings
