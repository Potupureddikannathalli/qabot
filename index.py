import os
import sys
from dotenv import load_dotenv

# Ensure Unicode characters (like Chinese/Japanese folder names) print correctly on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Ensure the root folder is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import load_directory
from src.chunking import chunk_documents
from src.embeddings import get_embeddings
from src.vector_store import get_vector_store

def main():
    load_dotenv()
    
    # 1. Fetch directories and parameters from environment
    data_dir = os.getenv("DATA_DIR", "data")
    db_dir = os.getenv("VECTOR_DB_DIR", "db")
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
    
    print("=" * 60)
    print("                 RAG PIPELINE: INDEXING STEP")
    print("=" * 60)
    print(f"Data Directory:       {os.path.abspath(data_dir)}")
    print(f"Database Directory:   {os.path.abspath(db_dir)}")
    print(f"Chunk Size:           {chunk_size} characters")
    print(f"Chunk Overlap:        {chunk_overlap} characters")
    print("-" * 60)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        print("Please create the folder and add documents (PDF, DOCX, TXT) first.")
        sys.exit(1)
        
    # Check for keys
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Neither GEMINI_API_KEY nor OPENAI_API_KEY is set in the environment.")
        print("Please configure your .env file with at least one API key.")
        sys.exit(1)

    # 2. Ingest Documents
    print("\nStep 1: Loading documents from directory...")
    raw_docs = load_directory(data_dir)
    if not raw_docs:
        print("No documents successfully loaded. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(raw_docs)} total document pages/sections.")

    # 3. Chunk Documents
    print("\nStep 2: Splitting documents into overlapping chunks...")
    chunks = chunk_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Generated {len(chunks)} total text chunks.")
    
    # 4. Generate Embeddings
    provider = os.getenv("EMBEDDING_PROVIDER", "openai" if os.getenv("OPENAI_API_KEY") else "gemini")
    print(f"\nStep 3: Generating text embeddings using {provider.upper()} API...")
    print("This will send embedding requests in batches. Please wait...")
    
    chunk_texts = [c["text"] for c in chunks]
    try:
        # Select embedding model based on provider
        emb_model = "text-embedding-3-small" if provider == "openai" else "models/text-embedding-004"
        embeddings = get_embeddings(chunk_texts, model=emb_model, batch_size=100, is_query=False)
        print(f"Successfully generated {len(embeddings)} embeddings vectors.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        sys.exit(1)

    # 5. Store in Vector Database
    print("\nStep 4: Storing chunks and embeddings in persistent vector database...")
    store = get_vector_store(db_dir)
    
    # Clear existing documents to start fresh
    store.clear()
    
    # Add documents and their embeddings
    store.add_documents(chunks, embeddings)
    
    print("-" * 60)
    print("INDEXING COMPLETED SUCCESSFULLY!")
    print(f"The vector store is persisted at '{db_dir}' and ready to be queried.")
    print("=" * 60)

if __name__ == "__main__":
    main()
