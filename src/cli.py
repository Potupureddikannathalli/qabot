import os
import sys
from dotenv import load_dotenv
from src.retriever import retrieve_context
from src.generator import generate_answer

# Ensure Unicode characters (like Chinese/Japanese folder names) print correctly on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def run_query(query, db_dir=None, k=4, verbose=True):
    """Executes a single RAG query, prints the results, and returns them."""
    if verbose:
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("-" * 60)
        print("Retrieving context from database...")
        
    # 1. Retrieve relevant chunks
    try:
        chunks = retrieve_context(query, db_dir=db_dir, k=k)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return None, []
        
    if verbose:
        print(f"Retrieved {len(chunks)} relevant document chunks.")
        print("Generating answer...")
        print("-" * 60)
        
    # 2. Generate answer
    answer, retrieved_chunks = generate_answer(query, chunks)
    
    if verbose:
        print("\nANSWER:")
        print(answer)
        print("-" * 60)
        print("SOURCES USED:")
        if not retrieved_chunks:
            print("No sources used.")
        else:
            # Print unique sources
            for idx, chunk in enumerate(retrieved_chunks):
                meta = chunk["metadata"]
                filename = meta.get("source", "unknown")
                page = meta.get("page", "unknown")
                
                print(f"[{idx+1}] File: {filename} | Page/Section: {page}")
                # Print a small snippet of the chunk for verification
                snippet = chunk["text"][:150].replace('\n', ' ')
                print(f"    Snippet: \"{snippet}...\"")
        print("=" * 60 + "\n")
        
    return answer, retrieved_chunks

def interactive_loop():
    """Starts an interactive command-line interface loop for the RAG Q&A Bot."""
    load_dotenv()
    db_dir = os.getenv("VECTOR_DB_DIR", "db")
    k = int(os.getenv("RETRIEVAL_K", 4))
    
    # Check for keys
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Neither GEMINI_API_KEY nor OPENAI_API_KEY is set in the environment.")
        print("Please configure your .env file with at least one API key.")
        return

    print("=" * 60)
    print("        RAG PIPELINE: INTERACTIVE DOCUMENT Q&A BOT")
    print("=" * 60)
    print("Type your question and press Enter.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nAsk a question > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting. Goodbye!")
                break
                
            run_query(query, db_dir=db_dir, k=k)
            
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
