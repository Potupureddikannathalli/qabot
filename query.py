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

from src.cli import run_query, interactive_loop

def main():
    load_dotenv()
    
    # Check if a single query was passed as command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        db_dir = os.getenv("VECTOR_DB_DIR", "db")
        k = int(os.getenv("RETRIEVAL_K", 4))
        
        # Check for keys
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            print("Error: Neither GEMINI_API_KEY nor OPENAI_API_KEY is set in the environment.")
            print("Please configure your .env file with at least one API key.")
            sys.exit(1)
            
        run_query(query, db_dir=db_dir, k=k)
    else:
        # Start interactive CLI loop
        interactive_loop()

if __name__ == "__main__":
    main()
