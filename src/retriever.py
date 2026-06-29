from src.embeddings import get_embeddings
from src.vector_store import get_vector_store

def retrieve_context(query, db_dir=None, k=4, force_numpy=False):
    """
    Given a natural language query, generates its embedding,
    queries the persistent vector database, and returns the top-k relevant chunks.
    """
    # 1. Embed query
    query_embeddings = get_embeddings([query], is_query=True)
    if not query_embeddings:
        return []
    query_vector = query_embeddings[0]
    
    # 2. Initialize vector store
    store = get_vector_store(db_dir=db_dir, force_numpy=force_numpy)
    
    # 3. Perform similarity search
    retrieved_chunks = store.query(query_vector, k=k)
    return retrieved_chunks
