import os
import pickle
import numpy as np

# Try importing chromadb. If not installed or fails, fall back to pure-numpy implementation.
try:
    import chromadb
    USE_CHROMA = True
except ImportError:
    USE_CHROMA = False

class NumPyVectorStore:
    """
    A pure-Python and NumPy fallback vector store that serializes database index to disk.
    Requires no C++ build tools and works everywhere.
    """
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.db_path = os.path.join(db_dir, "numpy_vector_store.pkl")
        os.makedirs(db_dir, exist_ok=True)
        self.embeddings = []
        self.documents = []
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    data = pickle.load(f)
                    self.embeddings = data.get("embeddings", [])
                    self.documents = data.get("documents", [])
                print(f"Loaded NumPy vector database from {self.db_path} ({len(self.documents)} chunks)")
            except Exception as e:
                print(f"Error loading NumPy database: {e}. Starting fresh.")
                self.embeddings = []
                self.documents = []
        else:
            self.embeddings = []
            self.documents = []

    def save(self):
        try:
            with open(self.db_path, "wb") as f:
                pickle.dump({
                    "embeddings": self.embeddings,
                    "documents": self.documents
                }, f)
            print(f"Saved NumPy vector database with {len(self.documents)} chunks.")
        except Exception as e:
            print(f"Failed to save NumPy database: {e}")

    def add_documents(self, documents, embeddings):
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.save()

    def query(self, query_embedding, k=4):
        if not self.embeddings:
            return []
            
        # Compute cosine similarity
        embeds_arr = np.array(self.embeddings)  # Shape (N, D)
        q_arr = np.array(query_embedding)       # Shape (D,)
        
        # Avoid division by zero
        q_norm = np.linalg.norm(q_arr)
        if q_norm == 0:
            q_norm = 1e-10
            
        embeds_norm = np.linalg.norm(embeds_arr, axis=1)
        embeds_norm[embeds_norm == 0] = 1e-10
        
        dot_product = np.dot(embeds_arr, q_arr)
        similarities = dot_product / (embeds_norm * q_norm)
        
        # Sort indices in descending order of similarity
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        retrieved = []
        for idx in top_k_indices:
            retrieved.append(self.documents[idx])
        return retrieved

    def clear(self):
        self.embeddings = []
        self.documents = []
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        print("Cleared NumPy database.")


class ChromaVectorStore:
    """
    A persistent vector store using ChromaDB.
    Persists database contents to a local directory.
    """
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.client = chromadb.PersistentClient(path=db_dir)
        # Using a deterministic distance metric (cosine similarity)
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Initialized ChromaDB at '{db_dir}'")

    def add_documents(self, documents, embeddings):
        if not documents:
            return
            
        # Create deterministic ids based on file source, page, and chunk index
        ids = []
        metadatas = []
        texts = []
        
        for idx, doc in enumerate(documents):
            meta = doc["metadata"]
            source = meta.get("source", "unknown").replace(" ", "_")
            page = meta.get("page", 1)
            chunk_idx = meta.get("chunk_index", 0)
            
            doc_id = f"{source}_p{page}_c{chunk_idx}_{idx}"
            ids.append(doc_id)
            metadatas.append(meta)
            texts.append(doc["text"])
            
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        print(f"Added {len(documents)} chunks to ChromaDB collection.")

    def query(self, query_embedding, k=4):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        retrieved = []
        if results and results["documents"] and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            for doc_text, meta in zip(docs, metas):
                retrieved.append({
                    "text": doc_text,
                    "metadata": meta
                })
        return retrieved

    def clear(self):
        # Delete and recreate collection to clear it
        try:
            self.client.delete_collection("rag_documents")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("Cleared ChromaDB collection.")


def get_vector_store(db_dir=None, force_numpy=False):
    """
    Returns the vector store database client. 
    Selects ChromaDB if available and not forced to use NumPy, 
    otherwise falls back to NumPyVectorStore.
    """
    if db_dir is None:
        db_dir = os.getenv("VECTOR_DB_DIR", "db")
        
    if USE_CHROMA and not force_numpy:
        try:
            return ChromaVectorStore(db_dir)
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}. Falling back to NumPyVectorStore.")
            return NumPyVectorStore(db_dir)
    else:
        return NumPyVectorStore(db_dir)
