import os
import sys
import shutil
import uuid
import streamlit as st
from dotenv import load_dotenv

# Ensure the root folder is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import retrieve_context
from src.generator import generate_answer
from src.ingestion import load_directory
from src.chunking import chunk_documents
from src.embeddings import get_embeddings
from src.vector_store import get_vector_store

# Load environmental variables
load_dotenv()

# 1. Initialize session state variables and load chats metadata
import json

CHATS_METADATA_FILE = os.path.join("db", "chats.json")

def load_chats_metadata():
    if os.path.exists(CHATS_METADATA_FILE):
        try:
            with open(CHATS_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chats metadata: {e}")
    return []

def save_chats_metadata(chats):
    try:
        os.makedirs(os.path.dirname(CHATS_METADATA_FILE), exist_ok=True)
        with open(CHATS_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(chats, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving chats metadata: {e}")

if "chats" not in st.session_state:
    st.session_state.chats = load_chats_metadata()
if "active_chat_id" not in st.session_state:
    if st.session_state.chats:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]
    else:
        st.session_state.active_chat_id = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if "suggestion_query" not in st.session_state:
    st.session_state.suggestion_query = None

# Helper to create a new chat
def create_new_chat():
    chat_id = str(uuid.uuid4())
    new_chat = {
        "id": chat_id,
        "title": "New Chat",
        "chat_history": []
    }
    st.session_state.chats.insert(0, new_chat)
    st.session_state.active_chat_id = chat_id
    # Create directories for files
    os.makedirs(os.path.join("data", chat_id), exist_ok=True)
    os.makedirs(os.path.join("db", chat_id), exist_ok=True)
    save_chats_metadata(st.session_state.chats)
    return chat_id

# Helper to delete a chat
def delete_chat(chat_id):
    st.session_state.chats = [c for c in st.session_state.chats if c["id"] != chat_id]
    chat_data = os.path.join("data", chat_id)
    chat_db = os.path.join("db", chat_id)
    try:
        if os.path.exists(chat_data):
            shutil.rmtree(chat_data)
        if os.path.exists(chat_db):
            shutil.rmtree(chat_db)
    except Exception as e:
        print(f"Error deleting chat directories: {e}")
        
    if st.session_state.active_chat_id == chat_id:
        if st.session_state.chats:
            st.session_state.active_chat_id = st.session_state.chats[0]["id"]
        else:
            st.session_state.active_chat_id = None
    save_chats_metadata(st.session_state.chats)

# Ensure there is at least one active chat session
if not st.session_state.chats or st.session_state.active_chat_id is None:
    create_new_chat()

active_chat_id = st.session_state.active_chat_id
active_chat = next((c for c in st.session_state.chats if c["id"] == active_chat_id), None)
if active_chat is None:
    if st.session_state.chats:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]
        active_chat = st.session_state.chats[0]
        active_chat_id = active_chat["id"]
    else:
        active_chat_id = create_new_chat()
        active_chat = st.session_state.chats[0]

# Sync session state to environment variables so all submodules see them
os.environ["GEMINI_API_KEY"] = st.session_state.gemini_api_key
os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

# App Page Config
st.set_page_config(
    page_title="InsightDocs AI - ChatGPT Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar visible by default
)

# Custom Styling (ChatGPT Light Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Font styles */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #374151;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #111827;
    }
    
    /* Main Layout - Light Theme */
    .stApp {
        background-color: #f9fafb !important;
        color: #1f2937 !important;
    }
    
    /* Sidebar Layout Override */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #374151 !important;
    }
    
    /* Elegant Title Header */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        padding: 25px 0 15px 0;
        margin-bottom: 25px;
        background: radial-gradient(circle at center, rgba(243, 244, 246, 0.6) 0%, rgba(249, 250, 251, 0.9) 100%);
        border-radius: 16px;
        border: 1px solid #e5e7eb;
    }
    
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        background: linear-gradient(135deg, #10a37f 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: 5px;
        margin-bottom: 0px;
    }
    
    /* Source Chunk cards */
    .source-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .source-card:hover {
        transform: translateY(-1px);
        border-color: #10a37f;
        box-shadow: 0 4px 20px rgba(16, 163, 127, 0.08);
    }
    
    .source-badge {
        display: inline-block;
        background: linear-gradient(90deg, #10a37f 0%, #059669 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 3px 8px;
        border-radius: 20px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Custom button styling */
    div.stButton > button {
        background: #ffffff !important;
        color: #374151 !important;
        border: 1px solid #d1d5db !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        border-color: #10a37f !important;
        color: #10a37f !important;
        background: #f0fdf4 !important;
    }
    
    /* Primary Action Buttons */
    .primary-btn div.stButton > button {
        background: linear-gradient(135deg, #10a37f 0%, #059669 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    .primary-btn div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(16, 163, 127, 0.3) !important;
        color: #ffffff !important;
    }

    /* Danger / Clear buttons styling */
    .danger-btn div.stButton > button {
        background: #ffffff !important;
        color: #dc2626 !important;
        border: 1px solid #fca5a5 !important;
    }
    
    .danger-btn div.stButton > button:hover {
        background: #fef2f2 !important;
        border-color: #dc2626 !important;
        color: #dc2626 !important;
    }
    
    /* Chat suggestion cards styling */
    .suggestion-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0px !important;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .suggestion-card:hover {
        border-color: #10a37f;
        background-color: #f9fafb;
        transform: translateY(-2px);
    }
    
    /* Target buttons inside suggestion card */
    .suggestion-card button {
        background: transparent !important;
        border: none !important;
        color: #374151 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 16px !important;
        width: 100% !important;
        height: 100% !important;
        white-space: normal !important;
        display: block !important;
    }
    
    .suggestion-card button:hover {
        color: #10a37f !important;
        background: transparent !important;
    }
    
    /* Custom style for inline radio select */
    div[data-testid="stRadio"] > label {
        font-weight: 600;
        color: #4b5563;
    }

    /* Chat bubble customizing and centering */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Center the chat input container */
    div[data-testid="stChatInput"] {
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Streamlit Chat inputs style */
    div[data-testid="stChatInput"] input {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
    }
    div[data-testid="stChatInput"] input:focus {
        border-color: #10a37f !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. Reusable Indexing Helper
def index_documents(data_dir, db_dir, chunk_size, chunk_overlap, provider, model_choice, force_numpy):
    """Ingests, chunks, embeds, and stores documents in the vector database."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not gemini_key and not openai_key:
        st.error("Cannot index: No active API Key found. Please configure a key in the settings.")
        return False
        
    try:
        raw_docs = load_directory(data_dir)
        store = get_vector_store(db_dir, force_numpy=force_numpy)
        
        if not raw_docs:
            store.clear()
            return True
            
        chunks = chunk_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            store.clear()
            return True
            
        chunk_texts = [c["text"] for c in chunks]
        
        if provider == "Google Gemini":
            os.environ["EMBEDDING_PROVIDER"] = "gemini"
            emb_model = "models/text-embedding-004"
        else:
            os.environ["EMBEDDING_PROVIDER"] = "openai"
            emb_model = "text-embedding-3-small"
            
        embeddings = get_embeddings(chunk_texts, model=emb_model, batch_size=100, is_query=False)
        
        store.clear()
        store.add_documents(chunks, embeddings)
        return True
    except Exception as e:
        st.error(f"Error during indexing pipeline: {e}")
        return False

# 3. Configuration Directory Setup
data_dir = os.path.join("data", active_chat_id)
db_dir = os.path.join("db", active_chat_id)

# Create folders if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(db_dir, exist_ok=True)

# Helper to fetch stored files details
def get_stored_files(data_dir):
    if not os.path.exists(data_dir):
        return []
    files = []
    for f in os.listdir(data_dir):
        path = os.path.join(data_dir, f)
        if os.path.isfile(path) and f.lower().endswith(('.pdf', '.docx', '.txt', '.md')):
            size_bytes = os.path.getsize(path)
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            
            ext = os.path.splitext(f)[1].lower()
            files.append({
                "name": f,
                "path": path,
                "type": ext.replace(".", "").upper(),
                "size": size_str,
                "raw_size": size_bytes
            })
    return files

# Resolve settings quietly from environment & keys
active_gemini_key = st.session_state.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
active_openai_key = st.session_state.openai_api_key or os.getenv("OPENAI_API_KEY", "")

if active_openai_key and (not active_gemini_key or os.getenv("LLM_PROVIDER", "openai") == "openai"):
    provider = "OpenAI"
    model_choice = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    emb_model = "text-embedding-3-small"
else:
    provider = "Google Gemini"
    model_choice = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    emb_model = "models/text-embedding-004"

# Settings parameters fallback
force_numpy = False
retrieval_k = int(os.getenv("RETRIEVAL_K", 4))
chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))

# Shared helper to handle uploaded files
def handle_uploaded_files(uploaded_files):
    if not uploaded_files:
        return
    new_files = []
    for uploaded_file in uploaded_files:
        dest_path = os.path.join(data_dir, uploaded_file.name)
        # Check if file already exists with same size
        exists = os.path.exists(dest_path) and os.path.getsize(dest_path) == uploaded_file.size
        if not exists:
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            new_files.append(uploaded_file.name)
            
    if new_files:
        st.toast(f"📥 Saved {len(new_files)} new file(s).")
        with st.spinner("Processing & Indexing..."):
            success = index_documents(
                data_dir=data_dir,
                db_dir=db_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                provider=provider,
                model_choice=model_choice,
                force_numpy=force_numpy
            )
            if success:
                st.toast("✅ Index updated successfully!")
                st.session_state.uploader_key += 1
                st.rerun()

# 4. Presentation Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">InsightDocs AI</h1>
    <p class="subtitle">Secure Retrieval-Augmented Generation Document Assistant</p>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar Layout Construction
with st.sidebar:
    st.markdown("### 💬 Chats Workspace")
    
    # New Chat Button
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("")
    
    # List Existing Chats
    for idx, chat in enumerate(st.session_state.chats):
        is_active = (chat["id"] == active_chat_id)
        c_col1, c_col2 = st.columns([7.8, 2.2])
        with c_col1:
            prefix = "➡️ 💬 " if is_active else "💬 "
            title_display = chat["title"]
            if len(title_display) > 20:
                title_display = title_display[:18] + "..."
            
            if st.button(f"{prefix}{title_display}", key=f"sel_chat_{chat['id']}", use_container_width=True):
                st.session_state.active_chat_id = chat["id"]
                st.rerun()
        with c_col2:
            st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
            if st.button("🗑️", key=f"del_chat_{chat['id']}", help="Delete chat"):
                delete_chat(chat["id"])
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Document list inside sidebar
    st.markdown("---")
    st.markdown("### 📚 Chat Documents")
    
    # Sidebar file uploader to allow multi-file uploads at any point
    sidebar_uploaded_files = st.file_uploader(
        "Upload files to this chat",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"sidebar_uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed"
    )
    if sidebar_uploaded_files:
        handle_uploaded_files(sidebar_uploaded_files)
        
    stored_files = get_stored_files(data_dir)
    if not stored_files:
        st.info("No files in this chat.")
    else:
        # Display list representation
        for file_info in stored_files:
            row_col1, row_col2 = st.columns([7.5, 2.5])
            with row_col1:
                icon = "📕" if file_info["type"] == "PDF" else "📘" if file_info["type"] == "DOCX" else "📄"
                st.markdown(f"<div style='line-height:1.2; overflow:hidden; text-overflow:ellipsis;'><strong>{icon} {file_info['name']}</strong><br><span style='font-size:0.75rem; color:#6b7280;'>{file_info['type']} • {file_info['size']}</span></div>", unsafe_allow_html=True)
            with row_col2:
                st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_file_{file_info['name']}", help=f"Delete {file_info['name']}"):
                    try:
                        os.remove(file_info["path"])
                        st.toast(f"Deleted {file_info['name']}")
                        with st.spinner("Updating index..."):
                            index_documents(
                                data_dir=data_dir,
                                db_dir=db_dir,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                provider=provider,
                                model_choice=model_choice,
                                force_numpy=force_numpy
                            )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error removing file: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
                        
        st.write("")
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("🚨 Clear All Docs", use_container_width=True, help="Clear all stored documents in this chat"):
            for file_info in stored_files:
                try:
                    os.remove(file_info["path"])
                except Exception:
                    pass
            with st.spinner("Purging vector database..."):
                store = get_vector_store(db_dir, force_numpy=force_numpy)
                store.clear()
            st.toast("Purged chat documents.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    
    # Response Settings in Sidebar (detail level radio removed as requested)
    st.markdown("### 💬 Chat Options")
    
    st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
    if st.button("🧹 Clear Chat History", use_container_width=True, help="Reset conversation history"):
        active_chat["chat_history"] = []
        save_chats_metadata(st.session_state.chats)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clean API Keys panel
    st.markdown("---")
    with st.expander("🔑 API Keys", expanded=False):
        gemini_input = st.text_input(
            "Gemini API Key", 
            value=st.session_state.gemini_api_key, 
            type="password",
            placeholder="AIzaSy..."
        )
        if gemini_input != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = gemini_input
            os.environ["GEMINI_API_KEY"] = gemini_input
            st.rerun()
            
        openai_input = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.openai_api_key, 
            type="password",
            placeholder="sk-proj-..."
        )
        if openai_input != st.session_state.openai_api_key:
            st.session_state.openai_api_key = openai_input
            os.environ["OPENAI_API_KEY"] = openai_input
            st.rerun()

# --- MAIN CONSOLE: Chat Workspace ---
# Render main content
if not stored_files:
    # Beautiful welcome screen with folder upload directly in chat center
    st.markdown("""
    <div style="text-align: center; margin-top: 60px; margin-bottom: 20px;">
        <span style="font-size: 4rem;">📁</span>
        <h2 style="font-weight: 800; color:#111827; margin-top: 10px;">Upload documents to start this chat</h2>
        <p style="color: #6b7280; font-size: 1rem; max-width: 500px; margin: 5px auto 25px auto;">
            Upload multiple PDF, DOCX, TXT, or MD files. The documents will be processed, chunked, and stored securely for this chat thread.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center uploader
    uploader_col1, uploader_col2, uploader_col3 = st.columns([1, 2, 1])
    with uploader_col2:
        uploaded_files = st.file_uploader(
            "Upload files for this chat",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed"
        )
        
        # Handle Uploaded files
        if uploaded_files:
            handle_uploaded_files(uploaded_files)

elif not active_chat["chat_history"]:
    # Beautiful welcome screen with prompt suggestions
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; margin-bottom: 35px;">
        <span style="font-size: 3.5rem;">🤖</span>
        <h2 style="font-weight: 700; margin-top: 10px; color:#111827;">How can I help you with your documents today?</h2>
        <p style="color: #6b7280; font-size: 0.95rem; margin-top: 5px;">Ask questions based on the uploaded files. Click below to try a sample prompt:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Grid of suggestions
    sug_col1, sug_col2 = st.columns(2, gap="medium")
    with sug_col1:
        st.markdown('<div class="suggestion-card">', unsafe_allow_html=True)
        if st.button("📝 Summarize key takeaways\nGet a quick summary of the uploaded documents.", key="sug_summary", use_container_width=True):
            st.session_state.suggestion_query = "Please provide a summary of the main points and key takeaways from the uploaded documents."
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="suggestion-card" style="margin-top:15px;">', unsafe_allow_html=True)
        if st.button("🔍 Find Action Items\nList tasks or action items mentioned in the files.", key="sug_actions", use_container_width=True):
            st.session_state.suggestion_query = "What are the key action items or next steps outlined in these documents?"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with sug_col2:
        st.markdown('<div class="suggestion-card">', unsafe_allow_html=True)
        if st.button("❓ Highlight Major Risks\nIdentify any potential risks, liabilities, or issues.", key="sug_risks", use_container_width=True):
            st.session_state.suggestion_query = "Are there any major risks, warnings, liabilities, or potential issues highlighted in the documents?"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="suggestion-card" style="margin-top:15px;">', unsafe_allow_html=True)
        if st.button("📅 Dates and Deadlines\nExtract all important dates and key deadlines.", key="sug_dates", use_container_width=True):
            st.session_state.suggestion_query = "Please extract all key dates, deadlines, or milestones mentioned in the documents and list them."
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Render active history
    for msg in active_chat["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("📚 View Grounded Citations & Sources"):
                    for idx, chunk in enumerate(msg["citations"]):
                        meta = chunk["metadata"]
                        filename = meta.get("source", "unknown")
                        page = meta.get("page", "unknown")
                        snippet = chunk["text"]
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="source-badge">Source {idx+1} : {filename} (Page/Section {page})</span>
                            <div style="font-size:0.92rem; line-height: 1.6; color: #374151; white-space: pre-line;">
                            {snippet}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# Chat Input Box and Suggestion processor (only if files are uploaded)
if stored_files:
    query = st.chat_input("Ask a question based on your uploaded documents...")
    if st.session_state.suggestion_query:
        query = st.session_state.suggestion_query
        st.session_state.suggestion_query = None

    if query:
        # 1. User Message
        with st.chat_message("user"):
            st.markdown(query)
        active_chat["chat_history"].append({"role": "user", "content": query})
        
        # Rename chat if it is the first query
        if active_chat["title"] == "New Chat":
            title = query[:30]
            if len(query) > 30:
                title += "..."
            active_chat["title"] = title
            
        save_chats_metadata(st.session_state.chats)
        
        # Check key
        active_gemini_key = st.session_state.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        active_openai_key = st.session_state.openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        if not active_gemini_key and not active_openai_key:
            with st.chat_message("assistant"):
                err_msg = "⚠️ No API keys configured. Please add a `GEMINI_API_KEY` or `OPENAI_API_KEY` in the API Keys panel in the sidebar."
                st.error(err_msg)
                active_chat["chat_history"].append({"role": "assistant", "content": err_msg})
                save_chats_metadata(st.session_state.chats)
        else:
            # Sync key to env var
            if st.session_state.gemini_api_key:
                os.environ["GEMINI_API_KEY"] = st.session_state.gemini_api_key
            if st.session_state.openai_api_key:
                os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
                
            # 2. Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("Searching document index and formulating response..."):
                    try:
                        # Get query embedding
                        query_embeddings = get_embeddings([query], model=emb_model, is_query=True)
                        if not query_embeddings:
                            err_msg = "Error: Failed to generate embedding for your query."
                            st.error(err_msg)
                            active_chat["chat_history"].append({"role": "assistant", "content": err_msg})
                            save_chats_metadata(st.session_state.chats)
                        else:
                            query_vector = query_embeddings[0]
                            
                            # Query store
                            store = get_vector_store(db_dir, force_numpy=force_numpy)
                            retrieved_chunks = store.query(query_vector, k=retrieval_k)
                            
                            # Generate answer
                            answer, chunks = generate_answer(
                                query, 
                                retrieved_chunks, 
                                model_name=model_choice
                            )
                            
                            st.markdown(answer)
                            
                            if chunks:
                                with st.expander("📚 View Grounded Citations & Sources"):
                                    for idx, chunk in enumerate(chunks):
                                        meta = chunk["metadata"]
                                        filename = meta.get("source", "unknown")
                                        page = meta.get("page", "unknown")
                                        snippet = chunk["text"]
                                        st.markdown(f"""
                                        <div class="source-card">
                                            <span class="source-badge">Source {idx+1} : {filename} (Page/Section {page})</span>
                                            <div style="font-size:0.92rem; line-height: 1.6; color: #374151; white-space: pre-line;">
                                            {snippet}
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Add to history
                            active_chat["chat_history"].append({
                                    "role": "assistant",
                                    "content": answer,
                                    "citations": chunks
                            })
                            save_chats_metadata(st.session_state.chats)
                            st.rerun()
                    except Exception as e:
                        err_msg = f"An error occurred during response generation: {e}"
                        st.error(err_msg)
                        active_chat["chat_history"].append({"role": "assistant", "content": err_msg})
                        save_chats_metadata(st.session_state.chats)
