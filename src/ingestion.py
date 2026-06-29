import os
import pypdf
import docx2txt

def clean_extracted_text(text):
    """
    Cleans extracted text by stripping leading/trailing whitespace, 
    filtering out lines that are just page numbers, and removing empty lines.
    """
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that contain only digits (likely standalone page numbers)
        if stripped.isdigit():
            continue
        # Skip typical header/footer artifacts like single characters or tiny page markers (e.g., "Page 1 of 10")
        if stripped.lower().startswith("page ") and len(stripped) < 15:
            continue
        # Otherwise, keep the line
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def load_pdf(file_path):
    """Loads a PDF file and extracts text page-by-page."""
    pages_data = []
    try:
        reader = pypdf.PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = clean_extracted_text(text)
            if text.strip():
                pages_data.append({
                    "text": text,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "page": idx + 1,
                        "type": "pdf"
                    }
                })
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
    return pages_data

def load_docx(file_path):
    """Loads a DOCX file and extracts its text contents."""
    pages_data = []
    try:
        text = docx2txt.process(file_path)
        text = clean_extracted_text(text)
        if text.strip():
            # For DOCX, we treat sections separated by multiple newlines as logical blocks
            # if possible, or just treat it as one doc with subparts.
            # Here we split by double newlines to simulate page-like logical sections
            paragraphs = text.split("\n\n")
            current_chunk = []
            section_idx = 1
            char_count = 0
            
            for para in paragraphs:
                if para.strip():
                    current_chunk.append(para)
                    char_count += len(para)
                    # Create logical pages/sections of around 2000 chars
                    if char_count > 2000:
                        pages_data.append({
                            "text": "\n\n".join(current_chunk),
                            "metadata": {
                                "source": os.path.basename(file_path),
                                "page": section_idx,
                                "type": "docx"
                            }
                        })
                        current_chunk = []
                        char_count = 0
                        section_idx += 1
            
            # Add remaining text
            if current_chunk:
                pages_data.append({
                    "text": "\n\n".join(current_chunk),
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "page": section_idx,
                        "type": "docx"
                    }
                })
    except Exception as e:
        print(f"Error loading DOCX {file_path}: {e}")
    return pages_data

def load_txt(file_path):
    """Loads a TXT file and extracts its contents."""
    pages_data = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text = clean_extracted_text(text)
        if text.strip():
            # Split by double newline to form logical page sections
            paragraphs = text.split("\n\n")
            current_chunk = []
            section_idx = 1
            char_count = 0
            
            for para in paragraphs:
                if para.strip():
                    current_chunk.append(para)
                    char_count += len(para)
                    if char_count > 2000:
                        pages_data.append({
                            "text": "\n\n".join(current_chunk),
                            "metadata": {
                                "source": os.path.basename(file_path),
                                "page": section_idx,
                                "type": "txt"
                            }
                        })
                        current_chunk = []
                        char_count = 0
                        section_idx += 1
            
            if current_chunk:
                pages_data.append({
                    "text": "\n\n".join(current_chunk),
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "page": section_idx,
                        "type": "txt"
                    }
                })
    except Exception as e:
        print(f"Error loading TXT {file_path}: {e}")
    return pages_data

def load_document(file_path):
    """Detects the extension of the document and invokes the correct loader."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext in [".txt", ".md"]:
        return load_txt(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return []

def load_directory(dir_path):
    """Loads all supported documents in a directory."""
    all_documents = []
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} does not exist.")
        return all_documents
        
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            docs = load_document(file_path)
            all_documents.extend(docs)
            print(f"Loaded {len(docs)} pages/sections from {filename}")
            
    return all_documents
