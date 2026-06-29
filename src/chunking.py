def split_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits text into chunks of maximum size `chunk_size` characters, 
    with a sliding window overlap of `chunk_overlap` characters.
    Aligns chunk boundaries to the nearest whitespace or punctuation 
    to avoid splitting words.
    """
    text_len = len(text)
    if text_len <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    
    while start < text_len:
        end = start + chunk_size
        
        if end >= text_len:
            chunks.append(text[start:])
            break
            
        # Look backward from 'end' up to 'chunk_overlap' chars to find a clean break (whitespace/punctuation)
        adjusted_end = end
        for i in range(end, max(start, end - chunk_overlap), -1):
            if i - 1 < text_len and text[i - 1] in [' ', '\n', '\t', '.', ',', ';', '!', '?']:
                adjusted_end = i
                break
                
        # If we couldn't find a clean break, just split at 'end'
        chunks.append(text[start:adjusted_end])
        
        # Next start starts before adjusted_end by chunk_overlap
        start = adjusted_end - chunk_overlap
        
        # Guard against stuck loops
        if start >= adjusted_end:
            start = adjusted_end
            
    return chunks

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Takes a list of documents (dicts with 'text' and 'metadata') and 
    splits their text into smaller chunks, replicating the metadata and 
    adding a chunk index.
    """
    chunked_docs = []
    
    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]
        
        # Skip empty documents
        if not text.strip():
            continue
            
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        for idx, chunk_text in enumerate(chunks):
            # Create a copy of metadata and add chunking details
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = idx
            
            chunked_docs.append({
                "text": chunk_text.strip(),
                "metadata": chunk_metadata
            })
            
    return chunked_docs
