import os
import google.generativeai as genai
from dotenv import load_dotenv

# Try importing OpenAI. If not installed, handle gracefully.
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Load env variables
load_dotenv()

def generate_answer(question, context_chunks, model_name=None):
    """
    Constructs a RAG prompt with retrieved context, prompts the LLM (Gemini or OpenAI),
    enforces context grounding, requires source citations, and returns the response.
    Respects any formatting constraints (e.g. short, brief, detailed) specified in the question.
    """
    # 1. Handle case with no context chunks retrieved
    if not context_chunks:
        return (
            "I'm sorry, but the provided documents do not contain the information needed to answer this question.", 
            []
        )
        
    # 2. Format context chunks for the prompt
    context_str = ""
    for idx, chunk in enumerate(context_chunks):
        meta = chunk["metadata"]
        filename = meta.get("source", "unknown")
        page = meta.get("page", "unknown")
        context_str += f"\n--- CONTEXT CHUNK {idx+1} ---\n"
        context_str += f"Source File: {filename}, Page/Section: {page}\n"
        context_str += f"Text Content:\n{chunk['text']}\n"
        
    # 3. Handle detail level constraint
    length_instruction = "Closely follow any specific formatting, style, or length guidelines requested in the user's question (e.g., 'in short', 'briefly', 'detailed', 'bullet points', etc.) if present. If no format or length is specified, provide a comprehensive explanation."

    # 4. Formulate the grounding system prompt
    prompt = f"""You are an expert Q&A assistant. Your goal is to answer the user's question using ONLY the provided context chunks below.
 
CONSTRAINTS:
1. Your answer must be strictly grounded in and directly supported by the context chunks provided. If the context does not contain the answer or if you cannot find enough information to answer, reply exactly with: "I'm sorry, but the provided documents do not contain the information needed to answer this question."
2. Do NOT use your own external training data or general knowledge. 
3. For every fact, statement, or claim, you must include an inline citation indicating the source file and page/section number, e.g., (Source: filename, Page/Section X) or [Source: filename, Page/Section X].
4. Maintain a professional, clear, and objective tone.
5. Format/Length: {length_instruction}
 
CONTEXT CHUNKS:
{context_str}
 
USER QUESTION: {question}
 
GROUNDED ANSWER:"""

    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    # 4. Invoke the appropriate LLM provider
    if openai_key and (not gemini_key or os.getenv("LLM_PROVIDER") == "openai"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library is not installed. Please run pip install openai.")
            
        try:
            client = OpenAI(api_key=openai_key)
            actual_model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            
            response = client.chat.completions.create(
                model=actual_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful, strictly grounded RAG bot."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip()
            return answer, context_chunks
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Error generating answer with OpenAI: {e}", context_chunks
            
    else:
        # Default to Gemini API
        from src.embeddings import init_gemini
        try:
            init_gemini()
            actual_model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            model = genai.GenerativeModel(actual_model_name)
            
            response = model.generate_content(prompt)
            answer = response.text.strip()
            
            return answer, context_chunks
        except Exception as e:
            print(f"Error calling Gemini generation API: {e}")
            return f"Error generating answer with Gemini: {e}", context_chunks
