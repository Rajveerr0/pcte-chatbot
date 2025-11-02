import os
import requests
from flask import Flask, json, render_template, request, jsonify
from dotenv import load_dotenv
from langchain.schema import Document
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    UnstructuredPDFLoader,
    TextLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urljoin, urlparse
import time
import re

load_dotenv()
app = Flask(__name__)

# 🔹 Paths
DATA_DIR = "data"
DB_DIR = "chroma_db"

# 🔹 Load API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ Missing OPENROUTER_API_KEY in .env file")

# 🔹 Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --------------------------------------------------------
# 1. Load Local Data (PDF, TXT, JSON)
# --------------------------------------------------------
def load_local_docs():
    docs = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created {DATA_DIR} directory")
        return docs

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        print(f"📄 Loading file: {file}")

        try:
            if file.endswith(".pdf"):
                try:
                    loader = PDFPlumberLoader(path)
                    loaded = loader.load()
                except Exception as e:
                    print(f"⚠️ PDFPlumber failed, trying UnstructuredPDFLoader: {e}")
                    loader = UnstructuredPDFLoader(path)
                    loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = f"local_pdf:{file}"
                docs.extend(loaded)
                print(f"✅ Loaded {len(loaded)} docs from {file}")

            elif file.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = f"local_txt:{file}"
                docs.extend(loaded)
                print(f"✅ Loaded {len(loaded)} docs from {file}")

            elif file.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):  # List of chunks
                    for chunk in data:
                        if "text" in chunk:
                            docs.append(Document(
                                page_content=chunk["text"], 
                                metadata=chunk.get("metadata", {"source": f"local_json:{file}"})
                            ))
                elif "content" in data:  # Single document
                    docs.append(Document(
                        page_content=data["content"], 
                        metadata=data.get("metadata", {"source": f"local_json:{file}"})
                    ))
                else:  # Try to convert the whole JSON to string
                    docs.append(Document(
                        page_content=str(data), 
                        metadata={"source": f"local_json:{file}"}
                    ))

                print(f"✅ Loaded {len(docs)} docs from {file}")

        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")

    return docs

# --------------------------------------------------------
# 2. Improved Website Crawling with Direct Content Extraction
# --------------------------------------------------------
def crawl_and_extract_content(base_url, max_pages=10):
    visited = set()
    to_visit = [base_url]
    documents = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    session_count = 0
    
    while to_visit and session_count < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
            
        try:
            print(f"🌐 Crawling: {url}")
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", 
                               "form", "button", "iframe", "noscript"]):
                element.decompose()
            
            # Remove common repetitive elements using class selectors
            for element in soup.select('.navbar, .menu, .sidebar, .ads, .advertisement, .header, .footer'):
                element.decompose()
            
            # Get clean text from the whole page
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk) > 10)
            
            # Only create document if we have substantial content
            if len(clean_text) > 300:
                doc = Document(
                    page_content=clean_text,
                    metadata={
                        "source": url,
                        "title": soup.title.string if soup.title else "No title",
                        "url": url
                    }
                )
                documents.append(doc)
                print(f"✅ Added content from {url} ({len(clean_text)} chars)")
            
            visited.add(url)
            session_count += 1
            
            # Find and add new links (focus on main content links)
            if session_count < max_pages:
                for a in soup.find_all("a", href=True):
                    link = a["href"]
                    if (link.startswith('#') or link.startswith('javascript:') or 
                        'facebook.com' in link or 'twitter.com' in link or 'linkedin.com' in link):
                        continue
                        
                    full_url = urljoin(url, link)
                    
                    # Only follow links from the same domain
                    if (urlparse(full_url).netloc == urlparse(base_url).netloc and
                       full_url not in visited and full_url not in to_visit):
                       to_visit.append(full_url)
            
            # Be polite - delay between requests
            time.sleep(0.5)
                        
        except Exception as e:
            print(f"⚠️ Error crawling {url}: {e}")
            visited.add(url)  # Mark as visited even if error

    print(f"✅ Extracted content from {len(documents)} pages")
    return documents

# --------------------------------------------------------
# Load website content
# --------------------------------------------------------
def load_website_docs():
    try:
        print("🌐 Starting website content extraction...")
        docs = crawl_and_extract_content("https://www.pcte.edu.in", max_pages=15)
        
        # If direct crawling didn't work well, try some specific important pages
        if len(docs) < 3:
            print("🔄 Trying specific important pages...")
            important_pages = [
                "https://www.pcte.edu.in/about-us",
                "https://www.pcte.edu.in/contact-us",
                "https://www.pcte.edu.in/courses",
                "https://www.pcte.edu.in/admissions"
            ]
            
            for page in important_pages:
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    resp = requests.get(page, headers=headers, timeout=10)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    
                    # Remove unwanted elements
                    for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                        element.decompose()
                    
                    text = soup.get_text()      
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    clean_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk) > 10)
                    
                    if len(clean_text) > 200:
                        doc = Document(
                            page_content=clean_text,
                            metadata={"source": page, "title": soup.title.string if soup.title else page}
                        )
                        docs.append(doc)
                        print(f"✅ Added content from {page}")
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"⚠️ Error loading {page}: {e}")
        
        return docs
        
    except Exception as e:
        print(f"⚠️ Website loading error: {e}")
        import traceback
        traceback.print_exc()
        return []

# --------------------------------------------------------
# 3. Create or Load Vector DB
# --------------------------------------------------------
def get_vectorstore():
    # Check if we should rebuild
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("📂 Loading existing ChromaDB...")
        try:
            return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        except:
            print("⚠️ Error loading existing DB, rebuilding...")
            import shutil
            shutil.rmtree(DB_DIR)

    print("🔄 Building new ChromaDB...")
    all_docs = []
    
    # Load local documents
    local_docs = load_local_docs()
    all_docs.extend(local_docs)
    print(f"📄 Loaded {len(local_docs)} local documents")
    
    # Load website documents
    website_docs = load_website_docs()
    all_docs.extend(website_docs)
    print(f"🌐 Loaded {len(website_docs)} website documents")

    if not all_docs:
        print("⚠️ No documents found! Put PDFs/TXT in /data or check website loader.")
        # Create a minimal document to avoid complete failure
        minimal_doc = Document(
            page_content="PCTE Group of Institutes is an educational institution. For specific information, please visit the official website or contact the administration.",
            metadata={"source": "fallback_content"}
        )
        all_docs.append(minimal_doc)

    print(f"✅ Total {len(all_docs)} documents before chunking")

    # 🔹 Chunking for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,  # Increased overlap for better context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunked_docs = splitter.split_documents(all_docs)

    print(f"✅ Chunked into {len(chunked_docs)} chunks")

    # Create vectorstore - Chroma now automatically persists
    vectorstore = Chroma.from_documents(
        documents=chunked_docs, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    
    print("💾 Vector database created")
    
    # Test the database with various queries
    test_queries = ["location", "address", "contact", "where is pcte", "pcte location"]
    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        if results and results[0].page_content.strip():
            print(f"🔍 Test query '{query}': Found {len(results[0].page_content)} chars")
            print(f"   Content: {results[0].page_content[:200]}...")
        else:
            print(f"🔍 Test query '{query}': No results")
    
    return vectorstore

# Initialize vectorstore
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) if vectorstore else None  # REDUCED from 10 to 4

# --------------------------------------------------------
# 4. OpenRouter API Call
# --------------------------------------------------------
def ask_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "PCTE Assistant",
    }
    data = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500  # REDUCED from 1000 to 500
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=45)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        try:
            error_msg = f"⚠️ LLM request failed: {response.text}"
            print(error_msg)
            return "I'm having trouble connecting to my knowledge base. Please try again later."
        except:
            error_msg = f"⚠️ LLM request failed: {e}"
            print(error_msg)
            return "I encountered a technical issue. Please try again."

# --------------------------------------------------------
# 5. Response Refinement Function
# --------------------------------------------------------
def refine_response(response, query):
    """Post-process to ensure conciseness"""
    refinement_prompt = f"""
The following is a response to the query: "{query}"
Current response: "{response}"

Please make this response more concise and professional:
- Reduce to 2-3 sentences maximum (under 80 words)
- Remove redundant information
- Keep only the most essential facts
- Ensure it sounds like a professional college assistant
- Maintain bullet points only for actual lists

Refined response:"""
    
    return ask_openrouter(refinement_prompt)

# --------------------------------------------------------
# 6. Enhanced Chat Route with Concise Responses
# --------------------------------------------------------
def extract_keywords(query):
    """Extract important keywords from the query for better search"""
    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "being", "been", 
                 "and", "or", "but", "if", "then", "else", "when", "where", "how", 
                 "what", "why", "your", "my", "our", "their", "this", "that", "these", 
                 "those", "am", "is", "are", "was", "were", "be", "been", "being"}
    
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    return keywords

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.json.get("msg", "").strip()

    if not user_input:
        return jsonify({"reply": "⚠️ Please enter a question."})

    # Handle greetings
    greetings = ["hi", "hlo", "hello", "hey", "heyy", "hii", "hai", "hola", "good morning", "good afternoon", "good evening"]
    if user_input.lower() in greetings:
        return jsonify({"reply": "Hello 👋 I'm your PCTE College Assistant. How can I help you today?"})

    if not retriever:
        return jsonify({"reply": "⚠️ No data available. Please check your document sources."})

    try:
        # Extract keywords and try multiple search strategies
        keywords = extract_keywords(user_input)
        search_queries = [user_input] + keywords
        
        all_docs = []
        for query in search_queries:
            docs = retriever.get_relevant_documents(query)
            all_docs.extend(docs)
        
        # Remove duplicates while preserving order
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content[:500])  # Hash first 500 chars to identify duplicates
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Debug: print what we found
        print(f"🔍 Found {len(unique_docs)} relevant documents for: {user_input}")
        for i, doc in enumerate(unique_docs[:3]):  # Show first 3 docs
            print(f"   Doc {i+1}: {doc.metadata.get('source', 'unknown')} - {len(doc.page_content)} chars")
            print(f"      Content: {doc.page_content[:200]}...")

        # Prepare context
        context_parts = []
        for i, doc in enumerate(unique_docs):
            if doc.page_content.strip():  # Only add non-empty content
                context_parts.append(f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}")
        
        context = "\n\n".join(context_parts) if context_parts else "No specific information available."

        # NEW: Concise, professional prompt
        prompt = f"""You are a professional PCTE college information assistant. Provide clear, concise answers to student queries.

**IMPORTANT GUIDELINES:**
- Keep answers under 80 words unless detailed explanation is explicitly requested
- Use bullet points ONLY for lists of items (facilities, software, etc.)
- Focus on key facts: what, where, when, how
- Be direct and avoid unnecessary explanations
- If multiple pieces of information exist, choose the most relevant
- Maintain professional but friendly tone
- Start directly with the answer, no introductions

**Context Information:**
{context}

**User Question:** {user_input}

**Instructions:**
1. Extract the most essential information from the context
2. Provide a concise 2-3 sentence answer
3. Only use bullet points for actual lists
4. If information is incomplete, briefly mention what's available

**Answer:**"""

        response = ask_openrouter(prompt)
        
        # NEW: Post-process for extra conciseness
        if len(response.split()) > 60:  # If response is too long (more than 60 words)
            print("🔄 Response too long, refining...")
            response = refine_response(response, user_input)
        
        return jsonify({"reply": response.strip()})
    
    except Exception as e:
        print(f"⚠️ Error in chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"reply": "Sorry, I encountered a technical error. Please try again."})

@app.route("/rebuild", methods=["POST"])
def rebuild_db():
    """Endpoint to manually rebuild the database"""
    global vectorstore, retriever
    try:
        if os.path.exists(DB_DIR):
            import shutil
            shutil.rmtree(DB_DIR)
        
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) if vectorstore else None
        return jsonify({"status": "success", "message": "Database rebuilt successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# --------------------------------------------------------
# Run
# --------------------------------------------------------
# Production configuration
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Changed to 7860 for Hugging Face
    app.run(host="0.0.0.0", port=port, debug=False)
