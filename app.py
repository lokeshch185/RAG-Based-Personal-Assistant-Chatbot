from flask import Flask, request, jsonify
import fitz  # PyMuPDF for PDF processing
import os
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import requests
import uuid
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Initialize Qdrant Vector Database
qdrant = qdrant_client.QdrantClient(":memory:")

collection_name = "portfolio_content"
if not qdrant.collection_exists(collection_name):
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

embedding_model = SentenceTransformer("all-mpnet-base-v2")

PDF_FOLDER = "portfolio"
os.makedirs(PDF_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    if len(text) <= chunk_size:
        chunks.append(text)
    else:
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 200:
                chunks.append(chunk)
    return chunks

def index_pdf(pdf_path):
    """Extract text from PDF and index it in Qdrant."""
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return False
    
    chunks = chunk_text(pdf_text)
    
    # Index each chunk
    for i, chunk in enumerate(chunks):
        vector = embedding_model.encode(chunk).tolist()
        point_id = str(uuid.uuid4())
        point = PointStruct(
            id=point_id, 
            vector=vector, 
            payload={"source": os.path.basename(pdf_path), "content": chunk, "chunk_id": i}
        )
        qdrant.upsert(collection_name=collection_name, points=[point])
    
    return True

def search_qdrant(query, top_k=3):
    """Perform semantic search in Qdrant to retrieve the most relevant content."""
    query_vector = embedding_model.encode(query).tolist()
    search_results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return [{"content": hit.payload["content"], "source": hit.payload["source"]} for hit in search_results]

@app.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint to answer questions based on indexed PDF content."""
    data = request.json
    message = data.get("message")
    
    if not message:
        return jsonify({"success": False, "message": "No message provided"}), 400
    
    if not GEMINI_API_KEY:
        return jsonify({"success": False, "message": "Missing Gemini API Key"}), 500

    try:
        collection_info = qdrant.get_collection(collection_name=collection_name)
        if collection_info.points_count == 0:
            pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
            if not pdf_files:
                return jsonify({
                    "success": True,
                    "response": "I don't have any portfolio documents loaded yet. Please add some PDFs to my knowledge base."
                }), 200
                
            for pdf_file in pdf_files:
                pdf_path = os.path.join(PDF_FOLDER, pdf_file)
                index_pdf(pdf_path)
                print(f"Indexed {pdf_file}")
        
        # Retrieve the most relevant text from stored data
        relevant_texts = search_qdrant(message, top_k=3)
        
        if not relevant_texts:
            return jsonify({
                "success": True,
                "response": "I don't have enough information to answer that question. My knowledge is based on the portfolio documents I've been provided."
            }), 200
        
        # Format context with source information
        context_parts = []
        for item in relevant_texts:
            context_parts.append(f"[From {item['source']}]: {item['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Gemini Prompt
        prompt = (f"You are an AI assistant for a portfolio website. Answer the following question based ONLY on the provided text from the portfolio documents:\n\n"
                  f"Context: {context}\n\n"
                  f"Question: {message}\n\n"
                  f"If the question cannot be answered based on the provided context, politely explain that you don't have that information in your portfolio documents. "
                  f"If you do answer, make sure to only use information from the provided context. "
                  f"Keep your answer concise, professional, and friendly as if you are representing the portfolio owner.")

        # Send request to Gemini API
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )
        
        response_json = response.json()
        ai_reply = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error generating response")
        
        return jsonify({"success": True, "response": ai_reply}), 200
    
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"success": False, "message": "An error occurred while processing your request"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)