from flask import Flask, request, jsonify
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

# Initialize Flask app
app = Flask(__name__)
@app.route('/')
def home():
    return 'Hello, Flask!'

if __name__ == "__main__":
    app.run(debug=True)
    
# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to chunk the extracted text
def chunk_text(text, chunk_size=500, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate embeddings for chunks in batches
def generate_embeddings_batched(chunks, batch_size=16):
    model = SentenceTransformer('all-MiniLM-L6-v2', device=torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=True, device=model.device)
        embeddings.extend(batch_embeddings)
    return embeddings

# Function to store embeddings in FAISS
def store_embeddings_in_faiss(embeddings):
    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexHNSWFlat(embeddings_np.shape[1], 32)
    index.add(embeddings_np)
    return index

# Function to query FAISS index
def query_faiss(query, index, chunks, model):
    query_embedding = model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=3)
    results = []
    for i in range(3):
        results.append({"chunk": chunks[I[0][i]], "similarity_score": D[0][i]})
    return results

# API Route for uploading PDF and extracting text
@app.route('/extract-text', methods=['POST'])
def extract_text():
    upload_dir = 'uploaded_files'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    pdf_file = request.files['file']
    pdf_path = f"uploaded_files/{pdf_file.filename}"
    pdf_file.save(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    return jsonify({"extracted_text": text[:500]})  # Just send the first 500 characters for brevity

# API Route for chunking the text
@app.route('/chunk-text', methods=['POST'])
def chunk_text_route():
    text = request.json.get('text')
    chunks = chunk_text(text)
    return jsonify({"total_chunks": len(chunks), "chunks": chunks[:3]})  # Show the first 3 chunks

# API Route for generating embeddings
@app.route('/generate-embeddings', methods=['POST'])
def generate_embeddings_route():
    text = request.json.get('text')
    chunks = chunk_text(text)
    embeddings = generate_embeddings_batched(chunks)
    return jsonify({"total_embeddings": len(embeddings), "first_embedding_shape": len(embeddings[0])})

# API Route for querying the FAISS index
@app.route('/query', methods=['POST'])
def query_route():
    query = request.json.get('query')
    text = request.json.get('text')
    chunks = chunk_text(text)
    embeddings = generate_embeddings_batched(chunks)
    index = store_embeddings_in_faiss(embeddings)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    results = query_faiss(query, index, chunks, model)
    return jsonify({"query_results": results})

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
    import time

# A function that sleeps for a given number of seconds
def slow_function():
    time.sleep(2)

# A function that simulates some calculations
def fast_function():
    return sum(range(1000))

# A function that calls both functions
def main():
    slow_function()
    fast_function()

if __name__ == "__main__":
    main()
import time

def slow_function():
    time.sleep(1)  # Simulate a slow function

def fast_function():
    time.sleep(0.1)  # Simulate a faster function

def main():
    for _ in range(3):
        slow_function()
        fast_function()
        fast_function()

if __name__ == "__main__":
    main()

