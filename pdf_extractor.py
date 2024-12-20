import PyPDF2
import numpy as np
import faiss
import langchain
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to chunk the extracted text
def chunk_text(text, chunk_size=500, chunk_overlap=150):  # Increased overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate embeddings for text chunks in batches (with GPU support)
def generate_embeddings_batched(chunks, batch_size=16):
    model = SentenceTransformer('all-MiniLM-L6-v2', device=torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=True, device=model.device)
        embeddings.extend(batch_embeddings)
    return embeddings

# Function to store embeddings in FAISS (using HNSW index for better speed)
def store_embeddings_in_faiss(embeddings):
    embeddings_np = np.array(embeddings).astype('float32')
    dim = embeddings_np.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)  # 32 is the number of neighbors to explore
    index.add(embeddings_np)
    return index

# Function to query FAISS index and retrieve the top N most similar chunks
def query_faiss(query, index, chunks, model, top_n=3):
    query_embedding = model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=top_n)
    print(f"\nTop {top_n} most relevant chunks:")
    for i in range(top_n):
        print(f"Chunk {i+1}: {chunks[I[0][i]]}\n")
        print(f"Similarity Score: {D[0][i]:.4f}\n")

# Example usage
pdf_path = "/Users/indrani/Downloads/Indrani_recent resume.pdf"  # Replace with your PDF file path
pdf_text = extract_text_from_pdf(pdf_path)
print(f"Extracted text (first 500 characters): {pdf_text[:500]}")

# Chunk the extracted text
chunks = chunk_text(pdf_text)
print(f"\nTotal Chunks: {len(chunks)}")
print(f"First Chunk: {chunks[0]}\n")

# Generate embeddings for the chunks
embeddings = generate_embeddings_batched(chunks)

# Store embeddings in FAISS
index = store_embeddings_in_faiss(embeddings)
print(f"Total embeddings stored in FAISS: {index.ntotal}")

# Query the FAISS index for relevant chunks based on a user query
query = "Tell me about the certifications in digital marketing"
query_faiss(query, index, chunks, SentenceTransformer('all-MiniLM-L6-v2'))

import logging

logging.basicConfig(filename='debug.log', level=logging.DEBUG)
logging.debug('Starting function X')
