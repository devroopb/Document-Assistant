
# Import SentenceTransformer to convert text into numerical embeddings.
from sentence_transformers import SentenceTransformer

# Import FAISS for efficient similarity searching.
import faiss

# Import NumPy for handling numerical arrays.
import numpy as np

# Import Streamlit so we can cache expensive operations.
import streamlit as st

# Cache the model and FAISS index so they aren't rebuilt
# every time Streamlit reruns the application.
@st.cache_resource(show_spinner=False)
def create_embeddings(chunks):
    """
    Convert document chunks into numerical embeddings.

    Each embedding represents the meaning of a text chunk.
    """

    # Load a lightweight, pre-trained embedding model.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Extract the text from each chunk.
    texts = [chunk["text"] for chunk in chunks]

    # Convert the text into numerical vectors.
    # normalize_embeddings makes cosine similarity searching easier.
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # FAISS requires embeddings in 32-bit floating-point format.
    embeddings = np.asarray(embeddings, dtype="float32")

    # Create a FAISS index using inner-product similarity.
    # Because embeddings are normalized, this is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])

    # Store all document embeddings in the FAISS index.
    index.add(embeddings)

    # Return the model and index for later searching.
    return model, index


def search_documents(question, model, index, chunks, top_k=20):
    """
    Find the most relevant document chunks for a user's question.
    """

    # Convert the user's question into an embedding.
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Convert the query embedding into FAISS-compatible format.
    query_embedding = np.asarray(query_embedding, dtype="float32")

    # Search the FAISS index for the top matching chunks.
    scores, indices = index.search(query_embedding, top_k)

    results = []

    # Retrieve the original text and page number for each match.
    for score, chunk_index in zip(scores[0], indices[0]):

        # FAISS returns -1 when there aren't enough results.
        if chunk_index == -1:
            continue

        results.append({
            "text": chunks[chunk_index]["text"],
            "page": chunks[chunk_index]["page"],
            "document": chunks[chunk_index].get("document", "Unknown document"),
            "score": float(score)
        })

    return results

