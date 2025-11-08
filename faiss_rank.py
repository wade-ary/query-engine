import os
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Client for OpenAI embeddings
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    organization=os.getenv("OPENAI_ORG_ID")
)


def embed_texts(texts: List[str], source: str) -> np.ndarray:
    # Convert a list of text strings into numerical embeddings using different models based on the data source
    # Uses PubMed-specific model for PubMed data, general model for OpenAlex, and science-focused model for Semantic Scholar
    
    source = source.lower()

    if source == "pubmed":
        model_name = "pritamdeka/S-PubMedBert-MS-MARCO" # PubMed specific model
    elif source == "openalex":
        model_name = "sentence-transformers/all-mpnet-base-v2"  # OpenAlex general-purpose model
    elif source == "semantic_scholar":
        model_name = "sentence-transformers/allenai-specter"  # Semantic Scholar Science focused model
    else:
        raise ValueError(f"Unknown source '{source}'. Expected 'pubmed', 'openalex', or 'semantic_scholar'.")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    # Create a FAISS search index from embeddings for fast similarity search
    # Normalizes the embeddings and builds an index optimized for inner product similarity
    
    # Normalize embeddings to unit length
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  
    index.add(embeddings)
    return index


def rerank_with_faiss(query: str, documents: list[dict], top_k: int = 10, source: str = "openalex") -> list[dict]:
    # Rank documents by comparing their titles to the query using FAISS similarity search
    # Returns the top documents with their similarity scores
    
    # Only keep docs with non-empty titles
    docs_with_text = [d for d in documents if d.get("title", "").strip()]
    # Use titles instead of abstracts
    texts = [d["title"] for d in docs_with_text]  
    
    # Embed titles
    doc_embs = embed_texts(texts, source)
    query_emb = embed_texts([query], source)[0].reshape(1, -1)
    faiss.normalize_L2(query_emb)

    # Build and search FAISS index over title embeddings
    index = build_faiss_index(doc_embs)
    scores, indices = index.search(query_emb, top_k)

    # Map back to documents and add the FAISS score
    top_docs = []
    for score, idx in zip(scores[0], indices[0]):
        doc = docs_with_text[idx].copy()  
        doc["_score"] = float(score)
        top_docs.append(doc)
    return top_docs



def get_top_titles(query: str, documents: list[dict], top_n: int = 5) -> tuple[list[str], list[str]]:
    # Two-stage ranking system: first rank by titles, then rerank abstracts
    # Returns the best matching titles and their corresponding abstracts
    
    # Coarse filtering using titles + domain-specific embeddings
    top_docs = rerank_with_faiss(query, documents, top_k=top_n * 2)  # retrieve a few extra

    # Extract abstracts from these docs
    abstracts = [doc["abstract"] for doc in top_docs if doc.get("abstract")]

    # Fine reranking using LlamaIndex and OpenAI embeddings
    reranked_abstracts = rerank_chunks(abstracts, query, top_k=top_n)

    # Match reranked abstracts back to their titles
    title_by_abstract = {doc["abstract"]: doc["title"] for doc in top_docs if doc.get("abstract")}
    reranked_titles = [title_by_abstract.get(abs_text, "") for abs_text in reranked_abstracts]

    return reranked_titles, reranked_abstracts


def rerank_chunks(
    chunks, 
    query, 
    top_k=5, 
    model_name="BAAI/bge-small-en-v1.5"
):
    """
    Rerank chunks using a basic HuggingFace embedding model.
    Default: BAAI/bge-small-en-v1.5 (lightweight, reliable, and fully supported).
    """

    # 1. Convert raw text → Documents
    documents = [Document(text=chunk) for chunk in chunks]

    # 2. Parse into nodes
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    # 3. Load lightweight HF embedding model (no OpenAI, no Mistral)
    embed_model = HuggingFaceEmbedding(model_name=model_name)

    # 4. Build vector index
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    # 5. Retrieve top_k similar chunks
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    results = retriever.retrieve(query)

    # 6. Extract ranked content
    return [res.node.get_content() for res in results]