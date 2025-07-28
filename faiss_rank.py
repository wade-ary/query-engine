import os
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding


# Client for OpenAI embeddings
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    organization=os.getenv("OPENAI_ORG_ID")
)


def embed_texts(texts: list[str], model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO") -> np.ndarray:
    """
    Embed a list of biomedical texts using a domain-specific SentenceTransformer model.
    
    Args:
        texts (list[str]): List of text strings (e.g., PubMed abstracts or queries).
        model_name (str): Hugging Face model name for domain-specific embeddings.

    Returns:
        np.ndarray: Matrix of text embeddings (shape: len(texts) x embedding_dim)
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for inner product similarity on normalized embeddings.
    """
    # Normalize embeddings to unit length
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product on normalized vectors equals cosine similarity
    index.add(embeddings)
    return index


def rerank_with_faiss(query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
    # Only keep docs with non-empty titles
    docs_with_text = [d for d in documents if d.get("title", "").strip()]
    texts = [d["title"] for d in docs_with_text]  # now using titles instead of abstracts

    # Embed titles
    doc_embs = embed_texts(texts)
    query_emb = embed_texts([query])[0].reshape(1, -1)
    faiss.normalize_L2(query_emb)

    # Build and search FAISS index over title embeddings
    index = build_faiss_index(doc_embs)
    scores, indices = index.search(query_emb, top_k)

    # Map back to documents and add the FAISS score
    top_docs = []
    for score, idx in zip(scores[0], indices[0]):
        doc = docs_with_text[idx].copy()  # keep abstract, etc.
        doc["_score"] = float(score)
        top_docs.append(doc)
    return top_docs



def get_top_titles(query: str, documents: list[dict], top_n: int = 5) -> tuple[list[str], list[str]]:
    """
    Run a two-stage reranking pipeline:
    1. Rank documents by title using FAISS + domain-specific embeddings
    2. Rerank corresponding abstracts using LlamaIndex + OpenAI embeddings

    Args:
        query (str): The user query
        documents (list[dict]): List of documents with 'title' and 'abstract'
        top_n (int): Number of top results to return

    Returns:
        tuple: (list of top titles, list of corresponding reranked abstracts)
    """
    # Step 1: Coarse filtering using titles + domain-specific embeddings
    top_docs = rerank_with_faiss(query, documents, top_k=top_n * 2)  # retrieve a few extra

    # Step 2: Extract abstracts from these docs
    abstracts = [doc["abstract"] for doc in top_docs if doc.get("abstract")]

    # Step 3: Fine reranking using LlamaIndex and OpenAI embeddings
    reranked_abstracts = rerank_chunks(abstracts, query, top_k=top_n)

    # Step 4: Match reranked abstracts back to their titles
    title_by_abstract = {doc["abstract"]: doc["title"] for doc in top_docs if doc.get("abstract")}
    reranked_titles = [title_by_abstract.get(abs_text, "") for abs_text in reranked_abstracts]

    return reranked_titles, reranked_abstracts


def rerank_chunks(chunks, query, top_k=5):
    # Convert strings to LlamaIndex Document objects
    documents = [Document(text=chunk) for chunk in chunks]

    # Parse them into nodes (chunks of text)
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    # Build a vector index using OpenAI embeddings
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    # Rerank by similarity to the original query
    results = retriever.retrieve(query)

    # Return the top reranked chunks (abstracts)
    top_chunks = [res.node.get_content() for res in results]
    return top_chunks