import os
import faiss
import numpy as np
from openai import OpenAI

# Client for OpenAI embeddings
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    organization=os.getenv("OPENAI_ORG_ID")
)


def embed_texts(texts: list[str], model: str = "text-embedding-ada-002") -> np.ndarray:
    """
    Embed a list of texts using OpenAI and return an array of embeddings.
    """
    batch_size = 100  # adjust based on rate limits
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        # Assume response.data is list of {embedding: [...], index: i}
        embeddings.extend([np.array(item.embedding, dtype="float32") for item in response.data])
    return np.vstack(embeddings)


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
    # only keep docs with non-empty abstracts
    docs_with_text = [d for d in documents if d.get("abstract","").strip()]
    texts = [d["abstract"] for d in docs_with_text]

    # now embed texts (none of these will be empty)
    doc_embs = embed_texts(texts)
    query_emb = embed_texts([query])[0].reshape(1, -1)
    faiss.normalize_L2(query_emb)

    # build index over the filtered docs
    index = build_faiss_index(doc_embs)
    scores, indices = index.search(query_emb, top_k)

    # map back into the original docs_with_text list
    top_docs = []
    for score, idx in zip(scores[0], indices[0]):
        doc = docs_with_text[idx].copy()
        doc["_score"] = float(score)
        top_docs.append(doc)
    return top_docs



def get_top_titles(query: str, documents: list[dict], top_n: int = 5) -> list[str]:
    """
    Return the titles of the top_n ranking articles for a given query.
    """
    top_docs = rerank_with_faiss(query, documents, top_k=top_n)
    return [doc.get('title', '') for doc in top_docs]