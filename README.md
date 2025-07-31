# Query Engine: Retrieval-Augmented Generation (RAG) Pipeline

## Overview

A LLM-powered research query engine. Given a user query, it expands, retrieves, ranks, and summarizes papers from academic sources using a modular RAG pipeline.

---

## Workflow

1. **LLM Query Expansion**  
   - The input query is expanded by a large language model (LLM) into *k* diverse, multi-hop keyword search phrases (*k* is configurable).
2. **Batch Retrieval**  
   - For each search phrase, up to *n* relevant papers are fetched from external APIs (PubMed, Semantic Scholar, OpenAlex). (*n* is configurable.)
3. **Semantic Title Ranking (FAISS)**  
   - All paper titles are embedded, and a FAISS index is built.
   - Cosine similarity is used to rank all titles by semantic closeness to the query.
   - The top *(n × 2)* papers are selected.
4. **Contextual Abstract Re-ranking (LlamaIndex)**  
   - Abstracts of selected papers are re-ranked with LlamaIndex to boost contextual relevance over simple title similarity.
5. **LLM Summarization (OpenAI)**  
   - Top *n* ranked papers are summarized using OpenAI’s models for easy reading and quick insight.

---

## Configuration

- `k`: Number of search phrases per query  
- `n`: Number of papers retrieved per search  
- `n`, `k` configurable in `main.py`
- `query`: User's research question
- `source`: Source for query (`"pubmed"`, `"semantic_scholar"`, `"openalex"`)
- `query`, `source` configurable via terminal arguments

Sample response for:  
query = "What are the recent advancements and challenges in applying large language models to clinical decision support, diagnostics, and patient care in medicine?"  
source = "pubmed"  
can be found in `response.txt`

---

## Instructions to run:

```bash
pip install -r requirements.txt
python main.py --query "your topic" --source "pubmed"/"semantic_scholar"/"openalex"




