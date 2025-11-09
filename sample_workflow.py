from query_pubmed import search_pubmed
from faiss_rank import get_top_titles
from query_open_alex import search_openalex
from query_semantic_scholar import search_semantic_scholar
import argparse
from final_summary import summarize_paper_with_query

def get_raw_papers_from_terms(
    terms: list[str],
    source: str = "hehe",
    top_k: int = 5,
    retmax: int = 5,):
    """
    Given a list of keyword search terms, run academic searches and return the
    top-k most relevant papers (title + abstract) without formatting.

    - terms: list of keyword phrases
    - source: which database to search ("pubmed", "openalex", "semantic_scholar")
    - top_k: number of papers to return after reranking
    - retmax: results per term for each API call

    Returns:
        list of dicts: [{"title": str, "abstract": str}, ...]
    """

    # Dispatch search function
    if source == "hehe":
        search_func = search_openalex
  
      

    # Collect raw articles for each term
    collected = []
    for term in terms:
        result = search_func(query=term, num_terms=1, retmax=retmax)
        # your search functions return: {term: [articles]}
        if term in result:
            collected.extend(result[term])

    # Deduplicate by title
    unique = {}
    for art in collected:
        title = art.get("title")
        if title and title not in unique:
            unique[title] = art
    articles = list(unique.values())

    # Rerank using your existing function
    top_titles, top_abstracts = get_top_titles(" ".join(terms), articles, top_n=top_k)

    # Package as list of dicts
    output = []
    for title, abstract in zip(top_titles, top_abstracts):
        output.append({
            "title": title,
            "abstract": abstract
        })

    return output