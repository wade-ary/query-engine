from query_pubmed import search_pubmed
from faiss_rank import get_top_titles
from query_open_alex import search_openalex
from query_semantic_scholar import search_semantic_scholar
import argparse


def get_raw_papers_from_terms_v2(
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


def get_raw_papers_from_terms(
    terms: list[str],
    source: str = "hehe",
    top_k: int = 5,
    retmax: int = 5,
):
    """
    Mock version of get_raw_papers_from_terms.
    Returns 5 sample papers in the same format:
    [{"title": str, "abstract": str}, ...]
    """

    sample_output = [
        {
            "title": "Impact of Diet on Cardiovascular Health",
            "abstract": "This study investigates dietary patterns and their influence on cardiovascular disease risk, focusing on saturated fats, fiber intake, and long-term metabolic outcomes."
        },
        {
            "title": "Hypertension as a Primary Contributor to Heart Disease",
            "abstract": "An analysis of blood pressure regulation, genetic predispositions, and environmental stressors that elevate the risk of myocardial infarction and stroke."
        },
        {
            "title": "Obesity and Its Role in Heart Disease Development",
            "abstract": "This review summarizes links between obesity, systemic inflammation, metabolic syndrome, and increased incidence of coronary artery blockage."
        },
        {
            "title": "Smoking and Cardiovascular Mortality Trends",
            "abstract": "A longitudinal study showing the correlation between smoking intensity, endothelial dysfunction, and elevated cardiovascular mortality rates."
        },
        {
            "title": "Diabetes Mellitus and Cardiovascular Complications",
            "abstract": "A clinical overview exploring how insulin resistance and chronic hyperglycemia accelerate atherosclerosis and elevate heart attack risk."
        },
    ]

    return sample_output[:top_k]