import os
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
from query_generator import generate_search_terms

# Base URLs for NCBI Entrez E-utilities 
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query: str,
                  num_terms: int = 10,
                  retmax: int = 20,
                  email: str = None,
                  api_key: str = None) -> Dict[str, List[Dict]]:
    """
    Given a user query, generate keyword search phrases, then fetch PubMed
    search results and abstracts for each phrase using Entrez ESearch, ESummary, and EFetch.

    Parameters:
    - query: Free-form user query
    - num_terms: How many keyword searches to generate
    - retmax: Maximum number of articles to return per search phrase
    - email: User email for NCBI Entrez identification (optional but recommended)
    - api_key: NCBI API key for higher rate limits (optional)

    Returns:
    A dict mapping each keyword phrase to a list of PubMed article metadata dicts,
    each including the abstract text.
    """
    # Generate candidate search phrases via LLM
    phrases = generate_search_terms(query, num_terms)
    results: Dict[str, List[Dict]] = {}

    # Prepare request params
    common_params = {
        "db": "pubmed",
        "retmax": retmax,
        "retmode": "json",
    }
    if email:
        common_params["email"] = email
    if api_key:
        common_params["api_key"] = api_key

    for phrase in phrases:
        # ESearch to get PMIDs
        esearch_params = {**common_params, "term": phrase}
        esearch_resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=esearch_params)
        esearch_resp.raise_for_status()
        id_list = esearch_resp.json().get("esearchresult", {}).get("idlist", [])

        articles: List[Dict] = []
        if id_list:
            # ESummary to fetch basic metadata
            summary_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json",
                **({"api_key": api_key} if api_key else {})
            }
            esummary_resp = requests.get(f"{EUTILS_BASE}/esummary.fcgi", params=summary_params)
            esummary_resp.raise_for_status()
            summary_data = esummary_resp.json().get("result", {})

            # EFetch to retrieve abstracts in XML
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
                **({"api_key": api_key} if api_key else {})
            }
            efetch_resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=fetch_params)
            efetch_resp.raise_for_status()
            xml_root = ET.fromstring(efetch_resp.text)

            # Build a map of PMID to abstract text
            abstract_map: Dict[str, str] = {}
            for article in xml_root.findall(".//PubmedArticle"):
                pmid = article.findtext(".//PMID") or ""
                texts = [elem.text or "" for elem in article.findall(".//AbstractText")]
                abstract_map[pmid] = " ".join(texts)

            # Combine metadata and abstract
            for uid, meta in summary_data.items():
                if uid == "uids":
                    continue
                if not isinstance(meta, dict):
                    continue
                meta_obj = meta.copy()
                meta_obj["abstract"] = abstract_map.get(uid, "")
                articles.append(meta_obj)

        results[phrase] = articles

    return results
