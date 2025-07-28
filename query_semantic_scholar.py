import requests
import json
import time
from typing import List, Dict

from query_generator import generate_search_terms

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

def search_semantic_scholar(query: str,
                             num_terms: int = 8,
                             retmax: int = 20,
                             email: str = None,
                             api_key: str = None) -> Dict[str, List[Dict]]:
    """
    Given a user query, generate keyword search phrases, then fetch Semantic Scholar
    results and abstracts using the Graph API.

    Parameters:
    - query: Free-form user query
    - num_terms: How many keyword searches to generate
    - retmax: Max number of results to return per phrase
    - email: (ignored)
    - api_key: (currently unused, no key required)

    Returns:
    A dict mapping each keyword phrase to a list of article metadata dicts,
    each including the abstract.
    """
    phrases = generate_search_terms(query, num_terms)
    results: Dict[str, List[Dict]] = {}

    for phrase in phrases:
        articles: List[Dict] = []
        retrieved = 0
        page_url = f"{BASE_URL}?query={phrase}&fields=title,abstract,year,authors,url,venue&limit={retmax}"

        while True:
            resp = requests.get(page_url)
            resp.raise_for_status()
            data = resp.json()

            if "data" in data:
                for paper in data["data"]:
                    articles.append(paper)
                    retrieved += 1
                    if retrieved >= retmax:
                        break

            if "token" not in data or retrieved >= retmax:
                break
            page_url = f"{BASE_URL}?token={data['token']}&fields=title,abstract,year,authors,url,venue&limit={retmax}"

            time.sleep(1)  # to be polite with rate limits

        results[phrase] = articles

    return results
