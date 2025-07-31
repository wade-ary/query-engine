import requests
import json
import time
from typing import List, Dict

from query_generator import generate_search_terms




BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def search_semantic_scholar(query: str,
                             num_terms: int = 10,
                             retmax: int = 20,
                             email: str = None,
                             api_key: str = None) -> Dict[str, List[Dict]]:
    # Search for academic papers in Semantic Scholar database using multiple keyword phrases
    # Converts user query into search terms, then fetches papers from Semantic Scholar Graph API
    """
    Given a user query, generate keyword search phrases, then fetch Semantic Scholar
    results and abstracts using the Graph API.
    """
    phrases = generate_search_terms(query, num_terms)
    results: Dict[str, List[Dict]] = {}

    headers = {}
    if api_key:
        headers["X-API-KEY"] = api_key

    for phrase in phrases:
        articles: List[Dict] = []
        retrieved = 0
        token = None

        while retrieved < retmax:
            params = {
                "query": phrase,
                "fields": "title,abstract",
                "limit": retmax
            }
            if token:
                params["token"] = token

            resp = requests.get(BASE_URL, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            for paper in data.get("data", []):
                articles.append(paper)
                retrieved += 1
                if retrieved >= retmax:
                    break

            token = data.get("token")
            if not token or retrieved >= retmax:
                break

            time.sleep(1)  # Respect API rate limits

        results[phrase] = articles

    return results
