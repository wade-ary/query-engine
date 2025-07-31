import requests
import time
from typing import List, Dict

from query_generator import generate_search_terms

BASE_URL = "https://api.openalex.org/works"

def search_openalex(query: str,num_terms: int = 10,
                    retmax: int = 20,
                    email: str = None,
                    api_key: str = None) -> Dict[str, List[Dict]]:
    # Search for academic papers in OpenAlex database using multiple keyword phrases
    # Converts user query into search terms, then fetches papers from OpenAlex API
    """
    Given a user query, generate keyword search phrases, then fetch academic papers
    from OpenAlex using the /works endpoint.

    Parameters:
    - query: User input string
    - num_terms: Number of keyword search terms to generate
    - retmax: Max number of results per search phrase
    - email: (ignored for OpenAlex)
    - api_key: (ignored for OpenAlex)

    Returns:
    A dict mapping each keyword phrase to a list of paper metadata dicts.
    """
    phrases = generate_search_terms(query, num_terms)
    results: Dict[str, List[Dict]] = {}

    for phrase in phrases:
        articles: List[Dict] = []
        retrieved = 0
        cursor = "*"
        headers = {"User-Agent": "query-engine/1.0"}

        while retrieved < retmax:
            params = {
                "search": phrase,
                "per-page": min(50, retmax - retrieved), 
                "cursor": cursor,
                "mailto": email or "aryamanwade@gmail.com"  # optional
            }

            response = requests.get(BASE_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            for item in data.get("results", []):
                primary_location = item.get("primary_location") or {}
                source = primary_location.get("source") or {}

                abstract_raw = item.get("abstract_inverted_index", "")
                if isinstance(abstract_raw, dict):
                    word_list = sorted([(pos, word) for word, pos_list in abstract_raw.items() for pos in pos_list])
                    abstract = " ".join([word for _, word in word_list])
                else:
                    abstract = abstract_raw or ""

                article = {
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "venue": source.get("name", "")
                }

                articles.append(article)
                retrieved += 1
                if retrieved >= retmax:
                    break


            if "meta" in data and "next_cursor" in data["meta"]:
                cursor = data["meta"]["next_cursor"]
            else:
                break  # no more results

            time.sleep(1)  # API rate limits

        results[phrase] = articles

    return results
