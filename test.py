import json
from query_pubmed import search_pubmed
from faiss_rank import get_top_titles
from query_open_alex import search_openalex
from query_semantic_scholar import search_semantic_scholar

def main():
    query = "How do LLMs work?"
    source = "openalex"
    if source == "pubmed":
        results = search_pubmed(
            query=query,
            num_terms=3,
            retmax=3,
            email="aryamanwade@gmail.com",
        )
    elif source == "openalex":
        results = search_openalex(query=query, num_terms=3, retmax=3)
    elif source == "semantic_scholar":
        results = search_semantic_scholar(query=query, num_terms=3, retmax=3)
    else:
        raise ValueError(f"Invalid source: {source}")

    all_articles = []
    for phrase, articles in results.items():
        print(f"Phrase {phrase!r} returned {len(articles)} articles")
        all_articles.extend(articles)
    
    print("All articles collected before deduplication:", len(all_articles))
    # Just print the first article's raw dict as JSON
    if all_articles:
        print("\nExample article structure:\n")
        print(json.dumps(all_articles[0], indent=2, ensure_ascii=False))
    else:
        print("No articles found!")

if __name__ == "__main__":
    main()
