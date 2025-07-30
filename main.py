from query_pubmed import search_pubmed
from faiss_rank import get_top_titles
from query_open_alex import search_openalex
from query_semantic_scholar import search_semantic_scholar


def main():
    # Step 1: Retrieve PubMed articles for the query
    query = "Application of LLMs in phyics"
    source = "semantic_scholar"
    if source == "pubmed":
        results = search_pubmed(
            query=query,
            num_terms=3,
            retmax=3,
            email="aryamanwade@gmail.com",  # NCBI recommends providing an email
    
        )
    elif source == "openalex":
        results = search_openalex(query=query, num_terms=3, retmax=3)
    elif source == "semantic_scholar":
        results = search_semantic_scholar(query=query, num_terms=3, retmax=3)
    else:
        raise ValueError(f"Invalid source: {source}")

    # Step 2: Flatten all article records across keyword phrases
    all_articles = []
    for phrase, articles in results.items():
        print(f"Phrase {phrase!r} returned {len(articles)} articles")
        all_articles.extend(articles)

    # Deduplicate by PMID
    print("All articles collected before deduplication:", len(all_articles))
    unique = {}
    for art in all_articles:
        title = art.get('title')
        if title and title not in unique:
            unique[title] = art
    all_articles = list(unique.values())
    print(f"Total unique articles collected: {len(all_articles)}\n")

    # Step 3: Rerank and extract top titles
    top_titles, top_abstracts = get_top_titles(query, all_articles, top_n=3)
    
    print("🔍 Top 5 Articles by Semantic Similarity:\n")
    for idx, (title, abstract) in enumerate(zip(top_titles, top_abstracts), start=1):
        print(f"{idx}. {title}")
        print(f"   Abstract: {abstract[:300].strip()}...\n")


if __name__ == '__main__':
    main()