from query_pubmed import search_pubmed
from faiss_rank import get_top_titles


def main():
    # Step 1: Retrieve PubMed articles for the query
    query = " stem cell research and regeneration of the optic nerve"
    results = search_pubmed(
        query=query,
        num_terms=3,
        retmax=10,
        email="aryamanwade@example.com",  # NCBI recommends providing an email
  
    )

    # Step 2: Flatten all article records across keyword phrases
    all_articles = []
    for phrase, articles in results.items():
        print(f"Phrase {phrase!r} returned {len(articles)} articles")
        all_articles.extend(articles)

    # Deduplicate by PMID
    unique = {}
    for art in all_articles:
        uid = art.get('uid')
        if uid and uid not in unique:
            unique[uid] = art
    all_articles = list(unique.values())
    print(f"Total unique articles collected: {len(all_articles)}\n")

    # Step 3: Rerank and extract top titles
    top_titles = get_top_titles(query, all_articles, top_n=5)
    print("Top 5 articles by semantic similarity:")
    for idx, title in enumerate(top_titles, start=1):
        print(f"  {idx}. {title}")


if __name__ == '__main__':
    main()