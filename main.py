from query_pubmed import search_pubmed
from faiss_rank import get_top_titles
from query_open_alex import search_openalex
from query_semantic_scholar import search_semantic_scholar
import argparse
from final_summary import summarize_paper_with_query
def main(query, source):

    # num_terms: Number of keyword searches per query
    num_terms = 10
    # retmax: Number of results returned per keyword serach
    retmax = 20

    # Retrieve articles for the query
    if source == "pubmed":
        results = search_pubmed(query=query,num_terms=num_terms,retmax=retmax)
    elif source == "openalex":
        results = search_openalex(query=query, num_terms=num_terms, retmax=retmax)
    elif source == "semantic_scholar":
        results = search_semantic_scholar(query=query, num_terms=num_terms, retmax=retmax)
    else:
        raise ValueError(f"Invalid source: {source}")

    # Flatten all article records across keyword phrases
    all_articles = []
    for phrase, articles in results.items():
        print(f"Phrase {phrase!r} returned {len(articles)} articles")
        all_articles.extend(articles)

    # Deduplicate by title
    print("All articles collected before deduplication:", len(all_articles))
    unique = {}
    for art in all_articles:
        title = art.get('title')
        if title and title not in unique:
            unique[title] = art
    all_articles = list(unique.values())
    print(f"Total unique articles collected: {len(all_articles)}\n")

    # Step 3: Rerank and extract top titles
    top_titles, top_abstracts = get_top_titles(query, all_articles, top_n=5)
    
    with open("response.txt", "w", encoding="utf-8") as f:
        f.write("Top 5 Articles by Semantic Similarity:\n\n")
        for idx, (title, abstract) in enumerate(zip(top_titles, top_abstracts), start=1):
            summary = summarize_paper_with_query(title, abstract, query)
            entry = f"{idx}. {title}\n   Summary: {summary}\n\n"
            print(entry)
            f.write(entry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RAG pipeline for multi-domain research articles")
    parser.add_argument('--query', required=True, help='Your search query')
    parser.add_argument('--source', required=True, choices=['pubmed', 'openalex', 'semantic_scholar'], help='Data source')
    args = parser.parse_args()

    main(args.query, args.source)