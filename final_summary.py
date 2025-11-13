import os
import json
from dotenv import load_dotenv
load_dotenv()

# Your LangChain model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini")   # or whichever model you are using


def summarize_papers_with_query(papers: list[dict], query: str) -> str:
    """
    Takes a list of papers (each a dict with title + abstract)
    and produces:
        - Mini-summary for each paper
        - Common themes across papers
        - Key differences between papers
    """

    # Prepare text block of all papers
    paper_block = ""
    for i, p in enumerate(papers, start=1):
        title = p.get("title", "No title")
        abstract = p.get("abstract", "No abstract available")
        paper_block += f"""
Paper {i}:
Title: {title}
Abstract: {abstract}

"""

    prompt = f"""
You are an expert scientific summarizer.

The user has this research question:
\"\"\"{query}\"\"\"

Below is a list of research papers (titles + abstracts).  
Your task:

1. For EACH paper:
   - Write a **3–5 sentence** summary explaining:
     - the main findings  
     - the methods (if relevant)
     - how it connects to the user's query  
   - Keep it very readable and beginner-friendly.

2. After all individual summaries, write:
   **Common Themes Across All Papers**  
   - Identify overlapping ideas, findings, or patterns.

3. Then write:
   **Key Differences Between the Papers**  
   - Highlight differing approaches, conclusions, or perspectives.

Return the full output as readable text.  
Do NOT return JSON.  
Do NOT add extra sections.
    
---
PAPERS:
{paper_block}
"""

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content.strip()