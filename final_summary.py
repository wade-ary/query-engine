import json
from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    organization=os.getenv("OPENAI_ORG_ID")
)

def summarize_paper_with_query(title, abstract, query):
    # Create a custom summary of each research paper that explains how it relates to the user's query
    # Uses GPT-4 to generate a concise 3-5 sentence summary highlighting key findings and relevance
    
    prompt = f"""
You are an expert scientific research assistant.

Given the following research paper title and abstract, and the user's query, write a concise summary (3-5 sentences) that explains how this paper relates to the query. Highlight any key findings, methods, or relevance to the question.

---
User query:
\"\"\"{query}\"\"\"

Paper title:
\"\"\"{title}\"\"\"

Paper abstract:
\"\"\"{abstract}\"\"\"

Summary:
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    summary = response.choices[0].message.content.strip()
    return summary