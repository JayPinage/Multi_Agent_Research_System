from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_web(query:str)->str:
    "Give me a title , link and a short summary of the top 5 results from the web for the given query"
    results = tavily.search(query=query,max_results=5)

    output = []
    for r in results['results']:
        output.append(f"Title: {r['title']}\nUrl: {r['url']}\nsnippet: {r['content'][:300]}\n")

    return "\n-----\n".join(output)

@tool
def scrape_url(url:str)->str:
    """scrape and return claen text from a given url for deeper reading"""
    try:
        response = requests.get(url,timeout=8,headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'html.parser')

        for i in soup(['script', 'style','nav','header','footer','aside']):
            i.decompose()
        return soup.get_text(separator=" ",strip=True)[:3000]
    except Exception as e:
        return f"URL not found or error occurred while scraping the URL: {e}"


