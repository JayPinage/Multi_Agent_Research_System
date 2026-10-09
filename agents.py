import re
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from tools import search_web, scrape_url

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[search_web],
        system_prompt=(
            "You are a research search agent. Find relevant, recent, reliable sources. "
            "Prefer web pages: articles, official documentation, reputable publications. "
            "Do not return PDF links. "
            "Return source titles, URLs, dates when available, and concise findings. "
            "Call only registered tools. Never invent tool names."
        ),
    )


def _scrape_one(url: str) -> str:
    try:
        result = scrape_url.invoke({"url": url})
        return f"Source URL: {url}\nExtracted content:\n{result}"
    except Exception as exc:
        return (
            f"Source URL: {url}\nCould not retrieve source: "
            f"{type(exc).__name__}: {exc}"
        )


class _DeterministicScrapeReader:
    """No LLM loop: picks web URLs (PDFs skipped) and scrapes them in parallel."""

    def invoke(self, payload: dict) -> dict:
        messages = payload.get("messages", [])
        user_text = ""

        if messages:
            last_message = messages[-1]
            if isinstance(last_message, tuple) and len(last_message) > 1:
                user_text = str(last_message[1])
            else:
                user_text = str(getattr(last_message, "content", last_message))

        urls = re.findall(r"https?://[^\s<>\"']+", user_text)
        web_urls = []
        for raw_url in urls:
            url = raw_url.rstrip(".,;:!?)}]")
            if ".pdf" in url.lower().split("?")[0]:
                continue
            if url not in web_urls:
                web_urls.append(url)

        if not web_urls:
            content = "No scrapable web page URL found in the search results."
            return {"messages": [HumanMessage(content=user_text), AIMessage(content=content)]}

        # Scrape up to 2 pages at the same time
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(_scrape_one, web_urls[:2]))

        content = "\n\n---\n\n".join(outputs)
        return {"messages": [HumanMessage(content=user_text), AIMessage(content=content)]}


def build_scrape_agent():
    return _DeterministicScrapeReader()


writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research writer.
Use only the supplied research as evidence.
Do not invent facts, quotations, statistics, or URLs.
Preserve source URLs so claims can be checked."""
    ),
    (
        "human",
        """Write a detailed research report.

Topic: {topic}

Research:
{research}

Use this structure:
1. Introduction
2. Key Findings (at least 3 well-explained points)
3. Limitations and uncertainties
4. Conclusion
5. Sources (actual URLs present in the research)

Be factual, professional, and clear."""
    ),
])
writer_chain = writer_prompt | llm | StrOutputParser()


critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict and constructive research critic."),
    (
        "human",
        """Review this report against the provided research.

Report:
{report}

Research:
{research}

Identify unsupported claims, missing evidence, weak reasoning,
missing sources, and important omissions.

Return:
Score: X/10

Strengths:
- ...

Areas to Improve:
- ...

One line verdict:
..."""
    ),
])
critic_chain = critic_prompt | llm | StrOutputParser()
