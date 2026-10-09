import os
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from tools import search_web, scrape_url, read_pdf

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


class ResearchEvaluation(BaseModel):
    relevance: int = Field(ge=1, le=10)
    coverage: int = Field(ge=1, le=10)
    faithfulness: int = Field(ge=1, le=10)
    citation_quality: int = Field(ge=1, le=10)
    overall_score: int = Field(ge=1, le=10)
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[search_web],
        system_prompt=(
            "You are a research search agent. Find relevant, recent, reliable sources. "
            "Prefer primary sources, official documentation, and reputable publications. "
            "Return source titles, URLs, dates when available, and concise findings. "
            "Call only registered tools. Never invent tool names."
        ),
    )


class _DeterministicScrapeReader:
    """
    Reader adapter compatible with the existing Streamlit call pattern.

    It deliberately avoids an LLM tool-calling loop, so the model cannot
    invent tool names such as `open_file`. Python selects a URL and invokes
    the registered scraper/PDF tools directly.
    """

    def invoke(self, payload: dict) -> dict:
        messages = payload.get("messages", [])
        user_text = ""

        if messages:
            last_message = messages[-1]
            if isinstance(last_message, tuple) and len(last_message) > 1:
                user_text = str(last_message[1])
            else:
                user_text = str(getattr(last_message, "content", last_message))

        # Prefer URLs in the search-results portion, excluding prompt examples.
        urls = re.findall(r"https?://[^\s<>\"']+", user_text)
        cleaned_urls = []
        for raw_url in urls:
            url = raw_url.rstrip(".,;:!?)}}]")
            if url not in cleaned_urls:
                cleaned_urls.append(url)

        if not cleaned_urls:
            content = (
                "Reader could not find a URL in the supplied search results. "
                "Try expanding the search results passed to the reader."
            )
            return {"messages": [HumanMessage(content=user_text), AIMessage(content=content)]}

        outputs = []
        # Read up to two sources to limit latency and provider/API load.
        for url in cleaned_urls[:2]:
            try:
                if ".pdf" in url.lower().split("?")[0]:
                    result = read_pdf.invoke({"url": url})
                elif "arxiv.org/abs/" in url.lower():
                    # arXiv abstract pages are HTML; scraping their abstract page
                    # is more reliable than treating /abs/ as a PDF URL.
                    result = scrape_url.invoke({"url": url})
                else:
                    result = scrape_url.invoke({"url": url})

                outputs.append(f"Source URL: {url}\\nExtracted content:\\n{result}")
            except Exception as exc:
                outputs.append(
                    f"Source URL: {url}\\nCould not retrieve source: "
                    f"{type(exc).__name__}: {exc}"
                )

        content = "\\n\\n---\\n\\n".join(outputs)
        return {"messages": [HumanMessage(content=user_text), AIMessage(content=content)]}


def build_scrape_agent():
    # Not an LLM agent by design: deterministic tool dispatch avoids invalid
    # model-generated tool calls such as `open_file`.
    return _DeterministicScrapeReader()


writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research writer.
Use only the supplied research as evidence.
Do not invent facts, quotations, statistics, or URLs.
Distinguish sourced facts from uncertainty.
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


evaluator = llm.with_structured_output(ResearchEvaluation)

evaluation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Evaluate the research report strictly. Score each criterion from 1 to 10:
- relevance: answers the requested topic
- coverage: covers important findings and limitations
- faithfulness: claims are supported by supplied evidence
- citation_quality: sources are identifiable and traceable
- overall_score: overall research quality

Do not assume claims are true merely because the report states them.
Flag unsupported claims and missing evidence. These are model-assisted
assessments, not proof of factual correctness. Do not claim independent
source verification unless verification evidence was supplied."""
    ),
    (
        "human",
        """Research topic:
{topic}

Retrieved research:
{research}

Final report:
{report}

Return the evaluation using the requested structured schema."""
    ),
])
evaluation_chain = evaluation_prompt | evaluator
