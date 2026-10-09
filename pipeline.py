import logging
import re
from urllib.parse import urlparse

from dotenv import load_dotenv

from agents import (
    build_search_agent,
    build_scrape_agent,
    writer_chain,
    critic_chain,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TOPIC_LENGTH = 500
MAX_SEARCH_CHARS = 6000
MAX_SCRAPED_CHARS = 6000


def validate_topic(topic: str) -> str:
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("Research topic cannot be empty.")
    if len(topic) > MAX_TOPIC_LENGTH:
        raise ValueError(f"Topic cannot exceed {MAX_TOPIC_LENGTH} characters.")
    if not any(c.isalnum() for c in topic):
        raise ValueError("Enter a valid research topic.")
    return topic


def extract_urls(text: str) -> list[str]:
    candidates = re.findall(r"https?://[^\s<>\]\[\"']+", text)
    urls = []
    for c in candidates:
        url = c.rstrip(".,);}")
        if urlparse(url).hostname:
            urls.append(url)
    return list(dict.fromkeys(urls))


def extract_agent_text(result: dict) -> str:
    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            combined = "\n".join(parts).strip()
            if combined:
                return combined
    raise RuntimeError("Agent returned no usable text.")


def run_research_pipeline(topic: str) -> dict:
    topic = validate_topic(topic)

    logger.info("Step 1: Search")
    search_result = build_search_agent().invoke({
        "messages": [("user", f"Research this topic: {topic}. "
                              "Find reliable sources and provide their URLs.")]
    })
    search_text = extract_agent_text(search_result)

    logger.info("Step 2: Scrape")
    reader_result = build_scrape_agent().invoke({
        "messages": [("user", search_text[:MAX_SEARCH_CHARS])]
    })
    scraped = extract_agent_text(reader_result)

    research = (
        f"SEARCH RESULTS:\n{search_text[:MAX_SEARCH_CHARS]}\n\n"
        f"SCRAPED EVIDENCE:\n{scraped[:MAX_SCRAPED_CHARS]}"
    )

    logger.info("Step 3: Write")
    report = writer_chain.invoke({"topic": topic, "research": research})

    logger.info("Step 4: Critic")
    feedback = critic_chain.invoke({"report": report, "research": research})

    return {
        "report": report,
        "feedback": feedback,
        "sources": extract_urls(report),
    }


if __name__ == "__main__":
    result = run_research_pipeline(input("Enter a research topic: "))

    print("\n" + "=" * 60 + "\nFINAL REPORT\n" + "=" * 60)
    print(result["report"])

    print("\n" + "=" * 60 + "\nCRITIC FEEDBACK\n" + "=" * 60)
    print(result["feedback"])

    print("\n" + "=" * 60 + "\nSOURCES\n" + "=" * 60)
    for url in result["sources"]:
        print("-", url)
