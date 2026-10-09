import ipaddress
import os
import socket
from io import BytesIO
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise RuntimeError("Missing TAVILY_API_KEY in your environment.")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
REQUEST_TIMEOUT = 15


def _validate_public_http_url(url: str) -> str:
    """Allow public HTTP(S) URLs only; block local/private destinations."""
    parsed = urlparse(url.strip())

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only valid HTTP(S) URLs are allowed.")

    host = parsed.hostname.lower()
    if host in {"localhost"} or host.endswith((".localhost", ".local")):
        raise ValueError("Local destinations are not allowed.")

    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError("Could not resolve URL host.") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Private or non-public destinations are not allowed.")

    return url.strip()


def _download_limited(url: str) -> requests.Response:
    """Download with a size cap and revalidate redirects."""
    safe_url = _validate_public_http_url(url)
    with requests.get(
        safe_url,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "ResearchMind/1.0"},
        stream=True,
        allow_redirects=False,
    ) as response:
        response.raise_for_status()

        # Do not automatically follow redirects to unvalidated destinations.
        if 300 <= response.status_code < 400:
            raise ValueError("Redirect detected. Submit the final destination URL.")

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise ValueError("File exceeds the 15 MB download limit.")

        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError("Download exceeded the 15 MB limit.")
            chunks.append(chunk)

        # Return a small response-like object with downloaded bytes.
        response._researchmind_content = b"".join(chunks)
        return response


@tool
def search_web(query: str) -> str:
    """Search the web and return titles, URLs, and short summaries."""
    try:
        results = tavily.search(query=query, max_results=5)
        output = []

        for result in results.get("results", []):
            output.append(
                f"Title: {result.get('title', 'Untitled')}\n"
                f"URL: {result.get('url', '')}\n"
                f"Summary: {str(result.get('content', ''))[:300]}\n"
            )

        return "\n-----\n".join(output) if output else "No search results found."
    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"


@tool
def scrape_url(url: str) -> str:
    """Extract readable text from a public HTML webpage URL."""
    try:
        response = _download_limited(url)
        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
            return "This URL appears to be a PDF. Use read_pdf(url) instead."

        if "html" not in content_type and content_type:
            return f"Unsupported webpage content type: {content_type}"

        soup = BeautifulSoup(response._researchmind_content, "html.parser")

        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            element.decompose()

        text = soup.get_text(separator=" ", strip=True)
        if not text:
            return "No readable HTML text was extracted."
        return text[:5000]

    except Exception as exc:
        return f"Could not scrape URL: {type(exc).__name__}: {exc}"


@tool
def read_pdf(url: str) -> str:
    """Extract text from a publicly accessible PDF URL, up to the first 15 pages."""
    try:
        from pypdf import PdfReader

        response = _download_limited(url)
        content_type = response.headers.get("Content-Type", "").lower()
        content = response._researchmind_content

        if "pdf" not in content_type and not content.startswith(b"%PDF"):
            return "The URL response does not appear to be a PDF."

        reader = PdfReader(BytesIO(content))
        page_text = []

        for page in reader.pages[:15]:
            page_text.append(page.extract_text() or "")

        extracted = "\n".join(page_text).strip()
        if not extracted:
            return "No extractable text found. The PDF may be scanned and require OCR."

        return extracted[:5000]
    except ImportError:
        return "PDF support is missing. Install pypdf with: pip install pypdf"
    except Exception as exc:
        return f"Could not read PDF: {type(exc).__name__}: {exc}"
