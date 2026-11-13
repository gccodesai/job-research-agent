"""Tools the agent can choose to call.

Each tool is a plain Python function plus a JSON "schema" that describes it
to the LLM. The LLM never runs code itself: it asks us to run a tool, we run
it, and we hand the result back. That loop is what makes this an *agent*.

Tools never raise exceptions to the agent. They return an error message
instead, so the agent can read it and try something else.
"""

import json

import requests
from bs4 import BeautifulSoup

MAX_PAGE_CHARS = 6000  # keep pages short so we stay within free-tier limits
HEADERS = {"User-Agent": "Mozilla/5.0 (job-research-agent; portfolio project)"}


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web with DuckDuckGo (free, no API key needed)."""
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=max_results)
    except Exception as e:  # network problems, rate limits, etc.
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    lines = [
        f"- {r.get('title', '')}\n  URL: {r.get('href', '')}\n  {r.get('body', '')}"
        for r in results
    ]
    return "\n".join(lines)


def fetch_webpage(url: str) -> str:
    """Download a web page and return its readable text."""
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return (
            f"Could not fetch {url}: {e}. Many job sites (LinkedIn, Indeed) block "
            "automated access; the user may need to paste the posting text instead."
        )
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + " ...[truncated]"
    return text or "The page had no readable text."


# Descriptions the LLM reads to decide which tool to use and when.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web. Use for company news, strategy, culture, "
                "recent announcements, leadership, and industry context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": (
                "Read the text of a web page. Use to read a job posting URL "
                "or a promising page found through web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL"},
                },
                "required": ["url"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "fetch_webpage": fetch_webpage,
}


def execute_tool(name: str, arguments_json: str) -> str:
    """Run the tool the LLM asked for and return its result as text."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'."
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return f"Error: could not parse tool arguments: {arguments_json}"
    try:
        return func(**args)
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}"
