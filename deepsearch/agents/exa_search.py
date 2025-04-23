import logging
import os
import re

import openai
from dotenv import load_dotenv
from exa_py import Exa

from deepsearch.models import SearchResult, SearchState

logger = logging.getLogger("deepsearch.exa")

CONTEXT_LENGTH = 30000
EXA_RESULTS_DIR = "exa_results/"
os.makedirs(EXA_RESULTS_DIR, exist_ok=True)

load_dotenv()

exa_api_key = os.environ.get("EXA_API_KEY")


def init_exa_client():
    """Initialize the Exa client."""
    if not exa_api_key:
        raise ValueError("EXA_API_KEY environment variable is not set")

    return Exa(api_key=exa_api_key)


def remove_markdown_urls(text: str) -> str:
    """Remove all markdown URLs from the text."""
    if not text:
        return text

    # Remove markdown URLs
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)
    # Remove markdown image URLs
    text = re.sub(r'!\[([^\]]+)\]\(([^)]+)\)', r'\1', text)
    # Strip any redundant whitespace
    text = text.strip()

    return text


def split_markdown_sections(text: str, max_results: int = 10) -> str:
    """Split the text into sections based on markdown headers.

    First of all, try to split by first heading (#).
    If there are any sections which exceed CONTEXT_LENGTH,
    split them by second heading (##).
    If there still be sections which exceed CONTEXT_LENGTH,
    split them by third heading (###).

    """
    if not text:
        return text

    # Split by first heading
    sections = re.split(r'(?<=\n)\s*#\s*', text)
    if len(sections) > 1:
        for i, section in enumerate(sections):
            if len(section) > CONTEXT_LENGTH:
                # Split by second heading
                sub_sections = re.split(r'(?<=\n)\s*##\s*', section)
                sections[i] = "\n".join(sub_sections)

    # If still too long, split by third heading
    for i, section in enumerate(sections):
        if len(section) > CONTEXT_LENGTH:
            sub_sections = re.split(r'(?<=\n)\s*###\s*', section)
            sections[i] = "\n".join(sub_sections)

    if len(sections) > max_results:
        # Limit the number of sections to max_results
        return sections[:max_results]

    return sections


def post_processing_markdown_result(text: str, max_results: int = 10) -> str:
    """Post-processing Markdown result.

    1. Remove all markdown URLs
    2. Split the text into sections based on markdown headers.
    3. Call the LLM to summarize each section.
    4. Call the LLM to summarize the whole text based on given summaries.
    5. Return the final summary.

    """
    text = remove_markdown_urls(text)
    sections = split_markdown_sections(text, max_results=max_results)

    for section in sections:
        if len(section) > CONTEXT_LENGTH:
            section = section[:CONTEXT_LENGTH]

    openai_client = openai.OpenAI(
        api_key=os.getenv("LLM_API_KEY", "no-need"),
        base_url=os.getenv("LLM_BASE_URL", "http://localmodel:65534/v1"),
    )

    text_summaries = []
    for section in sections:
        try:
            response = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Please summarize the following text "
                            "in a paragraph of around 5 - 7 sentences:\n\n"
                            f"{section}\n\n"
                        )
                    },
                ],
                temperature=0.1,
                max_tokens=500,
                top_p=1.0,
                model=os.getenv("LLM_MODEL_ID", "gpt-4.1-mini"),
            )
            text_summaries.append(response.choices[0].message.content)
        except Exception:
            continue

    summary = "\n".join(text_summaries)
    response = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": (
                    "Please summarize the following text "
                    "in a paragraph of around 5 - 7 sentences:\n\n"
                    f"{summary}\n\n"
                )
            },
        ],
        temperature=0.1,
        max_tokens=500,
        top_p=1.0,
        model=os.getenv("LLM_MODEL_ID", "gpt-4.1-mini"),
    )

    return response.choices[0].message.content


def perform_exa_search(
    query: str,
    max_results: int = 10,
) -> list[SearchResult]:
    """Perform search and get content using Exa API."""
    exa_client = init_exa_client()
    response = exa_client.search_and_contents(query, num_results=max_results)
    results = getattr(response, "results", [])
    search_results = []
    for i, result in enumerate(results):
        text = getattr(result, "text", "")
        if not text:
            logger.warning(f"Result {i} has no content")
            continue
        try:
            summary = post_processing_markdown_result(
                text,
                max_results=max_results,
            )
        except Exception:
            continue
        search_results.append(
            SearchResult(
                title=str(result.title),
                url=str(result.url),
                content=summary,
                score=getattr(result, "score", None),
            ),
        )
    return search_results


def exa_search_agent(state: SearchState, max_results: int = 10) -> SearchState:
    """Define an agent to perform Exa search.

    Args:
        state: The current search state with the refined query
        max_results: The maximum number of results to return

    Returns:
        Updated state with Exa search results

    """
    state.exa_results = []

    # Check if we have a query to search
    if not state.original_query:
        return state

    try:
        results = perform_exa_search(
            query=state.original_query,
            max_results=max_results,
        )
        state.exa_results = results
    except Exception as e:
        logger.error(f"Error in Exa search agent: {str(e)}", exc_info=True)

    return state
