import logging
import os

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


def perform_exa_search(
    query: str,
    max_results: int = 10,
    domains: list[str] | None = None,
) -> list[SearchResult]:
    """Perform search and get content using Exa API."""
    exa_client = init_exa_client()
    if not domains:
        response = exa_client.search_and_contents(
            query,
            num_results=max_results,
            summary=True,
        )
    else:
        response = exa_client.search_and_contents(
            query,
            num_results=max_results,
            summary=True,
            include_domains=domains,
        )
    results = getattr(response, "results", [])
    return [
        SearchResult(
            title=str(result.title),
            url=str(result.url),
            content=str(result.summary),
            score=getattr(result, "score", None),
        )
        for result in results
    ]


def exa_search_agent(
    state: SearchState,
    max_results: int = 10,
    field_to_update: str = "exa_results",
    domains: list[str] | None = None,
) -> SearchState:
    """Define an agent to perform Exa search.

    Args:
        state: The current search state with the refined query
        max_results: The maximum number of results to return

    Returns:
        Updated state with Exa search results

    """
    setattr(state, field_to_update, [])

    # Check if we have a query to search
    if not state.original_query:
        return state

    try:
        results = perform_exa_search(
            query=state.original_query,
            max_results=max_results,
            domains=domains,
        )
        setattr(state, field_to_update, results)
    except Exception as e:
        logger.error(f"Error in Exa search agent: {str(e)}", exc_info=True)

    return state
