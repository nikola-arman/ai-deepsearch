from typing import Generator, List
import os
import logging
import requests
from deepsearch.models import SearchState, SearchResult
from deepsearch.utils import to_chunk_data, wrap_thought

# Set up logging
logger = logging.getLogger("deepsearch.google")

# Get Google API key and search engine ID from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "no-need")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "no-need")

def google_search(query: str, max_results: int = 10) -> List[SearchResult]:
    """
    Perform a web search using Google Custom Search API.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of SearchResult objects
    """
    if not query:
        return []

    try:
        # Prepare the API request
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_SEARCH_ENGINE_ID,
            "q": query,
            "num": max_results
        }

        # Make the API request
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        data = response.json()

        print("Google search data:", data)
        
        # Extract results
        results = []
        if "items" in data:
            for item in data["items"]:
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    content=item.get("snippet", ""),
                    score=None  # Google doesn't provide relevance scores
                )
                results.append(result)
        
        return results

    except Exception as e:
        logger.error(f"Error in Google search: {str(e)}", exc_info=True)
        return []

def google_search_agent(state: SearchState, max_results: int = 10) -> Generator[bytes, None, SearchState]:
    """
    Uses Google Custom Search API to perform web searches.

    Args:
        state: The current search state
        max_results: Maximum number of results to return

    Returns:
        Updated state with Google search results
    """
    # Initialize results as empty list
    state.google_results = []

    # Check if we have a query to search
    if not state.original_query:
        yield to_chunk_data(
            wrap_thought(
                "Google search agent: No query",
                "No query provided for Google search"
            )
        )
        return state

    try:
        yield to_chunk_data(
            wrap_thought(
                "Google search agent: Starting search",
                f"Searching for: {state.original_query}"
            )
        )

        # Perform the search
        results = google_search(
            state.original_query,
            max_results=max_results
        )

        # Update the state
        state.google_results = results
        yield to_chunk_data(
            wrap_thought(
                "Google search agent: Complete",
                f"Found {len(results)} results"
            )
        )

    except Exception as e:
        logger.error(f"Error in Google search agent: {str(e)}", exc_info=True)
        yield to_chunk_data(
            wrap_thought(
                "Google search agent: Error",
                f"Error occurred during Google search: {str(e)}"
            )
        )

    return state 