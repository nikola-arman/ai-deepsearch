from typing import List, Dict, Any, Optional
import os
import logging
from dotenv import load_dotenv
from pymed import PubMed
from pymed.article import PubMedArticle
import datetime

from deepsearch.models import SearchState, SearchResult

# Set up logging
logger = logging.getLogger("deepsearch.pubmed")

# Load environment variables
load_dotenv()

# Get email for PubMed API (required by their terms of service)
pubmed_email = os.environ.get("PUBMED_EMAIL")


def init_pubmed_client():
    """Initialize the PubMed client."""
    if not pubmed_email:
        raise ValueError("PUBMED_EMAIL environment variable is not set")

    return PubMed(tool="Biomedical-DeepSearch", email=pubmed_email)


def convert_pubmed_results(pubmed_results: List[PubMedArticle]) -> List[SearchResult]:
    """Convert PubMed search results to our SearchResult model."""
    search_results = []

    for i, result in enumerate(pubmed_results):
        try:
            dict_result = result.toDict()
            # Extract content with validation
            title = dict_result.get("title", "Untitled")
            pubmed_id = dict_result.get("pubmed_id", "unknown") # .split("\n")[0] # idk what i am doing, but it seems to work

            methods = dict_result.get("methods", "")
            abstract = dict_result.get("abstract", "")
            conclusions = dict_result.get("conclusions", "")
            results = dict_result.get("results", "")

            authors = dict_result.get("authors", [])
            publication_date: Optional[datetime.date] = dict_result.get("publication_date", None)
            
            if isinstance(publication_date, datetime.date):
                publication_date = publication_date.strftime("%Y-%m-%d")
            else:
                publication_date = "unknown"
                
            contents = [
                ('Abstract', abstract),
                ('Methods', methods),
                ('Conclusions', conclusions),
                ('Results', results)
            ]
            
            search_results.extend([
                SearchResult(
                    title=title,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}",
                    content=f"{title}\n\n{label}: {content}",
                    score=None,
                    publication_date=publication_date,
                    authors=authors
                )
                for label, content in contents
                if content
            ])

        except Exception as e:
            logger.warning(f"Error processing PubMed result {i}: {str(e)}")
            continue

    return search_results


def pubmed_search_agent(state: SearchState) -> SearchState:
    """
    Fetches medical research articles from PubMed using the PyMed library.

    Args:
        state: The current search state with the refined query

    Returns:
        Updated state with search results
    """
    try:
        # Use the original query
        query = state.original_query

        logger.debug(f"Searching PubMed for: {query}")

        # Initialize PubMed client
        client = init_pubmed_client()

        # Perform the search
        search_response = client.query(
            query=query,
            max_results=30  # Fetch enough results for good indexing
        )

        # Convert the results to our model
        pubmed_results = convert_pubmed_results(search_response)
        print("DEBUG found", len(pubmed_results), "results for query ", query)
        logger.debug(f"Found {len(pubmed_results)} results from PubMed for query {query}")

        # Update the state
        state.pubmed_results = pubmed_results
    except Exception as e:
        logger.error(f"Error in PubMed search: {str(e)}", exc_info=True)
        # Set empty results on error
        state.pubmed_results = []

    return state
