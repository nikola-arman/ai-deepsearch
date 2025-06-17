from typing import Generator, Tuple, List
from rank_bm25 import BM25Okapi
import re
import logging
from deepsearch.schemas.agents import SearchState, SearchResult

# Set up logging
logger = logging.getLogger("deepsearch.bm25")

MINIMUM_TOKEN_COUNT = 15

def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 indexing."""
    if not text or not isinstance(text, str):
        return []

    # Basic tokenization: lowercase and split on non-alphanumeric characters
    tokens = re.split(r'\W+', text.lower())
    # Remove empty tokens and tokens that are just numbers
    tokens = [token for token in tokens if token and not token.isdigit()]
    return tokens

def create_bm25_index(search_results: List[SearchResult]):
    """Create a BM25 index from search results."""
    # Extract the content from search results
    texts = [result.content for result in search_results if result.content]

    # Guard against empty texts
    if not texts:
        return None, []

    # Tokenize all texts
    tokenized_texts = [tokenize(text) for text in texts]

    # Filter out empty tokenized texts
    valid_indices = [i for i, tokens in enumerate(tokenized_texts) if tokens and len(tokens) >= MINIMUM_TOKEN_COUNT]
    valid_tokenized_texts = [tokenized_texts[i] for i in valid_indices]

    # Guard against all empty tokenized texts
    if not valid_tokenized_texts:
        return None, []

    try:
        # Create BM25 index
        bm25 = BM25Okapi(valid_tokenized_texts)
        return bm25, valid_indices, valid_tokenized_texts
    except Exception as e:
        logger.error(f"Error creating BM25 index: {str(e)}")
        return None, [], []

def bm25_search(query: str, bm25, valid_indices: List[int], search_results: List[SearchResult], top_k: int = 5):
    """Search the BM25 index with the query."""
    if not bm25 or not valid_indices:
        return []

    # Tokenize the query
    tokenized_query = tokenize(query)

    if not tokenized_query:
        logger.warning("No valid tokens in query for BM25 search")
        return []

    try:
        # Get scores for all documents
        scores = bm25.get_scores(tokenized_query)

        # Create (index, score) pairs and sort by score in descending order
        scored_indices = [(valid_indices[i], score) for i, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        # Get the top k results
        top_indices = scored_indices[:top_k]
        
        if len(top_indices) == 0:
            logger.warning("No valid BM25 results found")
            return []

        # Map the results back to search results with scores
        bm25_results = []
        max_score, min_score = max([score for _, score in top_indices]), min([score for _, score in top_indices])

        for idx, score in top_indices:
            if idx < len(search_results):
                result = search_results[idx]
                norm_score = (score - min_score) / (max_score - min_score) if max_score != min_score else 0 
                bm25_results.append(
                    SearchResult(
                        id=result.id,
                        title=result.title,
                        url=result.url,
                        content=result.content,
                        score=float(norm_score)
                    )
                )

        return bm25_results
    except Exception as e:
        logger.error(f"Error in BM25 search: {str(e)}")
        return []

def bm25_search_agent(state: SearchState) -> SearchState:
    """
    Uses BM25 for on-the-fly keyword-based retrieval from search results.

    Args:
        state: The current search state with search results

    Returns:
        Updated state with BM25 search results and combined results
    """
    # Initialize combined results as an empty list
    state.combined_results = []

    # Check if we have search results to work with
    if not state.search_results or len(state.search_results) == 0:
        logger.info("No search results available for BM25 search")
        # If no search results, just use the FAISS results
        state.bm25_results = []
        state.combined_results = state.faiss_results if state.faiss_results else []
        return state

    # Ensure we have at least some content to work with
    valid_results = [r for r in state.search_results if r.content and len(r.content.strip()) > 0]
    if not valid_results:
        logger.info("No valid content in search results for BM25 search")
        # If no valid content in results, skip BM25
        state.bm25_results = []
        state.combined_results = state.faiss_results if state.faiss_results else []
        return state

    try:
        # Create a BM25 index from the search results
        bm25, valid_indices, tokenized_texts = create_bm25_index(valid_results)

        if not bm25:
            logger.warning("Failed to create BM25 index")
            state.bm25_results = []
            state.combined_results = state.faiss_results if state.faiss_results else []
            return state

        # Use the original query
        query = state.original_query

        # Search the index with the query
        bm25_results = bm25_search(
            query=query,
            bm25=bm25,
            valid_indices=valid_indices,
            search_results=valid_results,
            top_k=5
        )

        # Update the state
        state.bm25_results = bm25_results

    except Exception as e:
        # If any error occurs, just skip BM25
        logger.error(f"Error in BM25 search: {str(e)}", exc_info=True)
        state.bm25_results = []

    # Combine all results (will be used by the reasoning agent)
    all_results = []

    # Add FAISS results with higher priority
    if state.faiss_results:
        for result in state.faiss_results:
            if result not in all_results:
                all_results.append(result)

    # Add BM25 results
    if state.bm25_results:
        for result in state.bm25_results:
            if result not in all_results:
                all_results.append(result)

    # Add any remaining search results
    if state.search_results:
        for result in state.search_results:
            if result not in all_results:
                all_results.append(result)

    # Sort by score if available
    all_results.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)

    # Limit to top 10 combined results to avoid overwhelming the reasoning agent
    state.combined_results = all_results[:10]

    return state