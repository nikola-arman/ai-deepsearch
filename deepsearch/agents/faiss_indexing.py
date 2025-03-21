from typing import List, Optional, Tuple
import os
import numpy as np
import faiss
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from deepsearch.models import SearchState, SearchResult

# Set up logging
logger = logging.getLogger("deepsearch.faiss")

# Load environment variables
load_dotenv()

# Get the OpenAI-compatible API base URL and API key
openai_api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8080/v1")
openai_api_key = os.environ.get("OPENAI_API_KEY", "not-needed")


def init_embedding_model():
    """Initialize the embedding model using OpenAI-compatible API."""
    # Use OpenAI-compatible embeddings from the server
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_ID", "no-need"),
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base if not openai_api_key or openai_api_key == "not-needed" else None,
        dimensions=384  # Adjust based on your model
    )
    return embeddings


def create_faiss_index(embeddings, search_results: List[SearchResult]) -> Tuple[Optional[faiss.Index], Optional[List], Optional[List[int]]]:
    """Create a FAISS index from search results."""
    if not search_results:
        logger.warning("No search results provided for FAISS indexing")
        return None, None, None

    try:
        # Extract the content from search results
        texts = [result.content for result in search_results if result.content and len(result.content.strip()) > 0]

        if not texts:
            logger.warning("No valid text content in search results for FAISS indexing")
            return None, None, None

        # Track which search results have valid content
        valid_indices = [i for i, result in enumerate(search_results) if result.content and len(result.content.strip()) > 0]

        if not valid_indices:
            logger.warning("No valid indices for FAISS indexing")
            return None, None, None

        # Generate embeddings for each text
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        embedded_texts = embeddings.embed_documents(texts)

        # Create a FAISS index
        dimension = len(embedded_texts[0])
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize the vectors for cosine similarity
        faiss.normalize_L2(np.array(embedded_texts, dtype=np.float32))

        # Add the vectors to the index
        index.add(np.array(embedded_texts, dtype=np.float32))

        return index, embedded_texts, valid_indices
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}", exc_info=True)
        return None, None, None


def faiss_search(query: str, embeddings, index, embedded_texts: List, valid_indices: List[int], search_results: List[SearchResult], top_k: int = 5):
    """Search the FAISS index with the query."""
    if not index or not embedded_texts or not valid_indices or not query:
        logger.warning("Missing required parameters for FAISS search")
        return []

    try:
        # Generate embedding for the query
        query_embedding = embeddings.embed_query(query)

        # Normalize the query vector for cosine similarity
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding_np)

        # Search the index
        distances, indices = index.search(query_embedding_np, min(top_k, len(embedded_texts)))

        # Map the results back to search results with scores
        faiss_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(valid_indices):
                original_idx = valid_indices[idx]
                if original_idx < len(search_results):
                    result = search_results[original_idx]
                    # Convert distance to score (higher is better)
                    score = float(distances[0][i])
                    faiss_results.append(
                        SearchResult(
                            title=result.title,
                            url=result.url,
                            content=result.content,
                            score=score
                        )
                    )

        # Sort by score in descending order
        faiss_results.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)

        return faiss_results
    except Exception as e:
        logger.error(f"Error in FAISS search: {str(e)}", exc_info=True)
        return []


def faiss_indexing_agent(state: SearchState) -> SearchState:
    """
    Builds a FAISS index on-the-fly from Tavily search results and performs vector search.

    Args:
        state: The current search state with Tavily search results

    Returns:
        Updated state with FAISS search results
    """
    # Check if we have Tavily results to work with
    if not state.tavily_results or len(state.tavily_results) == 0:
        # If no results, skip this agent
        logger.info("No Tavily results available for FAISS indexing")
        state.faiss_results = []
        return state

    # Ensure we have at least some content to work with
    valid_results = [r for r in state.tavily_results if r.content and len(r.content.strip()) > 0]
    if not valid_results:
        # If no valid content in results, skip FAISS
        logger.info("No valid content in Tavily results for FAISS indexing")
        state.faiss_results = []
        return state

    try:
        # Initialize the embedding model
        embeddings = init_embedding_model()

        # Create a FAISS index from the search results
        index, embedded_texts, valid_indices = create_faiss_index(embeddings, valid_results)

        if not index or not embedded_texts or not valid_indices:
            logger.warning("Failed to create FAISS index")
            state.faiss_results = []
            return state

        # Use refined query if available, otherwise use original query
        query = state.refined_query if state.refined_query else state.original_query

        # Search the index with the query
        faiss_results = faiss_search(
            query=query,
            embeddings=embeddings,
            index=index,
            embedded_texts=embedded_texts,
            valid_indices=valid_indices,
            search_results=valid_results,
            top_k=5
        )

        # Update the state
        state.faiss_results = faiss_results
        logger.info(f"FAISS search found {len(faiss_results)} relevant results")
    except Exception as e:
        # If any error occurs, log it and return empty results
        logger.error(f"Error in FAISS search: {str(e)}", exc_info=True)
        state.faiss_results = []

    return state