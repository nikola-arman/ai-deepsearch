from typing import List, Optional, Tuple, Dict, Generator
import os
import numpy as np
import faiss
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import re

from deepsearch.schemas.agents import SearchState, SearchResult

# Set up logging
logger = logging.getLogger("deepsearch.faiss")

# Load environment variables
load_dotenv()

# Get the OpenAI-compatible API base URL and API key
openai_api_base = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
openai_api_key = os.environ.get("LLM_API_KEY", "no-need")


def batching(data: Generator, batch_size = 1):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def init_embedding_model():
    """Initialize the embedding model using OpenAI-compatible API."""
    try:
        # Use OpenAI-compatible embeddings from the server
        embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small"),  # Default to a known model
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            check_embedding_ctx_length=False,
        )

        # Test the embeddings with a simple query to catch any initialization issues
        try:
            test_embedding = embeddings.embed_query("test")
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError("Embedding model returned empty embeddings")
            logger.info(f"Embedding model initialized successfully. Vector dimension: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"Failed to generate test embedding: {str(e)}")
            raise

        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {str(e)}")
        raise


def sanitize_text_for_embedding(text):
    """Sanitize text for embedding to ensure it's a valid string."""
    if isinstance(text, str):
        return text.strip()
    elif isinstance(text, (list, tuple)) and len(text) > 0:
        # If it's a list or tuple, try to convert the first element to string
        first_element = text[0]
        if isinstance(first_element, (int, float)):
            # Convert numeric value to string
            logger.warning(f"Converting numeric value {first_element} to string for embedding")
            return str(first_element)
        elif isinstance(first_element, str):
            return first_element.strip()
        else:
            # Try to convert arbitrary object to string
            return str(first_element)
    elif isinstance(text, (int, float)):
        # Convert numeric value to string
        logger.warning(f"Converting numeric value {text} to string for embedding")
        return str(text)
    elif text is None:
        return ""
    else:
        # Try to convert arbitrary object to string
        logger.warning(f"Converting {type(text)} to string for embedding")
        return str(text)


def chunk_text(text: str, max_length: int = 2400) -> List[str]:
    """Split text into chunks of approximately max_length tokens.

    Args:
        text: Text to split into chunks
        max_length: Maximum number of characters per chunk (rough approximation of tokens)

    Returns:
        List of text chunks
    """
    # Basic sentence splitting on periods, exclamation marks, or question marks
    # followed by spaces and capital letters
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()[:max_length]
        sentence_length = len(sentence)

        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # If the current chunk has content, add it to chunks
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            # Start new chunk with current sentence
            current_chunk = [sentence]
            current_length = sentence_length

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def create_faiss_index(embeddings, search_results: List[SearchResult]) -> Tuple[Optional[faiss.Index], Optional[List], Optional[List[int]], Optional[Dict]]:
    """Create a FAISS index from search results."""
    if not search_results:
        logger.warning("No search results provided for FAISS indexing")
        return None, None, None, None

    try:
        # Extract the content from search results and ensure they're all valid strings
        texts = []
        valid_indices = []
        original_to_chunk_map = {}  # Map to track which chunks belong to which original text
        chunk_count = 0

        for i, result in enumerate(search_results):
            # Apply more robust content validation and sanitization
            if result.content is not None:
                # Sanitize the content to ensure it's a valid string
                sanitized_content = sanitize_text_for_embedding(result.content)
                # Only include non-empty content
                if sanitized_content and len(sanitized_content.strip()) > 0:
                    # Split content into chunks if it's too long
                    chunks = chunk_text(sanitized_content.strip())
                    for chunk in chunks:
                        texts.append(chunk)
                        valid_indices.append(i)
                        original_to_chunk_map[chunk_count] = i
                        chunk_count += 1
                else:
                    logger.warning(f"Skipping empty content after sanitization at index {i}")
            else:
                logger.warning(f"Skipping result with None content at index {i}")

        if not texts:
            logger.warning("No valid text content in search results for FAISS indexing")
            return None, None, None, None

        # Log the number of valid texts
        logger.info(f"Generating embeddings for {len(texts)} chunks")

        # Process texts in batches to isolate errors
        batch_size = 32
        embedded_texts = []
        final_texts = []
        final_indices = []
        final_chunk_map = {}

        for batch_idx, batch_texts in enumerate(batching(texts, batch_size)):
            try:
                # Additional type validation for the batch
                batch_texts = [
                    str(text) if not isinstance(text, str) else text 
                    for text in batch_texts
                ]
                
                # Get embeddings for the batch
                batch_embeddings = embeddings.embed_documents(batch_texts)

                logger.info(f"len(batch_texts): {len(batch_texts)}, len(batch_embeddings): {len(batch_embeddings)}")
                
                # Add each embedding and its associated data
                for i, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    embedded_texts.append(embedding)
                    final_texts.append(text)
                    original_idx = valid_indices[batch_idx * batch_size + i]
                    final_indices.append(original_idx)
                    final_chunk_map[len(embedded_texts) - 1] = original_to_chunk_map[batch_idx * batch_size + i]
            
            except Exception as e:
                logger.warning(f"Skipping batch at index {batch_idx} due to embedding error: {str(e)}")
                continue

        if not embedded_texts:
            logger.warning("No valid embeddings generated")
            return None, None, None, None

        # Create a FAISS index
        dimension = len(embedded_texts[0])
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize the vectors for cosine similarity
        faiss.normalize_L2(np.array(embedded_texts, dtype=np.float32))

        # Add the vectors to the index
        index.add(np.array(embedded_texts, dtype=np.float32))

        return index, embedded_texts, final_indices, final_chunk_map
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}", exc_info=True)
        return None, None, None, None


def faiss_search(query: str, embeddings, index, embedded_texts: List, valid_indices: List[int], search_results: List[SearchResult], chunk_to_original_map: Dict[int, int], top_k: int = 5):
    """Search the FAISS index with the query."""
    if not index or not embedded_texts or not valid_indices or not query:
        logger.warning("Missing required parameters for FAISS search")
        return []

    try:
        # Ensure query is a valid string
        if not isinstance(query, str):
            logger.warning(f"Converting non-string query of type {type(query)} to string")
            query = sanitize_text_for_embedding(query)

        if not query or not query.strip():
            logger.warning("Invalid query for FAISS search after sanitization")
            return []

        query = query.strip()

        # Generate embedding for the query
        try:
            query_embedding = embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            return []

        # Normalize the query vector for cosine similarity
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding_np)

        # Search the index - get more results initially since we'll combine chunks
        chunk_k = min(top_k * 3, len(embedded_texts))  # Get more chunks initially
        distances, indices = index.search(query_embedding_np, chunk_k)

        # Track scores by original document
        doc_scores = {}  # Map from original doc index to best score
        doc_chunks = {}  # Map from original doc index to matching chunks

        # Map the results back to original documents and combine scores
        for i, idx in enumerate(indices[0]):
            if idx < len(embedded_texts):
                # Get the original document index
                original_idx = chunk_to_original_map.get(int(idx), valid_indices[idx])
                if original_idx < len(search_results):
                    score = float(distances[0][i])

                    # Track the best score for each original document
                    if original_idx not in doc_scores or score > doc_scores[original_idx]:
                        doc_scores[original_idx] = score

                    # Store the matching chunk
                    if original_idx not in doc_chunks:
                        doc_chunks[original_idx] = []
                    doc_chunks[original_idx].append(embedded_texts[idx])

        # Create results using the best scores
        faiss_results = []
        for original_idx, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            if original_idx < len(search_results):
                result: SearchResult = search_results[original_idx]
                faiss_results.append(
                    SearchResult(
                        id=result.id,
                        title=result.title,
                        url=result.url,
                        content=result.content,
                        score=score,
                        query=query  # Add the query that produced this result
                    )
                )

        return faiss_results
    except Exception as e:
        logger.error(f"Error in FAISS search: {str(e)}", exc_info=True)
        return []

def faiss_indexing_agent(state: SearchState) -> SearchState:
    """
    Builds a FAISS index on-the-fly from search results and performs vector search.

    Args:
        state: The current search state with search results

    Returns:
        Updated state with FAISS search results
    """
    # Check if we have search results to work with
    if not state.search_results or len(state.search_results) == 0:
        # If no results, skip this agent
        logger.info("No search results available for FAISS indexing")
        state.faiss_results = []
        return state

    # Ensure we have at least some content to work with
    valid_results = [
        r for r in state.search_results if r.content and isinstance(r.content, str) and len(r.content.strip()) > 0
    ]

    if not valid_results:
        # If no valid content in results, skip FAISS
        logger.info("No valid content in search results for FAISS indexing")
        state.faiss_results = []
        return state

    try:
        # Initialize the embedding model
        try:
            embeddings = init_embedding_model()
        except Exception as e:
            logger.error(f"Failed to initialize embedding model. Skipping FAISS search: {str(e)}")
            state.faiss_results = []
            return state

        # Create a FAISS index from the search results
        index, embedded_texts, valid_indices, chunk_to_original_map = create_faiss_index(embeddings, valid_results)

        if not index or not embedded_texts or not valid_indices:
            logger.warning("Failed to create FAISS index")
            state.faiss_results = []
            return state

        # Store the chunk mapping in the state metadata
        state.metadata['chunk_to_original_map'] = chunk_to_original_map

        # Use the original query
        query = state.original_query

        # Validate query
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Invalid query for FAISS search")
            state.faiss_results = []
            return state

        # Search the index with the query
        faiss_results = faiss_search(
            query=query,
            embeddings=embeddings,
            index=index,
            embedded_texts=embedded_texts,
            valid_indices=valid_indices,
            search_results=valid_results,
            chunk_to_original_map=chunk_to_original_map,
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