
import os
os.environ['OPENAI_BASE_URL'] = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL"))
os.environ['OPENAI_API_KEY'] = os.getenv("LLM_API_KEY", 'no-need')

from typing import Dict, Any, Callable
from deepsearch.models import SearchState
from deepsearch.agents import (
    faiss_indexing_agent,
    bm25_search_agent,
    llama_reasoning_agent,
    deep_reasoning_agent,
    pubmed_search_agent
)
from starlette.concurrency import run_in_threadpool
from functools import partial
import asyncio

def sync2async(sync_func: Callable):
    async def async_func(*args, **kwargs):
        return await run_in_threadpool(partial(sync_func, *args, **kwargs))
    return async_func if not asyncio.iscoroutinefunction(sync_func) else sync_func

# Convert synchronous functions to asynchronous
async_faiss_indexing_agent = sync2async(faiss_indexing_agent)
async_bm25_search_agent = sync2async(bm25_search_agent)
async_llama_reasoning_agent = sync2async(llama_reasoning_agent)
async_deep_reasoning_agent = sync2async(deep_reasoning_agent)
async_pubmed_search_agent = sync2async(pubmed_search_agent)

import logging
logger = logging.getLogger(__name__)

async def run_simple_pipeline(query: str) -> Dict[str, Any]:
    """Run a simple pipeline without graph complexity."""
    try:
        # Initialize state
        state = SearchState(original_query=query)

        logger.info(f"Using query: {state.original_query}")

        # Step 1: PubMed Search
        logger.info("Step 1: Performing PubMed search...")
        try:
            state = await pubmed_search_agent(state)
            logger.info(f"  Found {len(state.pubmed_results)} PubMed results")
        except Exception as e:
            logger.error(f"  Error in PubMed search: {str(e)}", exc_info=True)
            state.pubmed_results = []

        # Step 2: FAISS Indexing (semantic search)
        logger.info("Step 2: Performing semantic search...")
        try:
            state = await async_faiss_indexing_agent(state)
            logger.info(f"  Found {len(state.faiss_results)} semantically relevant results")
        except Exception as e:
            logger.error(f"  Error in semantic search: {str(e)}", exc_info=True)
            state.faiss_results = []

        # Step 3: BM25 Search (keyword search)
        logger.info("Step 3: Performing keyword search...")
        try:
            state = await async_bm25_search_agent(state)
            logger.info(f"  Found {len(state.bm25_results)} keyword relevant results")
            logger.info(f"  Combined {len(state.combined_results)} total relevant results")
        except Exception as e:
            logger.error(f"  Error in keyword search: {str(e)}", exc_info=True)
            state.bm25_results = []
            # Ensure we have combined results even if BM25 fails
            if not state.combined_results:
                state.combined_results = state.faiss_results + state.pubmed_results

        # Step 4: LLM Reasoning
        logger.info("Step 4: Generating answer...")
        try:
            state = await async_llama_reasoning_agent(state)
            logger.info(f"  Confidence: {state.confidence_score}")
        except Exception as e:
            logger.error(f"  Error in reasoning: {str(e)}", exc_info=True)
            # Provide a fallback answer
            state.final_answer = "I'm sorry, but I couldn't generate a proper answer for your query due to a technical issue. Please try again with a different query."
            state.confidence_score = 0.1

        # Return the results
        sources = []
        if state.combined_results:
            for res in state.combined_results:
                sources.append({
                    "title": res.title,
                    "url": res.url
                })

        return {
            "original_query": state.original_query,
            "answer": state.final_answer,
            "confidence": state.confidence_score,
            "sources": sources[:5],  # Limit to 5 sources,
            "has_error": False
        }

    except Exception as e:
        # Handle any unexpected errors
        logger.error(f"Unexpected error in pipeline: {str(e)}", exc_info=True)
        return {
            "original_query": query,
            "answer": "An unexpected error occurred while processing your query. Please try again later.",
            "confidence": 0.0,
            "sources": [],
            "has_error": True
        }

async def run_deep_search_pipeline(query: str, max_iterations: int = 2) -> Dict[str, Any]:
    """Run the multi-query, iterative deep search pipeline with reasoning agent."""
    try:
        # Initialize state
        state = SearchState(original_query=query)

        # Instead of using query_expansion_agent, let the deep_reasoning_agent handle initial query generation
        # This avoids potential conflicts and allows for better reasoning about query generation
        logger.info("Step 1: Initial reasoning to analyze query and generate search queries...")
        try:
            # Initial call to deep_reasoning_agent will generate the queries
            state = await async_deep_reasoning_agent(state, max_iterations)
            logger.info(f"  Generated {len(state.generated_queries)} initial search queries")
        except Exception as e:
            logger.error(f"  Error in initial reasoning: {str(e)}", exc_info=True)
            # If reasoning fails, use just the original query
            state.generated_queries = [state.original_query]
            state.current_iteration = 1  # Ensure we don't skip the first iteration

        # Iterative search loop
        while not state.search_complete and state.current_iteration < max_iterations:
            iteration = state.current_iteration
            logger.info(f"Beginning search iteration {iteration}...")

            # Reset results for this iteration but keep accumulated results
            previous_results = state.combined_results.copy() if state.combined_results else []
            state.faiss_results = []
            state.bm25_results = []
            state.combined_results = []

            # Process each query in this iteration
            for i, query in enumerate(state.generated_queries):
                logger.info(f"  Processing query {i+1}/{len(state.generated_queries)}: {query}")

                # Create a temporary state for this query
                temp_state = SearchState(
                    original_query=query  # Use the current query as the original query for this temp state
                )

                # Step 2: PubMed Search for this query
                logger.info(f"    Performing PubMed search...")
                try:
                    temp_state = await async_pubmed_search_agent(temp_state)
                    # Tag results with the query that produced them
                    for result in temp_state.pubmed_results:
                        result.query = query
                    logger.info(f"    Found {len(temp_state.pubmed_results)} PubMed results")
                except Exception as e:
                    logger.error(f"    Error in PubMed search: {str(e)}", exc_info=True)

                # Step 3: FAISS Indexing (semantic search) for this query
                logger.info(f"    Performing semantic search...")
                try:
                    temp_state = await async_faiss_indexing_agent(temp_state)
                    # Tag results with the query that produced them
                    for result in temp_state.faiss_results:
                        result.query = query
                    logger.info(f"    Found {len(temp_state.faiss_results)} semantic results")
                except Exception as e:
                    logger.error(f"    Error in semantic search: {str(e)}", exc_info=True)

                # Step 4: BM25 Search (keyword search) for this query
                logger.info(f"    Performing keyword search...")
                try:
                    temp_state = await async_bm25_search_agent(temp_state)
                    # Tag results with the query that produced them
                    for result in temp_state.bm25_results:
                        result.query = query
                    logger.info(f"    Found {len(temp_state.bm25_results)} keyword results")
                except Exception as e:
                    logger.error(f"    Error in keyword search: {str(e)}", exc_info=True)

                # Collect results from this query
                state.faiss_results.extend(temp_state.faiss_results)
                state.bm25_results.extend(temp_state.bm25_results)
                if temp_state.combined_results:
                    state.combined_results.extend(temp_state.combined_results)

            # Add back previous results to ensure continuity
            if state.combined_results:
                # If we have combined results from this iteration, merge with previous
                seen_urls = {result.url for result in state.combined_results}
                for result in previous_results:
                    if result.url not in seen_urls:
                        state.combined_results.append(result)
            else:
                # No new combined results, use previous ones
                state.combined_results = previous_results

            # Make sure combined_results is populated even if BM25 didn't run
            if not state.combined_results:
                # Combine FAISS and PubMed results
                state.combined_results = state.faiss_results + state.pubmed_results

            # Deduplicate combined results by URL
            if state.combined_results:
                unique_results = {}

                for result in state.combined_results:
                    # Keep the highest scoring result for each URL
                    if result.url not in unique_results:
                        unique_results[result.url + result.content] = []
                    
                    unique_results[result.url + result.content].append(result)

                for k in list(unique_results.keys()):
                    unique_results[k] = sorted(
                        unique_results[k], 
                        key=lambda x: x.score, 
                        reverse=True
                    )[:3]

                state.combined_results = [
                    item 
                    for sublist in unique_results.values() 
                    for item in sublist
                ]

                logger.info(f"  Deduplicated to {len(state.combined_results)} unique results")

            # Step 5: Deep Reasoning - analyze results and decide whether to continue
            logger.info(f"  Analyzing search results and determining next steps...")
            try:
                state = await async_deep_reasoning_agent(state, max_iterations)
                logger.info(f"  Search complete: {state.search_complete}")
                if not state.search_complete:
                    logger.info(f"  Knowledge gaps identified: {len(state.knowledge_gaps)}")
                    logger.info(f"  New queries generated: {len(state.generated_queries)}")
            except Exception as e:
                logger.error(f"  Error in deep reasoning: {str(e)}", exc_info=True)
                # If reasoning fails, stop the search to avoid infinite loops
                state.search_complete = True
                state.final_answer = "I'm sorry, but I couldn't properly analyze the search results due to a technical issue. Please try again with a different query."
                state.confidence_score = 0.1

        # Prepare the response
        if not state.search_complete:
            # If we exited the loop due to max iterations, generate the final answer
            logger.info("Maximum iterations reached, generating final answer...")
            try:
                from deepsearch.agents.deep_reasoning import generate_final_answer
                state = await sync2async(generate_final_answer)(state)
            except Exception as e:
                logger.error(f"Error generating final answer: {str(e)}", exc_info=True)
                state.final_answer = "I reached the maximum number of search iterations but couldn't generate a comprehensive answer. Here's what I found: " + "\n".join(
                    [f"- {point}" for point in state.key_points]
                )
                state.confidence_score = 0.5

        # Return the results
        sources = []
        if state.combined_results:
            for res in state.combined_results[:5]:  # Limit to 5 sources
                sources.append({
                    "title": res.title,
                    "url": res.url
                })

        # Extract components from the final answer if available
        answer = state.final_answer
        key_points = state.key_points if state.key_points else []
        detailed_notes = state.detailed_notes if state.detailed_notes else None

        return {
            "original_query": state.original_query,
            "generated_queries": state.generated_queries,
            "iterations": state.current_iteration,
            "answer": answer,
            "key_points": key_points,
            "detailed_notes": detailed_notes,
            "confidence": state.confidence_score,
            "sources": sources,
            "has_error": False,
            "combined_results": [
                e.model_dump()
                for e in state.combined_results
            ]
        }
    except Exception as e:
        # Handle any unexpected errors
        logger.error(f"Unexpected error in deep search pipeline: {str(e)}", exc_info=True)
        return {
            "original_query": query,
            "answer": "An unexpected error occurred while processing your query. Please try again later.",
            "confidence": 0.0,
            "sources": [],
            "has_error": True,
            "combined_results": state.combined_results
        }

async def prompt(messages: list[dict[str, str]], **kwargs) -> str:
    assert len(messages) > 0, "received empty messages"
    query = messages[-1]['content']

    res: Dict = await run_deep_search_pipeline(query)
    
    sep = "-" * 30
    final_resp = res["answer"]

    # if len(res["sources"]) > 0:
    #     final_resp += "\n## References:\n"

    #     for item in res["sources"]:
    #         final_resp += "- [{title}]({url})\n".format(**item)

    if not res["has_error"]:
        if final_resp.strip()[-1] != '\n':
            final_resp += '\n'

        final_resp += "{sep}\nConfidence score: {confidence}".format(
            confidence=res["confidence"],
            sep=sep
        )

    return final_resp # , res
