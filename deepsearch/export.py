# import eai_http_middleware # do not remove this

from concurrent.futures import ThreadPoolExecutor, Future
from deepsearch.schemas.agents import SearchState
from deepsearch.agents import (
    tavily_search_agent,
    faiss_indexing_agent,
    bm25_search_agent,
    deep_reasoning_agent,
    brave_search_agent,
)
from deepsearch.agents.deep_reasoning import ReferenceBuilder, fast_generate_final_answer
import logging

logger = logging.getLogger(__name__)

import uuid
from enum import Enum

def random_uuid():
    return str(uuid.uuid4())

class Retriever(Enum):
    TAVILY = "tavily"
    BRAVE = "brave"
    EXA = "exa"
    TWITTER = "twitter"

def deepsearch(query: str, max_iterations: int = 3, retrievers: list[Retriever] = [Retriever.TAVILY, Retriever.BRAVE]):
    try:
        # Initialize state
        state = SearchState(original_query=query)
        logger.info("Step 1: Initial reasoning to analyze query and generate search queries...")

        try:
            state = deep_reasoning_agent(state, max_iterations)
            logger.info(f"  Generated {len(state.generated_queries)} initial search queries")
        except Exception as e:
            logger.error(f"  Error in initial reasoning: {str(e)}", exc_info=True)
            state.generated_queries = [state.original_query]
            state.current_iteration = 1

        # Iterative search loop
        while not state.search_complete and state.current_iteration < max_iterations:
            iteration = state.current_iteration
            logger.info(f"Beginning search iteration {iteration}...")

            previous_results = state.combined_results.copy() if state.combined_results else []

            state.tavily_results = []
            state.brave_results = []
            state.search_results = []
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

                # Step 1: Run Tavily and Brave searches in parallel
                logger.info(f"    Performing web searches in parallel...")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {}
                    
                    # Submit Tavily search if enabled
                    if Retriever.TAVILY in retrievers:
                        tavily_future = executor.submit(tavily_search_agent, temp_state)
                        futures['tavily'] = tavily_future

                    # time.sleep(1)
                    
                    # Submit Brave search if enabled
                    if Retriever.BRAVE in retrievers:
                        brave_future = executor.submit(brave_search_agent, temp_state, 10, True)  # use_ai_snippets=True
                        futures['brave'] = brave_future
                    
                    # Collect results as they complete
                    for search_type, future in futures.items():
                        try:
                            if search_type == 'tavily':
                                tavily_temp_state = future.result()
                                for result in tavily_temp_state.tavily_results:
                                    result.query = query
                                temp_state.tavily_results = tavily_temp_state.tavily_results
                                logger.info(f"    Found {len(temp_state.tavily_results)} Tavily results")
                            elif search_type == 'brave':
                                brave_temp_state = future.result()
                                for result in brave_temp_state.brave_results:
                                    result.query = query
                                temp_state.brave_results = brave_temp_state.brave_results
                                logger.info(f"    Found {len(temp_state.brave_results)} Brave results")
                        except Exception as e:
                            logger.error(f"    Error in {search_type} search: {str(e)}", exc_info=True)

                # Step 3.5: Twitter search for this query
                twitter_results = []
                # if Retriever.TWITTER in retrievers:
                #     logger.info("Performing Twitter search...")
                #     try:
                #         twitter_results = twitter_search(query)
                #         logger.info(f"    Found {len(twitter_results)} Twitter results")
                #     except Exception as e:
                #         logger.error(f"    Error in Twitter search: {str(e)}", exc_info=True)

                # Combine search results
                temp_state.search_results = (
                    temp_state.tavily_results
                    + temp_state.brave_results
                    + temp_state.exa_results
                    + twitter_results
                )
                logger.info(f"Combined {len(temp_state.search_results)} total search results")

                # Step 2: Run FAISS and BM25 searches in parallel
                logger.info(f"    Performing semantic and keyword searches in parallel...")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {}
                    
                    # Submit FAISS search
                    faiss_future = executor.submit(faiss_indexing_agent, temp_state, 10)
                    futures['faiss'] = faiss_future
                    
                    # Submit BM25 search
                    bm25_future = executor.submit(bm25_search_agent, temp_state, 10)
                    futures['bm25'] = bm25_future
                    
                    # Collect results as they complete
                    for search_type, future in futures.items():
                        try:
                            if search_type == 'faiss':
                                faiss_temp_state = future.result()
                                for result in faiss_temp_state.faiss_results:
                                    result.query = query
                                temp_state.faiss_results = faiss_temp_state.faiss_results
                                logger.info(f"    Found {len(temp_state.faiss_results)} semantic results")
                            elif search_type == 'bm25':
                                bm25_temp_state = future.result()
                                for result in bm25_temp_state.bm25_results:
                                    result.query = query
                                temp_state.bm25_results = bm25_temp_state.bm25_results
                                logger.info(f"    Found {len(temp_state.bm25_results)} keyword results")
                        except Exception as e:
                            logger.error(f"    Error in {search_type} search: {str(e)}", exc_info=True)
                            if search_type == 'faiss':
                                temp_state.faiss_results = []
                            elif search_type == 'bm25':
                                temp_state.bm25_results = []

                if len(temp_state.faiss_results) + len(temp_state.bm25_results) > 10:
                    temp_state.faiss_results = temp_state.faiss_results[:5]
                    temp_state.bm25_results = temp_state.bm25_results[:5]

                # Combine all results (will be used by the reasoning agent)
                all_results = []

                # Add FAISS results with higher priority
                if temp_state.faiss_results:
                    for result in temp_state.faiss_results:
                        if result not in all_results:
                            all_results.append(result)

                # Add BM25 results
                if temp_state.bm25_results:
                    for result in temp_state.bm25_results:
                        if result not in all_results:
                            all_results.append(result)

                # Add any remaining search results
                if temp_state.search_results:
                    for result in temp_state.search_results:
                        if result not in all_results:
                            all_results.append(result)

                # Sort by score if available
                all_results.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)

                # Limit to top 10 combined results to avoid overwhelming the reasoning agent
                temp_state.combined_results = all_results[:10]

                logger.info(f"    Found {len(temp_state.combined_results)} combined results")

                # Collect results from this query
                state.tavily_results.extend(temp_state.tavily_results)
                state.brave_results.extend(temp_state.brave_results)
                state.search_results.extend(temp_state.search_results)
                state.faiss_results.extend(temp_state.faiss_results)
                state.bm25_results.extend(temp_state.bm25_results)

                if temp_state.combined_results:
                    state.combined_results.extend(temp_state.combined_results)                

            # Add back previous results to ensure continuity
            if state.combined_results:
                # If we have combined results from this iteration, merge with previous
                seen_urls = {
                    result.url
                    for result in state.combined_results
                }

                for result in previous_results:
                    if result.url not in seen_urls:
                        state.combined_results.append(result)
            else:
                # No new combined results, use previous ones
                state.combined_results = previous_results

            # Make sure combined_results is populated even if BM25 didn't run
            if not state.combined_results:
                # Combine FAISS and search results
                state.combined_results = state.faiss_results + state.search_results

            # Deduplicate combined results by URL
            if state.combined_results:
                unique_results = {}

                for result in state.combined_results:
                    # if isinstance(result.score, float) and result.score < 0.3:
                    #     continue

                    # Keep the highest scoring result for each URL
                    key = result.url + "\n" + result.content

                    if key not in unique_results or (result.score is not None and
                        (unique_results[key].score is None or result.score > unique_results[key].score)):
                        unique_results[key] = result

                if len(unique_results) < 5:
                    need = 5 - len(unique_results)

                    for result in state.combined_results:
                        key = result.url + "\n" + result.content

                        if key not in unique_results:
                            unique_results[key] = result
                            need -= 1

                        if not need:
                            break

                state.combined_results = list(unique_results.values())
                logger.info(f"  Deduplicated to {len(state.combined_results)} unique results")

            # Step 8: Deep Reasoning - analyze results and decide whether to continue
            logger.info(f"  Analyzing search results and determining next steps...")
            try:
                state = deep_reasoning_agent(state, max_iterations)
                logger.info(f"  Search complete: {state.search_complete}")

                if not state.search_complete:
                    joined_gaps = '\n'.join([f'- {gap}' for gap in state.knowledge_gaps])
                    logger.info(f"  Knowledge gaps identified:\n{joined_gaps}")
                    logger.info(f"  New queries generated: {len(state.generated_queries)}")
            except Exception as e:
                logger.error(f"  Error in deep reasoning: {str(e)}", exc_info=True)

        logger.info("Generating final answer...")
        try:
            if state.final_answer:
                return None

            output = fast_generate_final_answer(state)
            
            return output

        except Exception as e:
            logger.error(f"  Error in final answer generation: {str(e)}", exc_info=True)

            ref_builder = ReferenceBuilder(state)

            points = []
            for kp in state.key_points:
                kp = kp.strip()
                points.append(ref_builder.embed_references(kp))

            logger.info("I couldn't generate a comprehensive answer due to technical issues. Here's what I found:\n" + "\n".join(f"- {point}" for point in points))
            return None

    except Exception as e:
        logger.info("An unexpected error occurred while processing your query. Please try again later.")
        return None
