import asyncio
from enum import Enum
import uuid
import eai_http_middleware # do not remove this

import os
import time

from app.oai_models import ChatCompletionStreamResponse, random_uuid
from deepsearch.magic import retry
from deepsearch.schemas import commons, twitter
from deepsearch.schemas.openai import ErrorResponse
from deepsearch.utils.oai_streaming import ChatCompletionResponseBuilder, create_streaming_response
from deepsearch.utils.streaming import wrap_thought, to_chunk_data

os.environ['TAVILY_API_KEY'] = 'no-need'
os.environ['OPENAI_BASE_URL'] = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
os.environ['OPENAI_API_KEY'] = os.getenv("LLM_API_KEY", 'no-need')
os.environ["EXA_API_KEY"] = "no-need"

from typing import Annotated, Dict, Any, Generator, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from deepsearch.schemas.agents import SearchState
from deepsearch.agents import (
    tavily_search_agent,
    faiss_indexing_agent,
    bm25_search_agent,
    deep_reasoning_agent,
    brave_search_agent,
    search_tavily,
    get_twitter_data_by_username,
    twitter_context_to_search_result,
    twitter_search
)
from deepsearch.agents.deep_reasoning import ReferenceBuilder, generate_final_answer

from json_repair import repair_json
import json
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

from langchain.prompts import PromptTemplate
from deepsearch.utils.streaming import wrap_chunk
import openai


class Retriever(Enum):
    TAVILY = "tavily"
    BRAVE = "brave"
    EXA = "exa"
    TWITTER = "twitter"

def write_state_to_file(state: SearchState):
    os.makedirs("output", exist_ok=True)
    with open("output/final_state.json", "w") as f:
        f.write(state.model_dump_json(indent=2))


def get_retriever_from_env() -> List[Retriever]:
    retriever_str = os.getenv("RETRIEVER", "tavily")
    retriever_list = retriever_str.split(",")
    retrievers = []
    for retriever in retriever_list:
        if retriever == Retriever.TAVILY.value:
            retrievers.append(Retriever.TAVILY)
        elif retriever == Retriever.BRAVE.value:
            retrievers.append(Retriever.BRAVE)
        elif retriever == Retriever.EXA.value:
            retrievers.append(Retriever.EXA)
        elif retriever == Retriever.TWITTER.value:
            retrievers.append(Retriever.TWITTER)
        else:
            raise ValueError(f"Invalid retriever: {retriever}")
    return retrievers


def extract_urls_from_report(report: str) -> set[str]:
    """
    Extract all URLs from a markdown report that appear in markdown links.

    Args:
        report: The markdown report text

    Returns:
        Set of URLs found in the report
    """
    import re
    # Match markdown links with nested square brackets: [text with [nested] brackets](url)
    url_pattern = r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]\((https?://[^)]+)\)'
    urls = set()

    for match in re.finditer(url_pattern, report):
        url = match.group(1)
        urls.add(url)

    return urls

def run_deep_search_pipeline(
    query: str,
    max_iterations: int = 3,
) -> Generator[bytes, None, dict[str, Any]]:
    """Run the multi-query, iterative deep search pipeline with reasoning agent."""
    try:
        # Initialize state
        state = SearchState(original_query=query)

        # usernames_obj = detect_twitter_usernames(query=query)
        # # Limit to at most 1 usernames to save API quota
        # twitter_usernames = usernames_obj.twitter_usernames[:1]

        # twitter_context = {}
        # for username in twitter_usernames:
        #     try:
        #         twitter_data = get_twitter_data_by_username(username)
        #         twitter_context[username] = twitter_data
        #     except Exception as e:
        #         logger.error(f"  Error in getting twitter data for {username}: {str(e)}", exc_info=True)
        #         continue

        # state.combined_results = twitter_context_to_search_result(twitter_context)

        # Instead of using query_expansion_agent, let the deep_reasoning_agent handle initial query generation
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
                yield wrap_thought(f"Searching for: {query}")

                # Create a temporary state for this query
                temp_state = SearchState(
                    original_query=query  # Use the current query as the original query for this temp state
                )

                retrievers = get_retriever_from_env()
                
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
                    faiss_future = executor.submit(faiss_indexing_agent, temp_state)
                    futures['faiss'] = faiss_future
                    
                    # Submit BM25 search
                    bm25_future = executor.submit(bm25_search_agent, temp_state)
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
                yield wrap_chunk(random_uuid(), "I'm sorry, but I couldn't properly analyze the search results due to a technical issue. Please try again with a different query.")
                return

        logger.info("Generating final answer...")
        try:
            if state.final_answer:
                yield wrap_chunk(random_uuid(), state.final_answer)
                return

            yield from generate_final_answer(state)

        except Exception as e:
            logger.error(f"  Error in final answer generation: {str(e)}", exc_info=True)

            ref_builder = ReferenceBuilder(state)

            points = []
            for kp in state.key_points:
                kp = kp.strip()
                points.append(ref_builder.embed_references(kp))

            yield wrap_chunk(random_uuid(), "I couldn't generate a comprehensive answer due to technical issues. Here's what I found:\n" + "\n".join(f"- {point}" for point in points))

    except Exception as e:
        logger.error(f"  Error in final answer generation: {str(e)}", exc_info=True)
        yield wrap_chunk(random_uuid(), "An unexpected error occurred while processing your query. Please try again later.")

class GeneratorValue:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen
        return self.value


TOOL_CALLS = [
    {
        "type": "function",
        "function": {
            "name": "research",
            # "description": "Research on a topic deeper and more comprehensive. Only use this tool when the asked question requires real time knowledge to answer, or when the user asks you to deep dive into a topic, or when you have already confirmed with the user.",
            "description": "Research on a topic deeper and more comprehensively. Unless the user prompt is a greeting, goodbye, or a question about yourself (like who are you, what can you do, etc), always use this tool to research on topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to research on"
                    }
                },
                "required": ["topic"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "search_internet",
    #         "description": "Search the internet for information on a topic. Only use this tool when the asked question can be quickly answered by a single search query, without the need to deep dive.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "query": {
    #                     "type": "string",
    #                     "description": "The query to search the internet for"
    #                 }
    #             },
    #             "required": ["query"],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # }
]

from .utils import refine_chat_history, refine_assistant_message

def prompt(messages: list[dict[str, str]], **kwargs) -> Generator[bytes, None, None]:
    assert len(messages) > 0, "received empty messages"

    system_prompt = ''

    if os.path.exists('system_prompt.txt'):
        with open('system_prompt.txt', 'r') as f:
            system_prompt = f.read()

    messages = refine_chat_history(messages, system_prompt=system_prompt)
    response_uuid = str(uuid.uuid4())

    base_url = os.getenv('LLM_BASE_URL')
    api_key = os.getenv('LLM_API_KEY')

    client = openai.Client(
        base_url=base_url,
        api_key=api_key
    )

    model_id = os.getenv('LLM_MODEL_ID', 'local-model')

    NO_STREAMING = False

    if NO_STREAMING:
        completion = retry(client.chat.completions.create, max_retry=3, first_interval=2, interval_multiply=2)(
            model=model_id,
            messages=messages,
            tools=TOOL_CALLS,
            tool_choice="auto",
            stream=False,
        )
    else:
        builder = ChatCompletionResponseBuilder()

        completion_it = create_streaming_response(
            base_url=base_url,
            api_key=api_key,
            messages=messages,
            model=model_id,
            tools=TOOL_CALLS,
        )
        
        for chunk in completion_it:
            if isinstance(chunk, ErrorResponse):
                raise openai.BadRequestError(chunk.message, response=None, body=None)

            builder.add_chunk(chunk)

            if chunk.choices[0].delta.content:
                chunk.id = response_uuid
                yield chunk

        completion = builder.build()

    messages.append(
        refine_assistant_message(completion.choices[0].message.model_dump())
    )

    loops = 0
    report = ''

    while completion.choices[0].message.tool_calls is not None and len(completion.choices[0].message.tool_calls) > 0:
        loops += len(completion.choices[0].message.tool_calls)

        for call in completion.choices[0].message.tool_calls:
            _id, _name = call.id, call.function.name
            _args = json.loads(call.function.arguments)

            if _name == 'research':
                yield wrap_thought(f'Start researching on {_args["topic"]}')

                for chunk in run_deep_search_pipeline(
                    _args['topic'],
                    max_iterations=3,
                ):
                    if isinstance(chunk, bytes):
                        chunk_str = chunk.decode('utf-8')
                    elif isinstance(chunk, ChatCompletionStreamResponse):
                        chunk_str = chunk.choices[0].delta.content
                    else:
                        chunk_str = str(chunk)

                    if "<action>" not in chunk_str:
                        report += chunk_str
                    yield chunk

                logger.info(f"report: {report}")

                with open('report.txt', 'w') as f:
                    f.write(report)

                return

            elif _name == 'search_internet':
                yield wrap_thought(f"Searching for {_args['query']}")

                try:
                    res = search_tavily(_args['query'])
                except Exception as e:
                    res = f"Error in search_internet: {str(e)}"

                logger.info(f"tavily search result: {res}")

                messages.append(
                    {
                        "role": "tool",
                        "content": str(res),
                        "tool_call_id": _id
                    }
                )

        if NO_STREAMING:
            completion = retry(client.chat.completions.create, max_retry=3, first_interval=2, interval_multiply=2)(
                model=model_id,
                messages=messages,
                tools=TOOL_CALLS,
                tool_choice="auto",
                stream=False,
            )
        else:
            builder = ChatCompletionResponseBuilder()

            completion_it = create_streaming_response(
                base_url=base_url,
                api_key=api_key,
                messages=messages,
                model=model_id,
                tools=TOOL_CALLS,
            )
            
            for chunk in completion_it:
                if isinstance(chunk, ErrorResponse):
                    raise openai.BadRequestError(chunk.message, response=None, body=None)

                builder.add_chunk(chunk)

                if chunk.choices[0].delta.content:
                    chunk.id = response_uuid
                    yield chunk

            completion = builder.build()
        
        messages.append(
            refine_assistant_message(completion.choices[0].message.model_dump())
        )