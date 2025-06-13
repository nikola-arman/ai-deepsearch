from enum import Enum
import uuid
import eai_http_middleware # do not remove this

import os

from deepsearch.magic import retry

os.environ['TAVILY_API_KEY'] = 'no-need'
os.environ['OPENAI_BASE_URL'] = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
os.environ['OPENAI_API_KEY'] = os.getenv("LLM_API_KEY", 'no-need')
os.environ["EXA_API_KEY"] = "no-need"

from typing import Annotated, Dict, Any, Generator, List
from deepsearch.models import SearchState
from deepsearch.agents import (
    tavily_search_agent,
    faiss_indexing_agent,
    bm25_search_agent,
    llama_reasoning_agent,
    deep_reasoning_agent,
    brave_search_agent,
    information_extraction_agent,
    fact_checking_agent,
    search_tavily
)
from deepsearch.agents.deep_reasoning import ReferenceBuilder, init_reasoning_llm
from app.utils import detect_research_intent, get_conversation_summary, reply_conversation

from json_repair import repair_json
import json
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

from langchain.prompts import PromptTemplate
from deepsearch.utils import to_chunk_data, wrap_step_start, wrap_step_finish, wrap_thought, wrap_chunk
import openai


class Retriever(Enum):
    TAVILY = "tavily"
    BRAVE = "brave"
    EXA = "exa"


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
                yield to_chunk_data(wrap_thought(f"Searching for: {query}"))

                # Create a temporary state for this query
                temp_state = SearchState(
                    original_query=query  # Use the current query as the original query for this temp state
                )

                retrievers = get_retriever_from_env()
                if Retriever.TAVILY in retrievers:
                    logger.info(f"    Performing Tavily web search...")

                    try:
                        temp_state = tavily_search_agent(temp_state)

                        for result in temp_state.tavily_results:
                            result.query = query

                        logger.info(f"    Found {len(temp_state.tavily_results)} web results")
                    except Exception as e:
                        logger.error(f"    Error in Tavily search: {str(e)}", exc_info=True)

                # Step 3: Brave Search for this query
                if Retriever.BRAVE in retrievers:
                    logger.info("Performing Brave web search...")
                    try:
                        temp_state = brave_search_agent(temp_state, use_ai_snippets=True)
                        # Tag results with the query that produced them
                        for result in temp_state.brave_results:
                            result.query = query
                        logger.info(f"Found {len(temp_state.brave_results)} web results")
                    except Exception as e:
                        logger.error(f"    Error in Brave search: {str(e)}", exc_info=True)

                # Combine search results
                temp_state.search_results = (
                    temp_state.tavily_results
                    + temp_state.brave_results
                    + temp_state.exa_results
                )
                logger.info(f"Combined {len(temp_state.search_results)} total search results")

                # Step 5: FAISS Indexing (semantic search) for this query
                logger.info(f"    Performing semantic search...")
                try:
                    temp_state = faiss_indexing_agent(temp_state)

                    for result in temp_state.faiss_results:
                        result.query = query

                    logger.info(f"    Found {len(temp_state.faiss_results)} semantic results")
                except Exception as e:
                    logger.error(f"    Error in semantic search: {str(e)}", exc_info=True)

                # Step 6: BM25 Search (keyword search) for this query
                logger.info(f"    Performing keyword search...")
                try:
                    temp_state = bm25_search_agent(temp_state)

                    for result in temp_state.bm25_results:
                        result.query = query

                    logger.info(f"    Found {len(temp_state.bm25_results)} keyword results")
                except Exception as e:
                    logger.error(f"    Error in keyword search: {str(e)}", exc_info=True)

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

                    if isinstance(result.score, float) and result.score < 0.3:
                        continue

                    # Keep the highest scoring result for each URL
                    key = result.url + "\n" + result.content

                    if key not in unique_results or (result.score is not None and
                        (unique_results[key].score is None or result.score > unique_results[key].score)):
                        unique_results[key] = result

                if len(unique_results) < 5:
                    need = 5 - len(unique_results)
                    
                    for result in state.combined_results:
                        if result.score is None:
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
                yield "I'm sorry, but I couldn't properly analyze the search results due to a technical issue. Please try again with a different query."
                return

        logger.info("Generating final answer...")
        try:
            if state.final_answer:
                yield state.final_answer
                return

            from deepsearch.agents.deep_reasoning import generate_final_answer
            from .utils import strip_thinking_content

            for msg in generate_final_answer(state):
                yield strip_thinking_content(msg)

        except Exception as e:
            logger.error(f"  Error in final answer generation: {str(e)}", exc_info=True)

            ref_builder = ReferenceBuilder(state)
            
            points = []
            for kp in state.key_points:
                kp = kp.strip()
                points.append(ref_builder.embed_references(kp))

            yield "I couldn't generate a comprehensive answer due to technical issues. Here's what I found:\n" + "\n".join(f"- {point}" for point in points)

    except Exception as e:
        logger.error(f"  Error in final answer generation: {str(e)}", exc_info=True)
        yield "An unexpected error occurred while processing your query. Please try again later."

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
            "description": "Research on a topic deeper and more comprehensive. Only use this tool when the asked question requires real time knowledge to answer, or when the user asks you to deep dive into a topic, or when you have already confirmed with the user.",
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

    client = openai.Client(
        base_url=os.getenv('LLM_BASE_URL'),
        api_key=os.getenv('LLM_API_KEY')
    )

    model_id = os.getenv('LLM_MODEL_ID', 'local-model')
    
    completion = retry(client.chat.completions.create, max_retry=3, first_interval=2, interval_multiply=2)(
        model=model_id,
        messages=messages,
        tools=TOOL_CALLS,
        tool_choice="auto",
    )

    if completion.choices[0].message.content:
        yield to_chunk_data(
            wrap_chunk(
                response_uuid,
                completion.choices[0].message.content
            )
        )

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
                yield to_chunk_data(wrap_thought(f'Start researching on {_args["topic"]}'))

                for chunk in run_deep_search_pipeline(
                    _args['topic'],
                    max_iterations=3,
                ):
                    if isinstance(chunk, bytes):
                        chunk_str = chunk.decode('utf-8')
                    else:
                        chunk_str = str(chunk)
                        
                    if "<action>" not in chunk_str:
                        report += chunk_str
                    yield chunk

                with open('report.txt', 'w') as f:
                    f.write(report)

                return
            
            elif _name == 'search_internet':
                yield to_chunk_data(wrap_thought(f"Searching for {_args['query']}"))

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

        completion = retry(client.chat.completions.create, max_retry=3, first_interval=2, interval_multiply=2)(
            model=model_id,
            messages=messages,
            tools=TOOL_CALLS if loops < 5 else openai._types.NOT_GIVEN,
            tool_choice="auto" if loops < 5 else openai._types.NOT_GIVEN,
        )
        
        if completion.choices[0].message.content:
            yield to_chunk_data(
                wrap_chunk(
                    response_uuid,
                    completion.choices[0].message.content
                )
            )
   
        messages.append(refine_assistant_message(completion.choices[0].message.model_dump()))
        