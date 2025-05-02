
import os
os.environ['OPENAI_BASE_URL'] = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL"))
os.environ['OPENAI_API_KEY'] = os.getenv("LLM_API_KEY", 'no-need')

from typing import Dict, Any, Callable, Generator, AsyncGenerator
from deepsearch.models import SearchState
from deepsearch.agents import (
    faiss_indexing_agent,
    bm25_search_agent,
    llama_reasoning_agent,
    deep_reasoning_agent,
    pubmed_search_agent,
    chess_xray_lesion_detector,
) 
from app.utils import (
    refine_assistant_message, 
    wrap_toolcall_request, 
    wrap_chunk, 
    to_chunk_data, 
    wrap_toolcall_response, 
    image_to_base64_uri, 
    wrap_thinking_chunk
)
from starlette.concurrency import run_in_threadpool
from functools import partial
import asyncio
import json
import uuid
from openai import AsyncClient
import openai


def sync2async(sync_func: Callable):
    async def async_func(*args, **kwargs):
        res = run_in_threadpool(partial(sync_func, *args, **kwargs))

        if isinstance(res, (Generator, AsyncGenerator)):
            return res

        return await res

    return async_func if not asyncio.iscoroutinefunction(sync_func) else sync_func

# Convert synchronous functions to asynchronous
async_faiss_indexing_agent = sync2async(faiss_indexing_agent)
async_bm25_search_agent = sync2async(bm25_search_agent)
async_llama_reasoning_agent = sync2async(llama_reasoning_agent)
async_deep_reasoning_agent = sync2async(deep_reasoning_agent)
async_pubmed_search_agent = sync2async(pubmed_search_agent)

async_chess_xray_lesion_detector = sync2async(chess_xray_lesion_detector.predict)
async_chess_xray_lesion_visualize = sync2async(chess_xray_lesion_detector.visualize)
async_chess_xray_lesion_quick_diagnose = sync2async(chess_xray_lesion_detector.quick_diagnose)

import logging
logger = logging.getLogger(__name__)

async def run_deep_search_pipeline(query: str, image_paths: list[str] = [], max_iterations: int = 3, response_uuid: str = str(uuid.uuid4())) -> AsyncGenerator[bytes, None]:
    """Run the multi-query, iterative deep search pipeline with reasoning agent."""
    try:
        # Initialize state
        state = SearchState(
            original_query=query, 
            image_paths=image_paths
        )

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

        yield await to_chunk_data(
            await wrap_thinking_chunk(
                response_uuid,
                f'Prepared {len(state.generated_queries)} search queries:\n'
            )
        )

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
                yield await to_chunk_data(
                    await wrap_thinking_chunk(
                        response_uuid,
                        f'Searching for {query}...\n'
                    )
                )

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
                    yield await to_chunk_data(
                        await wrap_thinking_chunk(
                            response_uuid,
                            f'Knowledge gaps identified:\n'
                        )
                    )
                    
                    for kg in state.knowledge_gaps:
                        yield await to_chunk_data(
                            await wrap_thinking_chunk(
                                response_uuid,
                                f'- {kg}\n'
                            )
                        )

                    yield await to_chunk_data(
                        await wrap_thinking_chunk(
                            response_uuid,
                            f'\nNew {len(state.generated_queries)} queries generated\n'
                        )
                    )

                    logger.info(f"  Knowledge gaps identified: {len(state.knowledge_gaps)}")
                    logger.info(f"  New queries generated: {len(state.generated_queries)}")

            except Exception as e:
                logger.error(f"  Error in deep reasoning: {str(e)}", exc_info=True)
                # If reasoning fails, stop the search to avoid infinite loops
                state.search_complete = True
                state.final_answer = "I'm sorry, but I couldn't properly analyze the search results due to a technical issue. Please try again with a different query."
                
                yield await to_chunk_data(
                    await wrap_thinking_chunk(
                        response_uuid,
                        f'Search complete: {state.search_complete}\n'
                    )
                )

                yield await to_chunk_data(
                    await wrap_chunk(
                        response_uuid,
                        state.final_answer
                    )
                )
                
                state.confidence_score = 0.1

        # Prepare the response
        if not state.search_complete:
            # If we exited the loop due to max iterations, generate the final answer
            logger.info("Maximum iterations reached, generating final answer...")
            try:
                from deepsearch.agents.deep_reasoning import generate_final_answer_stream

                for chunk in generate_final_answer_stream(state):
                    yield await to_chunk_data(
                        await wrap_chunk(
                            response_uuid,
                            chunk
                        )
                    )
                    
                    return

            except Exception as e:
                logger.error(f"Error generating final answer: {str(e)}", exc_info=True)
                state.final_answer = "I reached the maximum number of search iterations but couldn't generate a comprehensive answer. Here's what I found: " + "\n".join(
                    [f"- {point}" for point in state.key_points]
                )

                yield await to_chunk_data(
                    await wrap_chunk(
                        response_uuid,
                        state.final_answer
                    )
                )

                state.confidence_score = 0.5

    except Exception as e:
        yield await to_chunk_data(
            await wrap_thinking_chunk(
                response_uuid,
                f'Search complete: {state.search_complete}\n'
            )
        )

        yield await to_chunk_data(
            await wrap_chunk(
                response_uuid,
                "An unexpected error occurred while processing your query. Please try again later."
            )
        )


from .utils import get_attachments, preserve_upload_file, refine_chat_history
from typing import AsyncGenerator

TOOL_CALLS = [
    {
        "type": "function",
        "function": {
            "name": "research",
            "description": "Research on a scientific topic deeper and more comprehensive",
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
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Quick search for information on the internet to answer the question directly",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

async def quick_search(query: str) -> str:
    pass

async def execute_openai_compatible_toolcall(name: str, args: dict[str, str]) -> str:
    try:
        if name == 'search':
            return await quick_search(args['query'])
        
        if name == 'research':
            return await run_deep_search_pipeline(args['topic'])
        
        return f"Tool {name} not found"
    except Exception as e:
        logger.error(f"Error executing tool call {name} (args: {args}): {str(e)}", exc_info=True)
        return f"An error occurred while executing the tool call: {str(e)}"

async def prompt(messages: list[dict[str, str]], **kwargs) -> AsyncGenerator[bytes, None]:
    assert len(messages) > 0, "received empty messages"

    attachments = get_attachments(messages)
    attachment_paths = []

    if len(attachments) > 0:
        for attachment in attachments:
            path = await preserve_upload_file(attachment, attachment.split('/')[-1])
            attachment_paths.append(path)
            
            
    attachment_paths = [
        e
        for e in attachment_paths
        if os.path.splitext(e)[-1] in 
        [
            '.jpg', '.jpeg', '.png', 
            '.bmp', '.tiff', '.ico', 
            '.webp'
        ]
    ]
    
    system_prompt = ''

    if os.path.exists('system_prompt.txt'):
        with open('system_prompt.txt', 'r') as f:
            system_prompt = f.read()

    messages = refine_chat_history(messages, system_prompt=system_prompt)
    response_uuid = str(uuid.uuid4())

    if len(attachment_paths) > 0:
        calls = []

        for path in attachment_paths:
            calls.append(
                {
                    "id": 'call_' + str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "diagnose",
                        "arguments": json.dumps({
                            "image_path": path
                        })
                    }
                }
            )

        messages.append({
            "role": "assistant",
            "choices": [
                {
                    "tool_calls": calls
                }
            ]
        })
        
        for call, path in zip(calls, attachment_paths):
            file_basename = os.path.basename(path)

            yield await to_chunk_data(
                await wrap_thinking_chunk(
                    response_uuid,
                    f'Diagnosing: {file_basename}'
                )
            )

            vis, comment = await chess_xray_lesion_detector.xray_dianose_agent(path)

            if vis is not None:
                template = '''\
<p align="center">
    <img src="{uri}" width=480 alt><br>
    <em>{comment}</em>
</p>                

'''

                uri = image_to_base64_uri(vis)

                yield await to_chunk_data(
                    await wrap_chunk(
                        response_uuid,
                        template.format(uri=uri, comment=comment.replace('\n', '<br>'))
                    )
                )

            else:
                template = '''\
<blockquote cite="{file_basename}">
    <p>{comment}</p>
</blockquote>

'''
                yield await to_chunk_data(
                    await wrap_chunk(
                        response_uuid,
                        template.format(
                            file_basename=file_basename, 
                            comment=comment
                        )
                    )
                )

            messages.append({
                "role": "tool",
                "content": comment,
                "tool_call_id": call['id']
            })

    client = AsyncClient(
        base_url=os.getenv('LLM_BASE_URL'),
        api_key=os.getenv('LLM_API_KEY')
    )

    model_id = os.getenv('LLM_MODEL_ID')

    completion = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        tools=TOOL_CALLS,
        tool_choice="auto"
    )

    if completion.choices[0].message.content:
        yield await to_chunk_data(
            await wrap_chunk(
                response_uuid,
                completion.choices[0].message.content
            )
        )

    messages.append(
        await refine_assistant_message(completion.choices[0].message.model_dump())
    )

    loops = 0

    while completion.choices[0].message.tool_calls is not None and len(completion.choices[0].message.tool_calls) > 0:
        loops += 1

        for call in completion.choices[0].message.tool_calls:
            _id, _name = call.id, call.function.name
            _args = json.loads(call.function.arguments)
            
            if _name == 'research':
                yield await to_chunk_data(
                    await wrap_thinking_chunk(
                        response_uuid,
                        f'Researching on {_args["topic"]}'
                    )
                )

                async for chunk in await run_deep_search_pipeline(_args['topic']):
                    yield chunk

                return
            
            yield await to_chunk_data(
                await wrap_toolcall_request(
                    response_uuid,
                    _name,
                    _args
                )
            )

            result = await execute_openai_compatible_toolcall(_name, _args)
            
            yield await to_chunk_data(
                await wrap_toolcall_response(
                    response_uuid,
                    _name,
                    _args,
                    result
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _id,
                    "content": result
                }
            )

        completion = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=TOOL_CALLS if loops < 2 else (
                TOOL_CALLS[:1] if loops < 5 else openai._types.NOT_GIVEN
            ),
            tool_choice="auto"
        )

        if completion.choices[0].message.content:
            yield await to_chunk_data(
                await wrap_chunk(
                    response_uuid,
                    completion.choices[0].message.content
                )
            )

        messages.append(
            await refine_assistant_message(completion.choices[0].message.model_dump())
        )
