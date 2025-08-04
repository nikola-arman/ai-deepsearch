from typing import AsyncGenerator, Callable, Optional, Any
from app.oai_models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionStreamResponse, 
    LabRequest,
    ChatCompletionAdditionalParameters
)
import asyncio
import logging
from app import prompt
from functools import partial
from starlette.concurrency import run_in_threadpool
import inspect

logger = logging.getLogger(__name__)


def _safe_next(gen):
    """Safely gets the next item from a generator, returning None when exhausted."""
    try:
        return next(gen), False
    except StopIteration:
        return None, True


def sync_generator_to_async(sync_gen_func: Callable) -> Callable:
    """Convert a sync generator function to an async generator function."""
    async def async_generator(*args, **kwargs):
        # Run the sync generator function in a thread pool to get the generator
        sync_generator = await run_in_threadpool(partial(sync_gen_func, *args, **kwargs))
        
        # If it's not a generator, just return the result
        if not inspect.isgenerator(sync_generator):
            yield sync_generator
            return
            
        # Yield values from the sync generator without blocking
        while True:
            # Get the next value from the sync generator in a thread pool
            value, is_done = await run_in_threadpool(_safe_next, sync_generator)
            if is_done:
                break
            yield value
    
    return async_generator


def sync2async(sync_func: Callable):
    async def async_func(*args, **kwargs):
        return await run_in_threadpool(partial(sync_func, *args, **kwargs))
    return async_func if not asyncio.iscoroutinefunction(sync_func) else sync_func


async def handle_request(
    request: ChatCompletionRequest, 
    event: asyncio.Event,
    lab: Optional[LabRequest] = None, 
    additional_parameters: Optional[ChatCompletionAdditionalParameters] = None,
) -> AsyncGenerator[ChatCompletionStreamResponse | ChatCompletionResponse, None]:
    messages = request.messages
    assert len(messages) > 0, "No messages in the request"

    if not asyncio.iscoroutinefunction(prompt):
        # Convert sync generator function to async generator
        async_prompt_gen = sync_generator_to_async(prompt)
    else:
        async_prompt_gen = prompt
        
    # Call the async generator function
    async for chunk in async_prompt_gen(messages):
        if event.is_set():
            logger.info(f"[main] Event signal received, stopping the request")
            break
        yield chunk
