
import re
from typing import AsyncGenerator, Callable, Generator, Optional
import uuid

import regex
from langchain_core.messages import AIMessageChunk
from deepsearch.schemas.openai import ChatCompletionStreamResponse
import time

import json


def to_chunk_data(chunk: ChatCompletionStreamResponse) -> bytes:
    return ("data: " + json.dumps(chunk.model_dump()) + "\n\n").encode()


def wrap_thought(thought: str, thought_details: str = None, uuid_str: Optional[str] = None) -> ChatCompletionStreamResponse:
    if uuid_str is None:
        uuid_str = str(uuid.uuid4())

    template = f'''
<action>{thought}</action>

'''

    return ChatCompletionStreamResponse(
        id=uuid_str,
        object='chat.completion.chunk',
        created=int(time.time()),
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(
                    content=template,
                    role='tool'
                ),
            )
        ]
    )


def wrap_step_start(uuid_str: str, step: str) -> ChatCompletionStreamResponse:
    template = f'''<b>{step}</b>\n'''

    return ChatCompletionStreamResponse(
        id=uuid_str,
        object='chat.completion.chunk',
        created=int(time.time()),
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(
                    reasoning_content=template,
                    role='tool'
                ),
            )
        ]
    )


def wrap_step_finish(uuid_str: str, result_summary: str, result_details: str = None, is_error: bool = False) -> ChatCompletionStreamResponse:
    icon = "❌" if is_error else "✅"

    if result_details:
        template = f'''
<details>
<summary>
{icon} {result_summary}
</summary>
<p>
{result_details}
</p>
</details>
'''
    else:
        template = f'''\n{icon} {result_summary}\n'''

    return ChatCompletionStreamResponse(
        id=uuid_str,
        object='chat.completion.chunk',
        created=int(time.time()),
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(
                    reasoning_content=template,
                    role='tool'
                ),
            )
        ]
    )


def wrap_chunk(uuid: str, raw: str, role: str = 'assistant') -> ChatCompletionStreamResponse:
    return ChatCompletionStreamResponse(
        id=uuid,
        object='chat.completion.chunk',
        created=int(time.time()),
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(
                    content=raw,
                    role=role
                )
            )
        ]
    )


def handle_stream(stream: Generator[str, None, None], pattern: str, callback: Callable) -> Generator[str, None, None]:
    pattern_length = len(pattern)
    buffer = ''

    i, j = 0, 0 # i --> buffer, j --> pattern
    last_yield = 0
    for chunk in stream:
        buffer += chunk

        while i < len(buffer):
            if pattern[j] == '*' or pattern[j] == buffer[i]:
                if j == 0:
                    yield buffer[last_yield:i]
                    last_yield = i

                if pattern[j] != '*' or buffer[i] == pattern[j + 1]:
                    j += 1

                i += 1

            if j == pattern_length - 1:
                yield callback(buffer[last_yield:i])
                last_yield = i
                j = 0

            elif i < len(buffer) and pattern[j] != buffer[i] and pattern[j] != '*':
                if j != 0:
                    j = 0
                else:
                    i += 1

        if j == 0 and i > 0:
            yield buffer[last_yield:]
            last_yield = len(buffer)
            
    if last_yield < len(buffer):
        yield buffer[last_yield:]


def handle_stream_strip(stream: Generator[str, None, None], strip_pattern: regex.Pattern) -> Generator[str, None, None]:
    buffer = ''
    
    for chunk in stream:
        buffer += chunk

        partial_match = strip_pattern.search(buffer, partial=True)
        if not partial_match or (partial_match.span()[0] == partial_match.span()[1]):
            yield buffer
            buffer = ''
            continue
        
        if partial_match.partial:
            yield buffer[:partial_match.span()[0]]
            buffer = buffer[partial_match.span()[0]:]
            continue
        
        buffer = strip_pattern.sub('', buffer)
        yield buffer
        
        buffer = ''

    if buffer:
        yield buffer


def handle_llm_stream(stream: Generator[AIMessageChunk, None, None]) -> Generator[str, None, None]:
    for chunk in stream:
        yield chunk.content


def handle_stream_strip_begin_eoln(stream: Generator[str, None, None]) -> Generator[str, None, None]:
    pattern = regex.compile(r'^\n+', regex.DOTALL | regex.IGNORECASE)
    yield from handle_stream_strip(stream, pattern)


def handle_stream_strip_thinking(stream: Generator[str, None, None]) -> Generator[str, None, None]:
    pattern = regex.compile(r'<think>.*?</think>\n*', regex.DOTALL | regex.IGNORECASE)
    yield from handle_stream_strip_begin_eoln(handle_stream_strip(stream, pattern))


def handle_stream_strip_heading(stream: Generator[str, None, None]) -> Generator[str, None, None]:
    pattern = regex.compile(r'.*?##.*?\n', regex.DOTALL | regex.IGNORECASE)
    yield from handle_stream_strip(stream, pattern)
