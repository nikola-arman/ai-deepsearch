
from typing import Optional
import uuid
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