import time
import json
import uuid
from .models_openai import ChatCompletionStreamResponse
from pydantic import BaseModel
from typing import Any

def to_chunk_data(chunk: ChatCompletionStreamResponse) -> bytes:
    return ("data: " + json.dumps(chunk.model_dump()) + "\n\n").encode()
    

def wrap_thought(thought: str, thought_details: str = None, uuid_str: str = None) -> ChatCompletionStreamResponse:
    if uuid_str is None:
        uuid_str = str(uuid.uuid4())
        
    template = f'''
Thought: <b>{thought}</b>
'''
    
    if thought_details:
        template += f'''
<details>
    <summary>
    Details:
    </summary>
    <p>
    {thought_details}
    </p>
</details>
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

class Generator:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen
        return self.value