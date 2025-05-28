import os
import datetime
import logging
import base64
from .models import ChatCompletionStreamResponse
import time
import json
from typing import Any
from PIL import Image
from io import BytesIO
import numpy as np


logger = logging.getLogger(__name__)


def heif_to_jpeg(file_data_uri: str) -> str:
    file_data_base64 = file_data_uri.split(',')[-1]
    file_data = base64.b64decode(file_data_base64)
    img = Image.open(BytesIO(file_data))
    img = img.convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


async def preserve_upload_file(file_data_uri: str, file_name: str, preserve_attachments: bool = False) -> str:
    os.makedirs(os.path.join(os.getcwd(), 'uploads'), exist_ok=True)

    file_data_base64 = file_data_uri.split(',')[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    try:
        file_path = os.path.join(os.getcwd(), 'uploads', f"{timestamp}_{file_name}")

        if not preserve_attachments:
            return file_path

        file_data = base64.b64decode(file_data_base64)

        with open(file_path, 'wb') as f:
            f.write(file_data)

        return file_path
    except Exception as e:
        logger.error(f"Failed to preserve upload file: {e}")
        return None


async def get_attachments(content: list[dict[str, str]]) -> list[str]:
    attachments = []

    if isinstance(content, str):
        return []

    for item in content:

        if item.get('type', 'undefined') == 'file':
            file = item.get('file')
            data = file.get('file_data')
            filename = file.get('filename')

            if data and filename:
                attachments.append((data, filename))

        elif item.get('type', 'undefined') == 'image_url':
            image_url = item.get('image_url')
            name = image_url.get('name')
            url = image_url.get('url')

            if url and name:
                attachments.append((url, name))

    return attachments


async def refine_chat_history(messages: list[dict[str, str]], system_prompt: str, preserve_attachments: bool = False) -> list[dict[str, str]]:
    refined_messages = []

    has_system_prompt = False

    for message in messages:
        message: dict[str, str]

        if isinstance(message, dict) and message.get('role', 'undefined') == 'system':
            message['content'] += f'\n{system_prompt}'
            has_system_prompt = True
            refined_messages.append(message)
            continue

        if isinstance(message, dict) \
            and message.get('role', 'undefined') == 'user' \
            and isinstance(message.get('content'), list):

            content = message['content']
            text_input = ''
            attachments = []

            for item in content:
                if item.get('type', 'undefined') == 'text':
                    text_input += item.get('text') or ''

                elif item.get('type', 'undefined') == 'file':
                    file_item = item.get('file', {})
                    if 'file_data' in file_item and 'filename' in file_item:
                        file_path = await preserve_upload_file(
                            file_item.get('file_data', ''),
                            file_item.get('filename', ''),
                            preserve_attachments
                        )

                        if file_path:
                            attachments.append(file_path)

                elif item.get('type', 'undefined') == 'image_url':
                    file_item = item.get('image_url', {})

                    if 'url' in file_item:
                        file_path = await preserve_upload_file(
                            file_item.get('url', ''),
                            file_item.get('name', f'image_{len(attachments)}.jpg'),
                            preserve_attachments
                        )

                        if file_path:
                            attachments.append(file_path)

            if len(attachments) > 0:
                text_input += f'\nAttachments:\n'
                for attachment in attachments:
                    text_input += f'- {attachment}\n'

            refined_messages.append({
                "role": "user",
                "content": text_input
            })

        else:
            refined_messages.append(message)

    if not has_system_prompt and system_prompt != "":
        refined_messages.insert(0, {
            "role": "system",
            "content": system_prompt
        })

    if isinstance(refined_messages[-1], str):
        refined_messages[-1] = {
            "role": "user",
            "content": refined_messages[-1] + '\n/no_think'
        }

    return refined_messages


async def refine_assistant_message(
    assistant_message: dict[str, str]
) -> dict[str, str]:

    if 'content' in assistant_message:
        assistant_message['content'] = assistant_message['content'] or ""

    return assistant_message


async def wrap_chunk(uuid: str, raw: str, role: str = 'assistant') -> ChatCompletionStreamResponse:
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


async def wrap_thinking_chunk(uuid: str, raw: str) -> ChatCompletionStreamResponse:
    content = f"<think>{raw}</think>"
    return ChatCompletionStreamResponse(
        id=uuid,
        object='chat.completion.chunk',
        created=int(time.time()),
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(content=content)
            )
        ]
    )



async def wrap_toolcall_request(uuid: str, fn_name: str, args: dict[str, Any]) -> ChatCompletionStreamResponse:
    args_str = json.dumps(args, indent=2)

    template = f'''
Executing <b>{fn_name}</b>

<details>
<summary>
Arguments:
</summary>

```json
{args_str}
```

</details>
'''

    return ChatCompletionStreamResponse(
        id=uuid,
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


async def to_chunk_data(chunk: ChatCompletionStreamResponse) -> bytes:
    return ("data: " + json.dumps(chunk.model_dump()) + "\n\n").encode()


async def done_token() -> bytes:
    return "data: [DONE]\n\n".encode()


async def wrap_toolcall_response(
    uuid: str,
    fn_name: str,
    args: dict[str, Any],
    content: str
) -> ChatCompletionStreamResponse:

    result = f'''
<details>
<summary>
Response:
</summary>

{content}

</details>
<br>

'''

    return ChatCompletionStreamResponse(
        id=uuid,
        object='chat.completion.chunk',
        created=int(time.time()),
        model='unspecified',
        choices=[
            dict(
                index=0,
                delta=dict(
                    content=result,
                    role='tool'
                ),
            )
        ]
    )


def image_to_base64_uri(image: np.ndarray) -> str:
    buffer = BytesIO()
    image = Image.fromarray(image)
    image.save(buffer, format='JPEG')
    return f'data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode("utf-8")}'


def random_str(n: int) -> str:
    return os.urandom(n // 2).hex()
