import asyncio
import datetime
import logging
import os
import time
import uuid
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, wraps
from typing import (
    AsyncGenerator,
    Callable,
    ClassVar,
    Generator,
    Literal,
    Optional,
    Union,
)

import fastapi
import openai
import uvicorn
from app import prompt
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if not load_dotenv():
    logger.warning("hehe, .env not found")


def random_uuid() -> str:
    """Generate a random UUID string."""
    return str(uuid.uuid4())


@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None


class OpenAIBaseModel(BaseModel):
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    # Cache class field names
    field_names: ClassVar[Optional[set[str]]] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            # Get all class field names and their potential aliases
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names

        # Compare against both field names and aliases
        if any(k not in field_names for k in data):
            logger.warning(
                "The following fields were present in the request "
                "but ignored: %s",
                data.keys() - field_names,
            )
        return result


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(OpenAIBaseModel):
    role: str
    reasoning_content: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = Field(default_factory=list)


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: Optional[list[int]] = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    # Workaround: redefine fields name cache so that it's not
    # shared with the super class.
    field_names: ClassVar[Optional[set[str]]] = None
    top_logprobs: list[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: Optional[list[ChatCompletionLogProbsContent]] = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    # per OpenAI spec this is the default
    finish_reason: Optional[str] = "stop"
    # not part of the OpenAI spec but included in vLLM for legacy reasons
    stop_reason: Optional[Union[int, str]] = None


class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: Optional[int] = None


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
    prompt_logprobs: Optional[list[Optional[dict[int, Logprob]]]] = None


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


# a tool call delta where everything is optional
class DeltaToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    index: int = Field(default_factory=lambda: -1)
    function: Optional[DeltaFunctionCall] = None


class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class PromptErrorResponse(OpenAIBaseModel):
    type: Literal["error"] = "error"
    message: str
    details: Optional[str] = None


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


def sync2async(sync_func: Callable):
    async def async_func(*args, **kwargs):
        res = run_in_threadpool(partial(sync_func, *args, **kwargs))

        if isinstance(res, (Generator, AsyncGenerator)):
            return res

        return await res

    return async_func if not asyncio.iscoroutinefunction(sync_func) else sync_func


def limit_asyncio_concurrency(num_of_concurrent_calls: int):
    semaphore = asyncio.Semaphore(num_of_concurrent_calls)

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                res = func(*args, **kwargs)

                if isinstance(res, (Generator, AsyncGenerator)):
                    return res

                return await res

        return wrapper
    return decorator


async def stream_reader(s: AsyncGenerator[Union[str, bytes], None]):
    error_message = None
    response_uuid = str(uuid.uuid4())

    try:
        async for chunk in s:
            if chunk is None:
                continue

            if isinstance(chunk, str):
                chunk_model = ChatCompletionStreamResponse(
                    id=response_uuid,
                    object='chat.completion.chunk',
                    created=int(time.time()),
                    model='unspecified',
                    choices=[
                        dict(
                            index=0,
                            delta=dict(
                                content=chunk,
                                role='assistant'
                            )
                        )
                    ]
                )

                yield (f'data: {chunk_model.model_dump_json()}\n\n').encode('utf-8')
            else:
                yield chunk

    except openai.APIConnectionError as e:
        error_message = f"Failed to connect to language model: {e}"

    except openai.RateLimitError as e:
        error_message = f"Rate limit error: {e}"

    except openai.APIError as e:
        error_message = f"Language model returned an API Error: {e}"

    except Exception as err:
        error_message = "Unhandled error: " + str(err)

        import traceback
        logger.error(traceback.format_exc())

    finally:
        if error_message:
            yield (f'data: {PromptErrorResponse(message=error_message).model_dump_json()}\n\n').encode('utf-8')

        yield b'data: [DONE]\n\n'


def parse_args():
    parser = ArgumentParser(
        add_help="Yo!"
    )

    parser.add_argument(
        "-host", "--host",
        type=str,
        help="host",
        default="0.0.0.0",
        required=False
    )

    parser.add_argument(
        "-port", "--port",
        type=int,
        help="port",
        default=4000,
        required=False
    )

    parser.add_argument(
        "-log", "--log-level",
        type=str,
        help="log level",
        default="info",
        required=False
    )

    parser.add_argument(
        "-tka", "--timeout-keep-alive",
        type=int,
        help="timeout keep alive",
        default=300,
        required=False
    )

    parser.add_argument(
        "-ncc", "--num-of-concurrent-calls",
        type=int,
        default=-1,
        help="num of concurrent calls. -1 for no limit",
        required=False
    )

    return parser.parse_args()


options = parse_args()


def serve():
    api_app = fastapi.FastAPI()

    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # pre-setup
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    config = uvicorn.Config(
        api_app,
        loop=event_loop,
        host=options.host,
        port=options.port,
        log_level="info",
        timeout_keep_alive=options.timeout_keep_alive,
    )

    server = uvicorn.Server(config)

    async_prompt = sync2async(prompt)

    assert options.num_of_concurrent_calls != 0, "num_of_concurrent_calls cannot be 0"

    if options.num_of_concurrent_calls > 0:
        async_prompt = limit_asyncio_concurrency(options.num_of_concurrent_calls)(
            async_prompt
        )

    async def forward_messages(
        messages: list[dict[str, str]],
        **additional_kwargs,
    ) -> AsyncGenerator[Union[str, bytes], None]:
        nonlocal async_prompt

        res = await async_prompt(
            messages,
            **additional_kwargs
        )

        if isinstance(res, Generator):
            for item in res:
                yield item
        elif isinstance(res, AsyncGenerator):
            async for item in res:
                yield item
        else:
            yield res

    async def chat_completion_loop(
        messages: list[dict[str, str]],
        **additional_kwargs,
    ) -> AsyncGenerator[Union[str, bytes], None]:
        nonlocal async_prompt

        original_message = deepcopy(messages)

        if os.getenv("FORWARD_ALL_MESSAGES", "0") == "1":
            async for item in forward_messages(original_message, **additional_kwargs):
                yield item

            return

        toolcalls = [
            {
                "type": "function",
                "function": {
                    "name": "do_specialized",
                    "description": "Complete complex tasks in a specialized domain.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]

        client = openai.AsyncClient(
            api_key=os.getenv("LLM_API_KEY", 'no-need'),
            base_url=os.getenv("LLM_BASE_URL", 'http://localmodel:65534/v1'),
        )

        if 'name' in additional_kwargs or 'description' in additional_kwargs:
            name = additional_kwargs.get('name', 'a helpful agent')
            description = additional_kwargs.get('description', 'I help people complete their tasks as best as I can')

            messages[-1]['content'] = f'''
{messages[-1]['content']}
{"-" * 30}
Use the basic information below to quickly complete simple tasks like introducing, greeting,
or answering follow-up questions, etc. You are {name}, {description}
'''

        completion = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_ID", 'localmodel'),
            messages=messages,
            tools=toolcalls,
            tool_choice="auto",
            max_tokens=1024
        )

        # check if the completion request toolcall or not
        if (
            completion.choices[0].message.tool_calls is not None
            and len(completion.choices[0].message.tool_calls) > 0
        ):
            async for item in forward_messages(original_message, **additional_kwargs):
                yield item

        else:
            yield completion.choices[0].message.content

    @api_app.post("/prompt", response_model=None)
    async def prompt_handler(body: dict) -> Union[StreamingResponse, PlainTextResponse, JSONResponse]:
        nonlocal async_prompt

        if body.get('ping'):
            return PlainTextResponse("online")

        messages: list[dict[str, str]] = body.pop('messages', [])

        if len(messages) == 0:
            return JSONResponse(
                content=PromptErrorResponse(
                    message="Received empty messages"
                ).model_dump(),
                status_code=400
            )

        if isinstance(messages[-1], str):
            messages[-1] = {
                "role": "user",
                "content": messages[-1]
            }

        messages[-1].setdefault('role', 'user')

        current_time_utc_str = datetime.datetime.now(
            tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(messages[-1]['content'], str):
            messages[-1]['content'] += f'\nCurrent time is {current_time_utc_str} UTC'
        elif isinstance(messages[-1]['content'], list):
            for i in range(len(messages[-1]['content'])):
                if messages[-1]['content'][i].get('type') == 'text':
                    messages[-1]['content'][i]['text'] += f'\nCurrent time is {current_time_utc_str} UTC'

        try:
            stream = chat_completion_loop(messages, **body)
            return StreamingResponse(
                stream_reader(stream),
                media_type="text/event-stream"
            )
        except Exception as err:
            error_message = "Unexpected Error: " + str(err)

            import traceback
            logger.error(traceback.format_exc())

            return JSONResponse(
                content=PromptErrorResponse(
                    message=error_message
                ).model_dump(),
                status_code=500
            )

    event_loop.run_until_complete(server.serve())


if __name__ == '__main__':
    serve()
