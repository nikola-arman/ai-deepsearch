from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from deepsearch.utils.streaming import to_chunk_data
from app.handlers import handle_request 
from app.oai_models import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    random_uuid,
    LabRequest,
    ChatCompletionAdditionalParameters,
    ChatTemplate
)
import time
from typing import AsyncGenerator, Optional, Any
import logging
import asyncio

logger = logging.getLogger(__name__)
api_router = APIRouter()

class EventSignalHandler():
    def __init__(self):
        self.event_signals: dict[str, asyncio.Event] = {}

    def register_event_signal(self, event_id: str) -> asyncio.Event:
        event: asyncio.Event = asyncio.Event()
        logger.info(f"Request {event_id} - Registering event signal")
        self.event_signals[event_id] = event
        return event

    def emit_event_signal(self, event_id: str):

        if event_id in self.event_signals:
            logger.info(f"Request {event_id} - Emitting event signal")
            self.event_signals[event_id].set()
        else:
            logger.warning(f"Request {event_id} - Event signal not found")

    def unregister_event_signal(self, event_id: str):
        logger.info(f"Request {event_id} - Unregistering event signal")
        del self.event_signals[event_id]

event_handler = EventSignalHandler()

@api_router.post("/prompt")
async def chat_completions(request: ChatCompletionRequest, original_request: Request, background_tasks: BackgroundTasks):
    enqueued = time.time()
    ttft, tps, n_tokens = float("inf"), None, 0

    orig_data: dict[str, Any] = await original_request.json()
    lab_req_payload = (
        LabRequest.model_validate(orig_data)
        if orig_data.get("personality")
        else None
    )
    req_id = orig_data.get("id", request.request_id) or f"req-{random_uuid()}"
    logger.info(f"req_id: {req_id}")

    try:
        additional_parameters: Optional[ChatCompletionAdditionalParameters] = (
            ChatCompletionAdditionalParameters.model_validate(orig_data) 
            if orig_data.get("chat_template_kwargs")
            else ChatCompletionAdditionalParameters(chat_template_kwargs=ChatTemplate(enable_thinking=True))
        )
    except Exception as e:
        logger.error(f"Invalid additional parameters: {e}")
        additional_parameters = ChatCompletionAdditionalParameters(chat_template_kwargs=ChatTemplate(enable_thinking=True))

    additional_parameters.chat_template_kwargs.enable_thinking = True
    request.stream = True

    event = event_handler.register_event_signal(req_id)
    background_tasks.add_task(event_handler.unregister_event_signal, req_id)

    generator = handle_request(request, event, lab_req_payload, additional_parameters)

    if request.stream:

        async def to_bytes(gen: AsyncGenerator) -> AsyncGenerator[bytes, None]:
            nonlocal ttft, tps, n_tokens

            async for chunk in gen:
                current_time = time.time()

                n_tokens += 1
                ttft = min(ttft, current_time - enqueued)
                tps = n_tokens / (current_time - enqueued)

                yield to_chunk_data(chunk)

            logger.info(f"Request {req_id} - TTFT: {ttft:.2f}s, TPS: {tps:.2f} tokens/s")
            yield "data: [DONE]\n\n"

        return StreamingResponse(to_bytes(generator), media_type="text/event-stream")
    
    else:
        async for chunk in generator:
            current_time = time.time()

            n_tokens += 1
            ttft = min(ttft, current_time - enqueued)
            tps = n_tokens / (current_time - enqueued)

        logger.info(f"Request {req_id} - TTFT: {ttft:.2f}s, TPS: {tps:.2f} tokens/s")
        return JSONResponse(chunk.model_dump())

@api_router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    return await chat_completions(request)

@api_router.post("/cancel")
async def cancel(request: Request):
    orig_data: dict[str, Any] = await request.json()
    event_id = orig_data.get("id")

    if event_id:
        event_handler.emit_event_signal(event_id)

    return JSONResponse({"status": "ok"})