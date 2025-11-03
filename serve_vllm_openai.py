import asyncio
import json
import time
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from transformers import AutoTokenizer
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import EngineArgs
from ray import serve


fastapi_app = FastAPI()


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 2},
)
@serve.ingress(fastapi_app)
class VLLMOpenAI:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        engine_args = EngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        # Async engine enables token-by-token streaming
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @fastapi_app.get("/v1/models")
    async def list_models(self):
        now = int(time.time())
        data = [
            {
                "id": self.model_name,
                "object": "model",
                "created": now,
                "owned_by": "vllm",
            }
        ]
        return JSONResponse({"object": "list", "data": data})

    @fastapi_app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        body: Dict[str, Any] = await request.json()

        messages: List[Dict[str, str]] = body.get("messages", [])
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        # Convert OpenAI messages to a single prompt via chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Sampling parameters
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 512
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 1.0)
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        elif not isinstance(stop, list):
            stop = None

        params = SamplingParams(
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            stop=stop,
        )
        stream: bool = bool(body.get("stream", False))
        stream_options = body.get("stream_options") or {}
        include_usage = bool(stream_options.get("include_usage", False))

        request_id = f"req-{uuid.uuid4()}"

        # Add request to engine
        await self.engine.add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=params,
        )

        # Streaming path: send SSE deltas
        if stream:
            response_id = f"chatcmpl-{uuid.uuid4()}"
            async def event_generator():
                created = int(time.time())
                model = body.get("model") or self.model_name
                # First chunk: role delta
                first = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(first)}\n\n"

                prev_text_len = 0
                prompt_tokens = 0
                completion_tokens = 0

                async for output in self.engine.get_request_iterator(request_id):
                    # prompt tokens are available on every output
                    if hasattr(output, "prompt_token_ids") and output.prompt_token_ids is not None:
                        prompt_tokens = len(output.prompt_token_ids)
                    if not output.outputs:
                        continue
                    seq = output.outputs[0]
                    full_text: str = getattr(seq, "text", "") or ""
                    delta_text = full_text[prev_text_len:]
                    prev_text_len = len(full_text)
                    if hasattr(seq, "token_ids") and seq.token_ids is not None:
                        completion_tokens = len(seq.token_ids)

                    if delta_text:
                        chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Final chunk: finish + optional usage
                final_chunk: Dict[str, Any] = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                if include_usage:
                    final_chunk["usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming path: materialize the iterator and return one response
        final_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        async for output in self.engine.get_request_iterator(request_id):
            if hasattr(output, "prompt_token_ids") and output.prompt_token_ids is not None:
                prompt_tokens = len(output.prompt_token_ids)
            if not output.outputs:
                continue
            seq = output.outputs[0]
            final_text = getattr(seq, "text", "") or ""
            if hasattr(seq, "token_ids") and seq.token_ids is not None:
                completion_tokens = len(seq.token_ids)

        resp = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return JSONResponse(resp)


# Deployment graph for `serve run`
app = VLLMOpenAI.bind(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    tensor_parallel_size=1,
    dtype="auto",
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)
