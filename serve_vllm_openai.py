import json
import time
import uuid
from typing import Any, Dict, List

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from ray import serve
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs


@serve.deployment(ray_actor_options={"num_gpus": 1, "num_cpus": 2})
class VLLMDeployment:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_name = model_name

        # Initialize engine
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def list_models(self):
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": self.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm",
            }]
        })

    async def chat_completions(self, body: Dict[str, Any]):
        messages: List[Dict[str, str]] = body.get("messages", [])
        if not messages:
            return JSONResponse(
                {"error": "messages required"},
                status_code=400
            )

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Parse parameters
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 1.0)
        stream = body.get("stream", False)

        params = SamplingParams(
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
        )

        request_id = f"req-{uuid.uuid4()}"
        results_generator = self.engine.generate(prompt, params, request_id)

        if stream:
            return StreamingResponse(
                self._stream_response(results_generator, body),
                media_type="text/event-stream"
            )
        else:
            return await self._non_stream_response(results_generator, body)

    async def _stream_response(self, results_generator, body):
        """Generate streaming response chunks"""
        response_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())
        model = body.get("model", self.model_name)

        # First chunk with role
        yield f"data: {json.dumps({
            'id': response_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': model,
            'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]
        })}\n\n"

        prev_text = ""
        async for output in results_generator:
            if not output.outputs:
                continue

            text = output.outputs[0].text
            delta = text[len(prev_text):]
            prev_text = text

            if delta:
                yield f"data: {json.dumps({
                    'id': response_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': model,
                    'choices': [{'index': 0, 'delta': {'content': delta}, 'finish_reason': None}]
                })}\n\n"

        # Final chunk
        yield f"data: {json.dumps({
            'id': response_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': model,
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]
        })}\n\n"
        yield "data: [DONE]\n\n"

    async def _non_stream_response(self, results_generator, body):
        """Generate non-streaming response"""
        final_output = None
        async for output in results_generator:
            final_output = output

        text = final_output.outputs[0].text if final_output and final_output.outputs else ""
        prompt_tokens = len(final_output.prompt_token_ids) if final_output else 0
        completion_tokens = len(final_output.outputs[0].token_ids) if final_output and final_output.outputs else 0

        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", self.model_name),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })

    async def __call__(self, request: Request):
        """Route requests based on path and method"""
        path = request.url.path
        method = request.method

        if path == "/v1/models" and method == "GET":
            return await self.list_models()
        elif path == "/v1/chat/completions" and method == "POST":
            body = await request.json()
            return await self.chat_completions(body)
        else:
            return JSONResponse(
                {"error": f"Unknown endpoint: {method} {path}"},
                status_code=404
            )


entrypoint = VLLMDeployment.bind(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    tensor_parallel_size=1,
    dtype="auto",
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)
