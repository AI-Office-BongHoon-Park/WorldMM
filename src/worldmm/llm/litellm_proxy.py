"""
LiteLLM Proxy Model Wrapper.

Talks to an OpenAI-compatible LiteLLM proxy that ONLY supports streaming
chat completions. The proxy in question (chatgpt/* models) is backed by
the ChatGPT web frontend and exposes the standard /v1/chat/completions
SSE protocol, but not the OpenAI Responses API.

Drop-in shape-compatible with OpenAIModel.generate(prompt, text_format=...):
- prompt: str | List[Dict[str, Any]]
- text_format: Optional[type[BaseModel]]  -> returns Pydantic instance
- otherwise returns str.

Structured output is achieved by prompt engineering + Pydantic validation,
because the proxy does not support `responses.parse(text_format=...)`.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import sqlite3
import hashlib
import functools
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Type, Union

import requests
from filelock import FileLock
from pydantic import BaseModel

from .utils import dynamic_retry_decorator

logger = logging.getLogger(__name__)


# Models exposed by the local LiteLLM proxy (verified live 2026-05-16 KST).
# We pass these straight through; the proxy resolves them upstream.
MODEL_DICT: Dict[str, str] = {
    "chatgpt/gpt-5.4": "chatgpt/gpt-5.4",
    "chatgpt/gpt-5.5": "chatgpt/gpt-5.5",
    "chatgpt/gpt-5.3-codex": "chatgpt/gpt-5.3-codex",
    "chatgpt/codex-auto": "chatgpt/codex-auto",
    "chatgpt-gpt-5.4": "chatgpt/gpt-5.4",
    "chatgpt-gpt-5.5": "chatgpt/gpt-5.5",
    "chatgpt-gpt-5.3-codex": "chatgpt/gpt-5.3-codex",
    "chatgpt-codex-auto": "chatgpt/codex-auto",
}


class LiteLLMProxyError(Exception):
    """Raised when the proxy call or response cannot be turned into a usable result."""


def _cache_response(func):
    """Lightweight SQLite cache decorator, mirrors OpenAIModel.cache_response shape."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if args:
            prompt = args[0]
        else:
            prompt = kwargs.get("prompt")
        if prompt is None:
            raise ValueError("Missing required 'prompt' parameter for caching.")

        text_format = args[1] if len(args) > 1 else kwargs.get("text_format")

        key_data = {
            "prompt": prompt,
            "model": getattr(self, "model_name", None),
            "text_format": text_format.__name__ if text_format else None,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        lock_file = self.cache_file_name + ".lock"

        os.makedirs(os.path.dirname(self.cache_file_name) or ".", exist_ok=True)

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, message TEXT)"
            )
            conn.commit()
            c.execute("SELECT message FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                payload = json.loads(row[0])
                if text_format and isinstance(payload, dict):
                    return text_format(**payload)
                return payload

        result = func(self, *args, **kwargs)

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, message TEXT)"
            )
            if isinstance(result, BaseModel):
                serial = result.model_dump()
            else:
                serial = result
            c.execute(
                "INSERT OR REPLACE INTO cache (key, message) VALUES (?, ?)",
                (key_hash, json.dumps(serial, default=str)),
            )
            conn.commit()
            conn.close()

        return result

    return wrapper


def _pydantic_schema_hint(text_format: Type[BaseModel]) -> str:
    """Render a compact JSON schema hint for prompt injection."""
    try:
        schema = text_format.model_json_schema()
    except Exception:  # pragma: no cover - extremely unlikely with pydantic v2
        return f"a JSON object matching the Pydantic model `{text_format.__name__}`."
    # Strip noisy keys to keep the prompt tight.
    schema.pop("title", None)
    schema.pop("$defs", None)
    return json.dumps(schema, ensure_ascii=False)


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` fences if the model wrapped its answer."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the first line (``` or ```json) and the trailing fence.
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


class LiteLLMProxyModel:
    """LiteLLM-proxy streaming-only client with shape-compatible `generate` API."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
        request_timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        if model_name not in MODEL_DICT:
            if not (model_name.startswith("chatgpt/") or model_name.startswith("chatgpt-")):
                raise ValueError(
                    f"Unsupported LiteLLM proxy model: {model_name}. "
                    f"Known: {list(MODEL_DICT.keys())}"
                )
            resolved = model_name
        else:
            resolved = MODEL_DICT[model_name]

        self.model_name = resolved
        self.base_url = (base_url or os.getenv("LITELLM_BASE_URL") or "http://127.0.0.1:4000").rstrip("/")
        self.api_key = api_key or os.getenv("LITELLM_MASTER_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LiteLLMProxyError(
                "LITELLM_MASTER_KEY (or OPENAI_API_KEY) must be set for the proxy."
            )
        self.max_retries = max(1, max_retries)
        self.request_timeout = request_timeout
        self.kwargs = kwargs

        cache_root = cache_dir or ".cache"
        safe_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        self.cache_file_name = os.path.join(cache_root, f"litellm_cache_{safe_name}.db")

        logger.info(
            "Initialized LiteLLMProxyModel model=%s base_url=%s",
            self.model_name,
            self.base_url,
        )

    def _normalize_prompt(
        self, prompt: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return copy.deepcopy(prompt)

    def _inject_json_directive(
        self, messages: List[Dict[str, Any]], text_format: Type[BaseModel]
    ) -> List[Dict[str, Any]]:
        """Prepend / merge a system message that forces strict JSON output."""
        schema_hint = _pydantic_schema_hint(text_format)
        directive = (
            "You output ONLY a single valid JSON object that matches the following "
            "JSON schema. No prose. No markdown fences. No commentary. "
            f"Schema: {schema_hint}"
        )
        if messages and messages[0].get("role") == "system":
            existing = messages[0].get("content", "")
            if isinstance(existing, list):
                # multimodal content list -> merge as text item
                messages[0]["content"] = existing + [{"type": "text", "text": directive}]
            else:
                messages[0]["content"] = f"{existing}\n\n{directive}".strip()
        else:
            messages = [{"role": "system", "content": directive}] + messages
        return messages

    def _stream_chat(self, messages: List[Dict[str, Any]], **call_kwargs: Any) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
        }
        for k, v in {**self.kwargs, **call_kwargs}.items():
            if k in ("stream",):  # never let caller turn streaming off
                continue
            payload[k] = v

        with requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=self.request_timeout,
        ) as resp:
            if resp.status_code != 200:
                body = resp.text[:500]
                raise LiteLLMProxyError(
                    f"Proxy HTTP {resp.status_code}: {body}"
                )

            accumulated: List[str] = []
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    # The proxy sometimes injects HTML wait pages — surface fast.
                    raise LiteLLMProxyError(
                        f"Non-JSON SSE chunk from proxy: {data[:200]}"
                    )
                if "error" in chunk:
                    raise LiteLLMProxyError(f"Proxy error chunk: {chunk['error']}")
                for choice in chunk.get("choices", []) or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        accumulated.append(piece)
            return "".join(accumulated).strip()

    @_cache_response
    @dynamic_retry_decorator
    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Any:
        messages = self._normalize_prompt(prompt)
        if text_format is not None:
            messages = self._inject_json_directive(messages, text_format)

        raw = self._stream_chat(messages, **kwargs)

        if text_format is None:
            return raw

        text = _strip_code_fence(raw)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LiteLLMProxyError(
                f"Could not parse JSON from proxy reply: {exc}; raw={text[:400]!r}"
            ) from exc

        try:
            return text_format(**data)
        except Exception as exc:
            raise LiteLLMProxyError(
                f"Pydantic validation against {text_format.__name__} failed: {exc}; "
                f"data={data!r}"
            ) from exc

    def generate_batch(
        self,
        batch_prompts: List[Union[str, List[Dict[str, Any]]]],
        text_format: Optional[Type[BaseModel]] = None,
        max_workers: int = 4,
        **kwargs: Any,
    ) -> List[Any]:
        """Thread-pool batch generation. Errors propagate per-item as exceptions."""
        results: List[Any] = [None] * len(batch_prompts)

        def _work(idx_prompt):
            idx, prompt = idx_prompt
            return idx, self.generate(prompt, text_format=text_format, **kwargs)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for idx, result in pool.map(_work, enumerate(batch_prompts)):
                results[idx] = result
        return results

    def __repr__(self) -> str:  # pragma: no cover
        return f"LiteLLMProxyModel(model_name={self.model_name!r}, base_url={self.base_url!r})"
