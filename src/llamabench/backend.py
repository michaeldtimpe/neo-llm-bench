"""llama-server backend — OpenAI-compatible chat completions client.

Adapted from deluxe's oMLX backend for llama.cpp's ``llama-server`` (which
also speaks the OpenAI ``/v1/chat/completions`` API). Differences:

- Default base URL is the llama-server default (``http://127.0.0.1:8080``).
- Sampling fields (top_p, top_k, min_p, repeat_penalty, seed, stop) are sent
  as top-level body params, NOT inside ``extra_body`` — llama-server reads
  them directly off the OpenAI body.
- Health endpoint is ``GET /health`` (not ``/v1/models/status``).
- Per-model lifecycle (load / unload / swap) is owned by ``server.py``'s
  ``LlamaServer``; this client doesn't try to manage it. ``thermal_guard``
  is retained as a thin "wait for /health to return ok after a swap" helper.

Resilience features inherited from deluxe:
- Body-aware retry that distinguishes transient (model loading / slot
  unavailable / warming) from terminal (OOM, crashed, context-window-full)
  failures by inspecting the response text.
- Connection / read timeout retries.
- Empty-body-in-warmup-window retry (server just started, hasn't filled
  the JSON error envelope yet).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


logger = logging.getLogger(__name__)


@dataclass
class GenerationTiming:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_s: float = 0.0
    time_to_first_token_s: float = 0.0

    @property
    def decode_tok_per_s(self) -> float:
        if self.total_s <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / self.total_s


@dataclass
class ToolCallResponse:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    text: str = ""
    tool_calls: list[ToolCallResponse] = field(default_factory=list)
    finish_reason: str = ""
    timing: GenerationTiming = field(default_factory=GenerationTiming)
    retries: int = 0


# --- Retry classification ---------------------------------------------------

# llama-server returns "slot is unavailable" when all parallel slots are busy
# — that's a *transient* state (wait for a slot to free), distinct from
# "service unavailable" (terminal). Order in the markers tuples matters only
# in that classify_failure checks terminal first.
_TRANSIENT_BODY_MARKERS = (
    "loading", "swapping", "warming", "starting", "not yet ready",
    "slot is unavailable", "no slot available",
)
_TERMINAL_BODY_MARKERS = (
    "service unavailable", "out of memory", "oom", "shut down",
    "crashed", "failed to load", "context window full",
)
_WARMUP_WINDOW_S = 5.0
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BACKOFF_S = (1.0, 4.0, 16.0)


@dataclass
class RetryDecision:
    retry: bool
    reason: str
    delay_s: float = 0.0


def classify_failure(
    *,
    exc: Exception | None = None,
    status_code: int | None = None,
    body: str = "",
    elapsed_since_start_s: float = 0.0,
    attempt: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> RetryDecision:
    """Decide whether to retry based on the failure shape and elapsed time.

    Retry on:
      - connection / read timeouts (httpx.RequestError, httpx.TimeoutException)
      - 5xx with body containing transient markers (loading / swapping / slot busy)
      - 5xx with empty body during the warmup window (first 5s of a run)

    Fail fast on:
      - 4xx (our request bug, retrying won't help)
      - 5xx with terminal markers (service unavailable / crashed / OOM /
        failed to load / context window full)
      - 5xx with empty body AFTER warmup window (assume terminal)
      - any failure on the last attempt
    """
    if attempt + 1 >= max_attempts:
        return RetryDecision(retry=False, reason="exhausted-attempts")

    delay = _DEFAULT_BACKOFF_S[min(attempt, len(_DEFAULT_BACKOFF_S) - 1)]

    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout,
                        httpx.NetworkError, httpx.RemoteProtocolError)):
        return RetryDecision(retry=True, reason=f"transient-{type(exc).__name__}", delay_s=delay)

    if status_code is None:
        if exc is not None:
            return RetryDecision(retry=False, reason=f"unknown-error-{type(exc).__name__}")
        return RetryDecision(retry=False, reason="no-status-no-exception")

    if 400 <= status_code < 500:
        return RetryDecision(retry=False, reason=f"4xx-{status_code}")

    if 500 <= status_code < 600:
        body_lc = (body or "").lower()
        for marker in _TERMINAL_BODY_MARKERS:
            if marker in body_lc:
                return RetryDecision(retry=False, reason=f"5xx-terminal-{marker}")
        for marker in _TRANSIENT_BODY_MARKERS:
            if marker in body_lc:
                return RetryDecision(retry=True, reason=f"5xx-transient-{marker}", delay_s=delay)
        if not body_lc.strip() and elapsed_since_start_s < _WARMUP_WINDOW_S:
            return RetryDecision(retry=True, reason="5xx-empty-warmup", delay_s=delay)
        return RetryDecision(retry=False, reason="5xx-empty-post-warmup")

    return RetryDecision(retry=False, reason=f"unexpected-{status_code}")


# --- Backend ---------------------------------------------------------------


class BackendError(Exception):
    """Raised when the backend gives up after retries."""


class Backend:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "",
        timeout_s: float = 600.0,
        api_key: str = "",
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        if not api_key:
            api_key = os.environ.get("LLAMA_API_KEY", "")
        self.api_key = api_key
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            headers=headers,
        )
        self._created_at = time.monotonic()

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        repeat_penalty: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        # Kept for API compatibility with deluxe call-sites; llama-server
        # doesn't need a per-request num_ctx (server set at startup).
        num_ctx: int | None = None,
        on_retry: Callable[[RetryDecision, int], None] | None = None,
    ) -> ChatResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        if top_k >= 0:
            body["top_k"] = top_k
        if min_p > 0:
            body["min_p"] = min_p
        if repeat_penalty is not None:
            body["repeat_penalty"] = repeat_penalty
        if seed is not None:
            body["seed"] = seed
        if stop:
            body["stop"] = stop
        # num_ctx is a server-startup concern; ignored here. Logged at debug
        # so call-sites don't silently get the wrong context window.
        if num_ctx is not None:
            logger.debug(
                "chat() received num_ctx=%d but llama-server context size is "
                "set at server startup; ignoring", num_ctx,
            )

        attempt = 0
        last_decision: RetryDecision | None = None
        request_t0 = time.monotonic()

        while attempt < self.max_attempts:
            t0 = time.monotonic()
            try:
                resp = self._client.post("/v1/chat/completions", json=body)
                wall = time.monotonic() - t0
                if resp.status_code >= 400:
                    decision = classify_failure(
                        status_code=resp.status_code,
                        body=resp.text,
                        elapsed_since_start_s=time.monotonic() - request_t0,
                        attempt=attempt,
                        max_attempts=self.max_attempts,
                    )
                    last_decision = decision
                    logger.warning(
                        "backend %s status=%d body=%r decision=%s",
                        self.model, resp.status_code, resp.text[:200], decision,
                    )
                    if not decision.retry:
                        raise BackendError(
                            f"llama-server returned {resp.status_code}: {resp.text[:200]} "
                            f"({decision.reason})"
                        )
                    if on_retry:
                        on_retry(decision, attempt)
                    time.sleep(decision.delay_s)
                    attempt += 1
                    continue
                data = resp.json()
                choice = data["choices"][0]
                msg = choice["message"]
                usage = data.get("usage", {})

                timing = GenerationTiming(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_s=wall,
                )

                tc_list: list[ToolCallResponse] = []
                for tc in msg.get("tool_calls") or []:
                    fn = tc["function"]
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"_raw": args}
                    tc_list.append(ToolCallResponse(
                        id=tc.get("id", ""),
                        name=fn["name"],
                        arguments=args,
                    ))

                return ChatResponse(
                    text=msg.get("content") or "",
                    tool_calls=tc_list,
                    finish_reason=choice.get("finish_reason", ""),
                    timing=timing,
                    retries=attempt,
                )
            except (httpx.HTTPError, OSError) as exc:
                decision = classify_failure(
                    exc=exc,
                    elapsed_since_start_s=time.monotonic() - request_t0,
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                )
                last_decision = decision
                logger.warning(
                    "backend %s exception=%s decision=%s",
                    self.model, type(exc).__name__, decision,
                )
                if not decision.retry:
                    raise BackendError(
                        f"llama-server call failed: {type(exc).__name__}: {exc} "
                        f"({decision.reason})"
                    ) from exc
                if on_retry:
                    on_retry(decision, attempt)
                time.sleep(decision.delay_s)
                attempt += 1

        reason = last_decision.reason if last_decision else "unknown"
        raise BackendError(
            f"llama-server retries exhausted after {self.max_attempts} attempts ({reason})"
        )

    def chat_with_text_recovery(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Same as ``chat()``, but if the model emitted no structured tool
        calls *and* tools were offered, recover them from the text channel
        using the 5-shape parser in ``agents.loop._parse_text_tool_calls``.

        Wires deluxe lesson #13: "the text-channel parser fallback must be
        wired everywhere ``backend.chat()`` is called for an evaluation."
        Use this from BFCL ``raw`` mode and any other adapter where the
        agent loop's recovery isn't running.
        """
        resp = self.chat(messages=messages, tools=tools, **kwargs)
        if resp.tool_calls or not tools:
            return resp
        from llamabench.agents.loop import _parse_text_tool_calls
        known_names = [t.get("function", {}).get("name", "") for t in tools]
        recovered = _parse_text_tool_calls(resp.text, [n for n in known_names if n])
        if recovered:
            resp.tool_calls = [
                ToolCallResponse(id=tc.get("id", ""), name=tc["name"],
                                 arguments=tc.get("arguments", {}))
                for tc in recovered
            ]
        return resp

    def health(self) -> bool:
        """Return True if llama-server's /health says ok."""
        try:
            r = self._client.get("/health")
            if r.status_code != 200:
                return False
            return r.json().get("status") == "ok"
        except (httpx.HTTPError, ValueError):
            return False

    def list_models(self) -> list[str]:
        r = self._client.get("/v1/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def assert_models_available(self, required: list[str]) -> list[str]:
        """Confirm all required model IDs resolve via /v1/models. Returns missing list."""
        available = set(self.list_models())
        return [m for m in required if m not in available]

    # --- methods retained for API compatibility with deluxe call-sites ---
    # llama-server hosts exactly one model per process; lifecycle is owned by
    # ``llamabench.server.LlamaServer``, not by this client. These no-ops keep
    # call-sites that came from deluxe (which freely called unload_*) from
    # blowing up; they are *not* the way to swap models — use ``LlamaServer.swap``.

    def loaded_models(self) -> list[str]:
        try:
            return self.list_models()
        except httpx.HTTPError:
            return []

    def unload_model(self, model_id: str) -> bool:
        return False

    def unload_all_loaded(self, *, except_for: list[str] | None = None) -> dict[str, bool]:
        return {}

    def thermal_guard(self, target_model: str, settle_s: float = 2.0,
                      max_wait_s: float = 30.0) -> bool:
        """Wait briefly for /health to be ok and the model to be reported.

        Used after a ``LlamaServer.swap()`` to give the client a chance to
        observe the new server before issuing a chat call.
        """
        time.sleep(settle_s)
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            if self.health():
                try:
                    if not target_model or target_model in set(self.list_models()):
                        return True
                except httpx.HTTPError:
                    pass
            time.sleep(1.0)
        return False
