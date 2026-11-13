"""Connects to a free LLM provider using plain HTTP requests.

Both Google Gemini and Groq offer free API tiers (no credit card) and both
accept requests in the "OpenAI chat completions" format. Instead of using an
SDK, we send that request ourselves with `requests`. It's only a few lines,
nothing is hidden, and timeouts behave predictably.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from types import SimpleNamespace

import requests


@dataclass
class Provider:
    name: str
    base_url: str
    key_env_var: str
    default_model: str
    signup_url: str
    models_url: str
    auth_header: str  # header used to send the key when listing models
    backup_models: tuple[str, ...] = ()  # also tried by the connection test


PROVIDERS = {
    "gemini": Provider(
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env_var="GEMINI_API_KEY",
        default_model="gemini-3.5-flash",
        signup_url="https://aistudio.google.com/apikey",
        models_url="https://generativelanguage.googleapis.com/v1beta/models",
        auth_header="x-goog-api-key",
        # Tried in order if the main model is busy (503) or retired (404).
        backup_models=("gemini-3.5-flash-lite",),
    ),
    "groq": Provider(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        key_env_var="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        signup_url="https://console.groq.com/keys",
        models_url="https://api.groq.com/openai/v1/models",
        auth_header="Authorization",
        backup_models=("llama-3.3-70b-versatile",),
    ),
}


class LLMError(Exception):
    """The provider answered with an error (bad key, unknown model, rate limit...)."""

    def __init__(self, status: int, detail: str):
        self.status = status
        super().__init__(f"HTTP {status}: {detail}")


class ChatClient:
    """A tiny chat-completions client.

    It mimics the shape of the OpenAI SDK (client.chat.completions.create)
    so the agent code reads the same as most tutorials.
    """

    RETRY_STATUSES = {429, 500, 503}  # rate limited or temporarily overloaded
    SKIP_STATUSES = {404, 503}  # model retired or still busy: try the next one

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 60,
        fallback_models: tuple[str, ...] = (),
        retries: int = 2,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout
        self.fallback_models = fallback_models
        self.retries = retries
        self.last_model: str | None = None  # which model actually answered
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model: str, messages: list, tools: list | None = None):
        """Ask `model`; if it is busy or retired, fall back to the backups."""
        models = list(dict.fromkeys([model, *self.fallback_models]))
        error = None
        for name in models:
            try:
                reply = self._create_with_retries(name, messages, tools)
                self.last_model = name
                return reply
            except LLMError as e:
                if e.status not in self.SKIP_STATUSES:
                    raise
                error = e
        raise error

    def _create_with_retries(self, model: str, messages: list, tools: list | None):
        for attempt in range(self.retries + 1):
            try:
                return self._create_once(model, messages, tools)
            except LLMError as e:
                if e.status not in self.RETRY_STATUSES or attempt == self.retries:
                    raise
                time.sleep(2 * (attempt + 1))  # wait 2s, then 4s

    def _create_once(self, model: str, messages: list, tools: list | None):
        payload = {"model": model, "messages": messages}
        if tools:
            payload["tools"] = tools
        try:
            resp = requests.post(
                self.url, headers=self.headers, json=payload,
                timeout=(10, self.timeout),  # (connect, read) in seconds
            )
        except requests.Timeout:
            raise TimeoutError(f"No response after {self.timeout:.0f} seconds") from None
        if resp.status_code != 200:
            raise LLMError(resp.status_code, resp.text[:500])
        return _parse_reply(resp.json())


def _parse_reply(data: dict) -> SimpleNamespace:
    """Turn the JSON reply into objects: reply.choices[0].message.content etc."""
    msg = data["choices"][0]["message"]
    tool_calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        args = tc["function"].get("arguments") or "{}"
        if not isinstance(args, str):
            args = json.dumps(args)
        tool_calls.append(
            SimpleNamespace(
                id=tc.get("id") or f"call_{i}",
                function=SimpleNamespace(name=tc["function"]["name"], arguments=args),
                # Keep the original so we can send it back unchanged. Newer
                # Gemini models attach a "thought signature" they need to see again.
                raw=tc,
            )
        )
    message = SimpleNamespace(content=msg.get("content"), tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_client(provider_id: str, api_key: str | None = None) -> ChatClient:
    """Return a chat client pointed at the chosen provider."""
    provider = PROVIDERS[provider_id]
    key = api_key or os.getenv(provider.key_env_var)
    if not key:
        raise ValueError(
            f"No API key found for {provider.name}. Set {provider.key_env_var} "
            f"in your .env file (get a free key at {provider.signup_url})."
        )
    return ChatClient(provider.base_url, key, fallback_models=provider.backup_models)


def model_for(provider_id: str) -> str:
    """Model name, overridable with the MODEL environment variable."""
    return os.getenv("MODEL") or PROVIDERS[provider_id].default_model


def run_with_deadline(fn, seconds: float):
    """Run fn() but give up after `seconds`, no matter what it is stuck on."""
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        return future.result(timeout=seconds)
    except FutureTimeout:
        raise TimeoutError(f"No response after {seconds:.0f} seconds") from None
    finally:
        pool.shutdown(wait=False)


def diagnose(provider_id: str, api_key: str, model: str) -> list[tuple[bool, str]]:
    """Check network + key, then try a few models side by side.

    Returns a list of (passed, message) lines to show the user.
    """
    provider = PROVIDERS[provider_id]
    lines: list[tuple[bool, str]] = []

    # Step 1: list models (checks the network and the key, no AI call).
    value = f"Bearer {api_key}" if provider.auth_header == "Authorization" else api_key
    try:
        resp = requests.get(
            provider.models_url, headers={provider.auth_header: value}, timeout=15
        )
    except Exception as e:
        lines.append((False, f"Could not reach {provider.name}: {type(e).__name__}"))
        return lines
    if resp.status_code != 200:
        lines.append((False, f"{provider.name} rejected the key "
                             f"(HTTP {resp.status_code}): {resp.text[:300]}"))
        return lines
    data = resp.json()
    count = len(data.get("models", data.get("data", [])))
    lines.append((True, f"Reached {provider.name} and the key works ({count} models)."))

    # Step 2: ask each candidate model to say OK, all at the same time.
    candidates = list(dict.fromkeys([model, *provider.backup_models]))
    # No retries or fallbacks here: we want to see how each model does on its own.
    client = ChatClient(provider.base_url, api_key, timeout=30, retries=0)

    def try_model(name: str) -> tuple[bool, str]:
        started = time.time()
        try:
            reply = client.create(
                model=name,
                messages=[{"role": "user", "content": "Reply with just the word: OK"}],
            )
            text = (reply.choices[0].message.content or "").strip()[:30]
            return True, f"'{name}' works ({time.time() - started:.1f}s): {text}"
        except Exception as e:
            return False, f"'{name}' failed: {type(e).__name__}: {str(e)[:250]}"

    pool = ThreadPoolExecutor(max_workers=len(candidates))
    futures = [pool.submit(try_model, name) for name in candidates]
    for name, future in zip(candidates, futures):
        try:
            lines.append(future.result(timeout=40))
        except FutureTimeout:
            lines.append((False, f"'{name}' failed: no response after 40 seconds"))
    pool.shutdown(wait=False)
    return lines
