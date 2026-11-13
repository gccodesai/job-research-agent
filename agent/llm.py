"""Connects to a free LLM provider.

Both Google Gemini and Groq offer free API tiers (no credit card) and both
speak the OpenAI API format, so one client library works for either.
Switching providers = changing a base URL, an API key, and a model name.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

import requests
from openai import OpenAI


@dataclass
class Provider:
    name: str
    base_url: str
    key_env_var: str
    default_model: str
    signup_url: str
    models_url: str
    auth_header: str  # header name used to send the key when listing models


PROVIDERS = {
    "gemini": Provider(
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env_var="GEMINI_API_KEY",
        default_model="gemini-flash-latest",  # alias: always the current Flash model
        signup_url="https://aistudio.google.com/apikey",
        models_url="https://generativelanguage.googleapis.com/v1beta/models",
        auth_header="x-goog-api-key",
    ),
    "groq": Provider(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        key_env_var="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        signup_url="https://console.groq.com/keys",
        models_url="https://api.groq.com/openai/v1/models",
        auth_header="Authorization",
    ),
}


def make_client(provider_id: str, api_key: str | None = None) -> OpenAI:
    """Return an OpenAI-compatible client pointed at the chosen provider."""
    provider = PROVIDERS[provider_id]
    key = api_key or os.getenv(provider.key_env_var)
    if not key:
        raise ValueError(
            f"No API key found for {provider.name}. Set {provider.key_env_var} "
            f"in your .env file (get a free key at {provider.signup_url})."
        )
    # Fail fast with a visible error instead of waiting (and retrying) silently.
    return OpenAI(base_url=provider.base_url, api_key=key, timeout=60, max_retries=0)


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
    """Check network + key, then the model, one step at a time.

    Returns a list of (passed, message) lines to show the user.
    """
    provider = PROVIDERS[provider_id]
    lines: list[tuple[bool, str]] = []

    # Step 1: plain HTTP request listing models (no AI call, no OpenAI library).
    value = f"Bearer {api_key}" if provider.auth_header == "Authorization" else api_key
    try:
        resp = requests.get(
            provider.models_url, headers={provider.auth_header: value}, timeout=15
        )
    except Exception as e:
        lines.append((False, f"Could not reach {provider.name}: {type(e).__name__}"))
        return lines
    if resp.status_code != 200:
        lines.append((False, f"{provider.name} rejected the key (HTTP {resp.status_code}): "
                             f"{resp.text[:300]}"))
        return lines
    data = resp.json()
    names = [m.get("name", m.get("id", "")).removeprefix("models/")
             for m in data.get("models", data.get("data", []))]
    lines.append((True, f"Reached {provider.name} and the key works ({len(names)} models)."))
    if provider_id == "gemini":
        flash = [n for n in names if "flash" in n]
        lines.append((True, "Flash models on your key: " + ", ".join(flash[:15])))
    if model not in names and not model.endswith("-latest"):
        lines.append((False, f"'{model}' is not in your model list. Pick one listed above."))

    # Step 2: a tiny chat request through the OpenAI-compatible endpoint.
    client = make_client(provider_id, api_key)
    try:
        reply = run_with_deadline(
            lambda: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with just the word: OK"}],
            ),
            30,
        )
        text = reply.choices[0].message.content or "(empty reply)"
        lines.append((True, f"Model '{model}' answered: {text.strip()[:50]}"))
    except Exception as e:
        lines.append((False, f"Model '{model}' failed: {type(e).__name__}: {str(e)[:300]}"))
    return lines


def model_for(provider_id: str) -> str:
    """Model name, overridable with the MODEL environment variable."""
    return os.getenv("MODEL") or PROVIDERS[provider_id].default_model
