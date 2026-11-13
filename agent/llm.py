"""Connects to a free LLM provider.

Both Google Gemini and Groq offer free API tiers (no credit card) and both
speak the OpenAI API format, so one client library works for either.
Switching providers = changing a base URL, an API key, and a model name.
"""

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class Provider:
    name: str
    base_url: str
    key_env_var: str
    default_model: str
    signup_url: str


PROVIDERS = {
    "gemini": Provider(
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env_var="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
        signup_url="https://aistudio.google.com/apikey",
    ),
    "groq": Provider(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        key_env_var="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        signup_url="https://console.groq.com/keys",
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
    return OpenAI(base_url=provider.base_url, api_key=key)


def model_for(provider_id: str) -> str:
    """Model name, overridable with the MODEL environment variable."""
    return os.getenv("MODEL") or PROVIDERS[provider_id].default_model
