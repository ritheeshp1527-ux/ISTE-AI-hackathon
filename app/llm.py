"""OpenAI-compatible LLM client with retries and exponential backoff."""

import os
import time
from dotenv import load_dotenv
import openai
from openai import OpenAI

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


class LLMTimeout(Exception):
    """Raised when an LLM call exceeds maximum retries after timeouts."""
    pass


class LLMError(Exception):
    """Raised on LLM API or client errors."""
    pass


client = OpenAI(
    api_key=LLM_API_KEY or "dummy-api-key",
    base_url=LLM_BASE_URL,
)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the OpenAI-compatible chat completions endpoint with retry logic."""
    max_retries = 2
    backoffs = [1.0, 2.0]

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                timeout=20.0,
            )
            content = response.choices[0].message.content
            return content or ""
        except (openai.APITimeoutError, TimeoutError) as exc:
            if attempt < max_retries:
                time.sleep(backoffs[attempt])
                continue
            raise LLMTimeout(f"LLM request timed out after {attempt + 1} attempts: {exc}") from exc
        except openai.APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status == 429 or (status is not None and status >= 500):
                if attempt < max_retries:
                    time.sleep(backoffs[attempt])
                    continue
                raise LLMError(f"LLM server error ({status}): {exc}") from exc
            # Any other 4xx: raise immediately without retrying
            raise LLMError(f"LLM client error ({status}): {exc}") from exc
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise LLMError(f"LLM error: {exc}") from exc
            if attempt < max_retries:
                time.sleep(backoffs[attempt])
                continue
            raise LLMError(f"LLM error: {exc}") from exc

    raise LLMError("Exhausted retries without a response")
