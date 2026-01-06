import logging
from typing import List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from os import getenv

load_dotenv()


class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        # Lazy import to avoid cycle
        from app.utils.memory import get_memory_client

        mem0_client = get_memory_client()
        if not mem0_client:
            logging.warning("Memory client not available — using env fallback for categorization")
            client = OpenAI(
                base_url=getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=getenv("OPENAI_API_KEY")
            )
            model_to_use = getenv("OPENAI_MODEL", "gpt-4o-mini")
            temperature_to_use = 0.0
        else:
            llm_section = mem0_client.config.llm if hasattr(mem0_client.config, "llm") and mem0_client.config.llm else None
            llm_config = llm_section.config if llm_section else {}

            client = OpenAI(
                base_url=llm_config.get("base_url", getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
                api_key=llm_config.get("api_key", getenv("OPENAI_API_KEY"))
            )

            model_to_use = llm_config.get("model", getenv("OPENAI_MODEL", "gpt-4o-mini"))
            temperature_to_use = llm_config.get("temperature", 0.0)

        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
            {"role": "user", "content": memory}
        ]

        completion = client.beta.chat.completions.parse(
            model=model_to_use,
            messages=messages,
            response_format=MemoryCategories,
            temperature=temperature_to_use
        )

        parsed: MemoryCategories = completion.choices[0].message.parsed
        return [cat.strip().lower() for cat in parsed.categories]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        try:
            logging.debug(f"[DEBUG] Raw response: {completion.choices[0].message.content}")
        except:
            pass
        raise