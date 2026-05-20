from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-3.5-flash"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _build_client(self):
        from google import genai
        return genai.Client(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> str:
        if self._client is None:
            self._client = self._build_client()

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                text = response.text
                if not text:
                    raise ValueError("Gemini retornó respuesta vacía")
                return text

            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(f"Intento {attempt}/{max_retries} fallido: {e}. Esperando {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"Gemini falló tras {max_retries} intentos: {last_error}")
