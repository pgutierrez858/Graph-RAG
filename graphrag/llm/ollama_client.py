import json
import logging
import re
from typing import List, Dict, Any, Literal, Optional, Type, TypeVar, Union
import ollama
from pydantic import BaseModel
from ..config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class OllamaClient:
    def __init__(self):
        self.settings = get_settings()
        self.client = ollama.Client(host=self.settings.ollama_base_url)

    def chat(
            self,
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            temperature: float = 0.0,
            format: Union[Literal['', 'json'], Dict[str, Any], None] = None,
            think: bool = False,
    ) -> str:
        """Genera una respuesta usando Ollama.

        think=False (default) disables extended reasoning on models like qwen3
        that otherwise emit a long <think>…</think> block before answering.
        Set think=True only when you explicitly want chain-of-thought output.
        """
        model = model or self.settings.ollama_model
        logger.info("LLM chat → model=%s  messages=%d  think=%s", model, len(messages), think)
        response = self.client.chat(
            model=model,
            messages=messages,
            think=think,
            options={"temperature": temperature},
            format=format,
        )
        content = response.message.content or ""
        # Strip think blocks. Ollama sometimes strips the opening <think> tag but leaves </think>
        # in content, so we handle both forms:
        #   • full block:  <think>…</think>answer  → answer
        #   • orphan tag:  preamble…</think>answer  → answer
        if "</think>" in content:
            content = content[content.index("</think>") + len("</think>"):].strip()
        else:
            content = re.sub(r"<think>[\s\S]*?</think>\s*", "", content).strip()
        logger.info("LLM chat ✓ (%d chars)", len(content))
        return content

    def structured_output(
            self,
            prompt: str,
            schema: Type[T],
            system_prompt: str = "Eres un extractor de datos experto. Responde estrictamente en formato JSON.",
            model: Optional[str] = None,
            think: bool = False,
    ) -> T:
        """Fuerza al modelo a devolver un objeto validado contra un esquema Pydantic.

        think=False by default — structured outputs must never contain reasoning
        preambles; the schema fields are the only permitted output.
        """
        logger.info("LLM structured output → schema=%s  think=%s", schema.__name__, think)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]
        response_content = self.chat(
            messages=messages,
            model=model,
            format=schema.model_json_schema(),
            think=think,
        )
        return schema.model_validate_json(response_content)

    def structured_output_with_chat(
            self,
            messages: List[Dict[str, str]],
            schema: Type[T],
            model: Optional[str] = None,
            temperature: float = 0.0,
            think: bool = False,
    ) -> T:
        """Structured output preserving an existing conversation history.

        think=False by default for the same reason as structured_output.
        """
        logger.info("LLM structured output (chat) → schema=%s  think=%s", schema.__name__, think)
        response_content = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            format=schema.model_json_schema(),
            think=think,
        )
        return schema.model_validate_json(response_content)

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Genera embeddings usando Ollama."""
        model = model or self.settings.ollama_embedding_model
        logger.info("Embedding %d text(s) → model=%s", len(texts), model)
        embeddings = []
        for text in texts:
            response = self.client.embeddings(model=model, prompt=text)
            embeddings.append(list(response.embedding))
        logger.info("Embedding ✓")
        return embeddings

    @staticmethod
    def extract_json(response: str) -> Dict[str, Any]:
        """Extrae JSON de la respuesta."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            start = response.find('```json')
            if start != -1:
                start = response.find('\n', start) + 1
                end = response.find('```', start)
                if end != -1:
                    return json.loads(response[start:end].strip())

            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response[start:end + 1])

            raise ValueError("No se pudo extraer JSON de la respuesta")