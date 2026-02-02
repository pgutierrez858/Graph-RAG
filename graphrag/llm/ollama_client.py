import json
from typing import List, Dict, Any, Type, TypeVar
import ollama
from pydantic import BaseModel
from ..config import get_settings

# Definimos un TypeVar para mantener el tipado genérico de Pydantic
T = TypeVar('T', bound=BaseModel)

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


class OllamaClient:
    def __init__(self):
        self.settings = get_settings()
        self.client = ollama.Client(host=self.settings.ollama_base_url)

    def chat(
            self,
            messages: List[Dict[str, str]],
            model: str = None,
            temperature: float = 0.0,
            format: str = None
    ) -> str:
        """Genera una respuesta usando Ollama."""
        model = model or self.settings.ollama_model
        response = self.client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            format=format
        )
        return response['message']['content']

    def structured_output(
            self,
            prompt: str,
            schema: Type[T],
            system_prompt: str = "Eres un extractor de datos experto. Responde estrictamente en formato JSON.",
            model: str = None
    ) -> T:
        """
        Fuerza al modelo a devolver un objeto basado en un esquema de Pydantic.
        Ideal para Entity Extraction y Entity Resolution.
        """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]

        # Ollama acepta el JSON Schema del modelo de Pydantic directamente
        response_content = self.chat(
            messages=messages,
            model=model,
            format=schema.model_json_schema()
        )

        # Validar y convertir el string JSON a la instancia de Pydantic
        # Esto previene los KeyErrors
        return schema.model_validate_json(response_content)

    def embed(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Genera embeddings usando Ollama."""
        model = model or self.settings.ollama_embedding_model
        embeddings = []
        for text in texts:
            response = self.client.embeddings(model=model, prompt=text)
            embeddings.append(response['embedding'])
        return embeddings