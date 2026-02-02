import json
from typing import List, Type, Literal, Callable, Any, Dict, Optional
from pydantic import BaseModel
import ollama


class OllamaManager:
    """
    Manager para interactuar con Ollama localmente.
    Soporta Chat, JSON estructurado y Tool Calling.
    """

    def __init__(self, model: str = "llama3.1"):
        self.model = model

    def chat(
            self,
            messages: List[Dict[str, str]],
            tools: Optional[List[Callable]] = None
    ) -> Any:
        """
        Chat estándar y detección de herramientas.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=tools,
            )

            # Si hay llamadas a herramientas, las devolvemos
            if response.get('message', {}).get('tool_calls'):
                return response['message']['tool_calls']

            # Si es chat normal, devolvemos el contenido
            return response['message']['content']
        except Exception as e:
            return f"Error en chat: {str(e)}"

    def structured_output(
            self,
            prompt: str,
            schema: Type[BaseModel],
            system_prompt: str = "Eres un extractor de datos experto. Responde solo en JSON."
    ) -> BaseModel:
        """
        Fuerza al modelo a devolver un objeto basado en un esquema de Pydantic.
        """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]

        response = ollama.chat(
            model=self.model,
            messages=messages,
            format=schema.model_json_schema(),  # Ollama usa el esquema JSON de Pydantic
        )

        # Validar y convertir el string JSON a la instancia de Pydantic
        return schema.model_validate_json(response['message']['content'])