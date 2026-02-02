from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from ..llm.ollama_client import OllamaClient


# 1. Definimos los modelos de datos para la validación
class Entity(BaseModel):
    name: str = Field(..., description="The name of the entity")
    type: str = Field(..., description="The category of the entity")
    description: str = Field(..., description="A detailed description of the entity context")


class Relationship(BaseModel):
    source: str = Field(..., description="Name of the source entity")
    target: str = Field(..., description="Name of the target entity")
    type: str = Field(..., description="Type of relationship (e.g., INFLUENCED_BY, LIVES_IN)")
    description: str = Field(..., description="Detailed explanation of how they are related")
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


# 2. Modelo contenedor para la extracción completa
class GraphExtraction(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


class EntityExtractor:
    def __init__(self, entity_types: List[str] = None):
        self.client = OllamaClient()
        self.entity_types = entity_types or [
            "PERSON", "ORGANIZATION", "LOCATION",
            "EVENT", "CONCEPT", "TECHNOLOGY", "CULTURAL_ARTIFACT"
        ]

    def extract_entities_and_relationships(
            self,
            text: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extrae entidades y relaciones usando structured_output para evitar KeyErrors.
        """
        system_prompt = f"""You are an expert at extracting structured information for Knowledge Graphs.
        Your goal is to identify key entities and their connections from the text provided.

        ALLOWED ENTITY TYPES: {', '.join(self.entity_types)}
        """

        prompt = f"Extract entities and relationships from the following text:\n\n{text}"

        # Usamos el nuevo método que creamos en OllamaClient
        # Esto devuelve una instancia de GraphExtraction ya validada
        extraction: GraphExtraction = self.client.structured_output(
            prompt=prompt,
            schema=GraphExtraction,
            system_prompt=system_prompt
        )

        # Convertimos los modelos de Pydantic a diccionarios para mantener
        # compatibilidad con el resto de tu pipeline (TextProcessor)
        entities_dict = [e.model_dump() for e in extraction.entities]
        rels_dict = [r.model_dump() for r in extraction.relationships]

        return entities_dict, rels_dict

    def summarize_entity(self, entity_name: str, descriptions: List[str]) -> str:
        """Consolidación de descripciones (Entity Resolution step)."""
        system_prompt = "You are a specialist in knowledge synthesis. Create a single third-person summary."

        description_list = "\n- ".join(descriptions)
        user_message = f"Entity: {entity_name}\n\nDescriptions to merge:\n- {description_list}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self.client.chat(messages)

    def summarize_relationship(self, source: str, target: str, descriptions: List[str]) -> str:
        """Consolidación de relaciones."""
        prompt = f"Summarize the relationship between {source} and {target} based on: {descriptions}"
        # Aquí también podrías usar structured_output si quisieras un formato de resumen específico
        return self.client.chat([{"role": "user", "content": prompt}])