from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from ..graph.neo4j_manager import Neo4jManager
from ..llm.ollama_client import OllamaClient


class _CypherQuery(BaseModel):
    cypher: str


class Text2CypherRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.client = OllamaClient()
        self.few_shot_examples = []

    def add_few_shot_example(self, question: str, cypher: str):
        """Añade un ejemplo few-shot."""
        self.few_shot_examples.append({
            "question": question,
            "cypher": cypher
        })

    def generate_cypher(self, question: str) -> str:
        """
        Genera una query Cypher a partir de una pregunta en lenguaje natural.
        Uses structured output so the model is constrained to return only the
        Cypher string — no preamble, reasoning, or markdown wrapping.
        """
        schema = self.neo4j.get_schema()
        schema_str = self.neo4j.format_schema(schema)

        examples_str = ""
        if self.few_shot_examples:
            examples_str = "Examples:\n" + "\n".join([
                f"Question: {ex['question']}\nCypher: {ex['cypher']}"
                for ex in self.few_shot_examples
            ])

        system_prompt = (
            "You are an expert at converting natural language questions into Cypher queries for Neo4j.\n\n"
            f"Graph Schema:\n{schema_str}\n\n"
            f"{examples_str}\n\n"
            "Rules:\n"
            "1. Use only the node labels, relationship types, and properties shown in the schema.\n"
            "2. Output ONLY the Cypher query in the 'cypher' field — no explanations, no markdown.\n"
            "3. The query must be syntactically correct Neo4j Cypher."
        )

        result = self.client.structured_output_with_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            schema=_CypherQuery,
            temperature=0.0,
        )
        return result.cypher.strip()

    def retrieve(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Genera una query Cypher y la ejecuta.

        Returns:
            Tupla con (cypher_query, results)
        """
        cypher = self.generate_cypher(question)

        try:
            results = self.neo4j.execute_query(cypher)
            return cypher, results
        except Exception as e:
            print(f"Error ejecutando Cypher: {e}")
            print(f"Query generada: {cypher}")
            return cypher, []