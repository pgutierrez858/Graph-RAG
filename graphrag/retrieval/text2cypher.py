from typing import List, Dict, Any, Tuple
from ..graph.neo4j_manager import Neo4jManager
from ..llm.ollama_client import OllamaClient


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
        """
        schema = self.neo4j.get_schema()
        schema_str = self.neo4j.format_schema(schema)

        # Construir ejemplos few-shot
        examples_str = ""
        if self.few_shot_examples:
            examples_str = "Examples:\n" + "\n".join([
                f"Question: {ex['question']}\nCypher: {ex['cypher']}"
                for ex in self.few_shot_examples
            ])

        system_prompt = f"""You are an expert at converting natural language questions into Cypher queries for Neo4j.

Graph Schema:
{schema_str}

{examples_str}

Rules:
1. Use only the node labels, relationship types, and properties shown in the schema
2. Return ONLY the Cypher query, no explanations
3. Do not use markdown code blocks
4. Make sure the query is syntactically correct

Generate a Cypher query to answer the following question."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        cypher = self.client.chat(messages, temperature=0.0)

        # Limpiar la respuesta
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()

        return cypher

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