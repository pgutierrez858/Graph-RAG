from typing import List, Dict, Any
from ..llm.ollama_client import OllamaClient
from .retriever_tools import RetrieverTools
from typing import Literal
from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    """Schema for the LLM's routing decision."""

    tool: Literal["vector_search", "hybrid_search", "text2cypher"] = Field(
        ...,
        description="The name of the tool selected to handle the query."
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this tool was chosen based on the question."
    )
    query: str = Field(
        ...,
        description="The reformulated or original query to be passed to the tool."
    )


class RetrieverRouter:
    def __init__(self, retriever_tools: RetrieverTools):
        self.tools = retriever_tools
        self.client = OllamaClient()

    def route(self, question: str, conversation_history: List[Dict[str, str]] = None) -> RouterDecision:
        """
        Selecciona la mejor herramienta para responder la pregunta.
        """
        conversation_history = conversation_history or []

        # Obtener descripciones de herramientas
        tool_descriptions = self.tools.get_tool_descriptions()
        tools_str = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in tool_descriptions
        ])

        system_prompt = f"""You are a routing assistant that selects the best tool to answer a question.

Available tools:
{tools_str}

Analyze the question and select the most appropriate tool. Consider:
- Use vector_search for conceptual or semantic queries
- Use hybrid_search for queries that benefit from both semantic and keyword matching
- Use text2cypher for queries that require:
  * Counting or aggregation (e.g., "how many", "total", "average")
  * Complex graph traversals (e.g., "find all connections between X and Y")
  * Filtering by specific properties
  * Structured data retrieval
"""

        messages = [
                       {"role": "system", "content": system_prompt}
                   ] + conversation_history + [
                       {"role": "user", "content": f"Question: {question}"}
                   ]

        return self.client.structured_output_with_chat(messages, schema=RouterDecision)

    def retrieve(self, question: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Selecciona la herramienta apropiada y ejecuta la búsqueda.
        """
        decision = self.route(question, conversation_history)

        print("Decision: ", decision)

        tool_name = decision.tool
        query = decision.query or question

        result = self.tools.execute_tool(tool_name, query=query)
        result["routing_decision"] = decision

        return result
