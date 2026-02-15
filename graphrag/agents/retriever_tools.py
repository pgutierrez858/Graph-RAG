from typing import List, Dict, Any, Callable, TypedDict
from ..retrieval.vector_retriever import VectorRetriever, HybridRetriever
from ..retrieval.text2cypher import Text2CypherRetriever
from ..graph.neo4j_manager import Neo4jManager

# Definimos la estructura de la herramienta
class ToolData(TypedDict):
    function: Callable
    description: str

class RetrieverTools:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.vector_retriever = VectorRetriever(neo4j_manager)
        self.hybrid_retriever = HybridRetriever(neo4j_manager)
        self.text2cypher = Text2CypherRetriever(neo4j_manager)
        self.custom_tools: Dict[str, ToolData] = {}

    def register_custom_tool(self, name: str, function: Callable, description: str):
        """Registra una herramienta personalizada."""
        self.custom_tools[name] = {
            "function": function,
            "description": description
        }

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Obtiene las descripciones de todas las herramientas disponibles."""
        tools = [
            {
                "name": "vector_search",
                "description": "Search for relevant information using semantic similarity. Best for finding conceptually related content.",
                "parameters": {
                    "query": "The search query in natural language"
                }
            },
            {
                "name": "hybrid_search",
                "description": "Search using both semantic similarity and keyword matching. Good for balanced retrieval.",
                "parameters": {
                    "query": "The search query in natural language"
                }
            },
            {
                "name": "text2cypher",
                "description": "Query the knowledge graph using natural language. Best for structured queries, aggregations, and complex graph traversals.",
                "parameters": {
                    "query": "The question in natural language"
                }
            }
        ]

        # Añadir herramientas personalizadas
        for name, tool_info in self.custom_tools.items():
            tools.append({
                "name": name,
                "description": tool_info["description"],
                "parameters": {}
            })

        return tools

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Ejecuta una herramienta específica."""
        if tool_name == "vector_search":
            results = self.vector_retriever.retrieve(kwargs.get("query", ""))
            return {
                "tool": tool_name,
                "results": results,
                "context": [r["text"] for r in results]
            }

        elif tool_name == "hybrid_search":
            results = self.hybrid_retriever.retrieve(kwargs.get("query", ""))
            return {
                "tool": tool_name,
                "results": results,
                "context": [r["text"] for r in results]
            }

        elif tool_name == "text2cypher":
            cypher, results = self.text2cypher.retrieve(kwargs.get("query", ""))
            return {
                "tool": tool_name,
                "cypher": cypher,
                "results": results,
                "context": [str(r) for r in results]
            }

        elif tool_name in self.custom_tools:
            function = self.custom_tools[tool_name]["function"]
            results = function(**kwargs)
            return {
                "tool": tool_name,
                "results": results,
                "context": results if isinstance(results, list) else [str(results)]
            }

        else:
            raise ValueError(f"Unknown tool: {tool_name}")