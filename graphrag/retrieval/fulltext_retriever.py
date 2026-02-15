from typing import List, Dict, Any

from graphrag.config import get_settings
from graphrag.graph.neo4j_manager import Neo4jManager


class FullTextRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.settings = get_settings()
        self._create_fulltext_index()

    def _create_fulltext_index(self):
        """Crea un índice de texto completo."""
        query = """
        CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
        FOR (c:Chunk)
        ON EACH [c.text]
        """
        try:
            self.neo4j.execute_query(query)
        except Exception as e:
            print(f"Índice fulltext ya existe o error: {e}")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Recupera chunks usando búsqueda vectorial pura de texto completo.
        """
        top_k = top_k or self.settings.top_k_results

        cypher_query = """
        // Fulltext search
        CALL db.index.fulltext.queryNodes('chunk_fulltext', $query, {limit: $top_k})
        YIELD node, score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN node.text AS text,
               node.id AS chunk_id,
               score
        """

        results = self.neo4j.execute_query(cypher_query, {
            "query": query,
            "top_k": top_k
        })

        return results