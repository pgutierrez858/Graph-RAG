from typing import List, Dict, Any
from ..graph.neo4j_manager import Neo4jManager
from ..utils.embeddings import EmbeddingGenerator
from ..config import get_settings


class VectorRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.embedding_gen = EmbeddingGenerator()
        self.settings = get_settings()

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Recupera chunks relevantes usando búsqueda vectorial.
        """
        top_k = top_k or self.settings.top_k_results

        # Generar embedding de la query
        query_embedding = self.embedding_gen.embed_text(query)

        # Buscar chunks similares
        cypher_query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
        YIELD node AS chunk, score
        RETURN chunk.text AS text, 
               chunk.id AS chunk_id,
               score
        ORDER BY score DESC
        """

        results = self.neo4j.execute_query(cypher_query, {
            "query_embedding": query_embedding,
            "top_k": top_k
        })

        return results

    def retrieve_with_entities(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Recupera chunks relevantes junto con sus entidades.
        """
        top_k = top_k or self.settings.top_k_results
        query_embedding = self.embedding_gen.embed_text(query)

        cypher_query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
        YIELD node AS chunk, score
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:Entity)
        WITH chunk, score, collect(DISTINCT {
            name: e.name, 
            type: e.type, 
            summary: e.summary
        }) AS entities
        RETURN chunk.text AS text,
               chunk.id AS chunk_id,
               score,
               entities
        ORDER BY score DESC
        """

        results = self.neo4j.execute_query(cypher_query, {
            "query_embedding": query_embedding,
            "top_k": top_k
        })

        return results


class HybridRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.embedding_gen = EmbeddingGenerator()
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
        Recupera chunks usando búsqueda híbrida (vectorial + texto completo).
        """
        top_k = top_k or self.settings.top_k_results
        query_embedding = self.embedding_gen.embed_text(query)

        cypher_query = """
        CALL {
            // Vector search
            CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
            YIELD node, score
            WITH collect({node: node, score: score}) AS nodes, max(score) AS max
            UNWIND nodes AS n
            RETURN n.node AS node, (n.score / max) AS score
            UNION
            // Fulltext search
            CALL db.index.fulltext.queryNodes('chunk_fulltext', $query, {limit: $top_k})
            YIELD node, score
            WITH collect({node: node, score: score}) AS nodes, max(score) AS max
            UNWIND nodes AS n
            RETURN n.node AS node, (n.score / max) AS score
        }
        WITH node, max(score) AS score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN node.text AS text,
               node.id AS chunk_id,
               score
        """

        results = self.neo4j.execute_query(cypher_query, {
            "query_embedding": query_embedding,
            "query": query,
            "top_k": top_k
        })

        return results