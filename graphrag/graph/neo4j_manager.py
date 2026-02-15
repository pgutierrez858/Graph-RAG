from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from ..config import get_settings


class Neo4jManager:
    def __init__(self):
        settings = get_settings()
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )

    def close(self):
        """Cierra la conexión con Neo4j."""
        self.driver.close()

    def execute_query(
            self,
            query: str,
            parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Ejecuta una query en Neo4j."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def create_constraints(self):
        """Crea las constraints necesarias en Neo4j."""
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                self.execute_query(constraint)
            except Exception as e:
                print(f"Constraint ya existe o error: {e}")

    def create_vector_index(self, index_name: str = "chunk_embeddings"):
        """Crea un índice vectorial."""
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (c:Chunk)
        ON c.embedding
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: 768,
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        try:
            self.execute_query(query)
        except Exception as e:
            print(f"Índice vectorial ya existe o error: {e}")

    def get_schema(self) -> Dict[str, Any]:
        """Obtiene el schema del grafo."""
        node_props_query = """
        CALL db.schema.nodeTypeProperties()
        YIELD nodeType, propertyName, propertyTypes
        WITH nodeType, collect({property: propertyName, type: propertyTypes[0]}) as properties
        RETURN {labels: nodeType, properties: properties} AS output
        """

        rel_props_query = """
        CALL db.schema.relTypeProperties()
        YIELD relType, propertyName, propertyTypes
        WITH relType, collect({property: propertyName, type: propertyTypes[0]}) as properties
        RETURN {type: relType, properties: properties} AS output
        """

        rel_query = """
        CALL db.schema.visualization()
        YIELD nodes, relationships
        UNWIND relationships as rel
        RETURN {start: startNode(rel).name, type: type(rel), end: endNode(rel).name} AS output
        """

        try:
            node_props = self.execute_query(node_props_query)
            rel_props = self.execute_query(rel_props_query)
            relationships = self.execute_query(rel_query)

            return {
                "node_props": {item["output"]["labels"]: item["output"]["properties"]
                               for item in node_props},
                "rel_props": {item["output"]["type"]: item["output"]["properties"]
                              for item in rel_props},
                "relationships": [item["output"] for item in relationships]
            }
        except Exception as e:
            print(f"Error obteniendo schema: {e}")
            return {"node_props": {}, "rel_props": {}, "relationships": []}

    @staticmethod
    def format_schema(schema: Dict[str, Any]) -> str:
        """Formatea el schema para el prompt."""

        def format_props(props):
            return ", ".join([f"{p['property']}: {p['type']}" for p in props])

        formatted_node_props = [
            f"{label} {{{format_props(props)}}}"
            for label, props in schema["node_props"].items()
        ]

        formatted_rel_props = [
            f"{rel_type} {{{format_props(props)}}}"
            for rel_type, props in schema["rel_props"].items()
        ]

        formatted_rels = [
            f"(:{rel['start']})-[:{rel['type']}]->(:{rel['end']})"
            for rel in schema["relationships"]
        ]

        return "\n".join([
            "Node labels and properties:",
            "\n".join(formatted_node_props),
            "\nRelationship types and properties:",
            "\n".join(formatted_rel_props),
            "\nThe relationships:",
            "\n".join(formatted_rels),
        ])