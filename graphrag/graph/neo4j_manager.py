import logging
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from ..config import get_settings

# Suppress verbose GQL status notifications from the Neo4j driver
# (e.g. "The field `propertyTypes` will change output format in the next major version.")
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


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
        """
        Crea las constraints de unicidad para todos los labels de dominio.

        Nodos de infraestructura:
          - Document  → id
          - Chunk     → id

        Nodos de dominio (usan 'name' como clave natural, excepto Publication):
          - Person, ScientificTheory, Institution, Location, Award, ScientificField → name
          - Publication → title  (los títulos son la clave natural de un paper)
        """
        constraints = [
            # Infraestructura
            "CREATE CONSTRAINT document_id       IF NOT EXISTS FOR (d:Document)         REQUIRE d.id    IS UNIQUE",
            "CREATE CONSTRAINT chunk_id          IF NOT EXISTS FOR (c:Chunk)             REQUIRE c.id    IS UNIQUE",
            # Dominio — clave: name
            "CREATE CONSTRAINT person_name       IF NOT EXISTS FOR (p:Person)            REQUIRE p.name  IS UNIQUE",
            "CREATE CONSTRAINT theory_name       IF NOT EXISTS FOR (t:ScientificTheory)  REQUIRE t.name  IS UNIQUE",
            "CREATE CONSTRAINT institution_name  IF NOT EXISTS FOR (i:Institution)       REQUIRE i.name  IS UNIQUE",
            "CREATE CONSTRAINT location_name     IF NOT EXISTS FOR (l:Location)          REQUIRE l.name  IS UNIQUE",
            "CREATE CONSTRAINT award_name        IF NOT EXISTS FOR (a:Award)             REQUIRE a.name  IS UNIQUE",
            "CREATE CONSTRAINT field_name        IF NOT EXISTS FOR (f:ScientificField)   REQUIRE f.name  IS UNIQUE",
            # Dominio — clave: title
            "CREATE CONSTRAINT publication_title IF NOT EXISTS FOR (p:Publication)       REQUIRE p.title IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                self.execute_query(constraint)
            except Exception as e:
                print(f"Constraint ya existe o error: {e}")

    def create_vector_index(self, index_name: str = "chunk_embeddings"):
        """Crea un índice vectorial sobre los embeddings de los chunks."""
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

    def create_fulltext_index(self, index_name: str = "chunk_fulltext"):
        """Crea un índice de texto completo sobre el texto de los chunks."""
        query = f"""
        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
        FOR (c:Chunk)
        ON EACH [c.text]
        """
        try:
            self.execute_query(query)
        except Exception as e:
            print(f"Índice fulltext ya existe o error: {e}")

    def get_schema(self) -> Dict[str, Any]:
        """
        Obtiene el schema del grafo.

        Devuelve un diccionario con tres claves:
          - node_props    : {label → [{property, type}]}
          - rel_props     : {rel_type → [{property, type}]}
          - relationships : [{start, type, end}]
        """
        node_props_query = """
        CALL db.schema.nodeTypeProperties()
        YIELD nodeType, propertyName, propertyTypes
        WITH nodeType, collect({property: propertyName, type: propertyTypes[0]}) AS properties
        RETURN {labels: nodeType, properties: properties} AS output
        """

        rel_props_query = """
        CALL db.schema.relTypeProperties()
        YIELD relType, propertyName, propertyTypes
        WITH relType, collect({property: propertyName, type: propertyTypes[0]}) AS properties
        RETURN {type: relType, properties: properties} AS output
        """

        rel_query = """
        CALL db.schema.visualization()
        YIELD nodes, relationships
        UNWIND relationships AS rel
        RETURN {start: startNode(rel).name, type: type(rel), end: endNode(rel).name} AS output
        """

        try:
            node_props    = self.execute_query(node_props_query)
            rel_props     = self.execute_query(rel_props_query)
            relationships = self.execute_query(rel_query)

            return {
                "node_props": {
                    item["output"]["labels"]: item["output"]["properties"]
                    for item in node_props
                },
                "rel_props": {
                    item["output"]["type"]: item["output"]["properties"]
                    for item in rel_props
                },
                "relationships": [item["output"] for item in relationships],
            }
        except Exception as e:
            print(f"Error obteniendo schema: {e}")
            return {"node_props": {}, "rel_props": {}, "relationships": []}

    @staticmethod
    def format_schema(schema: Dict[str, Any]) -> str:
        """
        Formatea el schema como texto plano para incluirlo en un prompt LLM.

        Ejemplo de salida:
            Node labels and properties:
            Person {name: String, nationality: String, birth_year: Long, ...}
            ...
            Relationship types and properties:
            WORKED_AT {from_year: Long, to_year: Long, role: String}
            ...
            The relationships:
            (:Person)-[:WORKED_AT]->(:Institution)
            ...
        """
        def format_props(props: List[Dict[str, Any]]) -> str:
            return ", ".join(f"{p['property']}: {p['type']}" for p in props)

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