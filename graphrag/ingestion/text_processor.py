import uuid
from typing import List, Dict, Any
from tqdm import tqdm
from ..graph.neo4j_manager import Neo4jManager
from ..utils.chunking import chunk_text
from ..utils.embeddings import EmbeddingGenerator
from .entity_extractor import EntityExtractor


class TextProcessor:
    def __init__(
            self,
            neo4j_manager: Neo4jManager,
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            entity_types: List[str] = None,
    ):
        self.neo4j = neo4j_manager
        self.embedding_gen = EmbeddingGenerator()
        self.entity_extractor = EntityExtractor(entity_types=entity_types)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(
            self,
            text: str,
            document_id: str = None,
            metadata: Dict[str, Any] = None
    ):
        """
        Procesa un documento: lo divide en chunks, extrae entidades y relaciones,
        y lo almacena en Neo4j.
        """
        document_id = document_id or str(uuid.uuid4())
        metadata = metadata or {}

        # 1. Crear nodo de documento
        self._create_document_node(document_id, metadata)

        # 2. Dividir en chunks
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        print(f"Documento dividido en {len(chunks)} chunks")

        # 3. Procesar cada chunk
        for i, chunk in enumerate(tqdm(chunks, desc="Procesando chunks")):
            chunk_id = f"{document_id}_chunk_{i}"
            self._process_chunk(chunk_id, chunk, document_id, i)

        # 4. Consolidar entidades y relaciones
        self._consolidate_entities()
        self._consolidate_relationships()

        print(f"Documento {document_id} procesado exitosamente")

    def _create_document_node(self, document_id: str, metadata: Dict[str, Any]):
        """Crea un nodo de documento en Neo4j."""
        query = """
        MERGE (d:Document {id: $document_id})
        SET d += $metadata
        """
        self.neo4j.execute_query(query, {
            "document_id": document_id,
            "metadata": metadata
        })

    def _process_chunk(
            self,
            chunk_id: str,
            text: str,
            document_id: str,
            index: int
    ):
        """Procesa un chunk individual."""
        # Generar embedding
        embedding = self.embedding_gen.embed_text(text)

        # Crear nodo de chunk
        query = """
        MATCH (d:Document {id: $document_id})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.embedding = $embedding,
            c.index = $index
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding,
            "index": index,
            "document_id": document_id
        })

        # Extraer entidades y relaciones
        entities, relationships = self.entity_extractor.extract_entities_and_relationships(text)

        # Almacenar entidades
        for entity in entities:
            self._store_entity(entity, chunk_id)

        # Almacenar relaciones
        for rel in relationships:
            self._store_relationship(rel, chunk_id)

    def _store_entity(self, entity: Dict[str, Any], chunk_id: str):
        """Almacena una entidad y la conecta al chunk, manejando descripciones faltantes."""

        # Usamos .get() con un valor por defecto para evitar KeyErrors
        name = entity.get("name")
        entity_type = entity.get("type", "UNKNOWN")
        description = entity.get("description", "No description provided")

        if not name:
            return  # Si no hay nombre, no podemos crear el nodo

        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $type
        SET e.description = CASE 
            WHEN e.description IS NULL THEN [$description]
            ELSE e.description + [$description]
        END
        MERGE (c)-[:HAS_ENTITY]->(e)
        """
        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "name": name,
            "type": entity_type,
            "description": description
        })

    def _store_relationship(self, rel: Dict[str, Any], chunk_id: str):
        """Almacena una relación entre entidades con manejo de errores."""

        # Extraer campos con valores por defecto para evitar KeyError
        source = rel.get("source")
        target = rel.get("target")
        rel_type = rel.get("type", "RELATED_TO")
        description = rel.get("description", "No description available")
        strength = rel.get("strength", 0.5)

        # Validación crítica: Una relación necesita obligatoriamente origen y destino
        if not source or not target:
            print(f"Advertencia: Relación incompleta omitida en chunk {chunk_id}")
            return

        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (source:Entity {name: $source})
        MERGE (target:Entity {name: $target})
        MERGE (source)-[r:RELATIONSHIP {type: $type}]->(target)
        SET r.description = CASE 
            WHEN r.description IS NULL THEN [$description]
            ELSE r.description + [$description]
        END,
        r.strength = CASE 
            WHEN r.strength IS NULL THEN [$strength]
            ELSE r.strength + [$strength]
        END
        """
        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "source": source,
            "target": target,
            "type": rel_type,
            "description": description,
            "strength": strength
        })

    def _consolidate_entities(self):
        """Consolida las descripciones de las entidades."""
        query = """
        MATCH (e:Entity)
        WHERE size(e.description) > 1
        RETURN e.name AS name, e.description AS descriptions
        """
        entities = self.neo4j.execute_query(query)

        for entity in tqdm(entities, desc="Consolidando entidades"):
            summary = self.entity_extractor.summarize_entity(
                entity["name"],
                entity["descriptions"]
            )

            update_query = """
            MATCH (e:Entity {name: $name})
            SET e.summary = $summary
            """
            self.neo4j.execute_query(update_query, {
                "name": entity["name"],
                "summary": summary
            })

    def _consolidate_relationships(self):
        """Consolida las descripciones de las relaciones."""
        query = """
        MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
        WHERE size(r.description) > 1
        RETURN s.name AS source, t.name AS target, r.description AS descriptions, r.strength AS strengths
        """
        relationships = self.neo4j.execute_query(query)

        for rel in tqdm(relationships, desc="Consolidando relaciones"):
            summary = self.entity_extractor.summarize_relationship(
                rel["source"],
                rel["target"],
                rel["descriptions"]
            )

            avg_strength = sum(rel["strengths"]) / len(rel["strengths"])

            update_query = """
            MATCH (s:Entity {name: $source})-[r:RELATIONSHIP]->(t:Entity {name: $target})
            SET r.summary = $summary,
                r.avg_strength = $avg_strength
            """
            self.neo4j.execute_query(update_query, {
                "source": rel["source"],
                "target": rel["target"],
                "summary": summary,
                "avg_strength": avg_strength
            })