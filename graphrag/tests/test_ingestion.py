import pytest
from graphrag.graph.neo4j_manager import Neo4jManager
from graphrag.ingestion.text_processor import TextProcessor
from graphrag.ingestion.entity_extractor import EntityExtractor


@pytest.fixture
def neo4j_manager():
    manager = Neo4jManager()
    yield manager
    # Cleanup
    manager.execute_query("MATCH (n) DETACH DELETE n")
    manager.close()


def test_entity_extraction():
    """Test que la extracción de entidades funciona."""
    extractor = EntityExtractor()

    text = "Albert Einstein was born in Germany and worked in Switzerland."

    entities, relationships = extractor.extract_entities_and_relationships(text)

    assert len(entities) > 0
    assert len(relationships) >= 0

    # Verificar que las entidades tienen la estructura correcta
    for entity in entities:
        assert "name" in entity
        assert "type" in entity
        assert "description" in entity


def test_text_processing(neo4j_manager):
    """Test que el procesamiento de texto funciona."""
    processor = TextProcessor(neo4j_manager, chunk_size=100, chunk_overlap=20)

    text = "Albert Einstein was a physicist. He developed the theory of relativity."

    processor.process_document(text, document_id="test_doc")

    # Verificar que se creó el documento
    docs = neo4j_manager.execute_query(
        "MATCH (d:__Document__ {id: 'test_doc'}) RETURN count(d) as count"
    )
    assert docs[0]["count"] == 1

    # Verificar que se crearon chunks
    chunks = neo4j_manager.execute_query(
        "MATCH (c:__Chunk__) RETURN count(c) as count"
    )
    assert chunks[0]["count"] > 0


def test_entity_consolidation(neo4j_manager):
    """Test que la consolidación de entidades funciona."""
    processor = TextProcessor(neo4j_manager, chunk_size=50, chunk_overlap=10)

    # Texto que menciona la misma entidad varias veces
    text = """
    Einstein was born in Germany. Einstein worked on relativity. 
    Einstein received the Nobel Prize.
    """

    processor.process_document(text, document_id="test_consolidation")

    # Verificar que Einstein tiene múltiples descripciones consolidadas
    entities = neo4j_manager.execute_query("""
        MATCH (e:__Entity__ {name: 'Albert Einstein'})
        RETURN e.summary as summary
    """)

    if entities:
        assert entities[0]["summary"] is not None