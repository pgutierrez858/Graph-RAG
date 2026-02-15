import pytest
from graphrag.graph.neo4j_manager import Neo4jManager
from graphrag.retrieval.vector_retriever import VectorRetriever, HybridRetriever
from graphrag.retrieval.text2cypher import Text2CypherRetriever
from graphrag.ingestion.text_processor import TextProcessor


@pytest.fixture
def setup_database():
    """Setup con datos de prueba."""
    neo4j = Neo4jManager()
    neo4j.create_constraints()
    neo4j.create_vector_index()

    # Cargar datos de prueba
    processor = TextProcessor(neo4j, chunk_size=200, chunk_overlap=20)
    test_text = """
    Albert Einstein was a theoretical physicist. He developed the theory of relativity.
    Einstein was born in Ulm, Germany in 1879. He worked at the Swiss Patent Office.
    """
    processor.process_document(test_text, document_id="test_einstein")

    yield neo4j

    # Cleanup
    neo4j.execute_query("MATCH (n) DETACH DELETE n")
    neo4j.close()


def test_vector_retrieval(setup_database):
    """Test de vector retrieval."""
    retriever = VectorRetriever(setup_database)

    results = retriever.retrieve("Where was Einstein born?", top_k=3)

    assert len(results) > 0
    assert "text" in results[0]
    assert "score" in results[0]
    assert results[0]["score"] > 0


def test_hybrid_retrieval(setup_database):
    """Test de hybrid retrieval."""
    retriever = HybridRetriever(setup_database)

    results = retriever.retrieve("Einstein Germany", top_k=3)

    assert len(results) > 0
    assert "text" in results[0]
    assert "score" in results[0]


def test_text2cypher(setup_database):
    """Test de text2cypher."""
    retriever = Text2CypherRetriever(setup_database)

    cypher, results = retriever.retrieve("How many entities are there?")

    assert cypher is not None
    assert "MATCH" in cypher.upper()
    # Los resultados pueden estar vacíos si la query no es válida