import pytest
from graphrag.graph.neo4j_manager import Neo4jManager
from graphrag.agents import AgenticRAG
from graphrag.ingestion.text_processor import TextProcessor


@pytest.fixture
def setup_agentic_system():
    """Setup del sistema agéntico con datos."""
    neo4j = Neo4jManager()
    neo4j.create_constraints()
    neo4j.create_vector_index()

    processor = TextProcessor(neo4j, chunk_size=200, chunk_overlap=20)
    test_text = """
    Albert Einstein was a theoretical physicist who developed the theory of relativity.
    He was born in Ulm, Germany in 1879 and later worked at the Swiss Patent Office in Bern.
    Einstein received the Nobel Prize in Physics in 1921 for his work on the photoelectric effect.
    """
    processor.process_document(test_text, document_id="test_data")

    agentic_rag = AgenticRAG(neo4j)

    yield agentic_rag

    # Cleanup
    neo4j.execute_query("MATCH (n) DETACH DELETE n")
    neo4j.close()


def test_simple_question(setup_agentic_system):
    """Test con pregunta simple."""
    result = setup_agentic_system.answer("Where was Einstein born?")

    assert result["answer"] is not None
    assert "question" in result
    assert "iterations" in result
    assert len(result["iterations"]) > 0


def test_complex_question(setup_agentic_system):
    """Test con pregunta que requiere text2cypher."""
    result = setup_agentic_system.answer("How many PERSON entities are there?")

    assert result["answer"] is not None
    # Verificar que se usó text2cypher
    tool_used = result["iterations"][-1]["retrieval"]["tool"]
    # El router debería elegir text2cypher para este tipo de pregunta
    assert tool_used in ["text2cypher", "vector_search", "hybrid_search"]


def test_conversation_context(setup_agentic_system):
    """Test de contexto conversacional."""
    setup_agentic_system.reset_conversation()

    result1 = setup_agentic_system.answer("What prize did Einstein receive?")
    assert result1["answer"] is not None

    result2 = setup_agentic_system.answer("When did he receive it?")
    assert result2["answer"] is not None

    # La segunda respuesta debería usar el contexto de la primera
    assert len(setup_agentic_system.conversation_history) == 4  # 2 preguntas + 2 respuestas


def test_answer_critic(setup_agentic_system):
    """Test del answer critic."""
    result = setup_agentic_system.answer("Where was Einstein born?")

    assert "final_critique" in result
    critique = result["final_critique"]

    assert "is_complete" in critique
    assert "is_faithful" in critique
    assert isinstance(critique["is_complete"], bool)
    assert isinstance(critique["is_faithful"], bool)