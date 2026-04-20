"""
Tests for the RAGEvaluator.

Unit tests mock the LLM and Neo4j to run without external services.
Integration tests (marked @pytest.mark.integration) require a live Neo4j
instance and a running Ollama server — they share the same fixture used in
test_agents.py so the Einstein dataset is already available.
"""
import os
import json
from typing import cast
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd

from graphrag.evaluation import RAGEvaluator


def _make_evaluator(mock_rag=None, mock_neo4j=None):
    """Build a RAGEvaluator with fully mocked dependencies.

    OllamaClient is patched so no settings/env-vars are required.
    """
    rag = mock_rag or MagicMock()
    neo4j = mock_neo4j or MagicMock()
    with patch("graphrag.evaluation.evaluator.OllamaClient"):
        evaluator = RAGEvaluator(rag, neo4j)
    evaluator.client = MagicMock()
    return evaluator


# ---------------------------------------------------------------------------
# Unit tests – load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_loads_csv_with_semicolon_delimiter(self, tmp_path):
        csv_file = tmp_path / "bench.csv"
        csv_file.write_text("question;cypher\nHello;RETURN \"hi\"\n", encoding="utf-8")

        evaluator = _make_evaluator()
        df = evaluator.load_dataset(str(csv_file))

        assert list(df.columns) == ["question", "cypher"]
        assert len(df) == 1
        assert df.iloc[0]["question"] == "Hello"

    def test_loads_real_benchmark_file(self):
        bench_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "benchmark_data.csv"
        )
        evaluator = _make_evaluator()
        df = evaluator.load_dataset(bench_path)

        assert "question" in df.columns
        assert "cypher" in df.columns
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Unit tests – get_answer
# ---------------------------------------------------------------------------

class TestGetAnswer:
    def test_returns_answer_and_context_on_success(self):
        mock_rag = MagicMock()
        mock_rag.answer.return_value = {
            "answer": "Paris",
            "iterations": [
                {
                    "retrieval": {
                        "tool": "vector_search",
                        "context": ["France's capital is Paris."],
                    }
                }
            ],
            "final_critique": {},
        }
        evaluator = _make_evaluator(mock_rag=mock_rag)
        answer, context = evaluator.get_answer("What is the capital of France?")

        assert answer == "Paris"
        assert context == ["France's capital is Paris."]

    def test_returns_none_on_exception(self):
        mock_rag = MagicMock()
        mock_rag.answer.side_effect = RuntimeError("LLM offline")
        evaluator = _make_evaluator(mock_rag=mock_rag)
        answer, context = evaluator.get_answer("anything")

        assert answer is None
        assert context == []


# ---------------------------------------------------------------------------
# Unit tests – evaluate_context_recall
# ---------------------------------------------------------------------------

class TestEvaluateContextRecall:
    def _setup(self, llm_payload: dict):
        evaluator = _make_evaluator()
        mock_client = cast(MagicMock, evaluator.client)
        mock_client.chat.return_value = json.dumps(llm_payload)
        mock_client.extract_json.return_value = llm_payload
        return evaluator

    def test_returns_recall_score_from_llm(self):
        payload = {
            "sentences": ["Einstein won the Nobel Prize in 1921."],
            "attributions": [1],
            "recall": 1.0,
            "reasoning": "The context confirms the Nobel Prize.",
        }
        evaluator = self._setup(payload)
        result = evaluator.evaluate_context_recall(
            "When did Einstein win the Nobel Prize?",
            "Einstein won the Nobel Prize in 1921.",
            ["Einstein received the Nobel Prize in Physics in 1921."],
        )
        assert result["recall"] == 1.0

    def test_computes_recall_from_attributions_when_missing(self):
        payload = {"sentences": ["A.", "B."], "attributions": [1, 0]}
        evaluator = self._setup(payload)
        result = evaluator.evaluate_context_recall("q", "A. B.", ["context"])
        assert result["recall"] == pytest.approx(0.5)

    def test_defaults_to_zero_when_no_attributions(self):
        payload = {"sentences": [], "attributions": []}
        evaluator = self._setup(payload)
        result = evaluator.evaluate_context_recall("q", "", [])
        assert result["recall"] == 0.0


# ---------------------------------------------------------------------------
# Unit tests – evaluate_faithfulness
# ---------------------------------------------------------------------------

class TestEvaluateFaithfulness:
    def _setup(self, step1_payload: dict, step2_payload: dict):
        evaluator = _make_evaluator()
        mock_client = cast(MagicMock, evaluator.client)
        mock_client.chat.side_effect = [
            json.dumps(step1_payload),
            json.dumps(step2_payload),
        ]
        mock_client.extract_json.side_effect = [step1_payload, step2_payload]
        return evaluator

    def test_perfect_faithfulness(self):
        statements = {"statements": ["Einstein was born in Ulm.", "He worked in Bern."]}
        verdicts = {"verdicts": [1, 1], "reasoning": ["confirmed", "confirmed"]}
        evaluator = self._setup(statements, verdicts)
        result = evaluator.evaluate_faithfulness(
            "Where was Einstein born?", "Einstein was born in Ulm. He worked in Bern.", ["..."]
        )
        assert result["faithfulness"] == pytest.approx(1.0)

    def test_partial_faithfulness(self):
        statements = {"statements": ["A.", "B.", "C."]}
        verdicts = {"verdicts": [1, 0, 1], "reasoning": ["ok", "not found", "ok"]}
        evaluator = self._setup(statements, verdicts)
        result = evaluator.evaluate_faithfulness("q", "A. B. C.", ["ctx"])
        assert result["faithfulness"] == pytest.approx(2 / 3)

    def test_empty_answer_returns_zero(self):
        evaluator = _make_evaluator()
        mock_client = cast(MagicMock, evaluator.client)
        mock_client.chat.return_value = json.dumps({"statements": []})
        mock_client.extract_json.return_value = {"statements": []}
        result = evaluator.evaluate_faithfulness("q", "", [])
        assert result["faithfulness"] == 0.0
        assert result["statements"] == []

    def test_two_llm_calls_are_made(self):
        statements = {"statements": ["X."]}
        verdicts = {"verdicts": [1], "reasoning": ["ok"]}
        evaluator = self._setup(statements, verdicts)
        evaluator.evaluate_faithfulness("q", "X.", ["ctx"])
        assert cast(MagicMock, evaluator.client).chat.call_count == 2


# ---------------------------------------------------------------------------
# Unit tests – evaluate_answer_correctness
# ---------------------------------------------------------------------------

class TestEvaluateAnswerCorrectness:
    def _setup(self, answer_stmts, truth_stmts, classification):
        evaluator = _make_evaluator()
        mock_client = cast(MagicMock, evaluator.client)
        mock_client.chat.side_effect = [
            json.dumps({"statements": answer_stmts}),
            json.dumps({"statements": truth_stmts}),
            json.dumps(classification),
        ]
        mock_client.extract_json.side_effect = [
            {"statements": answer_stmts},
            {"statements": truth_stmts},
            classification,
        ]
        return evaluator

    def test_perfect_correctness(self):
        clf = {
            "classifications": [{"statement": "A.", "category": "TP", "reason": "matches"}],
            "tp_count": 1,
            "fp_count": 0,
            "fn_count": 0,
        }
        evaluator = self._setup(["A."], ["A."], clf)
        result = evaluator.evaluate_answer_correctness("q", "A.", "A.")
        assert result["answer_correctness"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_zero_correctness_all_fp(self):
        clf = {
            "classifications": [{"statement": "Wrong.", "category": "FP", "reason": "not in GT"}],
            "tp_count": 0,
            "fp_count": 1,
            "fn_count": 1,
        }
        evaluator = self._setup(["Wrong."], ["Correct."], clf)
        result = evaluator.evaluate_answer_correctness("q", "Wrong.", "Correct.")
        assert result["answer_correctness"] == pytest.approx(0.0)
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 1

    def test_f1_calculation(self):
        # TP=2, FP=1, FN=1 → precision=2/3, recall=2/3, F1=2/3
        clf = {"tp_count": 2, "fp_count": 1, "fn_count": 1, "classifications": []}
        evaluator = self._setup(["A.", "B.", "C."], ["A.", "B.", "D."], clf)
        result = evaluator.evaluate_answer_correctness("q", "A. B. C.", "A. B. D.")
        assert result["answer_correctness"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Unit tests – run_benchmark
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def test_output_columns_present(self):
        mock_neo4j = MagicMock()
        mock_neo4j.execute_query.return_value = [{"value": "Paris"}]

        mock_rag = MagicMock()
        mock_rag.answer.return_value = {
            "answer": "Paris",
            "iterations": [{"retrieval": {"tool": "vector_search", "context": ["ctx"]}}],
            "final_critique": {},
        }

        evaluator = _make_evaluator(mock_rag=mock_rag, mock_neo4j=mock_neo4j)
        df_input = pd.DataFrame(
            {"question": ["What is Paris?"], "cypher": ['RETURN "Paris"']}
        )
        results = evaluator.run_benchmark(df_input)

        assert set(["ground_truth", "answer", "latency", "retrieved_contexts"]).issubset(
            results.columns
        )
        assert results.iloc[0]["answer"] == "Paris"

    def test_handles_agent_failure_gracefully(self):
        mock_neo4j = MagicMock()
        mock_neo4j.execute_query.return_value = []

        mock_rag = MagicMock()
        mock_rag.answer.side_effect = RuntimeError("crash")

        evaluator = _make_evaluator(mock_rag=mock_rag, mock_neo4j=mock_neo4j)
        df_input = pd.DataFrame(
            {"question": ["bad question"], "cypher": ['RETURN "x"']}
        )
        results = evaluator.run_benchmark(df_input)
        assert results.iloc[0]["answer"] is None
        assert results.iloc[0]["retrieved_contexts"] == []

    def test_latency_is_non_negative(self):
        mock_neo4j = MagicMock()
        mock_neo4j.execute_query.return_value = []
        mock_rag = MagicMock()
        mock_rag.answer.return_value = {
            "answer": "ok",
            "iterations": [{"retrieval": {"tool": "vector_search", "context": []}}],
            "final_critique": {},
        }
        evaluator = _make_evaluator(mock_rag=mock_rag, mock_neo4j=mock_neo4j)
        df_input = pd.DataFrame({"question": ["q"], "cypher": ['RETURN "x"']})
        results = evaluator.run_benchmark(df_input)
        assert results.iloc[0]["latency"] >= 0


# ---------------------------------------------------------------------------
# Unit tests – evaluate_results
# ---------------------------------------------------------------------------

class TestEvaluateResults:
    def _df_with_one_row(self):
        return pd.DataFrame(
            {
                "question": ["q"],
                "cypher": ['RETURN "gt"'],
                "ground_truth": ["gt"],
                "answer": ["ans"],
                "latency": [0.5],
                "retrieved_contexts": [["ctx"]],
            }
        )

    def test_score_columns_added(self):
        evaluator = _make_evaluator()

        with (
            patch.object(evaluator, "evaluate_context_recall", return_value={"recall": 0.8}),
            patch.object(evaluator, "evaluate_faithfulness", return_value={"faithfulness": 0.9}),
            patch.object(evaluator, "evaluate_answer_correctness", return_value={"answer_correctness": 0.7}),
        ):
            result = evaluator.evaluate_results(self._df_with_one_row())

        assert "context_recall" in result.columns
        assert "faithfulness" in result.columns
        assert "answer_correctness" in result.columns
        assert result.iloc[0]["context_recall"] == pytest.approx(0.8)

    def test_missing_answers_filled_with_i_dont_know(self):
        """Rows with NaN answers should be replaced before scoring."""
        df = pd.DataFrame(
            {
                "question": ["q"],
                "cypher": ['RETURN "gt"'],
                "ground_truth": ["gt"],
                "answer": [None],
                "latency": [0.0],
                "retrieved_contexts": [["ctx"]],
            }
        )
        evaluator = _make_evaluator()

        with (
            patch.object(evaluator, "evaluate_context_recall", return_value={"recall": 0.0}),
            patch.object(evaluator, "evaluate_faithfulness", return_value={"faithfulness": 0.0}),
            patch.object(evaluator, "evaluate_answer_correctness", return_value={"answer_correctness": 0.0}) as mock_corr,
        ):
            evaluator.evaluate_results(df)

        # evaluate_answer_correctness received "I don't know", not None
        assert mock_corr.call_args[0][1] == "I don't know"


# ---------------------------------------------------------------------------
# Unit tests – print_summary
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_prints_without_error(self, capsys):
        df = pd.DataFrame(
            {
                "answer_correctness": [0.7, 0.8],
                "context_recall": [0.9, 0.6],
                "faithfulness": [1.0, 0.95],
                "latency": [1.2, 0.8],
            }
        )
        evaluator = _make_evaluator()
        evaluator.print_summary(df)
        captured = capsys.readouterr()
        assert "answer_correctness" in captured.out
        assert "faithfulness" in captured.out
        assert "context_recall" in captured.out

    def test_tolerates_missing_metric_columns(self):
        df = pd.DataFrame({"latency": [1.0]})
        evaluator = _make_evaluator()
        evaluator.print_summary(df)  # should not raise


# ---------------------------------------------------------------------------
# Integration tests (require Neo4j + Ollama)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def evaluation_system():
    """
    Full system fixture: ingest the Einstein test text into Neo4j,
    return a configured RAGEvaluator.  Cleaned up after the module.
    """
    pytest.importorskip("neo4j", reason="neo4j package not installed")

    from graphrag.graph.neo4j_manager import Neo4jManager
    from graphrag.agents import AgenticRAG
    from graphrag.ingestion.text_processor import TextProcessor

    neo4j = Neo4jManager()
    neo4j.create_constraints()
    neo4j.create_vector_index()

    processor = TextProcessor(neo4j, chunk_size=200, chunk_overlap=20)
    test_text = (
        "Albert Einstein was a theoretical physicist who developed the theory of "
        "relativity. He was born in Ulm, Germany in 1879 and later worked at the "
        "Swiss Patent Office in Bern. Einstein received the Nobel Prize in Physics "
        "in 1921 for his work on the photoelectric effect."
    )
    processor.process_document(test_text, document_id="eval_test_data")

    rag = AgenticRAG(neo4j)
    evaluator = RAGEvaluator(rag, neo4j)

    yield evaluator

    neo4j.execute_query("MATCH (n) DETACH DELETE n")
    neo4j.close()


@pytest.mark.integration
class TestIntegrationFullPipeline:
    def test_get_answer_returns_non_empty_string(self, evaluation_system):
        answer, _ = evaluation_system.get_answer("Where was Einstein born?")
        assert answer is not None
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_context_recall_score_in_range(self, evaluation_system):
        _, context = evaluation_system.get_answer("Where was Einstein born?")
        result = evaluation_system.evaluate_context_recall(
            "Where was Einstein born?",
            "Einstein was born in Ulm, Germany.",
            context or [""],
        )
        assert 0.0 <= result.get("recall", 0.0) <= 1.0

    def test_faithfulness_score_in_range(self, evaluation_system):
        answer, context = evaluation_system.get_answer("What did Einstein develop?")
        result = evaluation_system.evaluate_faithfulness(
            "What did Einstein develop?",
            answer or "I don't know",
            context or [""],
        )
        assert 0.0 <= result.get("faithfulness", 0.0) <= 1.0

    def test_answer_correctness_score_in_range(self, evaluation_system):
        answer, _ = evaluation_system.get_answer("When did Einstein receive the Nobel Prize?")
        result = evaluation_system.evaluate_answer_correctness(
            "When did Einstein receive the Nobel Prize?",
            answer or "I don't know",
            "Einstein received the Nobel Prize in Physics in 1921.",
        )
        assert 0.0 <= result.get("answer_correctness", 0.0) <= 1.0

    def test_run_benchmark_returns_correct_shape(self, evaluation_system):
        df = pd.DataFrame(
            {
                "question": [
                    "Hello",
                    "How many PERSON entities are in the knowledge graph?",
                    "Where was Einstein born?",
                ],
                "cypher": [
                    'RETURN "greeting"',
                    'MATCH (e:Entity {type: "PERSON"}) RETURN count(e) AS personCount',
                    'MATCH (e:Entity) WHERE toLower(e.name) CONTAINS "einstein" RETURN e.description LIMIT 1',
                ],
            }
        )
        results = evaluation_system.run_benchmark(df)
        assert len(results) == 3
        assert "ground_truth" in results.columns
        assert "answer" in results.columns
        assert "latency" in results.columns

    def test_evaluate_results_adds_score_columns(self, evaluation_system):
        df = pd.DataFrame(
            {
                "question": ["Where was Einstein born?"],
                "cypher": [
                    'MATCH (e:Entity) WHERE toLower(e.name) CONTAINS "einstein" RETURN e.description LIMIT 1'
                ],
                "ground_truth": ["Einstein was born in Ulm, Germany in 1879."],
                "answer": ["Einstein was born in Ulm."],
                "latency": [1.0],
                "retrieved_contexts": [["Einstein was born in Ulm, Germany."]],
            }
        )
        scored = evaluation_system.evaluate_results(df)
        for col in ("context_recall", "faithfulness", "answer_correctness"):
            assert col in scored.columns
            assert 0.0 <= scored.iloc[0][col] <= 1.0

    def test_full_benchmark_pipeline(self, evaluation_system):
        """End-to-end: load CSV → run_benchmark → evaluate_results → print_summary."""
        bench_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "benchmark_data.csv"
        )
        dataset = evaluation_system.load_dataset(bench_path)
        results = evaluation_system.run_benchmark(dataset)
        scored = evaluation_system.evaluate_results(results)
        evaluation_system.print_summary(scored)

        assert len(scored) == len(dataset)
        assert "answer_correctness" in scored.columns
