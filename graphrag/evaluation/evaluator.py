from typing import List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime
from ..llm.ollama_client import OllamaClient
from ..agents import AgenticRAG


class RAGEvaluator:
    def __init__(self, agentic_rag: AgenticRAG):
        self.rag = agentic_rag
        self.client = OllamaClient()

    def evaluate_context_recall(
            self,
            question: str,
            ground_truth: str,
            retrieved_context: List[str]
    ) -> Dict[str, Any]:
        """
        Evalúa qué tan bien el contexto recuperado cubre la respuesta correcta.
        """
        context_str = "\n".join(retrieved_context)

        system_prompt = """You are evaluating context recall for a RAG system.

Given a question, ground truth answer, and retrieved context, determine which sentences 
in the ground truth can be attributed to the context.

Respond with JSON in this format:
{
    "total_sentences": <number>,
    "attributed_sentences": <number>,
    "recall": <float between 0 and 1>,
    "reasoning": "brief explanation"
}"""

        user_message = f"""Question: {question}

Ground Truth Answer: {ground_truth}

Retrieved Context:
{context_str}

Evaluate context recall."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self.client.chat(messages, format="json")
        result = self.client.extract_json(response)

        return result

    def evaluate_faithfulness(
            self,
            question: str,
            answer: str,
            context: List[str]
    ) -> Dict[str, Any]:
        """
        Evalúa si la respuesta es fiel al contexto proporcionado.
        """
        context_str = "\n".join(context)

        # Paso 1: Descomponer la respuesta en declaraciones atómicas
        decompose_prompt = """Break down the following answer into atomic, self-contained statements.
Each statement should be independently verifiable.

Respond with JSON:
{
    "statements": ["statement 1", "statement 2", ...]
}"""

        decompose_messages = [
            {"role": "system", "content": decompose_prompt},
            {"role": "user", "content": f"Answer: {answer}"}
        ]

        decompose_response = self.client.chat(decompose_messages, format="json")
        statements_data = self.client.extract_json(decompose_response)
        statements = statements_data.get("statements", [])

        # Paso 2: Verificar cada declaración contra el contexto
        verify_prompt = """You are verifying if statements can be inferred from the given context.

For each statement, return 1 if it can be directly inferred from the context, 0 otherwise.

Respond with JSON:
{
    "verdicts": [1, 0, 1, ...],
    "reasoning": ["reason 1", "reason 2", ...]
}"""

        statements_str = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(statements)])

        verify_messages = [
            {"role": "system", "content": verify_prompt},
            {"role": "user", "content": f"""Context:
{context_str}

Statements to verify:
{statements_str}"""}
        ]

        verify_response = self.client.chat(verify_messages, format="json")
        verification = self.client.extract_json(verify_response)

        verdicts = verification.get("verdicts", [])
        faithfulness_score = sum(verdicts) / len(verdicts) if verdicts else 0.0

        return {
            "statements": statements,
            "verdicts": verdicts,
            "faithfulness": faithfulness_score,
            "reasoning": verification.get("reasoning", [])
        }

    def evaluate_answer_correctness(
            self,
            question: str,
            answer: str,
            ground_truth: str
    ) -> Dict[str, Any]:
        """
        Evalúa la corrección de la respuesta comparándola con el ground truth.
        """

        # Descomponer ambas respuestas en declaraciones
        def get_statements(text: str) -> List[str]:
            prompt = """Break down the following text into atomic statements.

Respond with JSON:
{
    "statements": ["statement 1", "statement 2", ...]
}"""
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
            response = self.client.chat(messages, format="json")
            data = self.client.extract_json(response)
            return data.get("statements", [])

        answer_statements = get_statements(answer)
        truth_statements = get_statements(ground_truth)

        # Clasificar cada declaración
        classify_prompt = """Classify each statement from the answer as:
- TP (true positive): Present in answer and supported by ground truth
- FP (false positive): Present in answer but not supported by ground truth
- FN (false negative): Present in ground truth but missing from answer

Respond with JSON:
{
    "classifications": [
        {"statement": "...", "category": "TP/FP/FN", "reason": "..."}
    ],
    "tp_count": <number>,
    "fp_count": <number>,
    "fn_count": <number>
}"""

        answer_str = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(answer_statements)])
        truth_str = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(truth_statements)])

        messages = [
            {"role": "system", "content": classify_prompt},
            {"role": "user", "content": f"""Answer statements:
{answer_str}

Ground truth statements:
{truth_str}"""}
        ]

        response = self.client.chat(messages, format="json")
        classification = self.client.extract_json(response)

        tp = classification.get("tp_count", 0)
        fp = classification.get("fp_count", 0)
        fn = classification.get("fn_count", 0)

        # Calcular métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "classifications": classification.get("classifications", []),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def evaluate_dataset(
            self,
            test_cases: List[Dict[str, str]]
    ) -> pd.DataFrame:
        """
        Evalúa el sistema RAG en un conjunto de casos de prueba.

        Args:
            test_cases: Lista de dicts con keys: 'question', 'ground_truth'

        Returns:
            DataFrame con resultados de evaluación
        """
        results = []

        for i, test_case in enumerate(test_cases):
            print(f"Evaluando caso {i + 1}/{len(test_cases)}: {test_case['question']}")

            question = test_case['question']
            ground_truth = test_case['ground_truth']

            # Obtener respuesta del sistema
            start_time = datetime.now()
            rag_result = self.rag.answer(question)
            latency = (datetime.now() - start_time).total_seconds()

            answer = rag_result['answer']
            context = rag_result['iterations'][-1]['retrieval']['context']

            # Evaluar
            context_recall = self.evaluate_context_recall(question, ground_truth, context)
            faithfulness = self.evaluate_faithfulness(question, answer, context)
            correctness = self.evaluate_answer_correctness(question, answer, ground_truth)

            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'answer': answer,
                'latency': latency,
                'context_recall': context_recall.get('recall', 0.0),
                'faithfulness': faithfulness.get('faithfulness', 0.0),
                'answer_f1': correctness.get('f1_score', 0.0),
                'precision': correctness.get('precision', 0.0),
                'recall': correctness.get('recall', 0.0),
                'num_iterations': len(rag_result['iterations']),
                'tool_used': rag_result['iterations'][-1]['retrieval']['tool']
            })

        df = pd.DataFrame(results)

        # Calcular estadísticas agregadas
        print("\n=== Resultados de Evaluación ===")
        print(f"Context Recall promedio: {df['context_recall'].mean():.3f}")
        print(f"Faithfulness promedio: {df['faithfulness'].mean():.3f}")
        print(f"Answer F1 promedio: {df['answer_f1'].mean():.3f}")
        print(f"Latencia promedio: {df['latency'].mean():.2f}s")

        return df