from typing import List, Dict, Any
from ..graph.neo4j_manager import Neo4jManager
from ..llm.ollama_client import OllamaClient
from .retriever_tools import RetrieverTools
from .retriever_router import RetrieverRouter
from .answer_critic import AnswerCritic


class AgenticRAG:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.client = OllamaClient()
        self.tools = RetrieverTools(neo4j_manager)
        self.router = RetrieverRouter(self.tools)
        self.critic = AnswerCritic()
        self.conversation_history = []

    def answer(self, question: str, max_iterations: int = 2) -> Dict[str, Any]:
        """
        Responde una pregunta usando el sistema agéntico.

        Args:
            question: Pregunta del usuario
            max_iterations: Número máximo de iteraciones de refinamiento

        Returns:
            Dict con la respuesta y metadatos
        """
        iterations = []
        current_question = question

        for iteration in range(max_iterations):
            # 1. Recuperar contexto
            retrieval_result = self.router.retrieve(
                current_question,
                self.conversation_history
            )

            context = retrieval_result["context"]

            # 2. Generar respuesta
            answer = self._generate_answer(current_question, context)

            # 3. Criticar respuesta
            critique = self.critic.critique(current_question, context, answer)

            iterations.append({
                "iteration": iteration + 1,
                "question": current_question,
                "retrieval": retrieval_result,
                "answer": answer,
                "critique": critique
            })

            # Si la respuesta es completa y fiel, terminar
            if critique["is_complete"] and critique["is_faithful"]:
                break

            # Si hay información faltante y no es la última iteración, refinar
            if critique["missing_info"] and iteration < max_iterations - 1:
                current_question = " ".join([
                    current_question,
                    "Additional questions:",
                    " ".join(critique["missing_info"])
                ])

        # Actualizar historial de conversación
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": iterations[-1]["answer"]
        })

        return {
            "question": question,
            "answer": iterations[-1]["answer"],
            "iterations": iterations,
            "final_critique": iterations[-1]["critique"]
        }

    def _generate_answer(self, question: str, context: List[str]) -> str:
        """Genera una respuesta usando el LLM."""
        context_str = "\n\n".join([f"[{i + 1}] {c}" for i, c in enumerate(context)])

        system_prompt = """You are a helpful assistant that answers questions based on provided context.

Rules:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Cite the context by referencing the source numbers [1], [2], etc.
4. Be concise but complete"""

        user_message = f"""Context:
{context_str}

Question: {question}

Answer:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        return self.client.chat(messages)

    def reset_conversation(self):
        """Reinicia el historial de conversación."""
        self.conversation_history = []