from typing import List, Dict, Any
from ..llm.ollama_client import OllamaClient


class AnswerCritic:
    def __init__(self):
        self.client = OllamaClient()

    def critique(self, question: str, context: List[str], answer: str) -> Dict[str, Any]:
        """
        Evalúa si la respuesta es completa y correcta.

        Returns:
            Dict con:
            - is_complete: bool
            - is_faithful: bool
            - missing_info: List[str] (preguntas adicionales si es necesario)
            - feedback: str
        """
        context_str = "\n\n".join([f"[{i + 1}] {c}" for i, c in enumerate(context)])

        system_prompt = """You are an expert at evaluating answers to questions based on provided context.

Your task is to determine:
1. Is the answer complete? (Does it fully address all parts of the question?)
2. Is the answer faithful? (Is it supported by the provided context?)
3. What information is missing, if any?

Respond ONLY with valid JSON in this format:
{
    "is_complete": true/false,
    "is_faithful": true/false,
    "missing_info": ["additional question 1", "additional question 2"],
    "feedback": "brief explanation"
}

If the answer is complete and faithful, missing_info should be an empty list."""

        user_message = f"""Question: {question}

Context:
{context_str}

Answer: {answer}

Evaluate this answer."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self.client.chat(messages, format="json")
        critique = self.client.extract_json(response)

        return critique