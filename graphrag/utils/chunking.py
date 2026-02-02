from typing import List
import tiktoken


def num_tokens_from_string(string: str, model: str = "qwen3:4b") -> int:
    """Cuenta el número de tokens en un string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(string))


def chunk_text(
        text: str,
        chunk_size: int,
        overlap: int,
        split_on_whitespace_only: bool = True
) -> List[str]:
    """
    Divide el texto en chunks con overlap.

    Args:
        text: Texto a dividir
        chunk_size: Tamaño máximo de cada chunk en caracteres
        overlap: Número de caracteres de overlap entre chunks
        split_on_whitespace_only: Si es True, solo divide en espacios

    Returns:
        Lista de chunks de texto
    """
    chunks = []
    index = 0

    while index < len(text):
        if split_on_whitespace_only:
            # Buscar el espacio anterior más cercano
            prev_whitespace = 0
            left_index = index - overlap

            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1

            # Buscar el próximo espacio
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)

            chunk = text[prev_whitespace:next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        else:
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            index += chunk_size

    return chunks