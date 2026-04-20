"""
Root conftest.

1. Fallback env vars so Settings() can be instantiated without a .env file.
   Real values in .env / the shell environment always take priority because
   os.environ.setdefault is a no-op when the key is already present.

2. Integration tests require explicit opt-in via --integration flag:
     uv run pytest                  → unit tests only  (fast, no services needed)
     uv run pytest --integration    → unit + integration tests (Neo4j + Ollama required)

   If --integration is passed but a service is not reachable, those tests are
   skipped with a clear message instead of hanging.
"""
import os
import socket
import pytest

# ---------------------------------------------------------------------------
# 1. Fallback settings
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "NEO4J_URI": "neo4j://127.0.0.1:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "password",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_MODEL": "qwen3:4b",
    "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
}

for _key, _value in _DEFAULTS.items():
    os.environ.setdefault(_key, _value)


# ---------------------------------------------------------------------------
# 2. --integration flag and service availability check
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires Neo4j and Ollama).",
    )


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    run_integration = config.getoption("--integration")

    if not run_integration:
        skip = pytest.mark.skip(reason="integration test — pass --integration to run")
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip)
        return

    # --integration passed: skip only if services are unreachable
    neo4j_up = _port_open("localhost", 7687)
    ollama_up = _port_open("localhost", 11434)
    if neo4j_up and ollama_up:
        return

    missing = []
    if not neo4j_up:
        missing.append("Neo4j :7687")
    if not ollama_up:
        missing.append("Ollama :11434")
    reason = f"service(s) not reachable: {', '.join(missing)}"

    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
