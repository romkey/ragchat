#!/usr/bin/env python3
"""
Minimal CLI chat with two RAG sources + Ollama.

Usage:
  python rag_chat.py "What is X?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_TEXT_FIELDS = "text,content,chunk,document,page_content"


def _truncate(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _payload_preview(payload: dict) -> str:
    try:
        return _truncate(json.dumps(payload, ensure_ascii=True))
    except (TypeError, ValueError):
        return "<unserializable payload>"


def format_http_error(
    exc: urllib.error.HTTPError,
    method: str,
    url: str,
    payload: Optional[dict] = None,
) -> str:
    body = ""
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except OSError:
        body = ""
    body = _truncate(body) if body else "<empty>"
    pieces = [
        f"{method} {url}",
        f"status={exc.code} reason={exc.reason}",
    ]
    if payload is not None:
        pieces.append(f"payload={_payload_preview(payload)}")
    pieces.append(f"response={body}")
    return "HTTP request failed: " + " | ".join(pieces)


def load_dotenv(path: Path) -> Dict[str, str]:
    """Load key=value pairs from a .env file."""
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if val and ((val[0] == val[-1]) and val[0] in {"'", '"'}):
            val = val[1:-1]
        if key:
            values[key] = val
    return values


def config_value(cli_value: Optional[str], env: Dict[str, str], key: str, default: str) -> str:
    if cli_value:
        return cli_value
    # Support docker-compose env_file and normal exported environment variables.
    runtime_value = os.environ.get(key, "")
    if runtime_value:
        return runtime_value
    return env.get(key, default)


def post_json(url: str, payload: dict, headers: Optional[dict] = None, timeout_s: int = 120) -> dict:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(format_http_error(exc, "POST", url, payload)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for POST {url}: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON response for POST {url}: {_truncate(body)}"
        ) from exc


def get_json(url: str, headers: Optional[dict] = None, timeout_s: int = 120) -> dict:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(format_http_error(exc, "GET", url, None)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for GET {url}: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON response for GET {url}: {_truncate(body)}"
        ) from exc


def ollama_base_url(api_url: str) -> str:
    """Extract base URL (scheme://host:port) from an Ollama API endpoint URL."""
    parsed = urllib.parse.urlsplit(api_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    base = api_url.split("/api/", 1)[0].rstrip("/")
    return base or "http://localhost:11434"


def ensure_ollama_model(model: str, api_url: str) -> None:
    """
    Ensure Ollama model exists on the target Ollama server.
    This function does not pull models automatically.
    It is best-effort so proxied endpoints that only expose chat/embed don't fail early.
    """
    base_url = ollama_base_url(api_url)
    show_url = f"{base_url}/api/show"
    tags_url = f"{base_url}/api/tags"
    candidates = [model]
    if ":" not in model:
        candidates.append(f"{model}:latest")

    # First: strong check via /api/show with common name variants.
    saw_404 = False
    for candidate in candidates:
        try:
            post_json(show_url, {"model": candidate}, timeout_s=60)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                saw_404 = True
                continue
            # Non-404 means endpoint exists but errored; surface it.
            raise RuntimeError(f"Failed checking Ollama model '{candidate}': HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            # Can't reach server at all: that's a real connectivity error.
            raise RuntimeError(f"Could not reach Ollama at {base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Unexpected response from Ollama while checking model '{candidate}'"
            ) from exc

    # If /api/show says missing, verify via /api/tags before failing.
    if saw_404:
        try:
            tags_data = get_json(tags_url, timeout_s=60)
            models = tags_data.get("models", [])
            names: List[str] = []
            if isinstance(models, list):
                for item in models:
                    if isinstance(item, dict):
                        name = item.get("name")
                        if isinstance(name, str):
                            names.append(name)
            wanted_base = model.split(":", 1)[0]
            for name in names:
                if name in candidates or name.split(":", 1)[0] == wanted_base:
                    return

            raise RuntimeError(
                textwrap.dedent(
                    f"""
                    Required Ollama model '{model}' is not installed on the configured Ollama server.
                    Configured endpoint: {api_url}

                    Install it on that Ollama server, then rerun:
                      ollama pull {model}

                    If this is a remote/containerized Ollama instance, run the pull command in that environment.
                    """
                ).strip()
            )
        except urllib.error.HTTPError:
            # Proxies often expose chat/embed but not management endpoints.
            print(
                f"Warning: could not verify model '{model}' via {tags_url}; continuing without preflight verification.",
                file=sys.stderr,
            )
            return
        except (urllib.error.URLError, json.JSONDecodeError):
            print(
                f"Warning: model verification endpoint unavailable for '{model}'; continuing.",
                file=sys.stderr,
            )
            return


def ollama_embed(question: str, embed_model: str, embed_url: str, timeout_s: int = 120) -> List[float]:
    """
    Create embedding using Ollama. Supports both /api/embed and /api/embeddings formats.
    """
    # Try new endpoint format first.
    try:
        data = post_json(embed_url, {"model": embed_model, "input": question}, timeout_s=timeout_s)
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            if isinstance(embeddings[0], list):
                return embeddings[0]
            if isinstance(embeddings[0], (float, int)):
                return [float(x) for x in embeddings]
    except (urllib.error.URLError, json.JSONDecodeError, KeyError):
        pass

    # Fallback to legacy /api/embeddings if caller provided /api/embed.
    fallback_url = embed_url
    if embed_url.endswith("/api/embed"):
        fallback_url = embed_url[:-len("/api/embed")] + "/api/embeddings"
    data = post_json(fallback_url, {"model": embed_model, "prompt": question}, timeout_s=timeout_s)
    vector = data.get("embedding")
    if not isinstance(vector, list):
        raise RuntimeError(f"Unexpected embedding response from Ollama: {data}")
    return [float(x) for x in vector]


def extract_payload_text(payload: dict, text_fields: Sequence[str]) -> str:
    for field in text_fields:
        val = payload.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Fallback: first string in payload.
    for _, val in payload.items():
        if isinstance(val, str) and val.strip():
            return val.strip()
    return json.dumps(payload, ensure_ascii=True)


def qdrant_search(
    qdrant_url: str,
    api_key: Optional[str],
    collection: str,
    vector: Sequence[float],
    limit: int,
    text_fields: Sequence[str],
    timeout_s: int = 120,
) -> List[Tuple[str, str, float]]:
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}/points/search"
    headers = {"api-key": api_key} if api_key else None
    data = post_json(
        url,
        {
            "vector": list(vector),
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        },
        headers=headers,
        timeout_s=timeout_s,
    )
    raw_points = data.get("result", [])
    if isinstance(raw_points, dict):
        raw_points = raw_points.get("points", [])

    results: List[Tuple[str, str, float]] = []
    for point in raw_points:
        payload = point.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        text = extract_payload_text(payload, text_fields)
        if not text:
            continue
        pid = str(point.get("id", "unknown"))
        score = float(point.get("score", 0.0))
        label = f"{collection}:id={pid}"
        results.append((label, text, score))
    return results


def build_messages(question: str, top1: Sequence[Tuple[str, str, float]], top2: Sequence[Tuple[str, str, float]]) -> List[dict]:
    """Build chat messages for Ollama API."""
    context_blocks: List[str] = []

    for label, chunk, score in top1:
        context_blocks.append(f"[RAG_SOURCE_1 | {label} | score={score:.3f}]\n{chunk}")
    for label, chunk, score in top2:
        context_blocks.append(f"[RAG_SOURCE_2 | {label} | score={score:.3f}]\n{chunk}")

    merged_context = "\n\n".join(context_blocks) if context_blocks else "No retrieval context available."

    system = textwrap.dedent(
        """
        You are a helpful assistant. Use the provided retrieval context from two RAG sources.
        Rules:
        - Prefer grounded answers from context.
        - If context is insufficient, say what is missing and provide best-effort answer.
        - Mention when the two sources disagree.
        - Keep response concise but complete.
        """
    ).strip()

    user = f"Question:\n{question}\n\nRetrieved context:\n{merged_context}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_ollama(
    messages: List[dict],
    model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.2,
    timeout_s: int = 120,
) -> str:
    """Call local Ollama /api/chat endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        data = post_json(ollama_url, payload, timeout_s=timeout_s)
        return data["message"]["content"]
    except KeyError as exc:
        raise RuntimeError(f"Unexpected Ollama response shape: {data}") from exc


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question using two Qdrant collections and Ollama."
    )
    parser.add_argument("question", help="Question to answer.")
    parser.add_argument("--env", default=".env", help="Path to .env config file (default: .env).")
    parser.add_argument("--collection1", help="Qdrant collection name for source 1.")
    parser.add_argument("--collection2", help="Qdrant collection name for source 2.")
    parser.add_argument("--model", help="Ollama chat model name.")
    parser.add_argument("--embed-model", help="Ollama embedding model name.")
    parser.add_argument("--top-k", type=int, help="Top chunks per source.")
    parser.add_argument("--ollama-url", help=f"Ollama chat endpoint (default: {DEFAULT_OLLAMA_URL}).")
    parser.add_argument("--ollama-embed-url", help=f"Ollama embedding endpoint (default: {DEFAULT_OLLAMA_EMBED_URL}).")
    parser.add_argument("--qdrant-url", help=f"Qdrant base URL (default: {DEFAULT_QDRANT_URL}).")
    parser.add_argument("--qdrant-api-key", help="Qdrant API key.")
    parser.add_argument("--text-fields", help=f"Comma-separated payload fields for text extraction (default: {DEFAULT_TEXT_FIELDS}).")
    parser.add_argument("--show-context", action="store_true", help="Print chosen chunks before final answer.")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    env = load_dotenv(Path(args.env).expanduser())

    model = config_value(args.model, env, "OLLAMA_MODEL", "llama3.2")
    embed_model = config_value(args.embed_model, env, "OLLAMA_EMBED_MODEL", model)
    ollama_url = config_value(args.ollama_url, env, "OLLAMA_CHAT_URL", DEFAULT_OLLAMA_URL)
    ollama_embed_url = config_value(args.ollama_embed_url, env, "OLLAMA_EMBED_URL", DEFAULT_OLLAMA_EMBED_URL)
    qdrant_url = config_value(args.qdrant_url, env, "QDRANT_URL", DEFAULT_QDRANT_URL)
    qdrant_api_key = config_value(args.qdrant_api_key, env, "QDRANT_API_KEY", "")
    collection1 = config_value(args.collection1, env, "QDRANT_COLLECTION_1", "")
    collection2 = config_value(args.collection2, env, "QDRANT_COLLECTION_2", "")
    top_k_raw = str(args.top_k) if args.top_k is not None else env.get("QDRANT_LIMIT", "3")
    text_fields_csv = config_value(args.text_fields, env, "QDRANT_TEXT_FIELDS", DEFAULT_TEXT_FIELDS)
    text_fields = [x.strip() for x in text_fields_csv.split(",") if x.strip()]

    try:
        top_k = int(top_k_raw)
    except ValueError:
        print(f"Invalid top-k/QDRANT_LIMIT: {top_k_raw}", file=sys.stderr)
        return 2

    if top_k < 1:
        print("--top-k must be >= 1", file=sys.stderr)
        return 2

    if not collection1:
        print("Missing collection 1. Set --collection1 or QDRANT_COLLECTION_1 in .env", file=sys.stderr)
        return 2
    if not collection2:
        print("Missing collection 2. Set --collection2 or QDRANT_COLLECTION_2 in .env", file=sys.stderr)
        return 2

    try:
        ensured = set()
        for model_name, endpoint_url in ((model, ollama_url), (embed_model, ollama_embed_url)):
            key = (model_name, ollama_base_url(endpoint_url))
            if key in ensured:
                continue
            ensure_ollama_model(model_name, endpoint_url)
            ensured.add(key)

        query_vec = ollama_embed(args.question, embed_model=embed_model, embed_url=ollama_embed_url)
        top1 = qdrant_search(
            qdrant_url=qdrant_url,
            api_key=qdrant_api_key or None,
            collection=collection1,
            vector=query_vec,
            limit=top_k,
            text_fields=text_fields,
        )
        top2 = qdrant_search(
            qdrant_url=qdrant_url,
            api_key=qdrant_api_key or None,
            collection=collection2,
            vector=query_vec,
            limit=top_k,
            text_fields=text_fields,
        )
    except RuntimeError as exc:
        print(f"Failed to retrieve RAG context: {exc}", file=sys.stderr)
        return 1

    if args.show_context:
        print("=== Selected context: Source 1 ===")
        for label, chunk, score in top1:
            print(f"\n[{label}] score={score:.3f}\n{chunk[:700]}\n")
        print("=== Selected context: Source 2 ===")
        for label, chunk, score in top2:
            print(f"\n[{label}] score={score:.3f}\n{chunk[:700]}\n")

    if not top1:
        print(f"No matches returned for collection: {collection1}", file=sys.stderr)
    if not top2:
        print(f"No matches returned for collection: {collection2}", file=sys.stderr)

    messages = build_messages(args.question, top1, top2)
    try:
        answer = call_ollama(messages, model=model, ollama_url=ollama_url)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(answer.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
