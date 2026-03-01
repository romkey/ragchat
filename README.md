# Two-Collection Qdrant RAG Chat (Ollama)

Simple command-line chat program that:

1. Takes a question as a command-line argument.
2. Retrieves relevant chunks from **two Qdrant collections**.
3. Sends those chunks to an LLM via **Ollama**.
4. Prints the answer.

## Requirements

- Python 3.9+
- Ollama running locally (chat + embeddings endpoint)
- At least one model pulled (example: `llama3.2`)
- Qdrant running with two collections that store payload text

## Quick Start

1. Start Ollama:

```bash
ollama serve
```

2. Create `.env` in the project directory:

```env
OLLAMA_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_URL=http://localhost:11434/api/chat
OLLAMA_EMBED_URL=http://localhost:11434/api/embed

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_1=collection_one
QDRANT_COLLECTION_2=collection_two
QDRANT_LIMIT=3
QDRANT_TEXT_FIELDS=text,content,chunk,document,page_content
```

3. Run:

```bash
python rag_chat.py "What are the main differences between source A and source B?"
```

## CLI options

```bash
python rag_chat.py "your question" \
  [--env .env] \
  [--collection1 <qdrant-collection>] \
  [--collection2 <qdrant-collection>] \
  [--model llama3.2] \
  [--embed-model nomic-embed-text] \
  [--top-k 3] \
  [--ollama-url http://localhost:11434/api/chat] \
  [--ollama-embed-url http://localhost:11434/api/embed] \
  [--qdrant-url http://localhost:6333] \
  [--qdrant-api-key <key>] \
  [--text-fields text,content,chunk,document,page_content] \
  [--show-context]
```

## Notes

- The script generates the question embedding via Ollama, then uses vector search in each Qdrant collection.
- The script auto-checks Ollama models and pulls missing chat/embedding models before answering.
- `QDRANT_TEXT_FIELDS` controls which payload fields are checked for text context (first matching string is used).
- CLI flags override `.env` values when both are present.
- Use `--show-context` to inspect selected chunks from each collection.

## Docker

Build locally:

```bash
docker build -t rag-chat:local .
```

Run:

```bash
docker run --rm \
  --env-file .env \
  rag-chat:local \
  "your question here"
```

You can also pass flags after the question, for example:

```bash
docker run --rm --env-file .env rag-chat:local \
  "your question here" --top-k 5 --show-context
```

## GitHub Actions Docker Publish

Workflow file: `.github/workflows/docker-publish.yml`

- Publishes to `ghcr.io/<owner>/<repo>`.
- Triggers only on pushes to version tags matching `v*` (example: `v1.2.3`).
- Uses the pushed Git tag as the Docker tag and also publishes `latest`.
- Runs only once per release push (no duplicate `main` + `tag` build), because branch pushes are not configured as triggers.
