# UAE Law RAG

This repo provides a full local stack (Milvus + Attu + Backend + Frontend) via Docker Compose.

Quick links:
- `docs/DEVELOPMENT_GUIDE.md` (full dev guide)
- `infra/docker/docker-compose.full.yml` (full stack compose)

## Docker Quickstart (from clone to chat)

### 1) Clone

```bash
git clone <your-repo>
cd uae_law_rag
```

### 2) Optional .env (remote providers + local reranker defaults)

If you use DashScope/OpenAI/DeepSeek or local reranker:

```
DASHSCOPE_API_KEY=your_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

RERANKER_MODEL_PATH=/Volumes/Workspace/Projects/RAG/tiic/models/bge-reranker-v2-m3
RERANKER_DEVICE=mps
RERANKER_TOP_N=10
```

### 3) Start full stack

```bash
docker compose -f infra/docker/docker-compose.full.yml up -d
```

### 4) Initialize DB

```bash
docker compose -f infra/docker/docker-compose.full.yml exec backend \
  bash -lc "PYTHONPATH=src python -m uae_law_rag.backend.scripts.init_db --drop --seed --seed-fts"
```

### 5) Update KB embedding config (optional)

Only needed if you change embedding provider/model (e.g., DashScope):

```bash
docker compose -f infra/docker/docker-compose.full.yml exec backend \
  bash -lc "python - <<'PY'\nimport sqlite3\nconn = sqlite3.connect('/app/.Local/uae_law_rag.db')\ncur = conn.cursor()\ncur.execute(\"update knowledge_base set embed_provider=?, embed_model=?, embed_dim=? where id='default'\", ('dashscope', 'text-embedding-v4', 1024))\nconn.commit()\nprint('kb updated')\nPY"
```

### 6) Initialize Milvus

`--embed-dim` must match `knowledge_base.embed_dim`:

```bash
docker compose -f infra/docker/docker-compose.full.yml exec backend \
  bash -lc "PYTHONPATH=src python -m uae_law_rag.backend.scripts.init_milvus --collection kb_default --embed-dim 1024 --metric-type COSINE --drop"
```

### 7) Set Run Config (global defaults)

```bash
docker compose -f infra/docker/docker-compose.full.yml exec backend \
  bash -lc "PYTHONPATH=src python -m uae_law_rag.backend.scripts.set_run_config"
```

To override defaults:

```bash
docker compose -f infra/docker/docker-compose.full.yml exec backend \
  bash -lc "PYTHONPATH=src python -m uae_law_rag.backend.scripts.set_run_config \
    --config-json '{\"model_provider\":\"dashscope\",\"model_name\":\"qwen3-max\",\"fusion_strategy\":\"rrf\"}' \
    --merge"
```

### 8) Open UI + Chat

- Frontend: http://localhost:5173/
- Backend health: http://localhost:18000/api/health

Send a chat (optional curl):

```bash
curl -sS -X POST http://127.0.0.1:18000/api/chat \
  -H 'Content-Type: application/json' \
  -H 'x-user-id: dev-user' \
  --data-binary @- <<'JSON' | python -m json.tool
{
  "query": "Your question here",
  "kb_id": "default",
  "debug": true
}
JSON
```

---

For more details (non-docker, pipeline steps, config priority), see `docs/DEVELOPMENT_GUIDE.md`.
