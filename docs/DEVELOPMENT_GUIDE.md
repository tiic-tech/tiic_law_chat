# DEVELOPMENT_GUIDE — UAE Law RAG

本指南面向开发者，覆盖从零环境到完整链路运行的全部步骤：依赖 → 配置 → 初始化 → 启动 → ingest/chat → 回放 → 重置 → 测试记录 → Docker 部署。

> 所有 Python 命令统一加：`PYTHONPATH=src`

---

## 0. 仓库结构速览

- `src/uae_law_rag/backend/`: FastAPI + pipelines（ingest/retrieval/generation/evaluator）
- `src/uae_law_rag/frontend/`: Vite + React 前端
- `infra/milvus/`: Milvus + Attu Docker Compose
- `playground/`: pytest gate 测试
- `docs/DEV_QUICKSTART_M1.md`: M1 最小闭环验证（已验证可用命令）

---

## 1. 运行依赖（必须）

### 1.1 系统依赖

- Python 3.11+（见 `pyproject.toml`）
- Node.js + pnpm（前端）
- Docker + Docker Compose（Milvus + Attu）
- `sqlite3` CLI（可选，便于验证 DB）

### 1.2 Python 依赖安装

推荐在虚拟环境内安装（最小可用闭环）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[backend,db,llamaindex-basic,parsing]"
```

可选扩展（按需）：

- `llamaindex-advance`：高级检索/本地模型
- `eval`：评估相关工具

### 1.3 前端依赖安装

```bash
cd src/uae_law_rag/frontend
pnpm install
```

---

## 2. Docker 依赖（Milvus + Attu）

项目内置 Milvus Compose：

```bash
cd infra/milvus
docker compose up -d
```

验证：

```bash
docker ps | grep milvus
nc -vz 127.0.0.1 19530
```

Attu UI（可选）：

- http://localhost:8000
- 连接地址：`host.docker.internal:19530`（非容器内时）

---

## 3. System Config（重点）

本系统存在多层配置来源，需要明确 **注入方式与优先级**。

### 3.1 配置优先级（Chat 相关）

Chat 服务的关键配置读取顺序（`chat_service._resolve_value`）：

1) **Request context**（`POST /api/chat` 的 `context` 字段）  
2) **KB 配置**（数据库表 `knowledge_base`）  
3) **Conversation settings**（`conversation.settings`）  
4) **Run Config**（数据库表 `run_config`）  
5) **默认值**（代码内默认）

### 3.2 `.env`（项目根目录）

`src/uae_law_rag/config.py` 会读取 **项目根目录** 下的 `.env`（仅作用于 Settings）：

关键字段（部分）：

- `LOCAL_MODELS`：是否默认使用本地模型（影响 LLM 默认 provider）
- `DEBUG`
- `PROJECT_ROOT`
- `UAE_LAW_RAG_DATABASE_URL`
- `UAE_LAW_RAG_DATA_RAW_PATH`
- `UAE_LAW_RAG_DATA_PARSED_PATH`
- `UAE_LAW_RAG_SAMPLE_PDF`
- `OPENAI_API_KEY`, `OPENAI_API_BASE`
- `DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`
- `QWEN_CHAT_MODEL`, `QWEN_MULTI_MODEL`, `QWEN_EMBED_MODEL`
- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_CHAT_MODEL`, `DEEPSEEK_REASONER_MODEL`
- `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_REQUEST_TIMEOUT_S`
- `DEVICE`
- `RERANKER_MODEL_PATH`
- `RERANKER_DEVICE`
- `RERANKER_TOP_N`

注意：

- Settings 只会读取 **自己定义的字段**。  
- Provider 相关字段（OpenAI/DashScope/DeepSeek）会被自动写入 `os.environ`，确保下游 SDK 可直接读取。  
- Milvus/Debug 等环境变量仍需显式 `export`（不通过 Settings 注入）。

### 3.3 Run Config（全局默认配置）

Run Config 是一张 **全局配置表**（`run_config`），用于保存系统的默认运行参数。  
它会在 **chat/ingest 请求时自动读取**，作为默认值基线。

写入方式：

```bash
PYTHONPATH=src python -m uae_law_rag.backend.scripts.set_run_config
```

可选合并（覆盖部分字段）：

```bash
PYTHONPATH=src python -m uae_law_rag.backend.scripts.set_run_config \
  --config-json '{"rerank_strategy":"none","rerank_top_k":5}' \
  --merge
```

完全自定义（传入完整 JSON）：

```bash
PYTHONPATH=src python -m uae_law_rag.backend.scripts.set_run_config \
  --config-json '{
    "model_provider": "dashscope",
    "model_name": "qwen3-max",
    "generation_config": {
      "temperature": 0.2,
      "top_p": 0.9
    },
    "keyword_top_k": 100,
    "vector_top_k": 30,
    "fusion_top_k": 20,
    "fusion_strategy": "rrf",
    "rerank_strategy": "bge_reranker",
    "rerank_model": "/Volumes/Workspace/Projects/RAG/tiic/models/bge-reranker-v2-m3",
    "rerank_config": {"device": "mps"},
    "rerank_top_k": 10
  }'
```

Run Config 支持的核心字段（默认快照）：

- Retrieval：`keyword_top_k`, `vector_top_k`, `fusion_top_k`, `rerank_top_k`,
  `fusion_strategy`, `rerank_strategy`, `rerank_model`, `rerank_config`, `metric_type`
- Generation：`model_provider`, `model_name`, `prompt_name`, `prompt_version`,
  `generation_config`, `prompt_config`, `postprocess_config`, `no_evidence_use_llm`
- Evaluator：`evaluator_config`
- Embed（默认值，KB 优先）：`embed_provider`, `embed_model`
- Ingest：`parser`, `parse_version`, `segment_version`

### 3.4 运行时环境变量（进程级）

以下变量通过 `os.getenv` 读取，必须在启动前显式 `export`：

- Milvus 连接：
  - `MILVUS_URI`（推荐，例如 `http://127.0.0.1:19530`）
  - 或 `MILVUS_HOST` + `MILVUS_PORT`
- DB 连接：
  - `UAE_LAW_RAG_DATABASE_URL` 或 `DATABASE_URL`
- 第三方模型 Key（remote provider）：
  - `OPENAI_API_KEY`
  - `DASHSCOPE_API_KEY`
  - `DEEPSEEK_API_KEY`
- 调试：
  - `UAE_LAW_RAG_DEBUG_DB=1`（打印 PRAGMA database_list）
  - `UAE_LAW_RAG_DEBUG_TRACEBACK=1`（500 错误时打印 traceback）
  - `SQL_ECHO=1`（SQLAlchemy echo）
- 数据目录：
  - `UAE_LAW_RAG_DATA_DIR`（覆盖 `.data` 根目录）

示例（本地 Milvus + SQLite）：

```bash
export MILVUS_URI="http://127.0.0.1:19530"
export UAE_LAW_RAG_DATABASE_URL="sqlite+aiosqlite:////absolute/path/to/.Local/uae_law_rag.db"
```

### 3.5 前端 Vite 配置

前端只读取 `src/uae_law_rag/frontend` 目录下 `.env.*`：

- `VITE_API_BASE`（默认 `/api`）
- `VITE_BACKEND_TARGET`（Vite proxy 目标，默认 `http://127.0.0.1:18000`）
- `VITE_SERVICE_MODE`（默认 `live`）

### 3.6 Provider / 模型切换方式

#### 3.6.1 Embedding Provider（检索向量）

来源：`knowledge_base` 表字段

- `embed_provider`
- `embed_model`
- `embed_dim`

默认 KB（`init_db --seed`）：

- `embed_provider=local`（本地 hash embedding）
- `embed_model=bge-small`
- `embed_dim=384`

切换示例（SQLite）：

```bash
sqlite3 .Local/uae_law_rag.db <<'SQL'
update knowledge_base
set embed_provider='ollama',
    embed_model='YOUR_EMBED_MODEL',
    embed_dim=YOUR_EMBED_DIM
where id='default';
SQL
```

Embedding provider allowlist（服务端硬编码）：

```
hash | local | mock | ollama | openai | dashscope | qwen
```

#### 3.6.2 LLM Provider（生成模型）

默认值：

- `LOCAL_MODELS=true` → `model_provider=ollama`，`model_name=OLLAMA_CHAT_MODEL`
- `LOCAL_MODELS=false` → `model_provider=dashscope`，`model_name=qwen3-max`

可在 `POST /api/chat` 的 `context` 中覆盖：

```bash
curl -sS -X POST http://127.0.0.1:18000/api/chat \
  -H 'Content-Type: application/json' \
  -H 'x-user-id: dev-user' \
  --data-binary @- <<'JSON' | python -m json.tool
{
  "query": "Explain rental rules",
  "kb_id": "default",
  "context": {
    "model_provider": "ollama",
    "model_name": "qwen2.5:1.5b",
    "generation_config": {
      "temperature": 0.2,
      "top_p": 0.9
    }
  },
  "debug": true
}
JSON
```

LLM provider allowlist（服务端硬编码）：

```
ollama | openai | dashscope | qwen | huggingface | hf | deepseek
openai_like | openai-like | mock | local | hash
```

本地模型说明：

- `ollama` 依赖本机 Ollama 服务（默认端口 11434）
- `local/hash/mock` 为本地确定性输出，仅用于离线/测试

DashScope 说明：

- 需要 `.env` 或环境变量提供 `DASHSCOPE_API_KEY`
- 推荐设置 `DASHSCOPE_BASE_URL`（若使用兼容模式地址）

#### 3.6.3 可覆盖的 Chat Context 字段（HTTP）

`ChatRequest.context` 支持（见 `schemas_http/chat.py`）：

- 检索：`keyword_top_k`, `vector_top_k`, `fusion_top_k`, `rerank_top_k`
- 策略：`fusion_strategy`, `rerank_strategy`
- 精排：`rerank_model`, `rerank_config`
- Embed：`embed_provider`, `embed_model`, `embed_dim`
- LLM：`model_provider`, `model_name`
- Prompt：`prompt_name`, `prompt_version`
- Evaluator：`evaluator_config`

额外支持（`extra="allow"`），如：

- `temperature`, `generation_config`, `prompt_config`, `postprocess_config`
- `output_fields`, `metric_type`, `file_id`, `document_id`
- `no_evidence_use_llm`

特殊说明：

- `vector_top_k=0` 会禁用向量检索（仅关键词检索）
- `rerank_top_k` 会映射为 LlamaIndex `SentenceTransformerRerank` 的 `top_n`

#### 3.6.4 Ingest 配置快照（KB / Request）

- KB 侧：`knowledge_base.chunking_config`
  - 常用字段：`window_size`, `window_metadata_key`, `original_text_metadata_key`
- Request 侧：`IngestRequest.ingest_profile`
  - `parser`（仅支持 `pymupdf4llm`）
  - `parse_version`
  - `segment_version`

#### 3.6.5 embed_dim 与 Milvus 初始化顺序（统一原则）

统一原则：

- **KB 的 `embed_dim` 与 Milvus collection 的 `embed_dim` 必须一致**
- 若修改 `embed_dim`，必须 **drop 并重建** Milvus collection，再重新 ingest

推荐顺序：

1) 先确定 embedding 模型与其维度（参照模型文档或本地验证）
2) 更新 KB 的 `embed_provider / embed_model / embed_dim`
3) 使用相同 `embed_dim` 执行 `init_milvus`
4) 执行 ingest（或重建数据）

#### 3.6.6 DashScope 快速启用（LLM + Embedding）

1) 确保 Key 注入：

```bash
# 方式 A：.env（推荐）
DASHSCOPE_API_KEY=your_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 方式 B：命令行显式注入
export DASHSCOPE_API_KEY=your_key
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

2) 更新 KB 的 embedding 配置（示例）：

```bash
sqlite3 .Local/uae_law_rag.db <<'SQL'
update knowledge_base
set embed_provider='dashscope',
    embed_model='text-embedding-v4',
    embed_dim=EMBED_DIM
where id='default';
SQL
```

> `EMBED_DIM` 必须与实际模型输出维度一致（可通过模型文档或本地调用确认）。

3) 使用相同 `embed_dim` 初始化 Milvus：

```bash
PYTHONPATH=src python -m uae_law_rag.backend.scripts.init_milvus \
  --collection kb_default \
  --embed-dim EMBED_DIM \
  --metric-type COSINE \
  --drop
```

4) Chat 时指定 DashScope LLM：

```bash
curl -sS -X POST http://127.0.0.1:18000/api/chat \
  -H 'Content-Type: application/json' \
  -H 'x-user-id: dev-user' \
  --data-binary @- <<'JSON' | python -m json.tool
{
  "query": "YOUR QUERY",
  "kb_id": "default",
  "context": {
    "model_provider": "dashscope",
    "model_name": "qwen3-max",
    "generation_config": {
      "temperature": 0.2
    }
  },
  "debug": true
}
JSON
```

#### 3.6.7 本地 Reranker 默认配置（.env）

若需要避免 HF 在线下载，可直接在 `.env` 中配置：

```
RERANKER_MODEL_PATH=/Volumes/Workspace/Projects/RAG/tiic/models/bge-reranker-v2-m3
RERANKER_DEVICE=mps
RERANKER_TOP_N=10
```

行为说明：

- 会话未显式设置时，默认启用 `rerank_strategy=bge_reranker`
- `RERANKER_MODEL_PATH` 会作为默认 `rerank_model`
- `RERANKER_DEVICE` 会作为默认 `rerank_config.device`
- `RERANKER_TOP_N` 会作为默认 `rerank_top_k`（并映射为 `SentenceTransformerRerank.top_n`）

---

## 4. 初始化系统（DB + Milvus）

### 4.1 初始化 SQLite（含默认 KB + FTS）

默认 DB 位置：`.Local/uae_law_rag.db`（可通过 `UAE_LAW_RAG_DATABASE_URL` 覆盖）

```bash
PYTHONPATH=src python -m uae_law_rag.backend.scripts.init_db --drop --seed --seed-fts
```

校验：

```bash
sqlite3 .Local/uae_law_rag.db "select kb_name, milvus_collection, embed_dim from knowledge_base;"
```

### 4.2 更新 KB 配置（Embed Provider/Model/Dim）

如果使用 dashscope embed model，使用以下bash命令

```bash
sqlite3 .Local/uae_law_rag.db <<'SQL'
update knowledge_base
set embed_provider='dashscope',
    embed_model='text-embedding-v4',
    embed_dim=1024
where id='default';
SQL
```

> 仅当你需要切换 embedding provider/model 时才需要更新此步。

### 4.3 初始化 Milvus Collection

```bash
PYTHONPATH=src python -m uae_law_rag.backend.scripts.init_milvus \
  --collection kb_default \
  --embed-dim EMBED_DIM \
  --metric-type COSINE \
  --drop
```

> `EMBED_DIM` 必须与 KB 配置一致（默认未改 KB 时为 384）。

### 4.4 写入 Run Config（全局默认值）

```bash
PYTHONPATH=src python -m uae_law_rag.backend.scripts.set_run_config
```

完全自定义（传入完整 JSON）：

```bash
PYTHONPATH=src python -m uae_law_rag.backend.scripts.set_run_config \
  --config-json '{
    "model_provider": "dashscope",
    "model_name": "qwen3-max",
    "generation_config": {
      "temperature": 0.2,
      "top_p": 0.9
    },
    "keyword_top_k": 100,
    "vector_top_k": 30,
    "fusion_top_k": 20,
    "fusion_strategy": "rrf",
    "rerank_strategy": "bge_reranker",
    "rerank_model": "/Volumes/Workspace/Projects/RAG/tiic/models/bge-reranker-v2-m3",
    "rerank_config": {"device": "mps"},
    "rerank_top_k": 10
  }'
```

> 建议顺序：init_db → 更新 KB → init_milvus → set_run_config → ingest

---

## 5. 启动后端与前端

### 5.1 启动后端（FastAPI）

```bash
PYTHONPATH=src uvicorn uae_law_rag.backend.main:app --host 127.0.0.1 --port 18000 --reload
```

健康检查：

```bash
curl -sS http://127.0.0.1:18000/api/health | python -m json.tool
```

### 5.2 启动前端（Vite）

```bash
cd src/uae_law_rag/frontend
pnpm dev
```

访问：http://localhost:5173/

---

## 6. Ingest Pipeline（导入）

说明：

- 当前 parser 仅支持 `pymupdf4llm`（需安装 `pymupdf` + `pymupdf4llm`，已包含在 `parsing` extra 中）
- `source_uri` 必须是后端可访问的 **绝对路径**

### 6.1 Dry Run

```bash
curl -sS -X POST "http://127.0.0.1:18000/api/ingest?debug=true" \
  -H "Content-Type: application/json" \
  -H "x-user-id: dev-user" \
  --data-binary @- <<'JSON' | python -m json.tool
{
  "kb_id": "default",
  "source_uri": "/ABSOLUTE/PATH/TO/demo.pdf",
  "file_name": "demo.pdf",
  "dry_run": true,
  "ingest_profile": { "parser": "pymupdf4llm" }
}
JSON
```

### 6.2 真正写入

```bash
curl -sS -X POST "http://127.0.0.1:18000/api/ingest?debug=true" \
  -H "Content-Type: application/json" \
  -H "x-user-id: dev-user" \
  --data-binary @- <<'JSON' | python -m json.tool
{
  "kb_id": "default",
  "source_uri": "/ABSOLUTE/PATH/TO/demo.pdf",
  "file_name": "demo.pdf",
  "dry_run": false,
  "ingest_profile": { "parser": "pymupdf4llm" }
}
JSON
```

### 6.3 DB 校验

```bash
sqlite3 .Local/uae_law_rag.db "select count(*) from node;"
sqlite3 .Local/uae_law_rag.db "select count(*) from node_vector_map;"
```

---

## 7. Chat Pipeline（检索 → 生成 → 评估）

```bash
curl -sS -X POST http://127.0.0.1:18000/api/chat \
  -H 'Content-Type: application/json' \
  -H 'x-user-id: dev-user' \
  --data-binary @- <<'JSON' | python -m json.tool
{
  "query": "YOUR QUERY",
  "kb_id": "default",
  "debug": true
}
JSON
```

预期：

- `status` 为 `success/partial/blocked`
- `citations` 有值（blocked 可能为 0）
- `debug` 中包含 `retrieval_record_id / generation_record_id / evaluation_record_id`

---

## 8. Records 回放

```bash
curl -sS http://127.0.0.1:18000/api/records/retrieval/<RETRIEVAL_RECORD_ID> | python -m json.tool
curl -sS http://127.0.0.1:18000/api/records/generation/<GENERATION_RECORD_ID> | python -m json.tool
curl -sS http://127.0.0.1:18000/api/records/evaluation/<EVALUATION_RECORD_ID> | python -m json.tool
```

---

## 9. 重置数据库（全链路）

```bash
# reset db
PYTHONPATH=src python -m uae_law_rag.backend.scripts.init_db --drop --seed --rebuild-fts

# reset milvus
PYTHONPATH=src python -m uae_law_rag.backend.scripts.init_milvus \
  --collection kb_default --embed-dim 384 --metric-type COSINE --drop
```

如需清空 Milvus Docker volumes：

```bash
cd infra/milvus
docker compose down -v
```

---

## 10. 测试与 Gate

### 10.1 Backend Pytest

```bash
PYTHONPATH=src pytest
```

### 10.2 Frontend PNPM

```bash
cd src/uae_law_rag/frontend
pnpm lint
pnpm typecheck
pnpm test
```

---

## 11. Pytest Record

未运行（未要求）。

---

## 12. PNPM Record

未运行（未要求）。

---

## 13. Docker 部署（整体打包）

当前仓库已提供完整 Docker 配置，位于 `infra/docker/`。

### 13.1 Backend Dockerfile（`infra/docker/Dockerfile.backend`）

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml uv.lock /app/
COPY src /app/src
RUN pip install --no-cache-dir -e ".[backend,db,llamaindex-basic,parsing]"
ENV PYTHONPATH=/app/src
EXPOSE 18000
CMD ["uvicorn", "uae_law_rag.backend.main:app", "--host", "0.0.0.0", "--port", "18000"]
```

### 13.2 Frontend Dockerfile（`infra/docker/Dockerfile.frontend`）

```dockerfile
FROM node:20-slim

WORKDIR /app
COPY src/uae_law_rag/frontend/package.json /app/
COPY src/uae_law_rag/frontend/pnpm-lock.yaml /app/
RUN corepack enable && pnpm install

COPY src/uae_law_rag/frontend /app

EXPOSE 5173
CMD ["pnpm", "dev", "--host", "0.0.0.0", "--port", "5173"]
```

### 13.3 全量 Compose（`infra/docker/docker-compose.full.yml`）

```yaml
name: uae-law-rag-full

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.16
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    command: >
      etcd
      -advertise-client-urls=http://0.0.0.0:2379
      -listen-client-urls=http://0.0.0.0:2379
      --data-dir=/etcd
    volumes:
      - etcd_data:/etcd
    ports:
      - "2379:2379"
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 10s
      timeout: 5s
      retries: 12

  minio:
    image: minio/minio:RELEASE.2024-12-18T13-15-44Z
    environment:
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-minioadmin}
      - MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-minioadmin}
    command: minio server /minio_data --console-address ":9001"
    volumes:
      - minio_data:/minio_data
    ports:
      - "9000:9000"
      - "9001:9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 12

  milvus:
    image: milvusdb/milvus:v2.4.16
    command: ["milvus", "run", "standalone"]
    depends_on:
      etcd:
        condition: service_healthy
      minio:
        condition: service_healthy
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-minioadmin}
      - MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-minioadmin}
      - COMMON_SECURITY_TLSMODE=0
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - milvus_data:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 10s
      timeout: 5s
      retries: 24

  attu:
    image: zilliz/attu:latest
    depends_on:
      milvus:
        condition: service_healthy
    environment:
      - MILVUS_URL=milvus:19530
    ports:
      - "8000:3000"
    restart: unless-stopped

  backend:
    build:
      context: ../..
      dockerfile: infra/docker/Dockerfile.backend
    environment:
      - MILVUS_URI=http://milvus:19530
      - UAE_LAW_RAG_DATABASE_URL=sqlite+aiosqlite:////app/.Local/uae_law_rag.db
      - UAE_LAW_RAG_DATA_DIR=/app/.data
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - OPENAI_API_BASE=${OPENAI_API_BASE:-}
      - DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:-}
      - DASHSCOPE_BASE_URL=${DASHSCOPE_BASE_URL:-}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-}
      - DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-}
    volumes:
      - ../../.Local:/app/.Local
      - ../../.data:/app/.data
    ports:
      - "18000:18000"
    depends_on:
      milvus:
        condition: service_healthy

  frontend:
    build:
      context: ../..
      dockerfile: infra/docker/Dockerfile.frontend
    environment:
      - VITE_BACKEND_TARGET=http://backend:18000
    ports:
      - "5173:5173"
    depends_on:
      - backend

volumes:
  etcd_data:
  minio_data:
  milvus_data:
```

说明：

- 前端容器运行 Vite dev server，`/api` 会通过 `VITE_BACKEND_TARGET` 代理到后端。
- 生产化部署可自行替换为静态构建 + Nginx（不在当前范围）。

### 13.4 Docker 启动步骤（给其他开发者）

```bash
git clone <your-repo>
cd uae_law_rag

# 如需 DashScope/OpenAI/DeepSeek，请先在仓库根目录 .env 写入 Key

# 1) 启动全量服务
docker compose -f infra/docker/docker-compose.full.yml up -d

# 2) 初始化 DB + Milvus
docker compose -f infra/docker/docker-compose.full.yml exec backend \
  bash -lc "PYTHONPATH=src python -m uae_law_rag.backend.scripts.init_db --drop --seed --seed-fts"

# 2.1) 更新 KB embed 配置（可选，仅当切换 embedding 模型时）
docker compose -f infra/docker/docker-compose.full.yml exec backend \
  bash -lc "python - <<'PY'\nimport sqlite3\nconn = sqlite3.connect('/app/.Local/uae_law_rag.db')\ncur = conn.cursor()\ncur.execute(\"update knowledge_base set embed_provider=?, embed_model=?, embed_dim=? where id='default'\", ('dashscope', 'text-embedding-v4', 1024))\nconn.commit()\nprint('kb updated')\nPY"

# 2.2) 初始化 Milvus
docker compose -f infra/docker/docker-compose.full.yml exec backend \
  bash -lc "PYTHONPATH=src python -m uae_law_rag.backend.scripts.init_milvus --collection kb_default --embed-dim 1024 --metric-type COSINE --drop"

# 如果未更新 KB embed_dim，请将 --embed-dim 改为 384

# 3) 写入 Run Config
docker compose -f infra/docker/docker-compose.full.yml exec backend \
  bash -lc "PYTHONPATH=src python -m uae_law_rag.backend.scripts.set_run_config"

# 4) 访问
# frontend: http://localhost:5173/
# backend:  http://localhost:18000/api/health
```

> Docker 环境中 ingest 的 `source_uri` 必须是容器可访问的路径（例如挂载到 `/app/.data/raw`）。
