# Development Environment Guide (Milvus + Attu + SQLite)

This document describes how to set up and use the **local visualization toolchain** for the UAE Law RAG project. The goal is to enable real‑time inspection of data storage and vector indexes **without blocking the main development workflow**.

---

## 1. Scope

This guide covers:

* SQLite database visualization
* Milvus vector database (Docker Compose)
* Attu (Milvus Web UI)
* Network / proxy requirements (Docker + Clash)

It is intended for **local development only**.

---

## 2. Architecture Overview

```
Python / FastAPI / RAG Pipeline
        |
        |  (SQLAlchemy / aiosqlite)
        v
SQLite (.Local/uae_law_rag.db)

        |
        |  (pymilvus)
        v
Milvus (Docker Compose)
        |
        v
Attu (Web UI @ localhost:8000)
```

Visualization is a **parallel toolchain** and does not affect ingestion or retrieval logic.

---

## 3. SQLite Visualization

### Database Path

The project uses a fixed local path:

```
.Local/uae_law_rag.db
```

Configured via:

```
UAE_LAW_RAG_DATABASE_URL=sqlite+aiosqlite:////absolute/path/to/.Local/uae_law_rag.db
```

### Recommended Tools

* TablePlus
* DB Browser for SQLite
* DBeaver

### Usage

1. Open the tool
2. Open database file: `.Local/uae_law_rag.db`
3. Observe tables and row changes during ingestion / query

No additional Python code is required.

---

## 4. Milvus (Docker Compose)

### Start Services

From project root:

```bash
cd infra/milvus
docker compose up -d
```

### Verify

```bash
docker ps | grep milvus
nc -vz 127.0.0.1 19530
```

Expected port:

```
19530 (Milvus gRPC)
```

---

## 5. Attu (Milvus Web UI)

### Pull & Run

```bash
docker run -d \
  --platform linux/amd64 \
  --name uae-law-rag-attu \
  -p 8000:3000 \
  -e MILVUS_URL=host.docker.internal:19530 \
  zilliz/attu:latest
```

### Open UI

```
http://localhost:8000
```

### Connection Settings

| Field          | Value                      |
| -------------- | -------------------------- |
| Milvus Address | host.docker.internal:19530 |
| Database       | default                    |
| Authentication | disabled                   |
| SSL            | disabled                   |
| Health Check   | enabled                    |

---

## 6. Network / Proxy Requirements (Critical)

In restricted networks (corporate / campus / CN), Docker cannot access Docker Hub directly and must use an HTTP proxy.

### Clash Configuration

Required:

```yaml
allow-lan: true
bind-address: 0.0.0.0
mixed-port: 7897
```

Verify:

```bash
lsof -iTCP:7897 -sTCP:LISTEN
# Expected: *:7897
```

### Shell Proxy Alias (recommended)

```bash
alias proxyon='export HTTP_PROXY="http://<LAN_IP>:7897"; export HTTPS_PROXY="http://<LAN_IP>:7897"; export ALL_PROXY="http://<LAN_IP>:7897"'
alias proxyoff='unset HTTP_PROXY HTTPS_PROXY ALL_PROXY'
```

### Docker Desktop Proxy Settings

Settings → Proxies:

```
HTTP  = http://<LAN_IP>:7897
HTTPS = http://<LAN_IP>:7897
```

Bypass:

```
localhost,127.0.0.1,*.local,docker.internal
```

Apply & Restart Docker Desktop.

### Validation

```bash
curl -I -x http://<LAN_IP>:7897 https://registry-1.docker.io/v2/
# Expect: HTTP/1.1 401 Unauthorized
```

```bash
docker pull hello-world
docker pull zilliz/attu:latest
```

---

## 7. Common Failure Modes

| Symptom                          | Cause                    | Fix                      |
| -------------------------------- | ------------------------ | ------------------------ |
| docker pull timeout              | Proxy not configured     | Configure Docker proxy   |
| Cannot connect to 127.0.0.1:7897 | Clash bound to localhost | Enable allow-lan         |
| Attu cannot reach Milvus         | Wrong host               | Use host.docker.internal |
| Port 19530 refused               | Milvus not started       | docker compose up        |

---

## 8. Engineering Notes

* Always use **LAN IP**, not `127.0.0.1`, for Docker proxy
* Avoid SOCKS5 for Docker registry
* Attu is a debugging tool, not a production dependency
* SQLite + Milvus visualization should never block pipeline development

---

## 9. Optional Enhancements

* Add `attu` service into `infra/milvus/docker-compose.yml`
* Add health-check script for Milvus startup
* Document collection schema contract
* Add ingestion verification checklist

---

## 10. Status

This setup is validated for:

* macOS (Apple Silicon)
* Docker Desktop
* Milvus standalone
* Attu v2.6.x
* Clash proxy environment

---

Maintained for internal development use.

---

# Appendix: Updated `infra/milvus/README.md` (with Attu)

Below is a drop‑in replacement for your current `infra/milvus/README.md`, integrating **Attu (Milvus Web UI)** as part of the standard local toolchain.

````md
# Milvus (Docker) for UAE Law RAG

This module provides a local Milvus standalone deployment via Docker Compose, plus an optional web‑based GUI (Attu) for inspection and debugging.

---

## Start

```bash
cd infra/milvus
cp .env.example .env   # optional
docker compose up -d
````

---

## Check

```bash
docker ps
```

Milvus gRPC endpoint:

```
localhost:19530
```

Optional quick probe:

```bash
nc -vz 127.0.0.1 19530
```

---

## Attu (Milvus Web UI)

Attu is an optional visualization tool for inspecting:

* Collections / schemas
* Entity counts
* Index status
* Vector dimensions

It does **not** affect runtime behavior and is for development/debugging only.

### Run Attu

```bash
docker run -d \
  --platform linux/amd64 \
  --name uae-law-rag-attu \
  -p 8000:3000 \
  -e MILVUS_URL=host.docker.internal:19530 \
  zilliz/attu:latest
```

### Open UI

```
http://localhost:8000
```

### Connection Settings

| Field          | Value                      |
| -------------- | -------------------------- |
| Milvus Address | host.docker.internal:19530 |
| Database       | default                    |
| Authentication | disabled                   |
| SSL            | disabled                   |

---

## Stop

```bash
docker compose down
```

---

## Stop and wipe data

```bash
docker compose down -v
```

---

## Notes

* Milvus data is stored in Docker volumes unless `-v` is used
* Attu runs as a separate container and can be removed independently:

```bash
docker rm -f uae-law-rag-attu
```

* For restricted networks, Docker must be configured with an HTTP proxy (see `README_dev.md`)

---

```
```
