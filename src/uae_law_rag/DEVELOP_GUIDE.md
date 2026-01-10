# DEVELOPMENT_GUIDE.md  
**Legal Evidence–First RAG Assistant (UAE Law MVP)**

---

## 0. 文档目的（必读）

本开发指南用于约束与指导 **UAE 法律证据链 RAG 助手 MVP** 的完整实现过程，确保：

- 聚焦 **真实法律业务管道**，而非平台或框架开发  
- 满足 **法律场景的严苛要求**：全量召回、可追溯、可审计  
- 避免过度设计与提前抽象  
- 产出一个 **可运行、可展示、可复盘、可面试讲清** 的全栈系统  

> **红线原则**  
> - 不引入高层平台抽象  
> - 不实现通用 RAG 框架  
> - 不为“未来可能性”牺牲当前可落地性  

---

## 1. 产品目标与范围定义

### 1.1 产品一句话定义

> **一个面向单一 UAE 法律文件的、以“证据链”为核心的法律检索与辅助分析系统。**

---

### 1.2 明确要解决的问题

- 给定关键词或查询：
  - **全量召回** 所有包含该关键词的法律原文证据
  - 通过 **向量语义检索补充召回**
  - 对候选证据进行 **融合与排序**
  - 在 **明确证据引用的前提下**，生成辅助性法律文本

---

### 1.3 明确不解决的问题（MVP 阶段）

- ❌ 多法律文件管理  
- ❌ 法律意见或裁判性结论  
- ❌ 案例法 / 判例法  
- ❌ 通用知识库平台  
- ❌ 多租户 / 权限系统  

---

## 2. 总体技术架构（业务管道优先）

### 2.1 核心技术栈（已锁定）

| 层级       | 技术                                       |
| ---------- | ------------------------------------------ |
| 后端 API   | FastAPI                                    |
| 前端       | React + Vite                               |
| ORM / DB   | SQLAlchemy + SQLite（生产可换 PostgreSQL） |
| 向量数据库 | Milvus                                     |
| RAG 管道   | LlamaIndex                                 |
| PDF 解析   | pymupdf4llm                                |
| LLM        | 可配置（本地或云）                         |

---

### 2.2 三条业务管道（唯一允许存在的“结构”）

#### Pipeline A：Ingestion（离线 / 后台）
PDF → 解析 → Document / Nodes → 
DB（结构化存储） + Milvus（向量存储）

#### Pipeline B：Retrieval（在线）
Query →
Keyword Retrieval（DB） +
Vector Retrieval（Milvus） →
Fusion →
Rerank →
RetrievalRecord 持久化

#### Pipeline C：Generation（在线）
Reranked Evidence →
LLM Generation（引用优先） →
Message / GenerationRecord


---

## 3. 数据系统设计（不抽象，但可迁移）

### 3.1 数据域划分（必须遵守）

#### ① Chat 域
- User
- Conversation
- Message

> 用于对话状态管理，**不承载检索或证据细节**

---

#### ② Knowledge Base（KB）域
- `kb_document`
- `kb_node`
- `kb_ingestion_job`（建议）

> 负责 **法律文本结构化与持久化**

---

#### ③ Retrieval / Generation 域（核心）
- `retrieval_record`
- `retrieval_hit`
- `generation_record`（建议）

> 负责 **检索融合、审计、复盘**

---

### 3.2 核心数据模型设计意图（非字段表）

#### kb_document
- 一份法律文件的逻辑抽象  
- 记录来源、hash、解析版本  
- **MVP 阶段仅允许 1 条记录**

---

#### kb_node
- 法律证据的最小存储单元  
- 一般对应：
  - Article (n)
  - 或 Article 内的合理分段  
- 必须包含：
  - 原文文本
  - 法律结构信息（article_id / page_range）

---

#### retrieval_record（重点）
一次完整检索行为的**审计快照**，必须包含：
- query 原文  
- 检索配置快照（topK、权重、模型名）  
- 时间戳  
- 所属 conversation / message  

---

#### retrieval_hit
一次检索中的单条证据命中记录，必须包含：
- node_id  
- 命中来源：`keyword / vector / both`  
- 分数信息：`raw_score / normalized_score`  
- 排名信息：`rank_before_rerank / rank_after_rerank`  

---

## 4. PDF Ingestion 规范（法律文件优先）

### 4.1 输入约束

- 单一 UAE 官方法律 PDF  
- 英文版本  
- 结构包含 `Article (n)`（如 Cabinet Resolution No. 109 of 2023）

---

### 4.2 解析原则

- **Article 是法律级证据最小单元**
- 不追求 sentence-level chunking
- 优先保证：
  - 法律语义完整
  - 引用可复核

---

### 4.3 Ingestion 验收标准

- Article (1) → Article (N) 全部被解析  
- 每个 node 可映射回：
  - 原 PDF
  - Article 编号  
- 重复 ingestion 不产生重复 nodes（hash 或唯一约束）

---

## 5. Keyword Retrieval（全量召回，DB 加速）

### 5.1 设计目标

- **Recall > Precision**
- 不允许遗漏包含关键词的原文证据

---

### 5.2 技术约束

- 必须基于 DB 层完成  
- 推荐使用：
  - SQLite FTS5（MVP）
  - PostgreSQL tsvector（生产）

---

### 5.3 验收标准

- 人工在 PDF 中查到的关键词位置，系统 **100% 返回**
- 命中结果可按 Article 分组展示

---

## 6. Vector Retrieval（补充召回）

### 6.1 定位

- **补充而非替代** keyword retrieval  
- 用于：
  - 同义词
  - 概念扩展
  - 上下文相关条文

---

### 6.2 Milvus 设计约束

- 单 collection  
- 必须存储：
  - node_id（主键）
  - embedding
  - doc_id / article_id 等 metadata

---

### 6.3 验收标准

- 能返回 topK node_id  
- 能与 DB 中 node 一一映射

---

## 7. Fusion + Rerank（法律级检索融合）

### 7.1 Fusion 原则

- 去重：按 node_id  
- 保留来源标签：`keyword / vector / both`  
- 规则优先级：`keyword 命中 > vector-only 命中`

---

### 7.2 Rerank 原则

- 只排序，不随意删证据  
- 所有排序变化必须可记录

---

### 7.3 RetrievalRecord 验收标准（核心）

- 能完整回答：  
  > “这次检索的结果是**如何一步步产生的**？”
- 能复盘：
  - keyword candidates  
  - vector candidates  
  - fused 列表  
  - reranked 顺序  

---

## 8. Generation（引用优先）

### 8.1 基本原则

- 生成文本 **只能基于 reranked evidence**
- 每条结论必须引用 ≥1 条 node
- 无证据 → 不生成

---

### 8.2 GenerationRecord（建议）

- 记录：
  - 使用模型
  - prompt 版本
  - answer
  - citations（结构化）

---

## 9. FastAPI 接口设计规范（薄层）

### 9.1 必要接口（最小集）

- `POST /conversations`
- `GET /conversations/{id}/messages`
- `POST /kb/ingest`
- `GET /search`（keyword-only）
- `POST /chat`（完整管道）
- `GET /retrieval/{record_id}`（审计回放）

---

### 9.2 API 设计原则

- 返回 **产品语义字段**，而非工程术语  
- 不在 API 层做业务逻辑  
- 不引入 service / domain 层  

---

## 10. 前端（React Vite）展示重点

### 10.1 核心页面

- Search（关键词证据检索）
- Chat（可选）
- Retrieval Record 回放页（强烈建议）

---

### 10.2 展示重点

- Article 分组  
- 命中次数  
- 原文高亮  
- 引用可点击  

> **证据清晰度 > UI 炫技**

---

## 11. 开发节奏建议（现实可控）

| Phase   | 目标              |
| ------- | ----------------- |
| Phase 0 | 项目骨架 + 约束   |
| Phase 1 | DB + Chat         |
| Phase 2 | KB Ingestion      |
| Phase 3 | Keyword Retrieval |
| Phase 4 | Fusion + Rerank   |
| Phase 5 | Generation        |
| Phase 6 | API               |
| Phase 7 | Frontend          |

---

## 12. 最终验收清单（不可妥协）

- [ ] 关键词全量命中可人工核对  
- [ ] 每条结果可追溯到原文 Article  
- [ ] 检索过程可审计、可复盘  
- [ ] 前端能清晰展示证据链  
- [ ] 系统复杂度与业务价值成正比  

---

## 13. 给未来自己的提醒（非常重要）

> 如果你发现自己开始：
> - 抽象通用接口  
> - 设计多 KB、多策略配置  
> - 写大量“未来可能用到”的代码  

**请立即停下，回到本指南第 1 页。**

---

**本指南即是 MVP 的“宪法”。  
所有实现必须以此为最高约束。**
