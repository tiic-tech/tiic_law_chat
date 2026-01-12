# src/uae_law_rag/backend/pipelines/ingest/embed.py

"""
[职责] embed：使用 LlamaIndex Embedding 抽象生成向量（对齐 Node 文本与元数据）。
[边界] 不负责 Milvus 写入；不负责 DB 写入；不实现自定义 pooling/truncation。
[上游关系] ingest/pipeline.py 调用 embed_texts/embed_nodes；上游传入 KB 的 provider/model/dim 配置。
[下游关系] ingest/persist_milvus.py 消费向量；NodeVectorMap 与 gate 依赖向量数量一致性。
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EmbeddingResult:
    """
    [职责] EmbeddingResult：单节点 embedding 结果快照（用于审计/落库前中间态）。
    [边界] 不携带原始 Node 全文；仅记录必要引用与向量。
    [上游关系] embed_nodes 产出。
    [下游关系] persist_milvus/persist_db 可消费并映射为 payload。
    """

    node_id: str
    vector: List[float]
    dim: int
    model: str
    provider: str


def _load_llama_index() -> Dict[str, Any]:
    """
    [职责] 延迟加载 LlamaIndex embedding 组件（BaseEmbedding/TextNode/NodeWithScore）。
    [边界] 仅负责 import；不做业务逻辑。
    [上游关系] embed_texts/embed_nodes 调用。
    [下游关系] _resolve_embedder/_build_text_nodes 使用返回的类对象。
    """
    try:
        from llama_index.core.base.embeddings.base import BaseEmbedding  # type: ignore  # docstring: BaseEmbedding 抽象
        from llama_index.core.schema import (  # type: ignore
            NodeWithScore,
            TextNode,
        )  # docstring: TextNode/NodeWithScore
    except Exception as exc:  # pragma: no cover - 依赖缺失场景
        raise ImportError("llama_index is required for embed") from exc  # docstring: 强制依赖
    return {
        "BaseEmbedding": BaseEmbedding,
        "TextNode": TextNode,
        "NodeWithScore": NodeWithScore,
    }


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    [职责] 过滤参数，仅保留目标函数支持的关键字。
    [边界] 不做值校验；只做参数名过滤。
    [上游关系] _resolve_embedder 调用。
    [下游关系] embedding 构造函数。
    """
    try:
        sig = inspect.signature(fn)  # docstring: 读取可用参数
    except (TypeError, ValueError):
        return {}  # docstring: 无法获取签名时回退为空
    return {k: v for k, v in kwargs.items() if k in sig.parameters}  # docstring: 保留受支持参数


def _build_hash_embedder(*, dim: int, model: str, provider: str) -> Any:
    """
    [职责] 构造本地 hash embedding（用于无外部依赖环境）。
    [边界] 非语义 embedding，仅用于测试/离线验证。
    [上游关系] _resolve_embedder 调用。
    [下游关系] embed_texts/embed_nodes 使用。
    """
    li = _load_llama_index()  # docstring: 加载 LlamaIndex 抽象
    BaseEmbedding = li["BaseEmbedding"]

    class _HashEmbedding(BaseEmbedding):
        """Simple deterministic embedding based on sha256."""  # docstring: 本地可复现嵌入

        def __init__(self, *, dim: int, model_name: str, provider_name: str) -> None:
            super().__init__(model_name=model_name)  # docstring: 初始化 BaseEmbedding
            self._dim = int(dim)  # docstring: 目标向量维度
            self._provider = str(provider_name)  # docstring: 记录 provider

        def _hash_to_vec(self, text: str) -> List[float]:
            h = hashlib.sha256(text.encode("utf-8")).digest()  # docstring: 生成哈希字节
            vals: List[float] = []
            seed = h
            while len(vals) < self._dim:
                for b in seed:
                    vals.append((b / 255.0) * 2.0 - 1.0)  # docstring: 映射到 [-1, 1]
                    if len(vals) >= self._dim:
                        break
                seed = hashlib.sha256(seed).digest()  # docstring: 扩展伪随机序列
            return vals[: self._dim]

        def _get_text_embedding(self, text: str) -> List[float]:
            return self._hash_to_vec(text)  # docstring: 文本向量

        def _get_query_embedding(self, query: str) -> List[float]:
            return self._hash_to_vec(query)  # docstring: 查询向量

    return _HashEmbedding(dim=dim, model_name=model, provider_name=provider)


def _resolve_embedder(
    *,
    provider: str,
    model: str,
    dim: Optional[int],
    embed_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    [职责] 根据 provider/model 构造 LlamaIndex BaseEmbedding 实例。
    [边界] 仅支持已接入的 embedding provider；未知 provider 直接报错。
    [上游关系] embed_texts/embed_nodes 调用。
    [下游关系] _embed_text_nodes 使用 embedder 生成向量。
    """
    li = _load_llama_index()  # docstring: 加载 LlamaIndex 抽象
    BaseEmbedding = li["BaseEmbedding"]

    provider_key = str(provider).strip().lower()  # docstring: 归一化 provider
    model_name = str(model).strip()
    cfg = embed_config or {}  # docstring: 额外配置透传

    if provider_key in {"mock", "local", "hash"}:
        embedder = _build_hash_embedder(  # docstring: 使用本地 hash embedding
            dim=int(dim or 128), model=model_name or "hash", provider=provider_key
        )
    elif provider_key == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding  # type: ignore  # docstring: Ollama embedding

        kwargs = {
            "model_name": model_name,
            "model": model_name,
            "embed_dim": dim,
            "output_dim": dim,
            **cfg,
        }  # docstring: Ollama 参数快照
        embedder = OllamaEmbedding(
            **_filter_kwargs(OllamaEmbedding.__init__, kwargs)
        )  # docstring: 构造 Ollama embedding
    elif provider_key in {"huggingface", "hf"}:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore  # docstring: HF embedding

        kwargs = {
            "model_name": model_name,
            "embed_dim": dim,
            "output_dim": dim,
            **cfg,
        }  # docstring: HuggingFace 参数快照
        embedder = HuggingFaceEmbedding(
            **_filter_kwargs(HuggingFaceEmbedding.__init__, kwargs)
        )  # docstring: 构造 HF embedding
    elif provider_key in {"dashscope", "qwen"}:
        from llama_index.embeddings.dashscope import DashScopeEmbedding  # type: ignore  # docstring: DashScope embedding

        kwargs = {
            "model_name": model_name,
            "model": model_name,
            "embed_dim": dim,
            "output_dim": dim,
            **cfg,
        }  # docstring: DashScope 参数快照
        embedder = DashScopeEmbedding(
            **_filter_kwargs(DashScopeEmbedding.__init__, kwargs)
        )  # docstring: 构造 DashScope embedding
    elif provider_key == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore  # docstring: OpenAI embedding

        kwargs = {
            "model": model_name,
            "model_name": model_name,
            "embed_dim": dim,
            "output_dim": dim,
            **cfg,
        }  # docstring: OpenAI 参数快照
        embedder = OpenAIEmbedding(
            **_filter_kwargs(OpenAIEmbedding.__init__, kwargs)
        )  # docstring: 构造 OpenAI embedding
    else:
        raise ValueError(f"unsupported embed provider: {provider}")  # docstring: 未接入 provider

    if not isinstance(embedder, BaseEmbedding):
        raise TypeError("embedding must be BaseEmbedding")  # docstring: 强制 LlamaIndex 抽象

    return embedder


def _build_text_nodes(
    *,
    texts: Sequence[str],
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Any]:
    """
    [职责] 构造 TextNode 列表（便于 embedding 与 metadata 透传）。
    [边界] 不做文本清洗；空文本由上游过滤。
    [上游关系] embed_texts/embed_nodes 调用。
    [下游关系] _embed_text_nodes 生成向量。
    """
    li = _load_llama_index()  # docstring: 加载 LlamaIndex 抽象
    TextNode = li["TextNode"]
    nodes: List[Any] = []
    metas = metadatas or []
    for i, text in enumerate(texts):
        meta = metas[i] if i < len(metas) else {}
        nodes.append(TextNode(text=str(text), metadata=dict(meta)))  # docstring: 构造 TextNode
    return nodes


async def _embed_text_nodes(embedder: Any, nodes: Sequence[Any]) -> Tuple[List[List[float]], List[Any]]:
    """
    [职责] 调用 BaseEmbedding 为 TextNode 列表生成向量。
    [边界] 仅使用 embedder 的公开 API；不处理模型缓存。
    [上游关系] embed_texts/embed_nodes 调用。
    [下游关系] embedding 向量与 NodeWithScore 列表。
    """
    li = _load_llama_index()  # docstring: 加载 LlamaIndex 抽象
    NodeWithScore = li["NodeWithScore"]

    texts = [str(getattr(n, "text", "")) for n in nodes]  # docstring: 提取文本序列
    if hasattr(embedder, "aget_text_embedding_batch"):
        vectors = await embedder.aget_text_embedding_batch(texts)  # docstring: 异步 batch embedding
    elif hasattr(embedder, "get_text_embedding_batch"):
        vectors = embedder.get_text_embedding_batch(texts)  # docstring: 同步 batch embedding
    elif hasattr(embedder, "aget_text_embedding"):
        vectors = [await embedder.aget_text_embedding(t) for t in texts]  # docstring: 异步逐条 embedding
    elif hasattr(embedder, "get_text_embedding"):
        vectors = [embedder.get_text_embedding(t) for t in texts]  # docstring: 同步逐条 embedding
    else:
        raise AttributeError("BaseEmbedding missing embedding methods")  # docstring: 强约束

    vecs = [list(map(float, v)) for v in vectors]  # docstring: 确保向量为 float list
    scored_nodes: List[Any] = []
    for node, vec in zip(nodes, vecs):
        setattr(node, "embedding", vec)  # docstring: 绑定向量到 TextNode
        scored_nodes.append(NodeWithScore(node=node, score=0.0))  # docstring: 构造 NodeWithScore
    return vecs, scored_nodes


async def embed_texts(
    *,
    texts: Sequence[str],
    provider: str,
    model: str,
    dim: Optional[int] = None,
    embed_config: Optional[Dict[str, Any]] = None,
) -> List[List[float]]:
    """
    [职责] 对文本列表进行 embedding（返回向量列表）。
    [边界] 不处理文本切分；不写入 DB/Milvus。
    [上游关系] ingest/pipeline.py 调用（embed 阶段）。
    [下游关系] persist_milvus 使用向量；gate 校验向量数量一致。
    """
    if not texts:
        return []  # docstring: 空输入直接返回

    embedder = _resolve_embedder(
        provider=provider, model=model, dim=dim, embed_config=embed_config
    )  # docstring: 构造 embedding
    nodes = _build_text_nodes(texts=texts)  # docstring: 构造 TextNode 列表
    vectors, _ = await _embed_text_nodes(embedder, nodes)  # docstring: 生成向量并绑定到节点

    if dim is not None:
        for vec in vectors:
            if len(vec) != int(dim):
                raise ValueError(f"embedding dim mismatch: {len(vec)} != {dim}")  # docstring: 维度一致性校验

    return vectors


async def embed_nodes(
    *,
    nodes: Sequence[Dict[str, Any]],
    provider: str,
    model: str,
    dim: Optional[int] = None,
    embed_config: Optional[Dict[str, Any]] = None,
) -> List[EmbeddingResult]:
    """
    [职责] 对节点列表进行 embedding（保留 node_id → vector 映射）。
    [边界] 不写入 DB/Milvus；不负责节点排序。
    [上游关系] ingest/segment.py 产出 node_dicts。
    [下游关系] persist_milvus/persist_db 可消费 EmbeddingResult。
    """
    if not nodes:
        return []  # docstring: 空输入直接返回

    texts = [str(n.get("text") or "") for n in nodes]  # docstring: 提取节点文本
    metadatas = [dict(n.get("meta_data") or {}) for n in nodes]  # docstring: 透传 metadata

    embedder = _resolve_embedder(
        provider=provider, model=model, dim=dim, embed_config=embed_config
    )  # docstring: 构造 embedding
    text_nodes = _build_text_nodes(texts=texts, metadatas=metadatas)  # docstring: 构造 TextNode 列表
    vectors, _ = await _embed_text_nodes(embedder, text_nodes)  # docstring: 生成向量并绑定到节点

    results: List[EmbeddingResult] = []
    for node, vec in zip(nodes, vectors):
        node_id = str(node.get("node_id") or node.get("id") or "")  # docstring: 解析 node_id
        if not node_id:
            raise ValueError("node_id is required for embed_nodes")  # docstring: 强制 node_id
        results.append(
            EmbeddingResult(
                node_id=node_id,
                vector=vec,
                dim=len(vec),
                model=model,
                provider=provider,
            )
        )

    if dim is not None:
        for res in results:
            if res.dim != int(dim):
                raise ValueError(f"embedding dim mismatch: {res.dim} != {dim}")  # docstring: 维度一致性校验

    return results


async def embed(
    *,
    texts: Sequence[str],
    provider: str,
    model: str,
    dim: Optional[int] = None,
    embed_config: Optional[Dict[str, Any]] = None,
) -> List[List[float]]:
    """
    [职责] embed_texts 的别名入口（兼容旧调用）。
    [边界] 行为与 embed_texts 一致。
    [上游关系] ingest/pipeline.py 的适配层（若调用 embed）。
    [下游关系] 返回向量列表。
    """
    return await embed_texts(  # docstring: 兼容旧函数名入口
        texts=texts,
        provider=provider,
        model=model,
        dim=dim,
        embed_config=embed_config,
    )
