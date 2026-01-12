#!/usr/bin/env python3
# src/uae_law_rag/backend/pipelines/ingest/segment.py

"""
[职责] segment：将 Markdown 解析为可检索的 Node 列表（保留法条结构与上下文窗口）。
[边界] 不做 PDF 解析；不做 embedding；不负责 DB/Milvus 写入。
[上游关系] ingest/pdf_parse.py 输出 markdown；ingest/pipeline.py 调用 segment_nodes。
[下游关系] ingest/persist_db.py 将节点落库；retrieval pipeline 依赖 node 元数据。
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _load_llama_index() -> Dict[str, Any]:
    """
    [职责] 延迟加载 LlamaIndex 解析组件（MarkdownElementNodeParser/SentenceWindowNodeParser）。
    [边界] 仅负责 import；不处理解析逻辑。
    [上游关系] segment_nodes 调用。
    [下游关系] _build_*_parser 使用返回的类对象。
    """
    try:
        from llama_index.core import Document  # type: ignore  # docstring: LlamaIndex Document
        from llama_index.core.node_parser import (  # type: ignore
            MarkdownElementNodeParser,
            SentenceWindowNodeParser,
        )  # docstring: LlamaIndex 节点解析器
    except Exception as exc:  # pragma: no cover - 依赖缺失场景
        raise ImportError("llama_index is required for segment") from exc  # docstring: 强制依赖

    return {
        "Document": Document,
        "MarkdownElementNodeParser": MarkdownElementNodeParser,
        "SentenceWindowNodeParser": SentenceWindowNodeParser,
    }


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    [职责] 过滤参数，仅保留目标函数支持的关键字。
    [边界] 不做值校验；只做参数名过滤。
    [上游关系] _build_markdown_parser/_build_sentence_window_parser 调用。
    [下游关系] 解析器构造函数。
    """
    try:
        sig = inspect.signature(fn)  # docstring: 读取可用参数
    except (TypeError, ValueError):
        return {}  # docstring: 无法获取签名时回退为空
    return {k: v for k, v in kwargs.items() if k in sig.parameters}  # docstring: 保留受支持参数


def _extract_markdown(parsed: Any) -> str:
    """
    [职责] 从 parsed 对象中提取 markdown 字符串。
    [边界] 仅支持 dict/attr 两种访问方式；其他类型报错。
    [上游关系] segment_nodes 输入。
    [下游关系] 解析器输入的 Document.text。
    """
    if isinstance(parsed, dict) and "markdown" in parsed:
        return str(parsed.get("markdown") or "")  # docstring: dict 形态解析
    if hasattr(parsed, "markdown"):
        return str(getattr(parsed, "markdown") or "")  # docstring: attr 形态解析
    raise ValueError("parsed markdown not found")  # docstring: 必须包含 markdown


def _extract_pages(parsed: Any) -> Optional[int]:
    """
    [职责] 从 parsed 对象中提取页数。
    [边界] 仅支持 dict/attr 两种访问方式；无法解析时返回 None。
    [上游关系] segment_nodes 输入。
    [下游关系] page fallback 与 gate 断言。
    """
    if isinstance(parsed, dict):
        v = parsed.get("pages")
        if isinstance(v, int):
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        return None  # docstring: dict 页数
    v = getattr(parsed, "pages", None)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None  # docstring: attr 页数


def _extract_article_id(text: str) -> Optional[str]:
    """
    [职责] 从标题文本中提取 article_id。
    [边界] 仅匹配常见 Article 规则；失败返回 None。
    [上游关系] _extract_section_marks 调用。
    [下游关系] Node.article_id 字段。
    """
    m = re.match(r"^\s*Article\s+([A-Za-z0-9IVXLC().-]+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    return f"Article {m.group(1)}"  # docstring: 统一 Article 形式


def _extract_section_marks(markdown: str) -> List[Tuple[int, str, Optional[str]]]:
    """
    [职责] 解析 Markdown 标题层级，构建 section_path 与 article_id 的位置索引。
    [边界] 仅处理 ATX 标题（#）；不解析 Setext 标题。
    [上游关系] segment_nodes 调用。
    [下游关系] _section_for_offset 使用。
    """
    marks: List[Tuple[int, str, Optional[str]]] = []
    stack: List[str] = []
    current_article: Optional[str] = None
    offset = 0
    for line in markdown.splitlines(keepends=True):
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            level = len(m.group(1))  # docstring: 标题层级
            title = m.group(2).strip()  # docstring: 标题文本
            stack = stack[: max(0, level - 1)] + [title]  # docstring: 更新层级栈
            article_id = _extract_article_id(title)  # docstring: 识别 Article 标题
            if article_id:
                current_article = article_id  # docstring: 更新当前 Article
            section_path = "/".join(stack)  # docstring: 生成 section_path
            marks.append((offset, section_path, current_article))
        offset += len(line)
    return marks


def _extract_page_marks(markdown: str, pages: Optional[int]) -> List[Tuple[int, int]]:
    """
    [职责] 从 Markdown 中提取页面标记（offset -> page）。
    [边界] 仅基于常见模式；无法识别时返回空列表。
    [上游关系] segment_nodes 调用。
    [下游关系] _page_for_offset 使用。
    """
    marks: List[Tuple[int, int]] = []
    comment_pattern = re.compile(r"<!--\s*page\s*[:=]\s*(\d+)\s*-->", re.IGNORECASE)
    for match in comment_pattern.finditer(markdown):
        marks.append((match.start(), int(match.group(1))))  # docstring: HTML 注释页码

    offset = 0
    for line in markdown.splitlines(keepends=True):
        line_stripped = line.strip()
        m = re.match(r"^Page\s+(\d+)$", line_stripped, flags=re.IGNORECASE)
        if m:
            marks.append((offset, int(m.group(1))))  # docstring: 纯文本页码行
        offset += len(line)

    if "\f" in markdown:
        page_no = 1
        for idx, ch in enumerate(markdown):
            if ch == "\f":
                page_no += 1
                marks.append((idx + 1, page_no))  # docstring: form-feed 视作分页

    if not marks and pages == 1:
        marks.append((0, 1))  # docstring: 单页文档默认归为 page=1

    marks.sort(key=lambda x: x[0])  # docstring: 按 offset 排序以便二分查找
    return marks


def _page_for_offset(offset: Optional[int], marks: Sequence[Tuple[int, int]]) -> Optional[int]:
    """
    [职责] 根据字符偏移量返回页码。
    [边界] 若缺少页码信息则返回 None。
    [上游关系] _build_payloads 调用。
    [下游关系] Node.page 字段。
    """
    if offset is None or not marks:
        return None
    page = None
    for mark_offset, mark_page in marks:
        if offset < mark_offset:
            break
        page = mark_page
    return page


def _section_for_offset(
    offset: Optional[int], marks: Sequence[Tuple[int, str, Optional[str]]]
) -> Tuple[Optional[str], Optional[str]]:
    """
    [职责] 根据字符偏移量返回 section_path 与 article_id。
    [边界] 若缺少标题信息则返回 (None, None)。
    [上游关系] _build_payloads 调用。
    [下游关系] Node.section_path / Node.article_id 字段。
    """
    if offset is None or not marks:
        return None, None
    section_path = None
    article_id = None
    for mark_offset, mark_section, mark_article in marks:
        if offset < mark_offset:
            break
        section_path = mark_section
        article_id = mark_article
    return section_path, article_id


def _node_text(node: Any) -> str:
    """
    [职责] 统一读取 LlamaIndex 节点文本。
    [边界] 仅覆盖常见属性；无法读取时返回空字符串。
    [上游关系] _build_payloads 调用。
    [下游关系] Node.text 字段。
    """
    if hasattr(node, "text"):
        return str(getattr(node, "text") or "")  # docstring: 直接取 text
    if hasattr(node, "get_content"):
        return str(node.get_content() or "")  # docstring: 兼容 get_content
    return ""


def _node_meta(node: Any) -> Dict[str, Any]:
    """
    [职责] 统一读取 LlamaIndex 节点元数据。
    [边界] 仅返回 dict；非 dict 时回退为空。
    [上游关系] _build_payloads 调用。
    [下游关系] Node.meta_data 字段。
    """
    meta = getattr(node, "metadata", None)
    return dict(meta) if isinstance(meta, dict) else {}


def _node_offsets(node: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    [职责] 读取 LlamaIndex 节点的字符偏移信息。
    [边界] 不做全文搜索；仅读节点自身属性/元数据。
    [上游关系] _build_payloads 调用。
    [下游关系] Node.start_offset / Node.end_offset 字段。
    """
    start = getattr(node, "start_char_idx", None)
    end = getattr(node, "end_char_idx", None)
    if isinstance(start, int) and isinstance(end, int):
        return start, end  # docstring: 使用 node 自带 offset
    meta = _node_meta(node)
    start_meta = meta.get("start_char_idx")
    end_meta = meta.get("end_char_idx")
    if isinstance(start_meta, int) and isinstance(end_meta, int):
        return start_meta, end_meta  # docstring: 使用 metadata offset
    return None, None


def _find_offsets(text: str, markdown: str, cursor: int) -> Tuple[Optional[int], Optional[int], int]:
    """
    [职责] 在 Markdown 中查找节点文本位置（fallback）。
    [边界] 可能受重复文本影响；找不到则返回 None。
    [上游关系] _build_payloads 调用。
    [下游关系] Node.start_offset / Node.end_offset 字段。
    """
    if not text:
        return None, None, cursor
    idx = markdown.find(text, cursor)
    if idx == -1:
        return None, None, cursor  # docstring: 未找到时保持 cursor
    end = idx + len(text)
    return idx, end, end  # docstring: 返回新 cursor


def _build_markdown_parser(chunking_config: Dict[str, Any]) -> Any:
    """
    [职责] 构造 MarkdownElementNodeParser（结构化 Markdown 切分）。
    [边界] 仅使用 LlamaIndex 内建解析器；参数按可用项传入。
    [上游关系] segment_nodes 调用。
    [下游关系] get_nodes_from_documents 输出基础节点。
    """
    li = _load_llama_index()  # docstring: 加载 LlamaIndex 组件
    parser_cls = li["MarkdownElementNodeParser"]
    kwargs = {
        "include_metadata": True,
        "include_prev_next_rel": True,
    }  # docstring: 基础参数快照
    parser = parser_cls(**_filter_kwargs(parser_cls.__init__, kwargs))  # docstring: 构造解析器
    return parser


def _build_sentence_window_parser(chunking_config: Dict[str, Any]) -> Any:
    """
    [职责] 构造 SentenceWindowNodeParser（上下文窗口切分）。
    [边界] 仅控制 window size 与 metadata key；不自定义分句算法。
    [上游关系] segment_nodes 调用。
    [下游关系] 输出 window nodes。
    """
    li = _load_llama_index()  # docstring: 加载 LlamaIndex 组件
    parser_cls = li["SentenceWindowNodeParser"]
    window_size = int(chunking_config.get("window_size") or chunking_config.get("sentence_window") or 2)
    window_key = str(chunking_config.get("window_metadata_key") or "window")
    original_key = str(chunking_config.get("original_text_metadata_key") or "original_text")
    kwargs = {
        "window_size": window_size,
        "window_metadata_key": window_key,
        "original_text_metadata_key": original_key,
    }  # docstring: SentenceWindow 参数快照

    if hasattr(parser_cls, "from_defaults"):
        parser = parser_cls.from_defaults(**_filter_kwargs(parser_cls.from_defaults, kwargs))  # docstring: 使用默认工厂
    else:
        parser = parser_cls(**_filter_kwargs(parser_cls.__init__, kwargs))  # docstring: 直接构造解析器
    return parser


def _apply_sentence_window(parser: Any, base_nodes: Sequence[Any], document: Any) -> List[Any]:
    """
    [职责] 运行 SentenceWindowNodeParser 获取 window nodes。
    [边界] 若 parser 不支持 nodes 输入则回退到 document 输入。
    [上游关系] segment_nodes 调用。
    [下游关系] window nodes 输出。
    """
    if hasattr(parser, "get_nodes_from_nodes"):
        return list(parser.get_nodes_from_nodes(base_nodes))  # docstring: 基于基础节点生成窗口
    if hasattr(parser, "get_nodes_from_documents"):
        return list(parser.get_nodes_from_documents([document]))  # docstring: 基于 document 生成窗口
    raise AttributeError("SentenceWindowNodeParser requires get_nodes_from_nodes/documents")  # docstring: 强约束


def _iter_nodes(nodes: Iterable[Any]) -> Iterable[Any]:
    """
    [职责] 统一迭代节点序列（避免 None）。
    [边界] 不做类型校验。
    [上游关系] _build_payloads 调用。
    [下游关系] 生成节点 payload。
    """
    return nodes or []


def _build_payloads(
    *,
    nodes: Sequence[Any],
    markdown: str,
    kind: str,
    element_type_fallback: str,
    section_marks: Sequence[Tuple[int, str, Optional[str]]],
    page_marks: Sequence[Tuple[int, int]],
    segment_version: str,
) -> List[Dict[str, Any]]:
    """
    [职责] 将 LlamaIndex nodes 映射为 NodePayload dict 列表。
    [边界] 不做 DB 写入；不生成 node_index（由外层统一排序后填充）。
    [上游关系] segment_nodes 调用。
    [下游关系] ingest/persist_db.py 批量落库。
    """
    payloads: List[Dict[str, Any]] = []
    cursor = 0
    for i, node in enumerate(_iter_nodes(nodes)):
        text_raw = _node_text(node)  # docstring: 提取节点文本
        text_stripped = text_raw.strip() if isinstance(text_raw, str) else str(text_raw).strip()
        if not text_stripped:
            continue  # docstring: 跳过空节点
        text = text_raw  # docstring: 保持原文本以确保与 start/end_offset 映射一致

        start, end = _node_offsets(node)  # docstring: 读取 offset
        if start is None or end is None:
            start, end, cursor = _find_offsets(text_raw, markdown, cursor)  # docstring: fallback 查找 offset

        section_path, article_id = _section_for_offset(start, section_marks)  # docstring: 标题映射
        page = _page_for_offset(start, page_marks)  # docstring: 页码映射

        meta = _node_meta(node)  # docstring: 透传原始 metadata
        if not article_id:
            article_id = _extract_article_id(text_stripped)  # docstring: 从文本补充 Article
        if not article_id and meta.get("article_id"):
            article_id = str(meta.get("article_id"))  # docstring: 从 metadata 补充 Article
        if not section_path and meta.get("section_path"):
            section_path = str(meta.get("section_path"))  # docstring: 从 metadata 补充 section
        element_type = str(meta.get("element_type") or meta.get("type") or element_type_fallback)
        meta_out = dict(meta)  # docstring: 先复制原始 metadata，避免丢失字段
        meta_out.update(
            {
                "source": "markdown",
                "element_type": element_type,
                "node_kind": kind,
                "segment_version": segment_version,
            }
        )  # docstring: 标准字段覆盖，保证合同稳定

        payloads.append(
            {
                "__sort_key": (start if start is not None else 10**12 + i, i),  # docstring: 稳定排序键
                "text": text,
                "page": page,
                "article_id": article_id,
                "section_path": section_path,
                "start_offset": start,
                "end_offset": end,
                "meta_data": meta_out,
            }
        )
    return payloads


async def segment_nodes(
    *,
    parsed: Any,
    chunking_config: Optional[Dict[str, Any]] = None,
    segment_version: str = "v1",
) -> List[Dict[str, Any]]:
    """
    [职责] 将 PDF 解析产物（Markdown）切分为 NodePayload 列表。
    [边界] 不做 embedding；不写入 DB；不处理 Milvus。
    [上游关系] ingest/pipeline.py 调用。
    [下游关系] ingest/persist_db.py 批量写入 NodeModel。
    """
    markdown = _extract_markdown(parsed)  # docstring: 获取 markdown
    if not markdown.strip():
        raise ValueError("markdown is empty")  # docstring: 必须有内容

    pages = _extract_pages(parsed)  # docstring: 获取页数
    cfg = chunking_config or {}  # docstring: 切分配置
    enable_window = bool(cfg.get("enable_sentence_window", True))

    li = _load_llama_index()  # docstring: 加载 LlamaIndex 组件
    Document = li["Document"]
    document = Document(text=markdown, metadata={"source": "markdown"})  # docstring: 构造文档输入

    section_marks = _extract_section_marks(markdown)  # docstring: 解析标题层级
    page_marks = _extract_page_marks(markdown, pages)  # docstring: 解析页码标记

    markdown_parser = _build_markdown_parser(cfg)  # docstring: Markdown 结构解析器
    base_nodes = list(markdown_parser.get_nodes_from_documents([document]))  # docstring: 生成基础节点

    payloads: List[Dict[str, Any]] = []
    payloads.extend(
        _build_payloads(
            nodes=base_nodes,
            markdown=markdown,
            kind="primary",
            element_type_fallback="markdown_element",
            section_marks=section_marks,
            page_marks=page_marks,
            segment_version=segment_version,
        )
    )  # docstring: 基础节点 payload

    if enable_window:
        window_parser = _build_sentence_window_parser(cfg)  # docstring: 句子窗口解析器
        window_nodes = _apply_sentence_window(window_parser, base_nodes, document)  # docstring: 生成窗口节点
        payloads.extend(
            _build_payloads(
                nodes=window_nodes,
                markdown=markdown,
                kind="window",
                element_type_fallback="sentence_window",
                section_marks=section_marks,
                page_marks=page_marks,
                segment_version=segment_version,
            )
        )  # docstring: 窗口节点 payload

    payloads.sort(key=lambda x: x["__sort_key"])  # docstring: 按阅读顺序排序

    nodes_out: List[Dict[str, Any]] = []
    for idx, payload in enumerate(payloads):
        payload.pop("__sort_key", None)  # docstring: 移除内部排序字段
        payload["node_index"] = idx  # docstring: 文档内节点序号
        if payload.get("page") is None and pages == 1:
            payload["page"] = 1  # docstring: 单页文档页码兜底
        nodes_out.append(payload)

    return nodes_out


async def segment(
    *,
    parsed: Any,
    chunking_config: Optional[Dict[str, Any]] = None,
    segment_version: str = "v1",
) -> List[Dict[str, Any]]:
    """
    [职责] segment_nodes 的别名入口（兼容旧调用）。
    [边界] 行为与 segment_nodes 一致。
    [上游关系] ingest/pipeline.py 的适配层（若调用 segment）。
    [下游关系] 返回标准 NodePayload 列表。
    """
    return await segment_nodes(  # docstring: 兼容旧函数名入口
        parsed=parsed, chunking_config=chunking_config, segment_version=segment_version
    )
