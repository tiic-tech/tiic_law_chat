# src/uae_law_rag/backend/pipelines/ingest/pdf_parse.py

"""
[职责] pdf_parse：使用 pymupdf4llm 将 PDF 转为 Markdown，并输出页数与解析元信息。
[边界] 不做 OCR；不做手写 PyMuPDF 文本抽取；仅负责结构化 Markdown 还原与最小元数据输出。
[上游关系] ingest/pipeline.py 调用 parse_pdf 作为 parse 步骤。
[下游关系] ingest/segment.py 消费 markdown 并生成节点；DB 写入文档元信息与页数快照。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _load_pymupdf4llm() -> Any:
    """
    [职责] 延迟加载 pymupdf4llm 以避免在无依赖环境下 import 失败。
    [边界] 只负责 import；不处理业务异常。
    [上游关系] parse_pdf 调用。
    [下游关系] _to_markdown 使用返回的模块对象。
    """
    try:
        import pymupdf4llm  # type: ignore  # docstring: 解析器依赖
    except Exception as exc:  # pragma: no cover - 依赖缺失场景
        raise ImportError("pymupdf4llm is required for pdf parsing") from exc  # docstring: 依赖缺失即失败
    return pymupdf4llm


def _load_pymupdf() -> Any:
    """
    [职责] 延迟加载 PyMuPDF（fitz）以读取页数信息。
    [边界] 仅用于页数统计；不做文本抽取。
    [上游关系] parse_pdf 调用。
    [下游关系] _page_count 使用返回的模块对象。
    """
    try:
        import fitz  # type: ignore  # docstring: PyMuPDF 依赖用于页数读取
    except Exception as exc:  # pragma: no cover - 依赖缺失场景
        raise ImportError("pymupdf (fitz) is required for page counting") from exc  # docstring: 页数读取依赖
    return fitz


def _page_count(pdf_path: Path) -> int:
    """
    [职责] 获取 PDF 页数（用于审计与 gate 断言）。
    [边界] 不做文本抽取；只读取元信息。
    [上游关系] parse_pdf 调用。
    [下游关系] 解析结果中的 pages 字段。
    """
    fitz = _load_pymupdf()  # docstring: 延迟导入以避免依赖在模块加载时失败
    with fitz.open(pdf_path) as doc:  # docstring: 仅打开文档获取页数
        return int(doc.page_count)  # docstring: 统一为 int 页数


def _normalize_markdown(raw: Any) -> str:
    """
    [职责] 归一化 pymupdf4llm 输出为单一 Markdown 字符串。
    [边界] 仅处理常见结构（str/list/tuple/dict）；其他类型直接报错。
    [上游关系] parse_pdf 的 _to_markdown 输出。
    [下游关系] 解析结果中的 markdown 字段。
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)):
        return "\n\n".join(str(x) for x in raw)  # docstring: 保留原顺序拼接为完整 Markdown
    if isinstance(raw, dict) and "markdown" in raw:
        return str(raw.get("markdown", ""))  # docstring: 兼容字典形态返回
    raise TypeError("pymupdf4llm output must be markdown string/list/dict")  # docstring: 保证 markdown 输出可用


def _to_markdown(pdf_path: Path) -> str:
    """
    [职责] 调用 pymupdf4llm 将 PDF 转为 Markdown（保持结构与顺序）。
    [边界] 不做额外清洗；不做内容裁剪；直接返回 Markdown。
    [上游关系] parse_pdf 调用。
    [下游关系] parse_pdf 的 markdown 输出。
    """
    mod = _load_pymupdf4llm()  # docstring: 获取 pymupdf4llm 模块
    if not hasattr(mod, "to_markdown"):
        raise AttributeError("pymupdf4llm.to_markdown is required")  # docstring: 强制使用官方入口

    to_markdown = getattr(mod, "to_markdown")  # docstring: 解析器主入口
    try:
        raw = to_markdown(str(pdf_path), write_images=False)  # docstring: 生成 Markdown（不写图片）
    except TypeError:
        raw = to_markdown(str(pdf_path))  # docstring: 兼容旧版本参数签名
    return _normalize_markdown(raw)  # docstring: 归一化输出为 Markdown 字符串


async def parse_pdf(*, pdf_path: str, parser_name: str = "pymupdf4llm", parse_version: str = "v1") -> Dict[str, Any]:
    """
    [职责] 解析 PDF 为 Markdown（用于后续法律友好切分）。
    [边界] 不进行 OCR；不负责分段与 embedding；不写入 DB。
    [上游关系] ingest/pipeline.py 调用（parse 阶段）。
    [下游关系] ingest/segment.py 使用 markdown 生成节点。
    """
    pdf_file = Path(pdf_path).expanduser().resolve()  # docstring: 标准化路径
    if not pdf_file.exists():
        raise FileNotFoundError(str(pdf_file))  # docstring: 明确输入文件不存在
    if not pdf_file.is_file():
        raise IsADirectoryError(str(pdf_file))  # docstring: 明确必须是文件路径（非目录/特殊文件）

    if parser_name != "pymupdf4llm":
        raise ValueError(f"unsupported parser: {parser_name}")  # docstring: 固定 parser 策略（guide 要求）

    markdown = _to_markdown(pdf_file)  # docstring: 使用 pymupdf4llm 生成 Markdown
    pages = _page_count(pdf_file)  # docstring: 使用 PyMuPDF 读取页数（不抽取文本）

    mod = _load_pymupdf4llm()  # docstring: 读取解析器版本用于审计/复现
    lib_version = getattr(mod, "__version__", None)
    meta_version = str(lib_version) if lib_version else parse_version

    meta = {
        "parser": parser_name,
        "version": meta_version,
        "lib": {"pymupdf4llm": str(lib_version)} if lib_version else {},
    }  # docstring: 解析器元信息快照（供审计/复现）

    return {
        "markdown": markdown,
        "pages": pages,
        "meta": meta,
    }  # docstring: 标准化解析结果（供 segment 使用）


async def parse(*, pdf_path: str, parser_name: str = "pymupdf4llm", parse_version: str = "v1") -> Dict[str, Any]:
    """
    [职责] parse_pdf 的别名入口（兼容旧调用）。
    [边界] 行为与 parse_pdf 一致。
    [上游关系] ingest/pipeline.py 的适配层（若调用 parse）。
    [下游关系] 返回与 parse_pdf 相同结构。
    """
    return await parse_pdf(  # docstring: 兼容旧函数名入口
        pdf_path=pdf_path, parser_name=parser_name, parse_version=parse_version
    )
