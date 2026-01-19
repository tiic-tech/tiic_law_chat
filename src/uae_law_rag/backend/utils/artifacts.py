# src/uae_law_rag/backend/utils/artifacts.py

from __future__ import annotations

import os
import re
from pathlib import Path

from typing import Dict, List

_PAGE_MARK_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->", re.IGNORECASE)

# docstring: repo relative default data root
DEFAULT_DATA_ROOT = ".data"


def get_repo_root() -> Path:
    """
    [职责] 尽力推断 repo root（以当前文件位置为基准）。
    [边界] 仅用于默认路径；生产环境建议显式设置 UAE_LAW_RAG_DATA_DIR。
    """
    # .../src/uae_law_rag/backend/utils/artifacts.py -> repo root = parents[5]
    # repo/
    #   src/uae_law_rag/backend/utils/artifacts.py
    return Path(__file__).resolve().parents[5]


def get_data_root() -> Path:
    """
    [职责] 获取运行时数据根目录（repo/.data）。
    [边界] 允许通过 UAE_LAW_RAG_DATA_DIR 覆盖；未设置时默认 repo/.data。
    """
    env = str(os.getenv("UAE_LAW_RAG_DATA_DIR", "") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (get_repo_root() / DEFAULT_DATA_ROOT).resolve()


def ensure_dir(p: Path) -> Path:
    """
    [职责] 确保目录存在。
    [边界] 失败则抛异常；调用方决定是否 best-effort。
    """
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_raw_dir() -> Path:
    """
    [职责] raw 输入目录（repo/.data/raw）。
    """
    return ensure_dir(get_data_root() / "raw")


def get_parsed_dir() -> Path:
    """
    [职责] parsed 工件目录（repo/.data/parsed）。
    """
    return ensure_dir(get_data_root() / "parsed")


def get_parsed_markdown_path(*, kb_id: str, file_id: str) -> Path:
    """
    [职责] 获取 parsed markdown 的稳定存放路径。
    [边界] 仅构造路径，不保证存在。
    """
    kb = str(kb_id or "").strip() or "default"
    fid = str(file_id or "").strip()
    if not fid:
        raise ValueError("file_id is required")
    # docstring: repo/.data/parsed/kb_<kb_id>/file_<file_id>/parsed.md
    base = get_parsed_dir() / f"kb_{kb}" / f"file_{fid}"
    ensure_dir(base)
    return base / "parsed.md"


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    """
    [职责] 原子写入文本文件（先写 tmp，再 replace）。
    [边界] Windows/跨盘 replace 可能失败；本项目默认本地开发环境。
    """
    path = path.resolve()
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def read_text(path: Path, encoding: str = "utf-8") -> str:
    """
    [职责] 读取文本文件。
    """
    return path.read_text(encoding=encoding)


def build_page_start_index(md: str) -> Dict[int, int]:
    """
    [职责] 在“全量 markdown 字符串”中，计算每个 page chunk 的起始 index（用于全量 offset -> 页内 offset 转换）。
    [边界] 依赖 <!-- page: N --> 标记；若缺失标记，返回 {1: 0}。
    """
    text = str(md or "")
    matches = list(_PAGE_MARK_RE.finditer(text))
    if not matches:
        return {1: 0}
    out: Dict[int, int] = {}
    for m in matches:
        try:
            page_no = int(m.group(1))
        except Exception:
            continue
        # 约定：page chunk 从 marker 开始（与 /records/page 的 split 逻辑对齐）
        out[page_no] = int(m.start())
    return out


def normalize_offsets_to_page_local(
    *,
    node_dicts: List[dict],
    markdown: str,
) -> List[dict]:
    """
    [职责] 基于 <!-- page: N --> 的全量起点，将 node_dicts 的 doc-level start/end 转换为页内 offset，
          写入 page_start_offset/page_end_offset（不覆盖原 start_offset/end_offset）。
    [边界] 若 node 未提供 page 或 start/end，则不写页内 offset；若缺 page mark，则按单页 {1:0} 处理。
    """
    page_start = build_page_start_index(markdown)
    if not page_start:
        page_start = {1: 0}

    out: List[dict] = []
    for n in node_dicts or []:
        d = dict(n or {})
        page = d.get("page")
        try:
            page_i = int(page) if page is not None else None
        except Exception:
            page_i = None

        if page_i is None:
            out.append(d)
            continue

        base = page_start.get(page_i)
        if base is None:
            # page 不在 index 中：异常数据，不写页内 offset
            out.append(d)
            continue

        def _coerce_int(v):
            if v is None:
                return None
            try:
                return int(v)
            except Exception:
                return None

        s = _coerce_int(d.get("start_offset"))
        e = _coerce_int(d.get("end_offset"))
        # docstring: 只写 page_*，不覆盖原 start/end（原 start/end 仍为 doc 全量 offset）
        if s is not None:
            page_s = s - int(base)
            if page_s < 0:
                raise ValueError(f"page_start_offset < 0 (page={page_i}, start_offset={s}, base={base})")
            d["page_start_offset"] = int(page_s)
        if e is not None:
            page_e = e - int(base)
            if page_e < 0:
                raise ValueError(f"page_end_offset < 0 (page={page_i}, end_offset={e}, base={base})")
            d["page_end_offset"] = int(page_e)

        # docstring: 记录 offset 模式（审计用）
        meta = dict(d.get("meta_data") or d.get("meta") or {})
        meta.setdefault("offset_mode", "doc_global+page_local")
        d["meta_data"] = meta
        out.append(d)

    return out
