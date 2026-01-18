# src/uae_law_rag/backend/utils/artifacts.py

from __future__ import annotations

import os
from pathlib import Path

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
