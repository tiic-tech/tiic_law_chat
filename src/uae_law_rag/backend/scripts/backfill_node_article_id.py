# src/uae_law_rag/backend/scripts/backfill_node_article_id.py
"""
[职责] Backfill NodeModel.article_id from section_path/text using rule-based regex.
[边界] 仅回填 article_id；不改动 text；不强制修改 section_path 语义。
[用途] 修复历史数据中 article_id 全为空导致的 citations locator 解释性不足。
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from typing import Optional, Tuple

ARTICLE_RE = re.compile(
    r"\bArticle\s*[\(\[]?\s*(\d+)\s*[\)\]]?\b",
    flags=re.IGNORECASE,
)


def _extract_article_id(s: str) -> Optional[str]:
    if not s:
        return None
    m = ARTICLE_RE.search(s)
    if not m:
        return None
    num = m.group(1)
    return f"Article {num}" if num else None


def _pick_article_source(section_path: Optional[str], text: Optional[str]) -> Tuple[Optional[str], str]:
    # Prefer section_path (often contains the heading), fallback to text
    for src, name in ((section_path, "section_path"), (text, "text")):
        if not src:
            continue
        aid = _extract_article_id(src)
        if aid:
            return aid, name
    return None, "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("select id, article_id, section_path, text from node")
    rows = cur.fetchall()

    updated = 0
    src_counts = {"section_path": 0, "text": 0, "none": 0}

    for r in rows:
        if (r["article_id"] or "").strip():
            continue
        aid, src = _pick_article_source(r["section_path"], r["text"])
        src_counts[src] = src_counts.get(src, 0) + 1
        if not aid:
            continue
        if not args.dry_run:
            cur.execute("update node set article_id = ? where id = ?", (aid, r["id"]))
        updated += 1

    if not args.dry_run:
        con.commit()

    print(f"updated={updated} total={len(rows)} src_counts={src_counts}")
    con.close()


if __name__ == "__main__":
    main()
