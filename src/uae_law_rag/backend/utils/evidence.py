# src/uae_law_rag/backend/utils/evidence.py

"""
[Responsibility] Evidence grouping utils: aggregate hits into debug.evidence structure.
[Boundary] Pure function; no DB/HTTP access; no ORM; only structural grouping and caps.
[Upstream] Used before chat_service/debug builds evidence.
[Downstream] EvidencePanel/debug consumes the output.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


__all__ = ["group_evidence_hits"]


_UNKNOWN_PAGE_KEY = "_"  # docstring: unknown page bucket key


def _read_field(obj: Any, key: str) -> Any:
    """
    [Responsibility] Read a field from object or mapping safely.
    [Boundary] No exception; missing fields return None.
    [Upstream] group_evidence_hits calls.
    [Downstream] hit field access.
    """
    if obj is None:
        return None  # docstring: empty object fallback
    if isinstance(obj, Mapping):
        return obj.get(key)  # docstring: mapping access
    return getattr(obj, key, None)  # docstring: attribute access


def _coerce_str(value: Any) -> str:
    """
    [Responsibility] Normalize to a stripped string.
    [Boundary] Empty values return empty string; no semantic cleaning.
    """
    return str(value or "").strip()  # docstring: string fallback


def _coerce_int(value: Any) -> Optional[int]:
    """
    [Responsibility] Convert to int; return None on failure.
    [Boundary] No exceptions; no range validation.
    """
    try:
        return int(value)  # docstring: force int
    except Exception:
        return None  # docstring: conversion fallback


def _coerce_cap(value: Any, default: int) -> int:
    """
    [Responsibility] Normalize cap params (non-negative int).
    [Boundary] Fallback to default on failure; negative -> 0.
    """
    v = _coerce_int(value)
    if v is None:
        v = int(default)  # docstring: default fallback
    if v < 0:
        v = 0  # docstring: negative -> zero
    return int(v)  # docstring: return non-negative int


def _normalize_source(value: Any) -> str:
    """
    [Responsibility] Normalize source field (lower + unknown fallback).
    [Boundary] No enum validation; keep extensibility.
    """
    src = _coerce_str(value).lower()  # docstring: source lower
    return src or "unknown"  # docstring: empty -> unknown


def _normalize_page(value: Any) -> Optional[int]:
    """
    [Responsibility] Normalize page (return only for >0).
    [Boundary] 0/None/non-numeric -> None.
    """
    page = _coerce_int(value)
    if page is None or page <= 0:
        return None  # docstring: invalid page fallback
    return int(page)  # docstring: return positive page


def _normalize_document_id(value: Any) -> Optional[str]:
    """
    [Responsibility] Normalize document_id.
    [Boundary] Empty -> None.
    """
    doc_id = _coerce_str(value)
    return doc_id or None  # docstring: empty fallback


def _normalize_node_id(value: Any) -> Optional[str]:
    """
    [Responsibility] Normalize node_id.
    [Boundary] Empty -> None.
    """
    node_id = _coerce_str(value)
    return node_id or None  # docstring: empty fallback


def group_evidence_hits(
    hits: Sequence[Any],
    *,
    max_documents: int = 20,
    max_nodes_per_document: int = 50,
    max_pages_per_document: int = 50,
    version: str = "v1",
) -> Dict[str, Any]:
    """
    [Responsibility] Group hits into debug.evidence structure and apply caps.
    [Boundary] Uses only node_id/document_id/page/source/file_id; no DB; no reordering.
    [Upstream] Called before chat_service/debug builds evidence.
    [Downstream] EvidencePanel/debug consumes the output.
    """
    max_documents_i = _coerce_cap(max_documents, 20)  # docstring: global doc cap
    max_nodes_i = _coerce_cap(max_nodes_per_document, 50)  # docstring: per-doc node cap
    max_pages_i = _coerce_cap(max_pages_per_document, 50)  # docstring: per-doc page cap

    document_ids: list[str] = []  # docstring: document_ids (dedupe, keep order)
    allowed_docs: set[str] = set()  # docstring: allowed doc set
    by_source: Dict[str, Dict[str, Any]] = {}  # docstring: evidence.by_source
    seen_node_ids: Dict[Tuple[str, str], set[str]] = {}  # docstring: (source, doc_id) -> seen node_ids

    total_hits_in = 0  # docstring: total hits in
    total_hits_used = 0  # docstring: total hits used
    dropped_missing_document_id = 0  # docstring: dropped hits missing document_id
    dropped_missing_node_id = 0  # docstring: dropped hits missing node_id
    unknown_page_count = 0  # docstring: hits with unknown page
    deduped_node_count = 0  # docstring: duplicate node_id drops

    for hit in hits or []:
        total_hits_in += 1  # docstring: hit counter
        doc_id = _normalize_document_id(_read_field(hit, "document_id"))  # docstring: doc_id fallback
        if not doc_id:
            dropped_missing_document_id += 1  # docstring: missing document_id
            continue  # docstring: no document_id, skip

        source = _normalize_source(_read_field(hit, "source"))  # docstring: source normalized
        node_id = _normalize_node_id(
            _read_field(hit, "node_id") or _read_field(hit, "nodeId") or _read_field(hit, "id")
        )  # docstring: node_id field compatibility
        if not node_id:
            dropped_missing_node_id += 1  # docstring: missing node_id
            continue  # docstring: no node_id, skip

        page = _normalize_page(_read_field(hit, "page"))  # docstring: page normalized
        if page is None:
            unknown_page_count += 1  # docstring: unknown page hit
            page_key = _UNKNOWN_PAGE_KEY
        else:
            page_key = str(page)

        if doc_id not in allowed_docs and len(document_ids) >= max_documents_i:
            continue  # docstring: exceed max_documents

        seen = seen_node_ids.get((source, doc_id))
        if seen is None:
            seen = set()
            seen_node_ids[(source, doc_id)] = seen

        if node_id in seen:
            deduped_node_count += 1  # docstring: deduped node_id
            continue  # docstring: skip duplicate node_id

        if len(seen) >= max_nodes_i:
            continue  # docstring: per-doc node cap

        source_bucket = by_source.get(source)
        if source_bucket is None:
            source_bucket = {"by_document": {}}  # docstring: init source bucket
            by_source[source] = source_bucket

        by_document = source_bucket["by_document"]
        doc_entry = by_document.get(doc_id)
        created_doc_entry = False
        if doc_entry is None:
            doc_entry = {"file_id": None, "pages": {}}  # docstring: init doc entry
            by_document[doc_id] = doc_entry
            created_doc_entry = True

        pages = doc_entry["pages"]
        if page_key != _UNKNOWN_PAGE_KEY and page_key not in pages:
            known_pages_n = sum(1 for key in pages.keys() if key != _UNKNOWN_PAGE_KEY)
            if known_pages_n >= max_pages_i:
                if created_doc_entry and not pages:
                    by_document.pop(doc_id, None)  # docstring: cleanup empty doc entry
                    if not by_document:
                        by_source.pop(source, None)  # docstring: cleanup empty source bucket
                continue  # docstring: per-doc page cap

        if page_key not in pages:
            pages[page_key] = []  # docstring: init page list

        pages[page_key].append(node_id)  # docstring: append node_id in order
        seen.add(node_id)  # docstring: track unique node_id
        total_hits_used += 1  # docstring: used hit count

        if doc_id not in allowed_docs:
            allowed_docs.add(doc_id)  # docstring: record allowed doc
            document_ids.append(doc_id)  # docstring: append in order

        file_id = _coerce_str(_read_field(hit, "file_id"))
        if file_id and not doc_entry.get("file_id"):
            doc_entry["file_id"] = file_id  # docstring: first non-empty file_id wins

    return {
        "version": _coerce_str(version) or "v1",
        "document_ids": document_ids,
        "by_source": by_source,
        "caps": {
            "max_documents": max_documents_i,
            "max_nodes_per_document": max_nodes_i,
            "max_pages_per_document": max_pages_i,
        },
        "meta": {
            "note": "node_ids only; fetch node detail via /api/records/node/{node_id}",
            "stats": {
                "dropped_missing_document_id": dropped_missing_document_id,
                "dropped_missing_node_id": dropped_missing_node_id,
                "unknown_page_count": unknown_page_count,
                "deduped_node_count": deduped_node_count,
                "total_hits_in": total_hits_in,
                "total_hits_used": total_hits_used,
            },
        },
    }  # docstring: evidence grouping output
