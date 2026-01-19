# playground/fastapi_gate/retrieval_strategy/evidence_replay_gate.py

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


DEFAULT_BASE_URL = "http://127.0.0.1:18000"
DEFAULT_USER_ID = "dev-user"
DEFAULT_KB_ID = "default"
DEFAULT_QUERY = "Financing"
DEFAULT_TIMEOUT_S = 60


@dataclass(frozen=True)
class GateResult:
    ok: bool
    node_id: Optional[str]
    document_id: Optional[str]
    page: Optional[int]


def _http_json(
    url: str, *, method: str, headers: Dict[str, str], payload: Optional[Dict[str, Any]], timeout_s: int
) -> Any:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers = dict(headers)
        headers["Content-Type"] = "application/json"
    req = Request(url=url, method=method, headers=headers, data=body)
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def _pick_node_id_from_evidence(evidence: Any) -> Optional[str]:
    """
    [职责] 从 debug.evidence 中挑一个有 page 的 node_id。
    [边界] 优先 reranked/fused/vector/keyword；跳过 unknown page。
    """
    if not isinstance(evidence, dict):
        return None
    by_source = evidence.get("by_source")
    if not isinstance(by_source, dict):
        return None

    def _pick_from_source(src: str) -> Optional[str]:
        bucket = by_source.get(src)
        if not isinstance(bucket, dict):
            return None
        by_document = bucket.get("by_document")
        if not isinstance(by_document, dict):
            return None
        for _doc_id, doc_entry in by_document.items():
            if not isinstance(doc_entry, dict):
                continue
            pages = doc_entry.get("pages")
            if not isinstance(pages, dict):
                continue
            for page_key, node_ids in pages.items():
                if page_key in ("_", "0"):
                    continue
                try:
                    if int(page_key) <= 0:
                        continue
                except Exception:
                    continue
                if isinstance(node_ids, list) and node_ids:
                    nid = str(node_ids[0] or "").strip()
                    if nid:
                        return nid
        return None

    for key in ("reranked", "fused", "vector", "keyword"):
        nid = _pick_from_source(key)
        if nid:
            return nid

    for key in by_source.keys():
        nid = _pick_from_source(str(key))
        if nid:
            return nid
    return None


def run_gate(
    *,
    base_url: str,
    user_id: str,
    kb_id: str,
    query: str,
    timeout_s: int,
) -> GateResult:
    try:
        # 1) chat(debug=true) -> debug.evidence
        url_chat = f"{str(base_url).rstrip('/')}/api/chat"
        chat = _http_json(
            url_chat,
            method="POST",
            headers={"x-user-id": str(user_id)},
            payload={"kb_id": str(kb_id), "query": str(query), "debug": True},
            timeout_s=int(timeout_s),
        )

        debug = (chat or {}).get("debug") if isinstance(chat, dict) else None
        evidence = (debug or {}).get("evidence") if isinstance(debug, dict) else None
        if not isinstance(evidence, dict):
            raise RuntimeError("missing debug.evidence")

        version = evidence.get("version")
        document_ids = evidence.get("document_ids") or []
        by_source = evidence.get("by_source") or {}
        if str(version) != "v1":
            raise RuntimeError(f"version mismatch {version!r}")
        if not isinstance(document_ids, list) or not document_ids:
            raise RuntimeError("empty document_ids")
        if not isinstance(by_source, dict) or not (("fused" in by_source) or ("reranked" in by_source)):
            raise RuntimeError("by_source missing fused/reranked")

        node_id = _pick_node_id_from_evidence(evidence)
        if not node_id:
            raise RuntimeError("no node_id found in evidence")

        # 2) node preview -> document_id + page
        url_node = f"{str(base_url).rstrip('/')}/api/records/node/{node_id}"
        node = _http_json(
            url_node,
            method="GET",
            headers={"x-user-id": str(user_id)},
            payload=None,
            timeout_s=int(timeout_s),
        )
        document_id = str((node or {}).get("document_id") or "").strip()
        page = (node or {}).get("page")
        if not document_id or not page:
            raise RuntimeError("node missing document_id/page")

        # 3) page replay
        url_page = (
            f"{str(base_url).rstrip('/')}/api/records/page"
            f"?document_id={document_id}&page={int(page)}&kb_id={str(kb_id)}&max_chars=8000"
        )
        page_resp = _http_json(
            url_page,
            method="GET",
            headers={"x-user-id": str(user_id)},
            payload=None,
            timeout_s=int(timeout_s),
        )

        content = str((page_resp or {}).get("content") or "")
        ok = bool(content.strip()) and f"page: {int(page)}" in content.lower()

        return GateResult(ok=ok, node_id=node_id, document_id=document_id, page=int(page))

    except (HTTPError, URLError) as e:
        raise RuntimeError(str(e)) from e
    except Exception as e:
        raise RuntimeError(str(e)) from e


def test_evidence_replay_gate() -> None:
    import os
    import pytest

    if not os.getenv("EVIDENCE_REPLAY_GATE"):
        pytest.skip("Set EVIDENCE_REPLAY_GATE=1 to run HTTP gate.")

    res = run_gate(
        base_url=os.getenv("EVIDENCE_REPLAY_GATE_BASE_URL", DEFAULT_BASE_URL),
        user_id=os.getenv("EVIDENCE_REPLAY_GATE_USER_ID", DEFAULT_USER_ID),
        kb_id=os.getenv("EVIDENCE_REPLAY_GATE_KB_ID", DEFAULT_KB_ID),
        query=os.getenv("EVIDENCE_REPLAY_GATE_QUERY", DEFAULT_QUERY),
        timeout_s=int(os.getenv("EVIDENCE_REPLAY_GATE_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))),
    )
    assert res.ok, f"evidence_replay_gate failed: node_id={res.node_id} doc={res.document_id} page={res.page}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="FastAPI gate: evidence replay via debug.evidence.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--user-id", default=DEFAULT_USER_ID)
    p.add_argument("--kb-id", default=DEFAULT_KB_ID)
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        res = run_gate(
            base_url=str(args.base_url),
            user_id=str(args.user_id),
            kb_id=str(args.kb_id),
            query=str(args.query),
            timeout_s=int(args.timeout_s),
        )
    except Exception as e:
        print(f"[evidence_replay_gate] ERROR: {e}", file=sys.stderr)
        return 1

    print(f"[evidence_replay_gate] node_id={res.node_id}")
    print(f"[evidence_replay_gate] document_id={res.document_id} page={int(res.page or 0)}")
    print(f"[evidence_replay_gate] gate={'PASS' if res.ok else 'FAIL'}")

    if bool(args.json):
        print(
            json.dumps(
                {"ok": res.ok, "node_id": res.node_id, "document_id": res.document_id, "page": res.page},
                ensure_ascii=False,
            )
        )

    return 0 if res.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
