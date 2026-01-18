# playground/fastapi_gate/retrieval_strategy/page_replay_gate.py
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional, Sequence
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


DEFAULT_BASE_URL = "http://127.0.0.1:18000"
DEFAULT_USER_ID = "dev-user"
DEFAULT_KB_ID = "default"
DEFAULT_QUERY = "Financing of public companies"
DEFAULT_TIMEOUT_S = 60


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="FastAPI gate: page replay endpoint.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--user-id", default=DEFAULT_USER_ID)
    p.add_argument("--kb-id", default=DEFAULT_KB_ID)
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        # 1) call keyword_recall_auto_detail to get a node_id sample
        url_auto = f"{str(args.base_url).rstrip('/')}/api/evaluator/keyword_recall_auto_detail"
        auto = _http_json(
            url_auto,
            method="POST",
            headers={"x-user-id": str(args.user_id)},
            payload={
                "kb_id": str(args.kb_id),
                "raw_query": str(args.query),
                "sample_n": 5,
            },
            timeout_s=int(args.timeout_s),
        )

        node_id = None
        for m in (auto or {}).get("metrics") or []:
            miss = (m or {}).get("missing_sample") or []
            extra = (m or {}).get("extra_sample") or []
            if miss:
                node_id = miss[0]
                break
            if extra:
                node_id = extra[0]
                break
        if not node_id:
            print("[page_replay_gate] ERROR: no sample node_id from keyword_recall_auto_detail", file=sys.stderr)
            return 2

        # 2) node preview -> get document_id + page
        url_node = f"{str(args.base_url).rstrip('/')}/api/records/node/{node_id}"
        node = _http_json(
            url_node,
            method="GET",
            headers={"x-user-id": str(args.user_id)},
            payload=None,
            timeout_s=int(args.timeout_s),
        )
        document_id = (node or {}).get("document_id")
        page = (node or {}).get("page")
        if not document_id or not page:
            print("[page_replay_gate] ERROR: node missing document_id/page", file=sys.stderr)
            return 2

        # 3) page replay
        url_page = (
            f"{str(args.base_url).rstrip('/')}/api/records/page"
            f"?document_id={document_id}&page={int(page)}&kb_id={str(args.kb_id)}&max_chars=8000"
        )
        page_resp = _http_json(
            url_page,
            method="GET",
            headers={"x-user-id": str(args.user_id)},
            payload=None,
            timeout_s=int(args.timeout_s),
        )

        content = str((page_resp or {}).get("content") or "")
        ok = bool(content.strip()) and f"page: {int(page)}" in content.lower()

        print(f"[page_replay_gate] node_id={node_id}")
        print(f"[page_replay_gate] document_id={document_id} page={int(page)}")
        print(f"[page_replay_gate] content_len={len(content)} ok={ok}")
        print(f"[page_replay_gate] gate={'PASS' if ok else 'FAIL'}")

        if bool(args.json):
            print(
                json.dumps(
                    {"ok": ok, "node_id": node_id, "document_id": document_id, "page": int(page)}, ensure_ascii=False
                )
            )

        return 0 if ok else 2

    except (HTTPError, URLError) as e:
        print(f"[page_replay_gate] ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[page_replay_gate] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
