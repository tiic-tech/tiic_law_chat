# playground/fastapi_gate/records/node_preview_gate.py

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:18000"
DEFAULT_USER_ID = "dev-user"
DEFAULT_KB_ID = "default"
DEFAULT_QUERY = "Financing of public companies"
DEFAULT_SAMPLE_N = 10
DEFAULT_TIMEOUT_S = 60


@dataclass(frozen=True)
class GateResult:
    ok: bool
    node_id: str
    has_excerpt: bool
    has_locator_fields: bool


def _http_json(
    url: str,
    *,
    method: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]],
    timeout_s: int,
) -> Dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = dict(headers)
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, method=str(method).upper(), headers=headers)
    with urlopen(req, timeout=int(timeout_s)) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _pick_any_sample_node_id(detail_resp: Dict[str, Any]) -> Optional[str]:
    metrics: List[Dict[str, Any]] = list(detail_resp.get("metrics") or [])
    for m in metrics:
        ms = list(m.get("missing_sample") or [])
        if ms:
            return str(ms[0])
        es = list(m.get("extra_sample") or [])
        if es:
            return str(es[0])
    return None


def run_gate(
    *,
    base_url: str,
    user_id: str,
    kb_id: str,
    raw_query: str,
    sample_n: int,
    timeout_s: int,
) -> GateResult:
    headers = {"x-user-id": str(user_id)}

    detail_url = f"{base_url.rstrip('/')}/api/evaluator/keyword_recall_auto_detail"
    detail = _http_json(
        detail_url,
        method="POST",
        headers=headers,
        payload={"kb_id": kb_id, "raw_query": raw_query, "sample_n": int(sample_n)},
        timeout_s=timeout_s,
    )

    node_id = _pick_any_sample_node_id(detail)
    if not node_id:
        print("[node_preview_gate] WARNING: no sample node_id found; try a different query/KB", file=sys.stderr)
        return GateResult(ok=False, node_id="", has_excerpt=False, has_locator_fields=False)

    node_url = f"{base_url.rstrip('/')}/api/records/node/{node_id}?kb_id={kb_id}&max_chars=800"
    node_view = _http_json(
        node_url,
        method="GET",
        headers=headers,
        payload=None,
        timeout_s=timeout_s,
    )

    excerpt = str(node_view.get("text_excerpt") or "")
    has_excerpt = len(excerpt.strip()) > 0

    has_locator_fields = ("page" in node_view) and ("article_id" in node_view) and ("section_path" in node_view)

    print(f"[node_preview_gate] node_id={node_id}")
    print(f"[node_preview_gate] excerpt_len={len(excerpt)} has_excerpt={has_excerpt}")
    print(f"[node_preview_gate] has_locator_fields={has_locator_fields}")

    ok = bool(node_view.get("node_id")) and bool(node_view.get("document_id")) and has_excerpt and has_locator_fields
    print(f"[node_preview_gate] gate={'PASS' if ok else 'FAIL'}")

    return GateResult(ok=ok, node_id=node_id, has_excerpt=has_excerpt, has_locator_fields=has_locator_fields)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FastAPI gate: node preview via records/node/{node_id}.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--user-id", default=DEFAULT_USER_ID)
    p.add_argument("--kb-id", default=DEFAULT_KB_ID)
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--sample-n", type=int, default=DEFAULT_SAMPLE_N)
    p.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        res = run_gate(
            base_url=str(args.base_url),
            user_id=str(args.user_id),
            kb_id=str(args.kb_id),
            raw_query=str(args.query),
            sample_n=int(args.sample_n),
            timeout_s=int(args.timeout_s),
        )
        return 0 if res.ok else 2
    except (HTTPError, URLError) as e:
        print(f"[node_preview_gate] HTTP ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[node_preview_gate] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
