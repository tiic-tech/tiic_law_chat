# playground/fastapi_gate/retrieval_strategy/query_plan_kwrecall_auto_gate.py

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
DEFAULT_TIMEOUT_S = 60


@dataclass(frozen=True)
class GateResult:
    ok: bool
    kb_id: str
    raw_query: str
    keywords_n: int
    metrics_n: int
    timing_ms: Dict[str, Any]


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


def run_gate(
    *,
    base_url: str,
    user_id: str,
    kb_id: str,
    raw_query: str,
    timeout_s: int,
) -> GateResult:
    qp_url = f"{base_url.rstrip('/')}/api/evaluator/query_plan"
    auto_url = f"{base_url.rstrip('/')}/api/evaluator/keyword_recall_auto"
    headers = {"x-user-id": str(user_id)}

    qp = _http_json(
        qp_url,
        method="POST",
        headers=headers,
        payload={"kb_id": kb_id, "raw_query": raw_query},
        timeout_s=timeout_s,
    )

    auto = _http_json(
        auto_url,
        method="POST",
        headers=headers,
        payload={"kb_id": kb_id, "raw_query": raw_query, "sample_n": 10},
        timeout_s=timeout_s,
    )

    qp_keywords: List[str] = list(((qp.get("analysis") or {}).get("keywords_list") or []))
    auto_keywords: List[str] = list(((auto.get("analysis") or {}).get("keywords_list") or []))
    metrics: List[Dict[str, Any]] = list(auto.get("metrics") or [])

    print("[query_plan_kwrecall_auto_gate] query_plan.keywords_n =", len(qp_keywords))
    print("[query_plan_kwrecall_auto_gate] auto.analysis.keywords_n =", len(auto_keywords))
    print("[query_plan_kwrecall_auto_gate] auto.metrics_n =", len(metrics))
    print("[query_plan_kwrecall_auto_gate] timing_ms =", auto.get("timing_ms") or {})

    # Gate: auto keywords should match query_plan keywords exactly (same builder rule_v1)
    same = qp_keywords == auto_keywords
    non_empty = len(auto_keywords) > 0
    aligned = len(metrics) == len(auto_keywords)

    ok = bool(same and non_empty and aligned)

    status = "PASS" if ok else "FAIL"
    if not same:
        print("[query_plan_kwrecall_auto_gate] MISMATCH: qp_keywords != auto_keywords")
        print("  qp_keywords  =", qp_keywords)
        print("  auto_keywords=", auto_keywords)

    print(f"[query_plan_kwrecall_auto_gate] gate={status}")
    return GateResult(
        ok=ok,
        kb_id=str(kb_id),
        raw_query=str(raw_query),
        keywords_n=len(auto_keywords),
        metrics_n=len(metrics),
        timing_ms=dict(auto.get("timing_ms") or {}),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FastAPI gate: query_plan -> keyword_recall_auto closed loop.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--user-id", default=DEFAULT_USER_ID)
    p.add_argument("--kb-id", default=DEFAULT_KB_ID)
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        res = run_gate(
            base_url=str(args.base_url),
            user_id=str(args.user_id),
            kb_id=str(args.kb_id),
            raw_query=str(args.query),
            timeout_s=int(args.timeout_s),
        )
        if args.json:
            print(json.dumps(res.__dict__, ensure_ascii=False, default=str))
        return 0 if res.ok else 2
    except (HTTPError, URLError) as e:
        print(f"[query_plan_kwrecall_auto_gate] HTTP ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[query_plan_kwrecall_auto_gate] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
