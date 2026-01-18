# playground/fastapi_gate/retrieval_strategy/keyword_recall_auto_detail_gate.py

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
    keywords_n: int
    metrics_n: int
    with_samples_n: int


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
    sample_n: int,
    timeout_s: int,
) -> GateResult:
    url = f"{base_url.rstrip('/')}/api/evaluator/keyword_recall_auto_detail"
    headers = {"x-user-id": str(user_id)}

    resp = _http_json(
        url,
        method="POST",
        headers=headers,
        payload={"kb_id": kb_id, "raw_query": raw_query, "sample_n": int(sample_n)},
        timeout_s=timeout_s,
    )

    keywords = list(((resp.get("analysis") or {}).get("keywords_list") or []))
    metrics: List[Dict[str, Any]] = list(resp.get("metrics") or [])

    with_samples = 0
    for m in metrics:
        ms = list(m.get("missing_sample") or [])
        es = list(m.get("extra_sample") or [])
        if len(ms) > 0 or len(es) > 0:
            with_samples += 1

    print("[keyword_recall_auto_detail_gate] keywords_n =", len(keywords))
    print("[keyword_recall_auto_detail_gate] metrics_n  =", len(metrics))
    print("[keyword_recall_auto_detail_gate] with_samples_n =", with_samples)

    ok = (len(keywords) > 0) and (len(metrics) == len(keywords))
    # docstring: samples 不强制每个 keyword 都有（取决于数据分布），但至少结构必须存在
    if metrics:
        m0 = metrics[0]
        if "missing_sample" not in m0 or "extra_sample" not in m0:
            ok = False
            print("[keyword_recall_auto_detail_gate] FAIL: missing_sample/extra_sample not present in metric")

    status = "PASS" if ok else "FAIL"
    print(f"[keyword_recall_auto_detail_gate] gate={status}")

    return GateResult(
        ok=ok,
        keywords_n=len(keywords),
        metrics_n=len(metrics),
        with_samples_n=with_samples,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FastAPI gate: keyword_recall_auto_detail returns samples.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--user-id", default=DEFAULT_USER_ID)
    p.add_argument("--kb-id", default=DEFAULT_KB_ID)
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--sample-n", type=int, default=DEFAULT_SAMPLE_N)
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
            sample_n=int(args.sample_n),
            timeout_s=int(args.timeout_s),
        )
        if args.json:
            print(json.dumps(res.__dict__, ensure_ascii=False, default=str))
        return 0 if res.ok else 2
    except (HTTPError, URLError) as e:
        print(f"[keyword_recall_auto_detail_gate] HTTP ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[keyword_recall_auto_detail_gate] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
