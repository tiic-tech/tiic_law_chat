# playground/sql_gate/test_seed_default_kb_gate.py

"""
[职责] seed gate：确保 init_db --seed 会创建 dev-user 与 default KB，并能被 /api/admin/kbs 查询到。
[边界] 不依赖外部 Milvus；仅验证 DB seed + API 可见性合同。
[上游关系] 依赖 scripts/init_db.py 的 seed 实现、FastAPI 路由 wiring。
[下游关系] 保障 M1 dev-loop：drop+seed 后立即可用 kb_id="default"。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
from httpx import ASGITransport, AsyncClient

from uae_law_rag.backend.main import app
from uae_law_rag.backend.scripts import init_db as init_db_mod


def _extract_kbs(payload: Any) -> List[Dict[str, Any]]:
    """
    [职责] 从 /api/admin/kbs 的响应中提取 KB 列表（兼容 list / {items:[]} / {data:[]}）。
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("items", "data", "kbs"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


@pytest.mark.asyncio
async def test_init_db_seed_default_kb_exposed_by_admin_api(tmp_path: Path) -> None:
    # docstring: 使用临时 sqlite 文件，避免污染本地 .Local
    db_path = tmp_path / "seed_gate.db"
    db_url = f"sqlite+aiosqlite:///{db_path.as_posix()}"

    # docstring: 直接调用 _run_async，等价于 CLI: init_db --drop --seed --db-url <...>
    result = await init_db_mod._run_async(
        db_url=db_url, drop=True, seed=True, seed_fts=True, rebuild_fts=True, echo=False
    )
    assert result.get("ok") is True
    assert result.get("created") is True
    assert result.get("seeded") is True
    assert result.get("seed_status") == "ok"
    seed = result.get("seed") or {}
    assert seed.get("kb_id") == "default"

    # docstring: 让后端 app 使用同一个 DB（依赖 db.engine 的 env/配置读取）
    os.environ["UAE_LAW_RAG_DATABASE_URL"] = db_url

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # docstring: admin kbs 查询
        r = await client.get("/api/admin/kbs", headers={"x-user-id": "dev-user"})
        assert r.status_code == 200, r.text
        data = r.json()
        kbs = _extract_kbs(data)

    def _kb_identifier(kb: Dict[str, Any]) -> str:
        """
        [职责] 兼容不同 API 输出：优先 kb_id，其次 id。
        """
        return str(kb.get("kb_id") or kb.get("id") or "").strip()

    assert any(_kb_identifier(kb) == "default" for kb in kbs), f"missing default KB. payload={data}"

    d = next(kb for kb in kbs if str(kb.get("kb_id")) == "default")
    assert str(d.get("id")).strip(), f"default KB missing db id. payload={data}"
