# playground/sql_gate/test_engine_gate.py

"""
[职责] engine gate：验证 db/engine.py 的最小可用性（可创建 engine、init_db、drop_db）。
[边界] 不跑 FastAPI；不引入业务 pipeline；只验证 DB 基础设施可用且不污染默认路径。
[上游关系] 依赖 backend/db/engine.py 与 backend/db/base.py、backend/db/models 注册。
[下游关系] 后续 services/api/deps 依赖 get_session 与 SessionLocal 的一致行为。
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from uae_law_rag.backend.db.engine import create_engine, drop_db, init_db


pytestmark = pytest.mark.sql_gate


@pytest.mark.asyncio
async def test_engine_init_and_drop(tmp_path) -> None:
    """Init DB creates tables; drop DB removes them (on isolated sqlite file)."""  # docstring: 防污染默认本地库
    db_file = tmp_path / "engine_gate.db"  # docstring: 独立临时 sqlite 文件
    url = f"sqlite+aiosqlite:///{db_file}"

    engine: AsyncEngine = create_engine(url=url, echo=False)  # docstring: 临时引擎
    try:
        await drop_db(engine=engine)  # docstring: 幂等（即使不存在也应安全）
        await init_db(engine=engine)  # docstring: create_all

        async with engine.connect() as conn:
            # SQLite introspection
            rows = (await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))).fetchall()
            names = {r[0] for r in rows}

        # 至少应包含核心业务表（不要求全部逐一列举）
        assert "user" in names
        assert "conversation" in names
        assert "message" in names
        assert "knowledge_base" in names
        assert "node" in names
        assert "retrieval_record" in names
        assert "generation_record" in names

        await drop_db(engine=engine)

        async with engine.connect() as conn:
            rows2 = (await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))).fetchall()
            names2 = {r[0] for r in rows2}

        # drop_all 后，这些表应该不存在（sqlite 仍可能保留 sqlite_sequence 等内部表）
        assert "user" not in names2
        assert "conversation" not in names2
        assert "message" not in names2
    finally:
        await engine.dispose()
