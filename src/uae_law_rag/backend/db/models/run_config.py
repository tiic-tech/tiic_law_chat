# src/uae_law_rag/backend/db/models/run_config.py

from __future__ import annotations

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base, TimestampMixin


class RunConfigModel(Base, TimestampMixin):
    """
    [职责] RunConfig：全局运行配置（默认值快照，持久化到 DB）。
    [边界] 仅保存配置快照；不负责业务逻辑或校验。
    [上游关系] scripts/set_run_config.py 写入；service 层读取并作为默认值。
    [下游关系] chat/ingest 等服务读取 config 作为默认策略。
    """

    __tablename__ = "run_config"

    name: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        default="default",
        comment="配置名称（默认 default）",
    )

    config: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        comment="运行配置快照（JSON）",
    )
