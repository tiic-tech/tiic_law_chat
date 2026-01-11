# src/uae_law_rag/backend/pipelines/base/timing.py

"""
[职责] timing 基础设施：为 pipelines 提供统一的阶段计时（ms）收集与导出能力，支撑审计、调试与性能门禁。
[边界] 不做分布式 tracing、不做 profiler；不负责日志落地；仅提供轻量计时器与可序列化的 timing_ms dict。
[上游关系] pipelines/* 在各阶段（parse/segment/embed/keyword/vector/fusion/rerank/generate 等）调用计时器。
[下游关系] DB models 的 timing_ms(JSON) 字段、schemas 的 timing_ms 合同、gate tests 的性能/结构断言可直接消费。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


def _now_ms() -> float:
    """
    [职责] 获取高精度时间戳（毫秒）。
    [边界] 仅用于相对耗时计算；不作为业务时间戳写入审计（业务时间戳用 datetime）。
    [上游关系] StageTimer / TimingCollector 调用。
    [下游关系] 阶段耗时统计（ms）。
    """
    return time.perf_counter() * 1000.0


@dataclass
class TimingCollector:
    """
    [职责] TimingCollector：统一收集 pipeline 各阶段耗时并导出 dict[str, float]（ms）。
    [边界] 不强制阶段命名集合；允许自定义 stage key；同名 stage 可累加或覆盖由调用方选择。
    [上游关系] pipelines 在每阶段用 stage(...) 包裹或显式 add_ms(...) 写入耗时。
    [下游关系] 写入 DB timing_ms 字段；或映射到 schemas 的 timing_ms；gate tests 可做结构断言。
    """

    _stages_ms: Dict[str, float] = field(default_factory=dict)
    _start_ms: float = field(default_factory=_now_ms)

    def reset(self) -> None:
        """
        [职责] 重置计时器（清空阶段耗时，重置总计时起点）。
        [边界] 不做线程安全保证；假设单请求/单协程内使用。
        [上游关系] pipeline 复用 collector 或重跑时调用。
        [下游关系] 后续 stage 统计与 total_ms。
        """
        self._stages_ms.clear()
        self._start_ms = _now_ms()

    def add_ms(self, key: str, ms: float, *, accumulate: bool = True) -> None:
        """
        [职责] 写入某阶段耗时（ms）。
        [边界] 不校验 key 语义；ms 需为非负数（负数会被截断为 0）。
        [上游关系] stage context manager 或调用方自行计算耗时后写入。
        [下游关系] to_dict() 输出的 timing_ms JSON。
        """
        k = str(key).strip()
        if not k:
            return
        v = float(ms)
        if v < 0:
            v = 0.0
        if accumulate:
            self._stages_ms[k] = self._stages_ms.get(k, 0.0) + v
        else:
            self._stages_ms[k] = v

    @contextmanager
    def stage(self, key: str, *, accumulate: bool = False) -> Iterator[None]:
        """
        [职责] stage：上下文管理器形式的阶段计时，退出时自动写入耗时（ms）。
        [边界] 默认不累加（accumulate=False）以避免重复包裹导致意外叠加；需要累加时显式设置 True。
        [上游关系] pipeline 编排处：with timing.stage("parse"): ...
        [下游关系] timing_ms 写入与审计。
        """
        start = _now_ms()
        try:
            yield
        finally:
            end = _now_ms()
            self.add_ms(key, end - start, accumulate=accumulate)

    def total_ms(self) -> float:
        """
        [职责] 返回从 collector 创建/重置起到当前的总耗时（ms）。
        [边界] total 与分阶段之和不必严格相等（可能存在未包裹阶段、并发等待等）。
        [上游关系] pipeline 在结束时读取。
        [下游关系] DB timing_ms["total"] 或返回结果中的总耗时字段（如需）。
        """
        return _now_ms() - self._start_ms

    def to_dict(self, *, include_total: bool = True, total_key: str = "total") -> Dict[str, float]:
        """
        [职责] 导出可 JSON 序列化的 timing dict（ms）。
        [边界] 仅导出 float；不包含嵌套结构；key 可自定义（默认 total）。
        [上游关系] pipeline 完成后调用。
        [下游关系] DB timing_ms JSON、schemas TimingSnapshot/TimingMs 合同对象。
        """
        out = dict(self._stages_ms)
        if include_total:
            out[total_key] = float(self.total_ms())
        return out

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        [职责] 获取某阶段耗时（ms）。
        [边界] key 不存在则返回 default。
        [上游关系] 调试/门禁需要读取特定阶段耗时。
        [下游关系] gate tests 或 debug 输出。
        """
        if key in self._stages_ms:
            return self._stages_ms[key]
        return default
