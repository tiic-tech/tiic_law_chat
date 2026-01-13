# src/uae_law_rag/backend/utils/constants.py

"""
[职责] 集中定义默认常量与协议字段名（prompt/gate/trace/timing/provider），降低跨模块硬编码。
[边界] 不包含运行时可变配置；不读取环境变量；不依赖业务具体实现。
[上游关系] services/pipelines/api 在构建请求/记录/响应时引用这些稳定字段与默认值。
[下游关系] schemas/db/logging 等使用一致字段名以便审计与回放。
"""

from __future__ import annotations


DEFAULT_PROMPT_NAME = "uae_law_grounded"  # docstring: 默认 prompt 名称
DEFAULT_PROMPT_VERSION = "v1"  # docstring: 默认 prompt 版本

PROMPT_NAME_KEY = "prompt_name"  # docstring: prompt 名称字段
PROMPT_VERSION_KEY = "prompt_version"  # docstring: prompt 版本字段

TRACE_ID_KEY = "trace_id"  # docstring: trace_id 字段
REQUEST_ID_KEY = "request_id"  # docstring: request_id 字段
PARENT_REQUEST_ID_KEY = "parent_request_id"  # docstring: parent_request_id 字段
CONVERSATION_ID_KEY = "conversation_id"  # docstring: conversation_id 字段
MESSAGE_ID_KEY = "message_id"  # docstring: message_id 字段
RETRIEVAL_RECORD_ID_KEY = "retrieval_record_id"  # docstring: retrieval_record_id 字段
GENERATION_RECORD_ID_KEY = "generation_record_id"  # docstring: generation_record_id 字段
EVALUATION_RECORD_ID_KEY = "evaluation_record_id"  # docstring: evaluation_record_id 字段

TRACE_TAGS_KEY = "trace_tags"  # docstring: trace tags 字段
TRACE_KEY = "trace"  # docstring: trace 快照字段

TRACE_FIELD_KEYS = (  # docstring: 结构化日志推荐字段集合
    TRACE_ID_KEY,
    REQUEST_ID_KEY,
    PARENT_REQUEST_ID_KEY,
    CONVERSATION_ID_KEY,
    MESSAGE_ID_KEY,
    RETRIEVAL_RECORD_ID_KEY,
    GENERATION_RECORD_ID_KEY,
    EVALUATION_RECORD_ID_KEY,
)

PROVIDER_SNAPSHOT_KEY = "provider_snapshot"  # docstring: provider 快照字段
TIMING_MS_KEY = "timing_ms"  # docstring: timing_ms 字段
TIMING_TOTAL_KEY = "total"  # docstring: timing_ms 的总耗时 key（短形式）
TIMING_TOTAL_MS_KEY = "total_ms"  # docstring: timing_ms 的总耗时 key（含单位）

DEBUG_KEY = "debug"  # docstring: debug 输出字段
RECORDS_KEY = "records"  # docstring: debug.records 字段

ERROR_KEY = "error"  # docstring: ErrorResponse 顶层字段
ERROR_CODE_KEY = "code"  # docstring: ErrorResponse.error.code 字段
ERROR_MESSAGE_KEY = "message"  # docstring: ErrorResponse.error.message 字段
ERROR_DETAIL_KEY = "detail"  # docstring: ErrorResponse.error.detail 字段

META_KEY = "meta"  # docstring: 通用 meta 字段
META_DATA_KEY = "meta_data"  # docstring: 通用 meta_data 字段

DEFAULT_EVAL_RULE_VERSION = "v0"  # docstring: evaluator 规则版本默认值
DEFAULT_EVAL_RETRIEVAL_TOPK = 10  # docstring: evaluator retrieval_topk 默认值
DEFAULT_EVAL_RETRIEVAL_MIN_HITS = 1  # docstring: evaluator retrieval_min_hits 默认值
DEFAULT_EVAL_REQUIRE_VECTOR_HITS = False  # docstring: evaluator require_vector_hits 默认值
DEFAULT_EVAL_REQUIRE_KEYWORD_HITS = False  # docstring: evaluator require_keyword_hits 默认值
DEFAULT_EVAL_MIN_ANSWER_CHARS = 20  # docstring: evaluator min_answer_chars 默认值
DEFAULT_EVAL_REQUIRE_STRUCTURED = False  # docstring: evaluator require_structured 默认值
DEFAULT_EVAL_REQUIRE_CITATIONS = True  # docstring: evaluator require_citations 默认值
DEFAULT_EVAL_MIN_CITATIONS = 1  # docstring: evaluator min_citations 默认值
DEFAULT_EVAL_CITATION_COVERAGE_THRESHOLD = 0.0  # docstring: evaluator citation_coverage_threshold 默认值
DEFAULT_EVAL_ENABLE_TOKEN_COUNT = False  # docstring: evaluator enable_token_count 默认值
