//docstring
// 职责: 检索命中列表的占位容器。
// 边界: 不执行外拉查询，不自行排序。
// 上游关系: EvidencePanel。
// 下游关系: NodePreview 联动由上层实现。
const RetrievalHitsTable = () => {
  return <div className="retrieval-hits-table" />
}

export default RetrievalHitsTable
