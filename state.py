from typing import TypedDict, Dict, Any, List

class ResearchState(TypedDict):
    query: str
    research_plan: Dict[str, Any]
    current_step: int
    completed_steps: List[int]
    research_findings: Dict[int, Dict[str, Any]]
    feedback: Dict[int, str]
    revision_count: int
    final_report: str
    status: str
