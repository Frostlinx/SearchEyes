"""
validator.py — 第4层：渲染-状态一致性校验器 (v2 Search/Report World)
====================================================================
检查 transition engine 输出的 StateDiff 是否与逻辑状态一致。
任何不一致都会被记录，防止幻觉污染 reward 和训练数据。

v2: 校验研究 workflow 的状态一致性（citations, results, phase transitions）。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from searcheyes.transition_engine import TransitionEngine
from searcheyes.research_contracts import ResearchState, ResearchPhase
from searcheyes.state_diff import StateDiff, DiffEventType


@dataclass
class ValidationResult:
    passed: bool = True
    mismatches: list[str] = field(default_factory=list)

    def add_mismatch(self, msg: str):
        self.passed = False
        self.mismatches.append(msg)

    def summary(self) -> str:
        if self.passed:
            return "Validation PASSED"
        return "Validation FAILED (" + str(len(self.mismatches)) + " mismatches): " + "; ".join(self.mismatches)


class Validator:
    """渲染-状态一致性校验器 (v2)"""

    def __init__(self, engine: TransitionEngine):
        self.engine = engine

    def validate(self, state: ResearchState, diff: StateDiff) -> ValidationResult:
        result = ValidationResult()

        for event in diff.events:
            # 1. Results consistency
            if event.event_type == DiffEventType.RESULTS_RETURNED:
                claimed_count = event.payload.get("result_count", 0)
                actual_count = len(self.engine.search_results)
                if claimed_count != actual_count:
                    result.add_mismatch(
                        "Result count mismatch: claimed " + str(claimed_count) +
                        ", actual " + str(actual_count)
                    )

            # 2. Document open consistency
            if event.event_type == DiffEventType.DOCUMENT_OPENED:
                result_id = event.payload.get("result_id")
                if state.opened_result_id != result_id:
                    result.add_mismatch(
                        "Document open mismatch: diff says " + str(result_id) +
                        ", state says " + str(state.opened_result_id)
                    )

            # 3. Evidence collection consistency
            if event.event_type == DiffEventType.EVIDENCE_COLLECTED:
                claimed_cid = event.payload.get("citation_id")
                actual_count = len(self.engine.citations)
                if claimed_cid and claimed_cid > actual_count:
                    result.add_mismatch(
                        "Citation ID " + str(claimed_cid) +
                        " exceeds actual citation count " + str(actual_count)
                    )

            # 4. Page navigation consistency
            if event.event_type == DiffEventType.PAGE_NAVIGATED:
                to_page = event.payload.get("to_page")
                if to_page and to_page != state.current_page:
                    result.add_mismatch(
                        "Page nav mismatch: diff says " + str(to_page) +
                        ", state says " + state.current_page
                    )

            # 5. Report submission consistency
            if event.event_type == DiffEventType.REPORT_SUBMITTED:
                if not state.report_submitted:
                    result.add_mismatch("Diff says report submitted but state disagrees")

        # 6. Phase-page consistency
        phase_page_map = {
            ResearchPhase.FORMULATE: ["formulate"],
            ResearchPhase.SEARCH: ["results"],
            ResearchPhase.BROWSE: [],  # document_N pages, checked by prefix
            ResearchPhase.SYNTHESIZE: ["report_submitted"],
        }
        valid_pages = phase_page_map.get(state.current_phase, [])
        if valid_pages and state.current_page not in valid_pages:
            result.add_mismatch(
                "Phase-page mismatch: phase=" + state.current_phase.value +
                " but page=" + state.current_page
            )
        if state.current_phase == ResearchPhase.BROWSE:
            if not state.current_page.startswith("document_"):
                result.add_mismatch(
                    "Browse phase but page=" + state.current_page +
                    " (expected document_N)"
                )

        # 7. State hash consistency
        if diff.new_state_hash and diff.new_state_hash != state.hash():
            result.add_mismatch(
                "State hash mismatch: diff says " + diff.new_state_hash +
                ", actual " + state.hash()
            )

        return result
