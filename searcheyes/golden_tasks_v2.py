"""
golden_tasks_v2.py — Phase 0 验证: 创建 golden test tasks + 契约验证
=====================================================================
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from searcheyes.research_contracts import (
    ResearchTask, ResearchState, ResearchPhase,
    SearchResult, DocumentView, CitationObject, FactSet,
)


def load_and_convert_golden_tasks(n: int = 5) -> list[ResearchTask]:
    tasks_path = Path(__file__).parent.parent / "data" / "tasks" / "rag_tasks.jsonl"
    tasks = []
    with open(tasks_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            vt_data = json.loads(line)
            rt = ResearchTask.from_visual_task(vt_data)
            tasks.append(rt)
    return tasks


def simulate_research_episode(task: ResearchTask) -> dict:
    state = ResearchState()
    citations: list[CitationObject] = []

    # Step 1: formulate -> search
    assert state.current_phase == ResearchPhase.FORMULATE
    state.current_phase = ResearchPhase.SEARCH
    state.query_history.append(task.goal[:50])
    state.current_page = "results"

    # Mock search results from wit_bindings
    search_results = []
    for binding in task.wit_bindings:
        sr = SearchResult(
            result_id=binding["product_id"],
            wit_id=binding["wit_id"],
            title=binding["caption"][:30],
            snippet=binding["caption"],
            relevance_score=0.8,
            virtual_url="wiki://" + binding["wit_id"],
            image_path=binding.get("image_filename", ""),
        )
        search_results.append(sr)
    state.current_result_ids = [sr.result_id for sr in search_results]

    assert state.current_phase == ResearchPhase.SEARCH
    assert len(state.current_result_ids) == len(task.wit_bindings)

    # Step 2: search -> browse (open GT result)
    gt_result = None
    for sr in search_results:
        if sr.wit_id == task.ground_truth_wit_id:
            gt_result = sr
            break
    if gt_result is None:
        gt_result = search_results[0]

    state.current_phase = ResearchPhase.BROWSE
    state.opened_result_id = gt_result.result_id
    state.current_page = "document_" + str(gt_result.result_id)

    doc = DocumentView(
        result_id=gt_result.result_id,
        wit_id=gt_result.wit_id,
        title=gt_result.title,
        body_text=gt_result.snippet,
        image_path=gt_result.image_path,
        source_url=gt_result.virtual_url,
    )

    assert state.current_phase == ResearchPhase.BROWSE
    assert state.opened_result_id == gt_result.result_id

    # Step 3: cite_source
    citation = CitationObject(
        citation_id=1,
        source_result_id=gt_result.result_id,
        source_wit_id=gt_result.wit_id,
        evidence_text="Evidence from " + gt_result.title,
        source_title=gt_result.title,
    )
    citations.append(citation)
    state.citation_count = len(citations)
    state.cited_wit_ids.append(citation.source_wit_id)

    assert state.citation_count == 1
    assert citation.matches_fact(task.ground_truth_wit_id)

    # Step 4: back_to_results
    state.current_phase = ResearchPhase.SEARCH
    state.opened_result_id = None
    state.current_page = "results"

    # Step 5: submit_report
    coverage = task.fact_set.coverage(citations)
    state.report_submitted = True
    state.current_phase = ResearchPhase.SYNTHESIZE

    assert coverage == 1.0, "Expected coverage 1.0, got " + str(coverage)
    assert state.report_submitted

    h = state.hash()
    assert len(h) == 8

    return {
        "task_id": task.task_id,
        "goal": task.goal,
        "steps": 5,
        "citations": len(citations),
        "coverage": coverage,
        "state_hash": h,
        "passed": True,
    }


def run_all_validations():
    sep = "=" * 60
    print(sep)
    print("Phase 0: v2 contract validation")
    print(sep)

    tasks = load_and_convert_golden_tasks(5)
    print("\nLoaded " + str(len(tasks)) + " golden tasks from rag_tasks.jsonl")

    for t in tasks:
        valid, errors = t.validate()
        status = "PASS" if valid else "FAIL"
        print("  [" + status + "] " + t.summary())
        if not valid:
            for e in errors:
                print("      FAIL: " + e)

    print("\n--- Simulating research episodes ---")
    results = []
    for task in tasks:
        try:
            result = simulate_research_episode(task)
            results.append(result)
            print("  [PASS] " + result["task_id"] + ": " + str(result["steps"]) + " steps, coverage=" + str(result["coverage"]) + ", hash=" + result["state_hash"])
        except Exception as exc:
            print("  [FAIL] " + task.task_id + ": " + str(exc))
            results.append({"task_id": task.task_id, "passed": False, "error": str(exc)})

    passed = sum(1 for r in results if r.get("passed"))
    print("\n" + sep)
    print("Results: " + str(passed) + "/" + str(len(results)) + " episodes passed")

    if passed == len(results):
        print("Phase 0 PASSED - contracts are valid, ready for Phase 1")
    else:
        print("Phase 0 FAILED - fix contracts")

    return passed == len(results)


if __name__ == "__main__":
    success = run_all_validations()
    sys.exit(0 if success else 1)
