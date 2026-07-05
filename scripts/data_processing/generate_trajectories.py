"""
第四阶段：长程轨迹数据集合成
================================
在 FSM 状态机中执行任务级随机游走，为每一步记录：
  1. 高清截图（全图 + Neural Zoom 局部）
  2. 所有 bounding_box 真值坐标
  3. Chain-of-Thought 推理过程模拟
  4. 动作决策与执行结果

每条轨迹格式为一个完整 episode：
  task_instruction → [step_0, step_1, ..., step_N] → task_result

输出目录结构：
  output/trajectories/
    ├── summary.json              # 全部轨迹汇总统计
    ├── trajectory_0000/
    │   ├── meta.json             # 任务指令 + 轨迹元数据
    │   ├── step_0_screenshot.png
    │   ├── step_0_zoom.png       # (如果该步有 zoom)
    │   ├── step_1_screenshot.png
    │   └── ...
    ├── trajectory_0001/
    │   └── ...
"""

import asyncio
import json
import random
import time
from pathlib import Path
from dataclasses import dataclass, asdict

from sandbox.renderer import SandboxRenderer
from sandbox.fsm import WorldFSM

OUTPUT_DIR = Path(__file__).parent / "output" / "trajectories"
SANDBOX_DIR = Path(__file__).parent / "sandbox"
CONFIG_PATH = SANDBOX_DIR / "world_config.json"

# 轨迹数量配置（演示用，生产环境改为 10000）
NUM_TRAJECTORIES = 20
MAX_STEPS_PER_TRAJECTORY = 10


@dataclass
class StepRecord:
    """单步记录"""
    step_idx: int
    state_id: str
    state_description: str
    action_type: str             # "click" | "zoom" | "observe"
    action_detail: dict          # {"x": ..., "y": ...} 或 {"bbox": [...], "factor": N}
    target_element: str | None   # 点击目标的 selector
    target_text: str | None      # 目标元素的文本内容
    cot_reasoning: str           # 模拟的 Chain-of-Thought
    transition_hit: bool         # 是否触发了有效状态跳转
    new_state_id: str | None     # 跳转后的状态 (仅 transition_hit=True)
    all_bboxes: dict             # 当前页面所有元素 bbox
    screenshot_file: str         # 截图文件名
    zoom_file: str | None        # zoom 截图文件名


def generate_cot(task_instruction: str, state_desc: str, bboxes: dict,
                 interactive_elems: list, action_type: str, target_sel: str = None,
                 target_text: str = None, step_idx: int = 0) -> str:
    """
    模拟 Agent 的 Chain-of-Thought 推理过程。
    这里用模板生成——后续可替换为真实 VLM 输出。
    """
    elem_list = ", ".join([
        f"{e['selector']}('{e.get('text', '')[:20]}')"
        for e in interactive_elems[:6]
    ])

    if action_type == "observe":
        return (
            f"[Step {step_idx}] 任务目标: {task_instruction}\n"
            f"当前页面: {state_desc}\n"
            f"观察: 页面上可见元素有 [{elem_list}]\n"
            f"思考: 首先需要观察整体布局，确认可交互元素的位置和文字。"
        )
    elif action_type == "zoom":
        return (
            f"[Step {step_idx}] 任务目标: {task_instruction}\n"
            f"当前页面: {state_desc}\n"
            f"观察: 页面截图中有部分微小文字（价格/标签）无法辨认。\n"
            f"思考: 需要对目标区域 {target_sel} 执行 Neural Zoom，获取高清局部视图后再做决策。\n"
            f"决策: zoom({target_sel})"
        )
    elif action_type == "click":
        return (
            f"[Step {step_idx}] 任务目标: {task_instruction}\n"
            f"当前页面: {state_desc}\n"
            f"观察: 可交互元素 [{elem_list}]\n"
            f"思考: 根据任务目标，我需要点击 {target_sel}('{target_text}') 来推进任务流程。\n"
            f"决策: click({target_sel})"
        )
    return ""


async def generate_single_trajectory(
    trajectory_id: int,
    task: dict,
    renderer: SandboxRenderer,
) -> dict:
    """
    生成一条完整的任务轨迹。
    返回轨迹元数据字典。
    """
    traj_dir = OUTPUT_DIR / f"trajectory_{trajectory_id:04d}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    fsm = WorldFSM(CONFIG_PATH)
    state_info = fsm.reset()

    steps: list[dict] = []
    task_instruction = task["instruction"]
    answer_state = task.get("answer_state", "")
    optimal_steps = task.get("optimal_steps", 3)
    task_completed = False
    total_zoom_count = 0

    for step_idx in range(MAX_STEPS_PER_TRAJECTORY):
        # ---- 渲染当前页面 ----
        html_path = fsm.get_html_path(SANDBOX_DIR)
        screenshot = await renderer.render_page(html_path)

        screenshot_file = f"step_{step_idx}_screenshot.png"
        with open(traj_dir / screenshot_file, "wb") as f:
            f.write(screenshot)

        # ---- 获取 bbox 真值 ----
        selectors = state_info.get("interactive_elements", [])
        bboxes = await renderer.get_all_bboxes(selectors)
        interactive = await renderer.get_interactive_bboxes()
        fsm.update_bboxes(bboxes)

        # ---- 决策：是否先 zoom ----
        zoom_file = None
        zoom_step = None

        # 50% 概率在结果页/详情页执行 zoom（模拟 Agent 需要看清细节）
        if ("results" in state_info["state_id"] or "detail" in state_info["state_id"]) \
           and random.random() < 0.5 and total_zoom_count < 3:

            # 随机选一个元素做 zoom
            zoom_candidates = [e for e in interactive if e.get("bbox")]
            if zoom_candidates:
                target_elem = random.choice(zoom_candidates)
                tb = target_elem["bbox"]
                zoom_factor = random.choice([3, 5, 8])

                cot_zoom = generate_cot(
                    task_instruction, state_info["description"],
                    bboxes, interactive, "zoom",
                    target_sel=target_elem["selector"],
                    step_idx=step_idx
                )

                zoom_bytes = await renderer.neural_zoom(
                    html_path, tb["x"], tb["y"], tb["width"], tb["height"], zoom_factor
                )
                zoom_file = f"step_{step_idx}_zoom_{zoom_factor}x.png"
                with open(traj_dir / zoom_file, "wb") as f:
                    f.write(zoom_bytes)

                zoom_step = StepRecord(
                    step_idx=step_idx,
                    state_id=state_info["state_id"],
                    state_description=state_info["description"],
                    action_type="zoom",
                    action_detail={
                        "bbox": [tb["x"], tb["y"], tb["width"], tb["height"]],
                        "factor": zoom_factor
                    },
                    target_element=target_elem["selector"],
                    target_text=target_elem.get("text", ""),
                    cot_reasoning=cot_zoom,
                    transition_hit=False,
                    new_state_id=None,
                    all_bboxes={k: v for k, v in bboxes.items() if v},
                    screenshot_file=screenshot_file,
                    zoom_file=zoom_file,
                )
                steps.append(asdict(zoom_step))
                total_zoom_count += 1
                step_idx_offset = 0.5  # 标记 zoom 是附加动作
                continue  # zoom 不消耗 step，紧接着做 click

        # ---- 决策：选择点击目标 ----
        transitions = state_info.get("transitions", [])
        if not transitions:
            # 终态，记录最后一步
            cot_final = generate_cot(
                task_instruction, state_info["description"],
                bboxes, interactive, "observe", step_idx=step_idx
            )
            final_step = StepRecord(
                step_idx=step_idx,
                state_id=state_info["state_id"],
                state_description=state_info["description"],
                action_type="observe",
                action_detail={},
                target_element=None,
                target_text=None,
                cot_reasoning=cot_final + "\n思考: 已到达终态，任务完成。",
                transition_hit=False,
                new_state_id=None,
                all_bboxes={k: v for k, v in bboxes.items() if v},
                screenshot_file=screenshot_file,
                zoom_file=zoom_file,
            )
            steps.append(asdict(final_step))
            task_completed = True
            break

        # 智能选择策略：70% 选正确路径，30% 随机探索
        if random.random() < 0.7 and answer_state:
            # 尝试走正确路径
            correct_trans = [t for t in transitions if t["target_state"] == answer_state]
            if not correct_trans:
                chosen_trans = random.choice(transitions)
            else:
                chosen_trans = correct_trans[0]
        else:
            chosen_trans = random.choice(transitions)

        target_sel = chosen_trans["element_selector"]
        target_bbox = bboxes.get(target_sel)

        if not target_bbox:
            break

        click_x = target_bbox["x"] + target_bbox["width"] / 2
        click_y = target_bbox["y"] + target_bbox["height"] / 2

        # 找目标文本
        target_text = ""
        for e in interactive:
            if e.get("selector") == target_sel:
                target_text = e.get("text", "")
                break

        cot_click = generate_cot(
            task_instruction, state_info["description"],
            bboxes, interactive, "click",
            target_sel=target_sel, target_text=target_text,
            step_idx=step_idx
        )

        # 执行点击
        action = {"type": "click", "x": click_x, "y": click_y}
        old_state = fsm.current_state_id
        new_state_info, done = fsm.step(action)
        transition_hit = new_state_info["state_id"] != old_state

        click_step = StepRecord(
            step_idx=step_idx,
            state_id=state_info["state_id"],
            state_description=state_info["description"],
            action_type="click",
            action_detail={"x": round(click_x, 1), "y": round(click_y, 1)},
            target_element=target_sel,
            target_text=target_text,
            cot_reasoning=cot_click,
            transition_hit=transition_hit,
            new_state_id=new_state_info["state_id"] if transition_hit else None,
            all_bboxes={k: v for k, v in bboxes.items() if v},
            screenshot_file=screenshot_file,
            zoom_file=zoom_file,
        )
        steps.append(asdict(click_step))

        if done:
            task_completed = True
            # 记录最终状态截图
            final_html = fsm.get_html_path(SANDBOX_DIR)
            final_screenshot = await renderer.render_page(final_html)
            final_file = f"step_{step_idx + 1}_final.png"
            with open(traj_dir / final_file, "wb") as f:
                f.write(final_screenshot)
            break

        state_info = new_state_info

    # ---- 计算轨迹指标 ----
    reached_answer = any(
        s.get("new_state_id") == answer_state or s.get("state_id") == answer_state
        for s in steps
    )

    meta = {
        "trajectory_id": trajectory_id,
        "task": task,
        "total_steps": len(steps),
        "zoom_count": total_zoom_count,
        "task_completed": task_completed,
        "reached_answer_state": reached_answer,
        "optimal_steps": optimal_steps,
        "efficiency_ratio": round(optimal_steps / max(len(steps), 1), 2),
        "state_sequence": [s["state_id"] for s in steps],
        "action_sequence": [s["action_type"] for s in steps],
        "steps": steps,
    }

    with open(traj_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    return meta


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载任务列表
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    tasks = config.get("tasks", [])

    if not tasks:
        tasks = [{"task_id": "explore", "instruction": "浏览商品并选购", "answer_state": "", "optimal_steps": 4}]

    renderer = SandboxRenderer(viewport_w=1280, viewport_h=720)
    await renderer.start()

    print("=" * 70)
    print("📦 第四阶段：长程轨迹数据集合成")
    print(f"   生成数量: {NUM_TRAJECTORIES} 条")
    print(f"   最大步数: {MAX_STEPS_PER_TRAJECTORY} 步/条")
    print(f"   任务模板: {len(tasks)} 个")
    print("=" * 70)

    all_metas = []
    t0 = time.time()

    for i in range(NUM_TRAJECTORIES):
        task = tasks[i % len(tasks)]
        meta = await generate_single_trajectory(i, task, renderer)
        all_metas.append({
            "trajectory_id": meta["trajectory_id"],
            "task_id": meta["task"]["task_id"],
            "instruction": meta["task"]["instruction"],
            "total_steps": meta["total_steps"],
            "zoom_count": meta["zoom_count"],
            "task_completed": meta["task_completed"],
            "reached_answer": meta["reached_answer_state"],
            "efficiency_ratio": meta["efficiency_ratio"],
            "state_sequence": meta["state_sequence"],
            "action_sequence": meta["action_sequence"],
        })

        status = "✅" if meta["task_completed"] else "⏳"
        answer = "🎯" if meta["reached_answer_state"] else "❌"
        print(f"  [{i+1:3d}/{NUM_TRAJECTORIES}] {status}{answer} task={task['task_id']:<16s} "
              f"steps={meta['total_steps']:2d} zooms={meta['zoom_count']} "
              f"eff={meta['efficiency_ratio']:.2f} states={' → '.join(meta['state_sequence'][:5])}...")

    elapsed = time.time() - t0

    # ---- 汇总统计 ----
    total_steps = sum(m["total_steps"] for m in all_metas)
    total_zooms = sum(m["zoom_count"] for m in all_metas)
    completed = sum(1 for m in all_metas if m["task_completed"])
    reached = sum(1 for m in all_metas if m["reached_answer"])
    avg_steps = total_steps / max(len(all_metas), 1)
    avg_eff = sum(m["efficiency_ratio"] for m in all_metas) / max(len(all_metas), 1)

    summary = {
        "generation_config": {
            "num_trajectories": NUM_TRAJECTORIES,
            "max_steps_per_trajectory": MAX_STEPS_PER_TRAJECTORY,
            "num_tasks": len(tasks),
            "generation_time_sec": round(elapsed, 1),
        },
        "statistics": {
            "total_steps": total_steps,
            "total_zoom_actions": total_zooms,
            "total_screenshots": total_steps,
            "total_zoom_screenshots": total_zooms,
            "avg_steps_per_trajectory": round(avg_steps, 1),
            "avg_efficiency_ratio": round(avg_eff, 2),
            "task_completion_rate": f"{completed}/{NUM_TRAJECTORIES} ({completed/NUM_TRAJECTORIES*100:.0f}%)",
            "answer_reach_rate": f"{reached}/{NUM_TRAJECTORIES} ({reached/NUM_TRAJECTORIES*100:.0f}%)",
        },
        "trajectories": all_metas,
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 70}")
    print(f"📊 合成完成统计")
    print(f"{'=' * 70}")
    print(f"  总轨迹数: {NUM_TRAJECTORIES}")
    print(f"  总步数:   {total_steps} (含 {total_zooms} 次 zoom)")
    print(f"  平均步数: {avg_steps:.1f} 步/轨迹")
    print(f"  完成率:   {completed}/{NUM_TRAJECTORIES} ({completed/NUM_TRAJECTORIES*100:.0f}%)")
    print(f"  正确率:   {reached}/{NUM_TRAJECTORIES} ({reached/NUM_TRAJECTORIES*100:.0f}%)")
    print(f"  平均效率: {avg_eff:.2f}")
    print(f"  耗时:     {elapsed:.1f}s")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    await renderer.close()


if __name__ == "__main__":
    asyncio.run(main())
