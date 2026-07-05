"""
transition_engine.py — 第①层：真状态转移引擎
=============================================
基于 FSM 的确定性状态机。
接收 Action → 输出 StateDiff（唯一的状态变化描述协议）。

search 动作通过 RAG（Qwen-VL-Embedding + ChromaDB）检索 WIT 知识库，
检索结果即为"商品"——RAG 就是搜索引擎本身。
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from searcheyes.state_diff import StateDiff, DiffEvent, DiffEventType
from searcheyes.screen_ir import PageFamily


@dataclass
class EnvState:
    """环境的完整可序列化状态"""
    current_page: str = "search"          # 当前页面 ID
    page_family: PageFamily = PageFamily.SEARCH
    selected_product_id: Optional[int] = None
    cart: list[int] = field(default_factory=list)
    active_modal: Optional[str] = None    # None 或 modal 名称
    active_dropdown: Optional[str] = None
    active_filter: dict[str, str] = field(default_factory=dict)

    def hash(self) -> str:
        s = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]


# 后备产品数据（仅当无 RAG 时使用，保证最低可用性）
_FALLBACK_PRODUCTS = [
    {"id": 1, "name": "Item A", "price": 500, "stock": 10, "color": "neutral",
     "image_path": "", "image_caption": "", "wit_id": ""},
    {"id": 2, "name": "Item B", "price": 800, "stock": 10, "color": "neutral",
     "image_path": "", "image_caption": "", "wit_id": ""},
]


def rag_facts_to_products(facts: list, images_dir: Path) -> dict[int, dict]:
    """将 RAG 检索到的 RagFact 列表转换为 products dict。

    每个 RagFact 映射为一个"商品"，价格由 wit_id hash 确定性生成。
    """
    products: dict[int, dict] = {}
    for idx, fact in enumerate(facts):
        pid = idx + 1
        img_filename = f"{fact.wit_id}.jpg"
        img_path = images_dir / img_filename
        # 确定性价格：同一 wit_id 始终相同价格
        price = (hash(fact.wit_id) % 1900) + 100
        products[pid] = {
            "id": pid,
            "name": fact.title or f"Item {pid}",
            "price": price,
            "stock": 10,
            "color": "neutral",
            "image_path": str(img_path) if img_path.exists() else "",
            "image_caption": fact.caption,
            "wit_id": fact.wit_id,
        }
    return products


class TransitionEngine:
    """
    确定性状态转移引擎。
    接收 (当前状态, 动作) → 返回 (新状态, StateDiff)

    search 动作通过 RAG 检索 WIT 知识库填充 products。
    """

    def __init__(
        self,
        rag: Any = None,
        images_dir: str | Path | None = None,
        ground_truth_wit_id: str = "",
        inject_ground_truth: bool = True,
    ):
        self.rag = rag  # MultimodalRAG 实例
        self.images_dir = Path(images_dir) if images_dir else Path("data/wit_subset_hf/images")
        self.ground_truth_wit_id = ground_truth_wit_id
        self.inject_ground_truth = inject_ground_truth  # False = 去掉拐杖，测真实检索
        self.products: dict[int, dict] = {}  # 空，由 RAG 填充

    def populate_products_from_rag(
        self, query_image: str = "", query_text: str = "",
    ) -> dict[int, dict]:
        """调用 RAG 检索 WIT 知识库，将结果转换为 products。

        如果 ground_truth_wit_id 不在检索结果中，则注入 GT（替换最低分结果），
        保证每个 episode 都可解。
        """
        if not self.rag:
            self.products = {p["id"]: dict(p) for p in _FALLBACK_PRODUCTS}
            return self.products

        try:
            from searcheyes.query_rewriter import rewrite as _rewrite_query
            cleaned_text = _rewrite_query(query_text) if query_text else ""
            # 初始 search：rewritten text + BM25 hybrid（搜索页截图是噪声，停用）
            facts = self.rag.get_rag_facts_combined(
                text=cleaned_text, top_k=6, use_hybrid=True,
            )
        except Exception as exc:
            print(f"[TransitionEngine] RAG 检索失败: {exc}")
            self.products = {p["id"]: dict(p) for p in _FALLBACK_PRODUCTS}
            return self.products

        if not facts:
            self.products = {p["id"]: dict(p) for p in _FALLBACK_PRODUCTS}
            return self.products

        # 保证 ground truth 出现在结果中（可关闭以测真实检索能力）
        if self.inject_ground_truth and self.ground_truth_wit_id:
            gt_in_results = any(f.wit_id == self.ground_truth_wit_id for f in facts)
            if not gt_in_results:
                try:
                    gt_fact = self.rag.get_fact_by_id(self.ground_truth_wit_id)
                    if gt_fact and len(facts) > 0:
                        facts[-1] = gt_fact  # 替换最低分的
                    elif gt_fact:
                        facts.append(gt_fact)
                except Exception:
                    pass

        self.products = rag_facts_to_products(facts, self.images_dir)
        return self.products

    def step(self, state: EnvState, action: str, params: dict = None) -> tuple[EnvState, StateDiff]:
        """
        执行一步状态转移。
        
        支持的动作:
          search       - 提交搜索
          click_product - 点击商品卡片 (params: product_id)
          buy           - 购买当前选中商品
          add_cart      - 加入购物车
          open_modal    - 打开弹窗 (params: modal_name)
          close_modal   - 关闭弹窗
          toggle_dropdown - 切换下拉 (params: dropdown_name)
          apply_filter  - 应用筛选 (params: key, value)
          back          - 返回上一页
          zoom          - 放大查看 (不改变状态，但记录在轨迹中)
        """
        params = params or {}
        old_hash = state.hash()
        events = []

        import copy
        new_state = copy.deepcopy(state)

        if action == "search":
            # RAG 驱动搜索：用当前截图/文本做 embedding 查询 ChromaDB
            query_image = params.get("query_image", "")
            query_text = params.get("query_text", "")
            if self.rag:
                self.populate_products_from_rag(query_image, query_text)
            elif not self.products:
                self.products = {p["id"]: dict(p) for p in _FALLBACK_PRODUCTS}
            new_state.current_page = "results"
            new_state.page_family = PageFamily.RESULTS
            events.append(DiffEvent(
                event_type=DiffEventType.PAGE_NAVIGATED,
                payload={"from_page": state.current_page, "to_page": "results"}
            ))

        elif action == "click_product":
            pid = params.get("product_id")
            if pid and pid in self.products:
                new_state.current_page = f"detail_{pid}"
                new_state.page_family = PageFamily.DETAIL
                new_state.selected_product_id = pid
                events.append(DiffEvent(
                    event_type=DiffEventType.PAGE_NAVIGATED,
                    payload={"from_page": state.current_page, "to_page": f"detail_{pid}"}
                ))

        elif action == "buy":
            pid = new_state.selected_product_id
            product = self.products.get(pid, {})
            if pid and product.get("stock", 0) > 0:
                old_stock = product["stock"]
                product["stock"] -= 1
                events.append(DiffEvent(
                    event_type=DiffEventType.INVENTORY_CHANGED,
                    payload={"product_id": pid, "old_stock": old_stock, "new_stock": old_stock - 1}
                ))
                events.append(DiffEvent(
                    event_type=DiffEventType.TOAST_SHOWN,
                    payload={"message": f"购买成功: {product.get('name', 'unknown')}", "type": "success"}
                ))

        elif action == "add_cart":
            pid = new_state.selected_product_id
            if pid:
                new_state.cart.append(pid)
                events.append(DiffEvent(
                    event_type=DiffEventType.CART_COUNT_CHANGED,
                    payload={"new_count": len(new_state.cart), "added_product": pid}
                ))
                events.append(DiffEvent(
                    event_type=DiffEventType.TOAST_SHOWN,
                    payload={"message": "已加入购物车", "type": "info"}
                ))

        elif action == "open_modal":
            modal_name = params.get("modal_name", "confirm")
            new_state.active_modal = modal_name
            events.append(DiffEvent(
                event_type=DiffEventType.MODAL_OPENED,
                payload={"modal_name": modal_name}
            ))

        elif action == "close_modal":
            new_state.active_modal = None
            events.append(DiffEvent(event_type=DiffEventType.MODAL_CLOSED))

        elif action == "toggle_dropdown":
            dd_name = params.get("dropdown_name", "sort")
            if new_state.active_dropdown == dd_name:
                new_state.active_dropdown = None
                events.append(DiffEvent(
                    event_type=DiffEventType.DROPDOWN_COLLAPSED,
                    payload={"dropdown_name": dd_name}
                ))
            else:
                new_state.active_dropdown = dd_name
                events.append(DiffEvent(
                    event_type=DiffEventType.DROPDOWN_EXPANDED,
                    payload={"dropdown_name": dd_name}
                ))

        elif action == "apply_filter":
            key = params.get("key", "color")
            value = params.get("value", "red")
            new_state.active_filter[key] = value
            events.append(DiffEvent(
                event_type=DiffEventType.FILTER_APPLIED,
                payload={"filter_key": key, "filter_value": value}
            ))

        elif action == "back":
            if new_state.page_family == PageFamily.DETAIL:
                new_state.current_page = "results"
                new_state.page_family = PageFamily.RESULTS
                new_state.selected_product_id = None
            elif new_state.page_family == PageFamily.RESULTS:
                new_state.current_page = "search"
                new_state.page_family = PageFamily.SEARCH
            events.append(DiffEvent(
                event_type=DiffEventType.PAGE_NAVIGATED,
                payload={"from_page": state.current_page, "to_page": new_state.current_page}
            ))

        elif action == "zoom":
            pass  # zoom 不改变逻辑状态

        diff = StateDiff(
            events=events,
            old_state_hash=old_hash,
            new_state_hash=new_state.hash()
        )

        return new_state, diff
