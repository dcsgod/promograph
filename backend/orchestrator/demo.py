"""One-shot end-to-end smoke test of the orchestrator (no server needed).

Run:  python -m backend.orchestrator.demo
"""
from __future__ import annotations

from backend.llm.client import llm_available
from backend.orchestrator.loop import run_turn


def _play(title: str, history: list[dict]) -> None:
    print(f"\n=== {title} ===")
    counts: dict[str, int] = {}
    text = []
    for ev in run_turn(history):
        counts[ev.type] = counts.get(ev.type, 0) + 1
        if ev.type == "token":
            text.append(ev.text)
        elif ev.type == "recommendation":
            print("  [recommendation]", ev.data["label"],
                  f"depth={ev.data['recommended_depth_pct']}%",
                  f"net=${ev.data['net_incremental_margin']}")
        elif ev.type == "tool":
            if ev.status == "start":
                print(f"  [tool] {ev.name} {ev.detail}")
        elif ev.type == "error":
            print("  [error]", ev.message)
    print("  events:", counts)
    print("  answer:", "".join(text).strip())


def main() -> None:
    print(f"LLM available: {llm_available()} "
          f"({'conversational' if llm_available() else 'deterministic fallback'} mode)")

    history = [{"role": "user", "content": "What happens if I run 20% off on ground coffee?"}]
    _play("Turn 1: promo question", history)

    history.append({"role": "user",
                    "content": "What's the optimal discount instead?"})
    _play("Turn 2: ask for optimal", history)

    # simulate a graph node click as a first-class turn
    sku = None
    from backend.orchestrator.tools import _svc
    row = _svc.store.find_sku("cola")
    sku = f"sku:{int(row['product_id'])}"
    history.append({"role": "user",
                    "content": f"[graph-click] User clicked the SKU node "
                               f"'{row['brand']} {row['sub_commodity_desc']}' ({sku})."})
    _play("Turn 3: node click drill-in", history)


if __name__ == "__main__":
    main()
