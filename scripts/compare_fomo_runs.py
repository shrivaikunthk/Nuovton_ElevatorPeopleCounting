"""Aggregate every FOMO sweep_*.json under runs/ and keras_fomo/runs/ into a
single sorted leaderboard printed to stdout (and saved to runs/leaderboard.csv).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect() -> list[dict]:
    rows: list[dict] = []
    for sweep_path in list(ROOT.glob("runs/**/sweep_*.json")) + list(ROOT.glob("keras_fomo/runs/**/sweep_*.json")):
        try:
            data = json.loads(sweep_path.read_text())
        except Exception:
            continue
        test = data.get("test_at_best", {})
        rows.append({
            "run": str(sweep_path.parent.relative_to(ROOT)),
            "backend": data.get("backend", "?"),
            "best_threshold": data.get("best_threshold", float("nan")),
            "test_mae": test.get("count_mae", float("nan")),
            "test_rmse": test.get("count_rmse", float("nan")),
            "test_bias": test.get("count_bias", float("nan")),
            "empty_fp": test.get("empty_false_positive_rate", float("nan")),
            "model": data.get("model", ""),
        })
    return rows


def fmt(rows: list[dict]) -> str:
    if not rows:
        return "no sweep_*.json files found yet."
    rows = sorted(rows, key=lambda r: (r["test_mae"], r["empty_fp"]))
    headers = ["run", "backend", "thr", "MAE", "RMSE", "bias", "empty_fp"]
    widths = [max(len(h), max(len(str(r[k] if k != "thr" else r["best_threshold"])) for r in rows))
              for h, k in zip(headers, ["run", "backend", "best_threshold", "test_mae", "test_rmse", "test_bias", "empty_fp"])]
    out = []
    out.append("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    out.append("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        out.append("  ".join([
            r["run"].ljust(widths[0]),
            r["backend"].ljust(widths[1]),
            f"{r['best_threshold']:.2f}".ljust(widths[2]),
            f"{r['test_mae']:.4f}".ljust(widths[3]),
            f"{r['test_rmse']:.4f}".ljust(widths[4]),
            f"{r['test_bias']:+.4f}".ljust(widths[5]),
            f"{r['empty_fp']:.3f}".ljust(widths[6]),
        ]))
    return "\n".join(out)


def main() -> None:
    rows = collect()
    print(fmt(rows))
    if rows:
        out_csv = ROOT / "runs" / "leaderboard.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nleaderboard -> {out_csv}")


if __name__ == "__main__":
    main()
