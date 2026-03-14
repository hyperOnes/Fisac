from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List

import mpmath as mp


def _finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _stats(values: Iterable[float]) -> Dict[str, float]:
    finite = _finite(values)
    n = len(finite)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "se": float("nan")}
    avg = float(mean(finite))
    sd = float(stdev(finite)) if n > 1 else 0.0
    se = float(sd / math.sqrt(n)) if n > 0 else float("nan")
    return {"n": n, "mean": avg, "std": sd, "se": se}


def _beta_ppf(q: float, a: float, b: float) -> float:
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        cdf = float(mp.betainc(a, b, 0.0, mid, regularized=True))
        if cdf < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson_ci(x: int, n: int, alpha: float = 0.05) -> List[float]:
    if n <= 0:
        return [float("nan"), float("nan")]
    if x == 0:
        lower = 0.0
    else:
        lower = _beta_ppf(alpha / 2.0, float(x), float(n - x + 1))

    if x == n:
        upper = 1.0
    else:
        upper = _beta_ppf(1.0 - alpha / 2.0, float(x + 1), float(n - x))
    return [float(lower), float(upper)]


def paired_sign_test_pvalue(deltas: Iterable[float]) -> float:
    vals = [float(v) for v in deltas if math.isfinite(float(v))]
    pos = sum(v > 0.0 for v in vals)
    neg = sum(v < 0.0 for v in vals)
    n = pos + neg
    if n == 0:
        return float("nan")
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / float(2**n)
    return float(min(1.0, 2.0 * tail))


def _enrich_row(row: Dict[str, float]) -> Dict[str, float]:
    out = dict(row)
    train_sps = float(out.get("train_steps_per_sec", out.get("steps_per_sec", float("nan"))))
    if math.isfinite(train_sps) and train_sps > 0.0:
        seconds_per_step = float(out.get("seconds_per_step", 1.0 / train_sps))
    else:
        seconds_per_step = float("nan")

    out["train_steps_per_sec"] = train_sps
    out["seconds_per_step"] = seconds_per_step
    out["peak_mem_gb"] = float(out.get("peak_mem_gb", float("nan")))

    liquid_mse = float(out["mse"])
    gru_mse = float(out["baseline_gru_mse"])
    ewma_mse = float(out["baseline_ewma_mse"])

    out["delta_mse_vs_gru"] = gru_mse - liquid_mse
    out["delta_mse_vs_ewma"] = ewma_mse - liquid_mse
    out["win_vs_gru"] = float(float(out["relative_post_shift_gain_vs_gru"]) > 0.0)
    out["win_vs_ewma"] = float(float(out["relative_post_shift_gain_vs_ewma"]) > 0.0)
    return out


def _fmt(mean_v: float, std_v: float, se_v: float, digits: int = 4) -> str:
    if not math.isfinite(mean_v):
        return "nan"
    return f"{mean_v:.{digits}f} +/- {std_v:.{digits}f} (SE {se_v:.{digits}f})"


def build_report(input_path: Path, output_dir: Path, alpha: float = 0.05) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(input_path.read_text())
    rows = [_enrich_row(r) for r in raw.get("rows", [])]
    rows = sorted(rows, key=lambda r: (int(r["seed"]), str(r["task"])))

    schema = {
        "schema_version": "1.0",
        "description": (
            "Paired seed-task benchmark rows. Each row compares Liquid vs GRU/EWMA under identical "
            "seed/task/config, and includes absolute MSE plus derived delta/gain metrics."
        ),
        "fields": [
            "seed",
            "task",
            "mse",
            "baseline_gru_mse",
            "baseline_ewma_mse",
            "delta_mse_vs_gru",
            "delta_mse_vs_ewma",
            "relative_post_shift_gain_vs_gru",
            "relative_post_shift_gain_vs_ewma",
            "adaptation_steps",
            "gru_adaptation_steps",
            "ewma_adaptation_steps",
            "train_steps_per_sec",
            "seconds_per_step",
            "peak_mem_gb",
            "pruned_total",
            "myelinated_total",
            "myelinated_experts_end",
            "alive_experts_end",
            "win_vs_gru",
            "win_vs_ewma",
        ],
    }

    paired_json_path = output_dir / "paired_rows.json"
    paired_csv_path = output_dir / "paired_rows.csv"

    paired_json_path.write_text(json.dumps({"schema": schema, "rows": rows}, indent=2))
    with paired_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema["fields"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in schema["fields"]})

    tasks = sorted({str(r["task"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})

    per_task: Dict[str, object] = {}
    for task in tasks:
        t_rows = [r for r in rows if str(r["task"]) == task]
        x_gru = sum(int(r["win_vs_gru"]) for r in t_rows)
        x_ewma = sum(int(r["win_vs_ewma"]) for r in t_rows)
        n = len(t_rows)

        per_task[task] = {
            "n_seeds": n,
            "liquid_mse": _stats(r["mse"] for r in t_rows),
            "gru_mse": _stats(r["baseline_gru_mse"] for r in t_rows),
            "ewma_mse": _stats(r["baseline_ewma_mse"] for r in t_rows),
            "delta_mse_vs_gru": _stats(r["delta_mse_vs_gru"] for r in t_rows),
            "delta_mse_vs_ewma": _stats(r["delta_mse_vs_ewma"] for r in t_rows),
            "relative_gain_vs_gru": _stats(r["relative_post_shift_gain_vs_gru"] for r in t_rows),
            "relative_gain_vs_ewma": _stats(r["relative_post_shift_gain_vs_ewma"] for r in t_rows),
            "adaptation_steps": _stats(r["adaptation_steps"] for r in t_rows),
            "gru_adaptation_steps": _stats(r["gru_adaptation_steps"] for r in t_rows),
            "ewma_adaptation_steps": _stats(r["ewma_adaptation_steps"] for r in t_rows),
            "train_steps_per_sec": _stats(r["train_steps_per_sec"] for r in t_rows),
            "seconds_per_step": _stats(r["seconds_per_step"] for r in t_rows),
            "peak_mem_gb": _stats(r["peak_mem_gb"] for r in t_rows),
            "win_rate_vs_gru": {
                "x": x_gru,
                "n": n,
                "rate": float(x_gru / max(1, n)),
                "cp95": clopper_pearson_ci(x_gru, n, alpha=alpha),
            },
            "win_rate_vs_ewma": {
                "x": x_ewma,
                "n": n,
                "rate": float(x_ewma / max(1, n)),
                "cp95": clopper_pearson_ci(x_ewma, n, alpha=alpha),
            },
            "paired_sign_test_delta_mse_vs_gru": {
                "p_value_two_sided": paired_sign_test_pvalue(r["delta_mse_vs_gru"] for r in t_rows),
            },
            "paired_sign_test_delta_mse_vs_ewma": {
                "p_value_two_sided": paired_sign_test_pvalue(r["delta_mse_vs_ewma"] for r in t_rows),
            },
        }

    x_gru_all = sum(int(r["win_vs_gru"]) for r in rows)
    x_ewma_all = sum(int(r["win_vs_ewma"]) for r in rows)
    n_all = len(rows)

    per_seed = []
    for seed in seeds:
        s_rows = [r for r in rows if int(r["seed"]) == seed]
        peak_vals = [r["peak_mem_gb"] for r in s_rows if math.isfinite(r["peak_mem_gb"])]
        per_seed.append(
            {
                "seed": seed,
                "liquid_mse_mean": float(mean(r["mse"] for r in s_rows)),
                "gru_mse_mean": float(mean(r["baseline_gru_mse"] for r in s_rows)),
                "ewma_mse_mean": float(mean(r["baseline_ewma_mse"] for r in s_rows)),
                "delta_mse_vs_gru_mean": float(mean(r["delta_mse_vs_gru"] for r in s_rows)),
                "delta_mse_vs_ewma_mean": float(mean(r["delta_mse_vs_ewma"] for r in s_rows)),
                "relative_gain_vs_gru_mean": float(mean(r["relative_post_shift_gain_vs_gru"] for r in s_rows)),
                "relative_gain_vs_ewma_mean": float(mean(r["relative_post_shift_gain_vs_ewma"] for r in s_rows)),
                "train_steps_per_sec_mean": float(mean(r["train_steps_per_sec"] for r in s_rows)),
                "seconds_per_step_mean": float(mean(r["seconds_per_step"] for r in s_rows)),
                "peak_mem_gb_mean": float(mean(peak_vals)) if peak_vals else float("nan"),
                "wins_vs_gru": int(sum(int(r["win_vs_gru"]) for r in s_rows)),
                "wins_vs_ewma": int(sum(int(r["win_vs_ewma"]) for r in s_rows)),
                "comparisons_per_seed": int(len(s_rows)),
            }
        )

    seed_level_overall = {
        "liquid_mse_mean": _stats(r["liquid_mse_mean"] for r in per_seed),
        "gru_mse_mean": _stats(r["gru_mse_mean"] for r in per_seed),
        "ewma_mse_mean": _stats(r["ewma_mse_mean"] for r in per_seed),
        "delta_mse_vs_gru_mean": _stats(r["delta_mse_vs_gru_mean"] for r in per_seed),
        "delta_mse_vs_ewma_mean": _stats(r["delta_mse_vs_ewma_mean"] for r in per_seed),
        "relative_gain_vs_gru_mean": _stats(r["relative_gain_vs_gru_mean"] for r in per_seed),
        "relative_gain_vs_ewma_mean": _stats(r["relative_gain_vs_ewma_mean"] for r in per_seed),
        "train_steps_per_sec_mean": _stats(r["train_steps_per_sec_mean"] for r in per_seed),
        "seconds_per_step_mean": _stats(r["seconds_per_step_mean"] for r in per_seed),
        "peak_mem_gb_mean": _stats(r["peak_mem_gb_mean"] for r in per_seed),
    }

    summary = {
        "source_file": str(input_path),
        "definitions": {
            "mse": "Mean squared error (MSE, unitless).",
            "delta_mse_vs_gru": "GRU_MSE - Liquid_MSE (positive means Liquid better).",
            "delta_mse_vs_ewma": "EWMA_MSE - Liquid_MSE (positive means Liquid better).",
            "relative_gain_vs_gru": "(post_shift_gru - post_shift_liquid) / max(post_shift_gru, 1e-8).",
            "relative_gain_vs_ewma": "(post_shift_ewma - post_shift_liquid) / max(post_shift_ewma, 1e-8).",
            "post_shift_region": (
                "After each regime change, collect errors from the next 80 steps (including the shift step), "
                "then average across all such windows."
            ),
            "train_steps_per_sec": "Online training/environment steps per second.",
            "seconds_per_step": "Wall-clock seconds per online step.",
            "peak_mem_gb": "Peak process RSS in GB (OS-reported high-water mark).",
        },
        "seed_level_uncertainty_unit": {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "note": "Primary uncertainty is computed across independent seeds (n=5 by default).",
        },
        "win_rates": {
            "overall_vs_gru": {
                "x": x_gru_all,
                "n": n_all,
                "rate": float(x_gru_all / max(1, n_all)),
                "cp95": clopper_pearson_ci(x_gru_all, n_all, alpha=alpha),
            },
            "overall_vs_ewma": {
                "x": x_ewma_all,
                "n": n_all,
                "rate": float(x_ewma_all / max(1, n_all)),
                "cp95": clopper_pearson_ci(x_ewma_all, n_all, alpha=alpha),
            },
            "per_task": {
                task: {
                    "vs_gru": per_task[task]["win_rate_vs_gru"],
                    "vs_ewma": per_task[task]["win_rate_vs_ewma"],
                }
                for task in tasks
            },
        },
        "per_task": per_task,
        "per_seed": per_seed,
        "seed_level_overall": seed_level_overall,
    }

    summary_json_path = output_dir / "summary_stats.json"
    summary_json_path.write_text(json.dumps(summary, indent=2))

    lines: List[str] = []
    lines.append("# Paired Benchmark Summary")
    lines.append("")
    lines.append("## Definitions")
    lines.append("- **Post-shift** means **after regime change**. Region: 80 steps after each shift (including shift step).")
    lines.append("- **MSE** values are unitless.")
    lines.append("- **Primary uncertainty unit**: seed-level aggregation (n=5 seeds).")
    lines.append("")
    lines.append("## Per-Task Paired Results (Seed-Level, n=5)")
    lines.append("")
    lines.append(
        "| Task | Liquid MSE (unitless) | GRU MSE (unitless) | Delta MSE vs GRU (MSE) | Relative Gain vs GRU | EWMA MSE (unitless) | Delta MSE vs EWMA (MSE) | Relative Gain vs EWMA | Train Steps/s | Seconds/Step (s) | Peak RAM (GB) | Win vs GRU x/n (95% CI) | Win vs EWMA x/n (95% CI) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for task in tasks:
        t = per_task[task]
        w_gru = t["win_rate_vs_gru"]
        w_ewma = t["win_rate_vs_ewma"]
        lines.append(
            "| "
            + f"{task}"
            + " | "
            + _fmt(t["liquid_mse"]["mean"], t["liquid_mse"]["std"], t["liquid_mse"]["se"])
            + " | "
            + _fmt(t["gru_mse"]["mean"], t["gru_mse"]["std"], t["gru_mse"]["se"])
            + " | "
            + _fmt(t["delta_mse_vs_gru"]["mean"], t["delta_mse_vs_gru"]["std"], t["delta_mse_vs_gru"]["se"])
            + " | "
            + _fmt(t["relative_gain_vs_gru"]["mean"], t["relative_gain_vs_gru"]["std"], t["relative_gain_vs_gru"]["se"])
            + " | "
            + _fmt(t["ewma_mse"]["mean"], t["ewma_mse"]["std"], t["ewma_mse"]["se"])
            + " | "
            + _fmt(t["delta_mse_vs_ewma"]["mean"], t["delta_mse_vs_ewma"]["std"], t["delta_mse_vs_ewma"]["se"])
            + " | "
            + _fmt(t["relative_gain_vs_ewma"]["mean"], t["relative_gain_vs_ewma"]["std"], t["relative_gain_vs_ewma"]["se"])
            + " | "
            + _fmt(t["train_steps_per_sec"]["mean"], t["train_steps_per_sec"]["std"], t["train_steps_per_sec"]["se"])
            + " | "
            + _fmt(t["seconds_per_step"]["mean"], t["seconds_per_step"]["std"], t["seconds_per_step"]["se"])
            + " | "
            + _fmt(t["peak_mem_gb"]["mean"], t["peak_mem_gb"]["std"], t["peak_mem_gb"]["se"])
            + " | "
            + f"{w_gru['x']}/{w_gru['n']} ({w_gru['cp95'][0]:.3f}, {w_gru['cp95'][1]:.3f})"
            + " | "
            + f"{w_ewma['x']}/{w_ewma['n']} ({w_ewma['cp95'][0]:.3f}, {w_ewma['cp95'][1]:.3f})"
            + " |"
        )

    overall_gru = summary["win_rates"]["overall_vs_gru"]
    overall_ewma = summary["win_rates"]["overall_vs_ewma"]
    lines.append("")
    lines.append("## Overall Win Rates")
    lines.append(
        f"- vs GRU: {overall_gru['x']}/{overall_gru['n']} (95% CP CI: {overall_gru['cp95'][0]:.3f}, {overall_gru['cp95'][1]:.3f})"
    )
    lines.append(
        f"- vs EWMA: {overall_ewma['x']}/{overall_ewma['n']} (95% CP CI: {overall_ewma['cp95'][0]:.3f}, {overall_ewma['cp95'][1]:.3f})"
    )

    lines.append("")
    lines.append("## Raw Paired Rows")
    lines.append(f"- JSON: `{paired_json_path}`")
    lines.append(f"- CSV: `{paired_csv_path}`")

    summary_md_path = output_dir / "summary_table.md"
    summary_md_path.write_text("\n".join(lines))

    return {
        "paired_json": str(paired_json_path),
        "paired_csv": str(paired_csv_path),
        "summary_json": str(summary_json_path),
        "summary_markdown": str(summary_md_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build seed-level paired benchmark stats/report from benchmark_robust output.")
    parser.add_argument("--input", type=str, required=True, help="Path to benchmark_robust JSON output")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write report artifacts")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided CI alpha (default 0.05 => 95%% CI)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outputs = build_report(Path(args.input), Path(args.output_dir), alpha=args.alpha)
    print(json.dumps(outputs, indent=2))
