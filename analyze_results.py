"""
analyze_results.py — Cross-model comparison tool for table header detection experiment.

Usage:
  # Merge original + retry runs (best result per task wins):
  python analyze_results.py results/run_*_qwen30b results/run_*_qwen30b_retry results/run_*_capped_retry

  # Compare multiple models:
  python analyze_results.py results/run_*_qwen30b results/run_*_llama8b

  # Or point at root results dir (auto-discovers all runs):
  python analyze_results.py --results-dir results/

  # Save comparison to specific output dir:
  python analyze_results.py results/run_* --output-dir analysis/

Output:
  analysis/
    comparison_by_model.csv/xlsx          — F1/P/R per model (main table for paper)
    comparison_by_model_prompt.csv        — model × prompt
    comparison_by_model_format.csv        — model × format (JSON vs HTML)
    comparison_by_model_prompt_format.csv — model × prompt × format
    comparison_by_model_source.csv        — model × dataset source
    comparison_by_model_size.csv          — model × table size bin
    comparison_by_model_type.csv          — model × header type (col/proj/spanning)
    summary.txt                           — human-readable overall comparison
    all_responses.csv/parquet             — merged flat table of all responses
                                           (best result per task after dedup)
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

METRIC_COLS = [
    "f1_mean", "f1_median",
    "precision_mean", "recall_mean",
    "jaccard_mean", "exact_match_mean", "partial_match_mean",
    "header_coverage_mean",
    "support_mean", "pred_count_mean", "tp_mean", "fp_mean", "fn_mean",
]

DISPLAY_METRICS = [
    ("F1",        "f1_mean"),
    ("Precision", "precision_mean"),
    ("Recall",    "recall_mean"),
    ("Jaccard",   "jaccard_mean"),
    ("Exact",     "exact_match_mean"),
    ("Partial",   "partial_match_mean"),
]


# =========================
# DATA LOADING
# =========================

def find_checkpoints(run_dirs: List[Path]) -> List[Tuple[Path, Path]]:
    """
    For each run dir, find the most recent checkpoint.
    Returns list of (run_dir, checkpoint_path).
    """
    found = []
    for rd in run_dirs:
        ckpt_dir = rd / "checkpoints"
        if not ckpt_dir.exists():
            logging.warning(f"No checkpoints/ in {rd}, skipping")
            continue
        ckpts = sorted(ckpt_dir.glob("checkpoint_*.json"))
        if not ckpts:
            logging.warning(f"No checkpoint files in {ckpt_dir}, skipping")
            continue
        found.append((rd, ckpts[-1]))
    return found


def load_checkpoint(ckpt_path: Path) -> Tuple[Dict, List[Dict]]:
    """Load checkpoint, return (metadata, all_records)."""
    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    meta    = ckpt.get("metadata", {})
    records = (
        ckpt.get("responses", [])
        + ckpt.get("api_failed_requests", [])
        + ckpt.get("parse_failed_requests", [])
    )
    return meta, records


def load_all_runs(run_dirs: List[Path]) -> pd.DataFrame:
    """Load all runs into a single flat DataFrame."""
    all_rows = []
    ckpts = find_checkpoints(run_dirs)
    if not ckpts:
        logging.error("No valid checkpoints found.")
        sys.exit(1)

    for rd, ckpt_path in ckpts:
        meta, records = load_checkpoint(ckpt_path)
        logging.info(
            f"Loaded {len(records)} records from {rd.name} "
            f"(model={meta.get('model_alias', '?')})"
        )
        for rec in records:
            # Ensure model_alias present (older runs may lack it)
            if "model_alias" not in rec:
                rec["model_alias"] = meta.get("model_alias", rd.name.split("_")[-1])
            if "model" not in rec:
                rec["model"] = meta.get("model", "unknown")
            rec["_run_dir"] = str(rd)
            all_rows.append(rec)

    df = pd.DataFrame(all_rows)

    # Coerce numeric columns
    num_cols = [
        "f1", "precision", "recall", "jaccard",
        "exact_match", "partial_match", "header_coverage",
        "support", "pred_count", "tp", "fp", "fn",
        "duration_sec", "prompt_tokens", "completion_tokens", "total_tokens",
        "column_headers_f1", "projected_row_headers_f1", "spanning_f1",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logging.info(f"Total records loaded: {len(df)} from {len(ckpts)} runs")
    return df

# =========================
# DEDUP / MERGE LOGIC
# =========================

def build_task_key(r: Dict) -> str:
    """
    Unique key per (model, prompt, table, format) task.
    Used to deduplicate when merging original + retry runs.
    """
    return "|".join([
        str(r.get("model_alias", r.get("model", "?"))),
        str(r.get("prompt_name", "?")),
        str(r.get("source_group", "?")),
        str(r.get("source_stem",  "?")),
        str(r.get("table_index",  "0")),
        str(r.get("table_format", "?")),
    ])


def merge_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    When multiple runs cover the same task (original + retry + capped_retry),
    keep the BEST result per task key ranked by:
      1. api_success=True  > api_success=False
      2. parse_success=True > False
      3. f1 (highest wins)
      4. completion_capped=False > True  (prefer complete responses)

    This ensures retried tasks replace their failed/capped originals.
    """
    if df.empty:
        return df

    df = df.copy()
    df["_task_key"] = df.apply(build_task_key, axis=1)

    # Score for ranking: higher = better
    df["_api_ok"]   = pd.to_numeric(df.get("api_success",   False), errors="coerce").fillna(0)
    df["_parse_ok"] = pd.to_numeric(df.get("parse_success", False), errors="coerce").fillna(0)
    df["_f1"]       = pd.to_numeric(df.get("f1", 0),                errors="coerce").fillna(0)
    df["_not_capped"] = 1 - pd.to_numeric(
        df.get("completion_capped", False), errors="coerce").fillna(0)

    # Sort so best result comes first for each task
    df = df.sort_values(
        ["_task_key", "_api_ok", "_parse_ok", "_f1", "_not_capped"],
        ascending=[True, False, False, False, False]
    )

    before = len(df)
    df = df.drop_duplicates(subset=["_task_key"], keep="first")
    after  = len(df)
    dupes  = before - after

    if dupes > 0:
        logging.info(
            f"Merged {before} records → {after} unique tasks "
            f"(replaced {dupes} duplicates with better retry results)"
        )

    df = df.drop(columns=["_task_key", "_api_ok", "_parse_ok", "_f1", "_not_capped"],
                 errors="ignore")
    return df.reset_index(drop=True)




# =========================
# AGGREGATION
# =========================

def agg_group(df: pd.DataFrame, group_cols: List[str],
               metric_prefix: str = "") -> pd.DataFrame:
    """Group by given columns and compute mean/median of key metrics."""
    if df.empty:
        return pd.DataFrame()

    base_metrics = [
        "f1", "precision", "recall", "jaccard",
        "exact_match", "partial_match", "header_coverage",
        "support", "pred_count", "tp", "fp", "fn",
        "duration_sec", "completion_tokens", "prompt_tokens",
    ]
    if metric_prefix:
        base_metrics = [m for m in [
            f"{metric_prefix}_f1",
            f"{metric_prefix}_precision",
            f"{metric_prefix}_recall",
        ] if m in df.columns]

    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        row: Dict[str, Any] = {}
        ks = [key] if len(group_cols) == 1 else list(key)
        for col, val in zip(group_cols, ks):
            row[col] = val
        row["n"]            = len(sub)
        row["api_success"]  = float(pd.to_numeric(
            sub.get("api_success", pd.Series(dtype=float)), errors="coerce").mean())
        row["parse_success"] = float(pd.to_numeric(
            sub.get("parse_success", pd.Series(dtype=float)), errors="coerce").mean())
        if "completion_capped" in sub.columns:
            row["capped_rate"] = float(pd.to_numeric(
                sub["completion_capped"], errors="coerce").mean())

        for m in base_metrics:
            if m not in sub.columns:
                continue
            s = pd.to_numeric(sub[m], errors="coerce")
            if s.notna().any():
                alias = m.replace(f"{metric_prefix}_", "") if metric_prefix else m
                row[f"{alias}_mean"]   = float(s.mean())
                row[f"{alias}_median"] = float(s.median())
                row[f"{alias}_std"]    = float(s.std())

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# COMPARISON TABLES
# =========================

def build_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model — the main paper table."""
    return agg_group(df, ["model_alias"])


def build_model_prompt(df: pd.DataFrame) -> pd.DataFrame:
    return agg_group(df, ["model_alias", "prompt_name"])


def build_model_format(df: pd.DataFrame) -> pd.DataFrame:
    return agg_group(df, ["model_alias", "table_format"])


def build_model_prompt_format(df: pd.DataFrame) -> pd.DataFrame:
    return agg_group(df, ["model_alias", "prompt_name", "table_format"])


def build_model_source(df: pd.DataFrame) -> pd.DataFrame:
    return agg_group(df, ["model_alias", "source_group"])


def build_model_size(df: pd.DataFrame) -> pd.DataFrame:
    return agg_group(df, ["model_alias", "table_rows_bin"])


def build_model_type(df: pd.DataFrame) -> pd.DataFrame:
    """Header type breakdown per model — requires has_type_info=True rows."""
    if "has_type_info" not in df.columns:
        return pd.DataFrame()
    sub = df[df["has_type_info"] == True].copy()
    if sub.empty:
        return pd.DataFrame()

    frames = []
    for ht in ["column_headers", "projected_row_headers", "spanning"]:
        col_f1 = f"{ht}_f1"
        if col_f1 not in sub.columns:
            continue
        tmp = agg_group(sub, ["model_alias"], metric_prefix=ht)
        if not tmp.empty:
            tmp.insert(1, "header_type", ht)
            frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_pivot(df: pd.DataFrame,
                index: str, columns: str, values: str = "f1_mean") -> pd.DataFrame:
    """Pivot table: index × columns → metric value. Useful for paper tables."""
    agg = agg_group(df, [index, columns])
    if agg.empty or values not in agg.columns:
        return pd.DataFrame()
    pivot = agg.pivot(index=index, columns=columns, values=values)
    pivot.columns.name = None
    pivot = pivot.round(3)
    return pivot


# =========================
# SUMMARY TEXT
# =========================

def write_summary(df: pd.DataFrame, out_path: Path, run_dirs: List[Path]):
    models = sorted(df["model_alias"].dropna().unique())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("CROSS-MODEL COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Runs analysed: {len(run_dirs)}\n")
        f.write(f"Models: {', '.join(models)}\n")
        f.write(f"Total records: {len(df)}\n\n")

        # Overall by model
        f.write("OVERALL METRICS BY MODEL\n" + "-" * 40 + "\n")
        overall = build_model_summary(df)
        if not overall.empty:
            cols = ["model_alias", "n", "api_success", "parse_success"] + \
                   [k for m, k in DISPLAY_METRICS if k in overall.columns]
            f.write(overall[[c for c in cols if c in overall.columns]]
                    .to_string(index=False) + "\n\n")

        # JSON vs HTML by model
        f.write("JSON vs HTML — F1 BY MODEL\n" + "-" * 40 + "\n")
        try:
            pivot_fmt = build_pivot(df, "model_alias", "table_format", "f1_mean")
            if not pivot_fmt.empty:
                f.write(pivot_fmt.to_string() + "\n\n")
        except Exception as e:
            f.write(f"  (pivot error: {e})\n\n")

        # Best prompt per model
        f.write("BEST PROMPT PER MODEL (by F1)\n" + "-" * 40 + "\n")
        mp = build_model_prompt(df)
        if not mp.empty and "f1_mean" in mp.columns:
            best = mp.loc[mp.groupby("model_alias")["f1_mean"].idxmax()]
            for _, row in best.iterrows():
                f.write(
                    f"  {row['model_alias']:15s}  "
                    f"prompt={row['prompt_name']:25s}  "
                    f"F1={row.get('f1_mean',0):.3f}  "
                    f"P={row.get('precision_mean',0):.3f}  "
                    f"R={row.get('recall_mean',0):.3f}\n"
                )
            f.write("\n")

        # Header type breakdown
        f.write("HEADER TYPE BREAKDOWN (pubtables, F1 mean)\n" + "-" * 40 + "\n")
        mt = build_model_type(df)
        if not mt.empty and "f1_mean" in mt.columns:
            f.write(mt[["model_alias", "header_type", "n", "f1_mean", "f1_median"]]
                    .to_string(index=False) + "\n\n")

        # By table size
        f.write("BY TABLE SIZE (F1 mean)\n" + "-" * 40 + "\n")
        try:
            pivot_size = build_pivot(df, "model_alias", "table_rows_bin", "f1_mean")
            if not pivot_size.empty:
                # Sort columns
                size_order = ["<=10", "11-25", ">25"]
                ordered = [c for c in size_order if c in pivot_size.columns]
                f.write(pivot_size[ordered].to_string() + "\n\n")
        except Exception as e:
            f.write(f"  (pivot error: {e})\n\n")

        # Truncation analysis
        if "completion_capped" in df.columns:
            f.write("TRUNCATION ANALYSIS (completion_capped)\n" + "-" * 40 + "\n")
            capped_df  = df[df["completion_capped"] == True]
            uncapped_df = df[df["completion_capped"] == False]
            for model in models:
                cap = capped_df[capped_df["model_alias"] == model]
                unc = uncapped_df[uncapped_df["model_alias"] == model]
                cap_f1 = cap["f1"].mean() if len(cap) else float("nan")
                unc_f1 = unc["f1"].mean() if len(unc) else float("nan")
                f.write(
                    f"  {model:15s}  capped={len(cap)}/{len(df[df['model_alias']==model])} "
                    f"({len(cap)/max(1,len(df[df['model_alias']==model])):.0%})  "
                    f"F1_capped={cap_f1:.3f}  F1_full={unc_f1:.3f}\n"
                )
            f.write("\n")

    logging.info(f"Summary → {out_path}")


# =========================
# SAVE
# =========================

def save_views(views: Dict[str, pd.DataFrame], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, vdf in views.items():
        if vdf.empty:
            continue
        vdf.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        logging.info(f"  Saved {name}.csv ({len(vdf)} rows)")

    # All views in one XLSX
    try:
        with pd.ExcelWriter(out_dir / "comparison_all.xlsx", engine="openpyxl") as w:
            for name, vdf in views.items():
                if not vdf.empty:
                    vdf.to_excel(w, sheet_name=name[:31], index=False)
        logging.info("  Saved comparison_all.xlsx")
    except Exception as e:
        logging.warning(f"XLSX save failed: {e}")


def save_flat_responses(df: pd.DataFrame, out_dir: Path):
    """Save the merged flat table of all responses."""
    # Drop raw_response and parsed_headers to keep file manageable
    slim = df.drop(columns=[c for c in ["raw_response", "parsed_headers",
                                         "tokens_used", "_run_dir",
                                         "true_headers_raw", "true_headers_1based",
                                         "true_headers_by_type_raw"]
                             if c in df.columns], errors="ignore")
    slim.to_csv(out_dir / "all_responses.csv", index=False, encoding="utf-8-sig")
    logging.info(f"  Saved all_responses.csv ({len(slim)} rows)")

    try:
        slim.to_parquet(out_dir / "all_responses.parquet", index=False)
        logging.info("  Saved all_responses.parquet")
    except Exception:
        pass  # parquet optional


# =========================
# MAIN
# =========================

def discover_runs(results_dir: Path) -> List[Path]:
    """Find all run_* subdirectories inside results_dir."""
    runs = sorted([p for p in results_dir.iterdir()
                   if p.is_dir() and p.name.startswith("run_")])
    logging.info(f"Discovered {len(runs)} run directories in {results_dir}")
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model analysis for table header detection experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_dirs", nargs="*", metavar="RUN_DIR",
        help="One or more run_* directories to compare. "
             "If omitted, uses --results-dir to discover all runs.",
    )
    parser.add_argument(
        "--results-dir", default=None, metavar="DIR",
        help="Root results directory. Auto-discovers all run_* subdirs.",
    )
    parser.add_argument(
        "--output-dir", default="analysis", metavar="DIR",
        help="Where to write comparison outputs.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="ALIAS",
        help="Filter to specific model aliases (e.g. qwen30b llama8b).",
    )
    args = parser.parse_args()

    # Resolve run directories
    run_dirs: List[Path] = []
    if args.run_dirs:
        for rd in args.run_dirs:
            p = Path(rd)
            if not p.exists():
                logging.warning(f"Run dir not found: {p}")
            else:
                run_dirs.append(p)
    elif args.results_dir:
        run_dirs = discover_runs(Path(args.results_dir))
    else:
        parser.error("Provide run dirs as arguments or use --results-dir.")

    if not run_dirs:
        logging.error("No run directories found."); sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    df_raw = load_all_runs(run_dirs)

    # Merge: keep best result per task when retries exist
    df = merge_runs(df_raw)
    logging.info(
        f"After merge: {len(df)} unique tasks "
        f"(from {len(df_raw)} total records across {len(run_dirs)} runs)"
    )

    # Filter by model if requested
    if args.models:
        df = df[df["model_alias"].isin(args.models)]
        logging.info(f"Filtered to models: {args.models} → {len(df)} records")

    if df.empty:
        logging.error("No records after filtering."); sys.exit(1)

    # Build views
    views = {
        "comparison_by_model":              build_model_summary(df),
        "comparison_by_model_prompt":       build_model_prompt(df),
        "comparison_by_model_format":       build_model_format(df),
        "comparison_by_model_prompt_format":build_model_prompt_format(df),
        "comparison_by_model_source":       build_model_source(df),
        "comparison_by_model_size":         build_model_size(df),
        "comparison_by_model_type":         build_model_type(df),
    }

    # Pivot tables for paper
    try:
        views["pivot_model_x_prompt_F1"] = build_pivot(df, "model_alias", "prompt_name")
        views["pivot_model_x_format_F1"] = build_pivot(df, "model_alias", "table_format")
        views["pivot_model_x_size_F1"]   = build_pivot(df, "model_alias", "table_rows_bin")
    except Exception as e:
        logging.warning(f"Pivot build error: {e}")

    save_views(views, out_dir)
    save_flat_responses(df, out_dir)
    write_summary(df, out_dir / "summary.txt", run_dirs)

    # Print quick overview to console
    print("\n" + "=" * 70)
    print("QUICK OVERVIEW — F1 by model")
    print("=" * 70)
    overall = views["comparison_by_model"]
    if not overall.empty and "f1_mean" in overall.columns:
        for _, row in overall.sort_values("f1_mean", ascending=False).iterrows():
            alias = str(row.get("model_alias", "?"))
            print(
                f"  {alias:<15}  "
                f"n={int(row.get('n', 0)):5d}  "
                f"F1={float(row.get('f1_mean', 0)):.3f}  "
                f"P={float(row.get('precision_mean', 0)):.3f}  "
                f"R={float(row.get('recall_mean', 0)):.3f}  "
                f"Exact={float(row.get('exact_match_mean', 0)):.3f}"
            )

    if "comparison_by_model_format" in views:
        fmt_df = views["comparison_by_model_format"]
        if not fmt_df.empty and "f1_mean" in fmt_df.columns:
            print("\n" + "-" * 70)
            print("F1 by model × format")
            print("-" * 70)
            for _, row in fmt_df.iterrows():
                alias = str(row.get("model_alias", "?"))
                fmt   = str(row.get("table_format", "?"))
                print(
                    f"  {alias:<15}  "
                    f"[{fmt:<4}]  "
                    f"F1={float(row.get('f1_mean', 0)):.3f}  "
                    f"P={float(row.get('precision_mean', 0)):.3f}  "
                    f"R={float(row.get('recall_mean', 0)):.3f}"
                )

    print(f"\nFull analysis → {out_dir}/")


if __name__ == "__main__":
    main()