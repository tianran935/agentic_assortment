from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_barh(df: pd.DataFrame, label_col: str, value_col: str, title: str, out_path: Path, top_n: int = 15) -> None:
    plot_df = df.head(top_n).copy().iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df[label_col], plot_df[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualization for transformed transaction + UPC data.")
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("outputs/stats"),
        help="Directory containing statistics outputs from describe_chedone.py.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--value-column",
        default="net_amt",
        choices=["gross_amt", "net_amt", "mkdn_amt", "item_qty", "meas_qty"],
        help="Transaction metric to visualize as the primary value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    value_col = args.value_column

    sales_by_upc = pd.read_csv(args.stats_dir / "sales_by_upc.csv").sort_values(value_col, ascending=False)
    sales_by_upc["upc_label"] = (
        sales_by_upc["upc_description"].fillna("UNKNOWN").astype(str).str.slice(0, 45)
        + " | "
        + sales_by_upc["upc_id"].fillna("NA").astype(str)
    )

    sales_by_category = pd.read_csv(args.stats_dir / "sales_by_category.csv").sort_values(value_col, ascending=False)
    sales_by_category["category_label"] = (
        sales_by_category["category_name"].fillna("UNKNOWN").astype(str).str.slice(0, 50)
        + " | "
        + sales_by_category["category_id"].fillna("NA").astype(str)
    )

    daily_value = pd.read_csv(args.stats_dir / "daily_value_trend.csv").sort_values("txn_day")
    store_value = pd.read_csv(args.stats_dir / "store_day_value.csv").groupby("store_id", dropna=False)[value_col].sum().reset_index()
    store_value = store_value.sort_values(value_col, ascending=False)

    plt.figure(figsize=(10, 6))
    positive_values = sales_by_upc[sales_by_upc[value_col] > 0][value_col]
    plt.hist(positive_values, bins=50)
    plt.title(f"Distribution of {value_col} Across UPCs")
    plt.xlabel(value_col)
    plt.ylabel("UPC count")
    plt.tight_layout()
    plt.savefig(args.figure_dir / "value_distribution_by_upc.png", dpi=180)
    plt.close()

    save_barh(
        sales_by_upc,
        "upc_label",
        value_col,
        f"Top 15 UPC by {value_col}",
        args.figure_dir / "top_upc_by_value.png",
    )
    save_barh(
        sales_by_category,
        "category_label",
        value_col,
        f"Top 15 Categories by {value_col}",
        args.figure_dir / "top_categories_by_value.png",
    )

    plt.figure(figsize=(11, 6))
    plt.plot(daily_value["txn_day"].astype(str), daily_value[value_col], linewidth=1.8)
    plt.title(f"Daily Trend of {value_col}")
    plt.xlabel("txn_day")
    plt.ylabel(value_col)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(args.figure_dir / "daily_value_trend.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(store_value["store_id"].astype(str), store_value[value_col])
    plt.title(f"Store-level {value_col}")
    plt.xlabel("store_id")
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(args.figure_dir / "store_value.png", dpi=180)
    plt.close()

    concentration = sales_by_upc[[value_col]].dropna().sort_values(value_col, ascending=False).reset_index(drop=True)
    concentration["upc_rank_share"] = (concentration.index + 1) / len(concentration)
    concentration["value_share"] = concentration[value_col].cumsum() / concentration[value_col].sum()
    plt.figure(figsize=(8, 6))
    plt.plot(concentration["upc_rank_share"], concentration["value_share"], label="Observed")
    plt.plot([0, 1], [0, 1], linestyle="--", label="45-degree line")
    plt.title(f"Value Concentration Curve ({value_col})")
    plt.xlabel("Share of UPCs")
    plt.ylabel("Cumulative share of value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.figure_dir / "value_concentration_curve.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
