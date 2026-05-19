from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("/root/autodl-tmp/data_cleaning/outputs/cleaned/transactions/transactions_with_upc.csv")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/data_cleaning/outputs/sku_pool")
VALID_METRICS = ("net_amt", "gross_amt", "item_qty", "line_count")


def aggregate_category_skus(
    input_csv: Path,
    metric: str = "net_amt",
    top_k: int = 50,
    chunk_size: int = 300_000,
    category_names: list[str] | None = None,
) -> pd.DataFrame:
    if metric not in VALID_METRICS:
        raise ValueError(f"Unsupported metric: {metric}. Valid metrics: {', '.join(VALID_METRICS)}")

    metric_store: dict[tuple[str, str, str, str], dict[str, float]] = {}

    usecols = [
        "category_id",
        "category_name",
        "upc_id",
        "upc_description",
        "net_amt",
        "gross_amt",
        "item_qty",
    ]

    for chunk in pd.read_csv(input_csv, usecols=usecols, chunksize=chunk_size):
        if category_names:
            chunk = chunk.loc[chunk["category_name"].isin(category_names)].copy()
            if chunk.empty:
                continue
        for col in ("net_amt", "gross_amt", "item_qty"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)
        chunk["line_count"] = 1

        grouped = (
            chunk.groupby(["category_id", "category_name", "upc_id", "upc_description"], dropna=False)[
                ["net_amt", "gross_amt", "item_qty", "line_count"]
            ]
            .sum()
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            key = (
                "" if pd.isna(row.category_id) else str(row.category_id),
                "" if pd.isna(row.category_name) else str(row.category_name),
                "" if pd.isna(row.upc_id) else str(row.upc_id),
                "" if pd.isna(row.upc_description) else str(row.upc_description),
            )
            if key not in metric_store:
                metric_store[key] = {
                    "net_amt": 0.0,
                    "gross_amt": 0.0,
                    "item_qty": 0.0,
                    "line_count": 0.0,
                }
            metric_store[key]["net_amt"] += float(row.net_amt)
            metric_store[key]["gross_amt"] += float(row.gross_amt)
            metric_store[key]["item_qty"] += float(row.item_qty)
            metric_store[key]["line_count"] += float(row.line_count)

    rows = []
    for (category_id, category_name, upc_id, upc_description), values in metric_store.items():
        rows.append(
            {
                "category_id": category_id,
                "category_name": category_name,
                "upc_id": upc_id,
                "upc_description": upc_description,
                "net_amt": round(values["net_amt"], 2),
                "gross_amt": round(values["gross_amt"], 2),
                "item_qty": round(values["item_qty"], 2),
                "line_count": int(values["line_count"]),
            }
        )

    all_skus = pd.DataFrame(rows)
    all_skus = all_skus.loc[
        all_skus["category_name"].fillna("").astype(str).str.strip().ne("")
        & all_skus["upc_id"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    all_skus = all_skus.sort_values(
        ["category_name", metric, "line_count", "upc_description"],
        ascending=[True, False, False, True],
    )
    all_skus["rank_within_category"] = all_skus.groupby("category_name")[metric].rank(
        method="first",
        ascending=False,
    )
    top_skus = all_skus.loc[all_skus["rank_within_category"] <= top_k].copy()
    top_skus["rank_within_category"] = top_skus["rank_within_category"].astype(int)
    top_skus = top_skus.sort_values(["category_name", "rank_within_category"])
    return top_skus


def random_sample_from_top(
    top_skus: pd.DataFrame,
    n: int,
    seed: int | None = None,
    category_names: list[str] | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be positive.")

    if category_names:
        sample_source = top_skus.loc[top_skus["category_name"].isin(category_names)].copy()
    else:
        sample_source = top_skus.copy()

    rng = random.Random(seed)
    sampled_frames: list[pd.DataFrame] = []

    for category_name, group in sample_source.groupby("category_name", dropna=False):
        if len(group) < n:
            if strict:
                raise ValueError(
                    f"Category '{category_name}' only has {len(group)} candidates in top list, cannot sample {n}."
                )
            continue
        chosen_indices = sorted(rng.sample(list(group.index), n))
        sampled = group.loc[chosen_indices].copy()
        sampled["sample_n"] = n
        sampled["sample_seed"] = "" if seed is None else seed
        sampled_frames.append(sampled)

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    sampled_df["sample_rank_within_category"] = sampled_df.groupby("category_name").cumcount() + 1
    return sampled_df.sort_values(["category_name", "sample_rank_within_category"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-category top-50 SKU table and sample n final SKUs from each category."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-top50", help="Aggregate category-level top K SKU table.")
    build_parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build_parser.add_argument("--metric", default="net_amt", choices=VALID_METRICS)
    build_parser.add_argument("--top-k", type=int, default=50)
    build_parser.add_argument("--chunk-size", type=int, default=300_000)
    build_parser.add_argument(
        "--category-names",
        nargs="*",
        default=None,
        help="Optional list of category_name values. If provided, only build top-k for these categories.",
    )

    sample_parser = subparsers.add_parser("sample-n", help="Sample n SKUs from each category's top list.")
    sample_parser.add_argument("--top-csv", type=Path, default=DEFAULT_OUTPUT_DIR / "top_50_skus_by_category.csv")
    sample_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    sample_parser.add_argument("--n", type=int, required=True)
    sample_parser.add_argument("--seed", type=int, default=None)
    sample_parser.add_argument(
        "--category-names",
        nargs="*",
        default=None,
        help="Optional list of category_name values to sample. If omitted, sample from every category in top-csv.",
    )
    sample_parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip categories with fewer than n candidates instead of raising an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "build-top50":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        top_skus = aggregate_category_skus(
            input_csv=args.input_csv,
            metric=args.metric,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            category_names=args.category_names,
        )
        out_name = (
            f"top_{args.top_k}_skus_selected_categories.csv"
            if args.category_names
            else f"top_{args.top_k}_skus_by_category.csv"
        )
        top_skus.to_csv(args.output_dir / out_name, index=False)
        return

    if args.command == "sample-n":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        top_skus = pd.read_csv(args.top_csv)
        sampled = random_sample_from_top(
            top_skus=top_skus,
            n=args.n,
            seed=args.seed,
            category_names=args.category_names,
            strict=not args.non_strict,
        )
        seed_suffix = "noseed" if args.seed is None else f"seed_{args.seed}"
        out_name = f"sampled_skus_n_{args.n}_{seed_suffix}.csv"
        sampled.to_csv(args.output_dir / out_name, index=False)
        return


if __name__ == "__main__":
    main()
