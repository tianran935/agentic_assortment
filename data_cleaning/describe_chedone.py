from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_upc(path_base: Path) -> pd.DataFrame:
    parquet_path = path_base.with_suffix(".parquet")
    csv_path = path_base.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Could not find {parquet_path.name} or {csv_path.name}")


def update_sum_map(target: dict[tuple, float], chunk: pd.DataFrame, keys: list[str], value_col: str) -> None:
    grouped = chunk.groupby(keys, dropna=False)[value_col].sum(min_count=1).reset_index()
    for row in grouped.itertuples(index=False):
        key = tuple(getattr(row, key) for key in keys)
        value = getattr(row, value_col)
        target[key] += 0.0 if pd.isna(value) else float(value)


def update_count_map(target: dict[tuple, float], chunk: pd.DataFrame, keys: list[str], value_col: str) -> None:
    grouped = chunk.groupby(keys, dropna=False)[value_col].count().reset_index()
    for row in grouped.itertuples(index=False):
        key = tuple(getattr(row, key) for key in keys)
        target[key] += float(getattr(row, value_col))


def export_map(target: dict[tuple, float], keys: list[str], value_col: str, out_path: Path) -> None:
    rows = [list(key) + [value] for key, value in target.items()]
    out_df = pd.DataFrame(rows, columns=keys + [value_col]).sort_values(value_col, ascending=False)
    out_df.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descriptive statistics for transformed transaction + UPC data.")
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("outputs/cleaned"),
        help="Directory containing cleaned outputs from clean_chedone.py.",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("outputs/stats"),
        help="Directory for descriptive statistics outputs.",
    )
    parser.add_argument(
        "--value-column",
        default="net_amt",
        choices=["gross_amt", "net_amt", "mkdn_amt", "item_qty", "meas_qty"],
        help="Transaction metric to treat as the primary analysis value.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Rows per chunk when scanning cleaned transaction CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.stats_dir.mkdir(parents=True, exist_ok=True)

    upc_df = load_upc(args.clean_dir / "upc_cleaned")
    txn_csv = args.clean_dir / "transactions_cleaned.csv"
    merged_csv = args.clean_dir / "transactions_with_upc.csv"
    value_col = args.value_column

    upc_counts = pd.DataFrame(
        [
            ("unique_upc", upc_df["upc_id"].nunique(dropna=True)),
            ("unique_subclass", upc_df["subclass_id"].nunique(dropna=True)),
            ("unique_class", upc_df["class_id"].nunique(dropna=True)),
            ("unique_category", upc_df["category_id"].nunique(dropna=True)),
            ("unique_group", upc_df["group_id"].nunique(dropna=True)),
            ("unique_department", upc_df["department_id"].nunique(dropna=True)),
        ],
        columns=["metric", "value"],
    )
    upc_counts.to_csv(args.stats_dir / "upc_structure_summary.csv", index=False)

    row_count = 0
    unique_txn: set[str] = set()
    unique_household: set[str] = set()
    unique_upc: set[str] = set()
    unique_store: set[str] = set()
    value_sum = 0.0
    value_sq_sum = 0.0
    value_count = 0
    value_min = None
    value_max = None
    date_sales: dict[tuple, float] = defaultdict(float)
    store_date_sales: dict[tuple, float] = defaultdict(float)

    for chunk in pd.read_csv(txn_csv, chunksize=args.chunk_size):
        row_count += len(chunk)
        unique_txn.update(chunk["txn_id"].dropna().astype(str).unique().tolist())
        unique_household.update(chunk["household_id"].dropna().astype(str).unique().tolist())
        unique_upc.update(chunk["upc_id"].dropna().astype(str).unique().tolist())
        unique_store.update(chunk["store_id"].dropna().astype(str).unique().tolist())

        values = pd.to_numeric(chunk[value_col], errors="coerce").dropna()
        if not values.empty:
            value_sum += float(values.sum())
            value_sq_sum += float((values ** 2).sum())
            value_count += len(values)
            value_min = float(values.min()) if value_min is None else min(value_min, float(values.min()))
            value_max = float(values.max()) if value_max is None else max(value_max, float(values.max()))

        chunk["txn_dte"] = pd.to_datetime(chunk["txn_dte"], errors="coerce")
        chunk["txn_day"] = chunk["txn_dte"].dt.strftime("%Y-%m-%d")
        update_sum_map(date_sales, chunk.dropna(subset=["txn_day"]), ["txn_day"], value_col)
        update_sum_map(store_date_sales, chunk.dropna(subset=["txn_day"]), ["store_id", "txn_day"], value_col)

    value_mean = value_sum / value_count if value_count else None
    value_var = value_sq_sum / value_count - value_mean ** 2 if value_count and value_mean is not None else None
    value_std = value_var ** 0.5 if value_var is not None and value_var >= 0 else None

    txn_summary = pd.DataFrame(
        [
            ("row_count", row_count),
            ("unique_txn", len(unique_txn)),
            ("unique_household", len(unique_household)),
            ("unique_upc", len(unique_upc)),
            ("unique_store", len(unique_store)),
            (f"{value_col}_sum", value_sum),
            (f"{value_col}_mean", value_mean),
            (f"{value_col}_std", value_std),
            (f"{value_col}_min", value_min),
            (f"{value_col}_max", value_max),
        ],
        columns=["metric", "value"],
    )
    txn_summary.to_csv(args.stats_dir / "transaction_structure_summary.csv", index=False)

    metric_rows = []
    for metric_col in ["gross_amt", "net_amt", "mkdn_amt", "item_qty", "meas_qty"]:
        total = 0.0
        sq_total = 0.0
        count = 0
        min_value = None
        max_value = None
        for chunk in pd.read_csv(txn_csv, chunksize=args.chunk_size):
            values = pd.to_numeric(chunk[metric_col], errors="coerce").dropna()
            if values.empty:
                continue
            total += float(values.sum())
            sq_total += float((values ** 2).sum())
            count += len(values)
            min_value = float(values.min()) if min_value is None else min(min_value, float(values.min()))
            max_value = float(values.max()) if max_value is None else max(max_value, float(values.max()))
        mean = total / count if count else None
        var = sq_total / count - mean ** 2 if count and mean is not None else None
        std = var ** 0.5 if var is not None and var >= 0 else None
        metric_rows.append(
            {
                "metric_name": metric_col,
                "count": count,
                "sum": total,
                "mean": mean,
                "std": std,
                "min": min_value,
                "max": max_value,
            }
        )
    pd.DataFrame(metric_rows).to_csv(args.stats_dir / "transaction_metric_summary.csv", index=False)

    sales_by_department: dict[tuple, float] = defaultdict(float)
    sales_by_group: dict[tuple, float] = defaultdict(float)
    sales_by_category: dict[tuple, float] = defaultdict(float)
    sales_by_class: dict[tuple, float] = defaultdict(float)
    sales_by_upc: dict[tuple, float] = defaultdict(float)
    line_count_by_upc: dict[tuple, float] = defaultdict(float)
    matched_rows = 0
    unmatched_rows = 0

    for chunk in pd.read_csv(merged_csv, chunksize=args.chunk_size):
        matched_rows += int(chunk["upc_description"].notna().sum())
        unmatched_rows += int(chunk["upc_description"].isna().sum())
        update_sum_map(sales_by_department, chunk, ["department_id", "department_name"], value_col)
        update_sum_map(sales_by_group, chunk, ["group_id", "group_name"], value_col)
        update_sum_map(sales_by_category, chunk, ["category_id", "category_name"], value_col)
        update_sum_map(sales_by_class, chunk, ["class_id", "class_name"], value_col)
        update_sum_map(sales_by_upc, chunk, ["upc_id", "upc_description"], value_col)
        update_count_map(line_count_by_upc, chunk, ["upc_id", "upc_description"], value_col)

    export_map(sales_by_department, ["department_id", "department_name"], value_col, args.stats_dir / "sales_by_department.csv")
    export_map(sales_by_group, ["group_id", "group_name"], value_col, args.stats_dir / "sales_by_group.csv")
    export_map(sales_by_category, ["category_id", "category_name"], value_col, args.stats_dir / "sales_by_category.csv")
    export_map(sales_by_class, ["class_id", "class_name"], value_col, args.stats_dir / "sales_by_class.csv")
    export_map(sales_by_upc, ["upc_id", "upc_description"], value_col, args.stats_dir / "sales_by_upc.csv")
    export_map(line_count_by_upc, ["upc_id", "upc_description"], "line_count", args.stats_dir / "line_count_by_upc.csv")

    pd.read_csv(args.stats_dir / "sales_by_upc.csv").head(100).to_csv(args.stats_dir / "top_100_upc_by_value.csv", index=False)
    pd.DataFrame([[k[0], v] for k, v in date_sales.items()], columns=["txn_day", value_col]).sort_values("txn_day").to_csv(
        args.stats_dir / "daily_value_trend.csv", index=False
    )
    pd.DataFrame([[k[0], k[1], v] for k, v in store_date_sales.items()], columns=["store_id", "txn_day", value_col]).sort_values(
        ["store_id", "txn_day"]
    ).to_csv(args.stats_dir / "store_day_value.csv", index=False)

    total_rows = matched_rows + unmatched_rows
    pd.DataFrame(
        [
            ("matched_rows", matched_rows),
            ("unmatched_rows", unmatched_rows),
            ("matched_share", matched_rows / total_rows if total_rows else 0.0),
        ],
        columns=["metric", "value"],
    ).to_csv(args.stats_dir / "merge_quality_summary.csv", index=False)


if __name__ == "__main__":
    main()
