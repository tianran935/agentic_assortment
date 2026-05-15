from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chedone_config import (
    CHE_DONE_DEFAULT_INPUT,
    TRANSACTION_COLUMNS,
    TRANSACTION_DATE_COLUMNS,
    TRANSACTION_ID_COLUMNS,
    TRANSACTION_NUMERIC_COLUMNS,
    TRANSACTION_SOURCE,
    UPC_COLUMNS,
    UPC_ID_COLUMNS,
    UPC_SOURCE,
)

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None


def normalize_id(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .str.replace(r"\.$", "", regex=True)
    )
    return normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def read_csv_auto(path: Path, *, chunksize: int | None = None) -> pd.DataFrame | pd.io.parsers.TextFileReader:
    return pd.read_csv(path, dtype="string", low_memory=False, chunksize=chunksize)


def clean_upc(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [col.strip().lower() for col in df.columns]
    rename_map = {old: new for old, new in zip(df.columns, UPC_COLUMNS)}
    df = df.rename(columns=rename_map)
    df = df[UPC_COLUMNS]
    for col in df.columns:
        if col in UPC_ID_COLUMNS:
            df[col] = normalize_id(df[col])
        elif col == "numeric_size_qty":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("string").str.strip().replace({"": pd.NA})
    return df


def clean_transaction_chunk(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [col.strip().lower() for col in df.columns]
    rename_map = {old: new for old, new in zip(df.columns, TRANSACTION_COLUMNS)}
    df = df.rename(columns=rename_map)
    df = df[TRANSACTION_COLUMNS]

    for col in TRANSACTION_ID_COLUMNS:
        df[col] = normalize_id(df[col])
    for col in TRANSACTION_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["txn_tm"] = pd.to_datetime(df["txn_tm"], errors="coerce")
    df["txn_dte"] = pd.to_datetime(df["txn_dte"], errors="coerce")
    return df


def write_outputs(df: pd.DataFrame, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(base_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def build_upc_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_count": len(df),
                "unique_upc": df["upc_id"].nunique(dropna=True),
                "unique_subclass": df["subclass_id"].nunique(dropna=True),
                "unique_class": df["class_id"].nunique(dropna=True),
                "unique_category": df["category_id"].nunique(dropna=True),
                "unique_group": df["group_id"].nunique(dropna=True),
                "unique_department": df["department_id"].nunique(dropna=True),
                "missing_upc_description": int(df["upc_description"].isna().sum()),
            }
        ]
    )


def process_batch(chunks: list[pd.DataFrame], n_jobs: int) -> list[pd.DataFrame]:
    if n_jobs > 1 and Parallel is not None and delayed is not None:
        return Parallel(n_jobs=n_jobs, backend="loky")(delayed(clean_transaction_chunk)(chunk) for chunk in chunks)
    return [clean_transaction_chunk(chunk) for chunk in chunks]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean transformed cheDONE transaction and UPC data.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=CHE_DONE_DEFAULT_INPUT,
        help="Directory containing transformed_store_item_969_1.csv and transformed_upc.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cleaned"),
        help="Directory for cleaned CSV/Parquet outputs.",
    )
    parser.add_argument(
        "--txn-chunk-size",
        type=int,
        default=200_000,
        help="Number of rows per transaction chunk.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for transaction chunk cleaning.",
    )
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=4,
        help="How many chunks to collect before dispatching to joblib.",
    )
    return parser.parse_args()


def append_csv(df: pd.DataFrame, path: Path, *, write_header: bool) -> None:
    df.to_csv(path, mode="a", index=False, header=write_header)


def flush_batch(
    chunks: list[pd.DataFrame],
    upc_join_df: pd.DataFrame,
    cleaned_csv_path: Path,
    merged_csv_path: Path,
    n_jobs: int,
    write_header_state: dict[str, bool],
    accumulators: dict[str, object],
) -> None:
    cleaned_batch = process_batch(chunks, n_jobs=n_jobs)
    for cleaned_df in cleaned_batch:
        merged_df = cleaned_df.merge(upc_join_df, on="upc_id", how="left")

        append_csv(cleaned_df, cleaned_csv_path, write_header=write_header_state["cleaned"])
        append_csv(merged_df, merged_csv_path, write_header=write_header_state["merged"])
        write_header_state["cleaned"] = False
        write_header_state["merged"] = False

        accumulators["row_count"] += len(cleaned_df)
        accumulators["txn_ids"].update(cleaned_df["txn_id"].dropna().astype(str).unique().tolist())
        accumulators["household_ids"].update(cleaned_df["household_id"].dropna().astype(str).unique().tolist())
        accumulators["upc_ids"].update(cleaned_df["upc_id"].dropna().astype(str).unique().tolist())
        accumulators["store_ids"].update(cleaned_df["store_id"].dropna().astype(str).unique().tolist())
        if cleaned_df["txn_dte"].notna().any():
            accumulators["min_date"] = (
                cleaned_df["txn_dte"].min()
                if accumulators["min_date"] is None
                else min(accumulators["min_date"], cleaned_df["txn_dte"].min())
            )
            accumulators["max_date"] = (
                cleaned_df["txn_dte"].max()
                if accumulators["max_date"] is None
                else max(accumulators["max_date"], cleaned_df["txn_dte"].max())
            )
        for col in TRANSACTION_NUMERIC_COLUMNS:
            accumulators["metric_sums"][col] += float(cleaned_df[col].sum(skipna=True))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    upc_path = args.input_dir / UPC_SOURCE
    txn_path = args.input_dir / TRANSACTION_SOURCE

    upc_df = clean_upc(read_csv_auto(upc_path))
    write_outputs(upc_df, args.output_dir / "upc_cleaned")
    build_upc_summary(upc_df).to_csv(args.output_dir / "upc_data_overview.csv", index=False)

    upc_join_df = upc_df[
        [
            "upc_id",
            "upc_description",
            "subclass_id",
            "class_id",
            "class_name",
            "category_id",
            "category_name",
            "group_id",
            "group_name",
            "department_id",
            "department_name",
            "dept_section_id",
            "dept_section_name",
            "numeric_size_qty",
            "size_uom_cd",
        ]
    ].copy()

    cleaned_csv_path = args.output_dir / "transactions_cleaned.csv"
    merged_csv_path = args.output_dir / "transactions_with_upc.csv"
    for path in [cleaned_csv_path, merged_csv_path]:
        if path.exists():
            path.unlink()

    accumulators = {
        "row_count": 0,
        "txn_ids": set(),
        "household_ids": set(),
        "upc_ids": set(),
        "store_ids": set(),
        "min_date": None,
        "max_date": None,
        "metric_sums": {col: 0.0 for col in TRANSACTION_NUMERIC_COLUMNS},
    }
    write_header_state = {"cleaned": True, "merged": True}

    chunk_iter = read_csv_auto(txn_path, chunksize=args.txn_chunk_size)
    pending_chunks: list[pd.DataFrame] = []
    for raw_chunk in chunk_iter:
        pending_chunks.append(raw_chunk)
        if len(pending_chunks) >= args.chunk_batch_size:
            flush_batch(
                pending_chunks,
                upc_join_df,
                cleaned_csv_path,
                merged_csv_path,
                args.n_jobs,
                write_header_state,
                accumulators,
            )
            pending_chunks = []

    if pending_chunks:
        flush_batch(
            pending_chunks,
            upc_join_df,
            cleaned_csv_path,
            merged_csv_path,
            args.n_jobs,
            write_header_state,
            accumulators,
        )

    txn_summary = {
        "row_count": accumulators["row_count"],
        "unique_txn": len(accumulators["txn_ids"]),
        "unique_household": len(accumulators["household_ids"]),
        "unique_upc": len(accumulators["upc_ids"]),
        "unique_store": len(accumulators["store_ids"]),
        "min_txn_dte": accumulators["min_date"],
        "max_txn_dte": accumulators["max_date"],
    }
    for key, value in accumulators["metric_sums"].items():
        txn_summary[f"{key}_sum"] = value
    pd.DataFrame([txn_summary]).to_csv(args.output_dir / "transaction_data_overview.csv", index=False)

    matched_rows = 0
    total_rows = 0
    for merged_chunk in pd.read_csv(merged_csv_path, chunksize=args.txn_chunk_size):
        matched_rows += int(merged_chunk["upc_description"].notna().sum())
        total_rows += len(merged_chunk)
    matched_share = matched_rows / total_rows if total_rows else 0.0
    with open(args.output_dir / "merge_quality.txt", "w", encoding="utf-8") as handle:
        handle.write(f"UPC merge matched share: {matched_share:.4%}\n")
        handle.write(f"Matched rows: {matched_rows}\n")
        handle.write(f"Total rows: {total_rows}\n")


if __name__ == "__main__":
    main()
