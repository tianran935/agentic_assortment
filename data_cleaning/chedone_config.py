from __future__ import annotations

from pathlib import Path


CHE_DONE_DEFAULT_INPUT = Path("/root/autodl-tmp/data/cheDONE")

TRANSACTION_SOURCE = "transformed_store_item_969_1.csv"
UPC_SOURCE = "transformed_upc.csv"

TRANSACTION_COLUMNS = [
    "txn_id",
    "card_nbr",
    "store_id",
    "txn_tm",
    "txn_dte",
    "household_id",
    "upc_id",
    "gross_amt",
    "net_amt",
    "mkdn_amt",
    "item_qty",
    "meas_qty",
]

UPC_COLUMNS = [
    "upc_id",
    "upc_description",
    "generic_cic_id",
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

TRANSACTION_ID_COLUMNS = {
    "txn_id",
    "card_nbr",
    "store_id",
    "household_id",
    "upc_id",
}

UPC_ID_COLUMNS = {
    "upc_id",
    "generic_cic_id",
    "subclass_id",
    "class_id",
    "category_id",
    "group_id",
    "department_id",
    "dept_section_id",
}

TRANSACTION_NUMERIC_COLUMNS = ["gross_amt", "net_amt", "mkdn_amt", "item_qty", "meas_qty"]
TRANSACTION_DATE_COLUMNS = ["txn_tm", "txn_dte"]
