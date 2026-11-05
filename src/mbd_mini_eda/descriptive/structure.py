"""Cấu trúc, vai trò cột và quan hệ giữa các bảng."""
import pandas as pd
from ..engine import inventory
from ..statistics import result

from ..models import AnalysisContext, SectionResult


def run(context: AnalysisContext) -> SectionResult:
    tables, columns = inventory(context)
    relationships = pd.DataFrame([
        {"left": name, "right": "clients", "keys": "client_id", "cardinality": "many_to_one"}
        for name in ("targets", "transactions", "geo", "dialog")
    ])
    return result(context, "descriptive.structure", {
        "table_inventory": tables.copy(), "column_roles": columns.copy(),
        "relationship_summary": relationships}, grain="table", scope="full_file_metadata")
