"""Validate the executed notebook, report package, source inventory and aggregate consistency."""
import base64
from html.parser import HTMLParser
import json
from pathlib import Path

import nbformat
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"deliverables/eda"


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections=[]
        self.images=[]
        self.links=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="section":self.sections.append(attrs.get("id"))
        if tag=="img":self.images.append(attrs["src"])
        if tag=="a":self.links.append(attrs["href"])


def validate():
    nb=nbformat.read(ROOT/"notebooks/eda.ipynb",as_version=4)
    nbformat.validate(nb)
    assert nb.metadata["eda_execution"]["status"]=="ready"
    code=[c for c in nb.cells if c.cell_type=="code"]
    assert all(c.execution_count is not None for c in code)
    assert not [o for c in code for o in c.outputs if o.output_type=="error"]
    manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]=="ready"
    assert manifest["validation"]["input_metadata_unchanged"]
    for item in json.loads((OUT/"input_inventory.json").read_text(encoding="utf-8")):
        p=ROOT/item["path"]
        assert p.stat().st_size==item["size"] and p.stat().st_mtime_ns==item["mtime_ns"],str(p)
    tables={name:pd.read_csv(OUT/"tables"/(name+".csv")) for name in manifest["tables"]}
    n=manifest["validation"]["predictors_described"]
    assert len(tables["feature_statistics_train"])==n
    assert len(tables["feature_target_associations"])==n
    assert tables["feature_statistics_train"].feature.is_unique
    assert set(tables["feature_statistics_train"].feature)==set(tables["feature_target_associations"].feature)
    assert tables["target_monthly"].rows.sum()==manifest["validation"]["features_rows"]
    assert tables["target_monthly"].positive_n.sum()==manifest["validation"]["positives"]
    expected=set(map(tuple,tables["schema"].query("table not in ['features','train']")[["table","column"]].values))
    observed=set(map(tuple,tables["missing_values"][["table","column"]].values))
    assert expected.issubset(observed),expected-observed
    for name in ("feature_outliers_train","amount_outliers_by_currency"):
        t=tables[name]
        classified=t[t.iqr_applicable]
        assert (classified.iqr_n.between(0,classified.log_valid_n)).all()
        z=t[t.zscore_n.notna()]
        assert (z.zscore_n.between(0,z.log_valid_n)).all()
        assert t.loc[~t.iqr_applicable,"iqr_n"].isna().all()
        assert (classified.iqr_n==classified.log_below_n+classified.log_above_n).all()
    main=tables["amount_outliers_by_currency"].query("currency == 11").iloc[0]
    prior=pd.read_csv(ROOT/"docs/propensity_feasibility/tail_review/amount_log_iqr_by_currency.csv")
    previous=prior[prior.currency_group.astype(str)=="11"].iloc[0]
    assert main.analysis_scale=="log1p"
    assert int(main.log_below_n)==int(previous.log1p_below)
    assert int(main.log_above_n)==int(previous.log1p_above)
    assert int(main.iqr_n)==int(previous.log1p_outside)
    assert int(main.ln_iqr_n)==int(previous.ln_outside)
    parser=ReportParser()
    parser.feed((OUT/"report.html").read_text(encoding="utf-8"))
    assert parser.sections==[f"step{i}" for i in range(1,8)]
    assert len(parser.images)==len(manifest["figures"])
    for src in parser.images:
        assert src.startswith("data:image/png;base64,")
        assert base64.b64decode(src.split(",",1)[1]).startswith(b"\x89PNG\r\n\x1a\n")
    for href in parser.links:
        assert href.startswith("#") or (OUT/href).is_file(),href
    notebook_images=sum("image/png" in o.get("data",{}) for c in code for o in c.outputs)
    assert notebook_images==len(manifest["figures"])
    result={"status":"ready","code_cells_executed":len(code),"errors":0,"tables":len(tables),
            "figures":len(parser.images),"predictors":n,"input_metadata_unchanged":True,
            "all_7_report_sections":True,"self_contained_images":True,"notebook_images":notebook_images,
            "log_iqr_matches_prior_audit":True,"currency11_log_iqr_n":int(main.iqr_n)}
    (ROOT/"outputs/eda_execution/validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":validate()
