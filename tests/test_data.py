"""Dataset parsing, label mapping, patient-ID integrity and validation checks."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from oncogene.data.load import (build_sample_metadata, parse_platform_table,
                                parse_series_matrix)
from oncogene.data import validate

SERIES = """!Series_title\t"Toy series"
!Series_geo_accession\t"GSE0"
!Sample_title\t"101-Normal"\t"101-Tumor"\t"102-Normal"\t"102-Tumor"
!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"\t"GSM4"
!Sample_characteristics_ch1\t"Sex: female"\t"Sex: female"\t"Sex: female"\t"Sex: female"
!Sample_characteristics_ch1\t"age: 40 yr"\t"age: 40 yr"\t"age: 51 yr"\t"age: 51 yr"
!Sample_characteristics_ch1\t"cancer stage: I"\t"cancer stage: I"\t"cancer stage: IIA"\t"cancer stage: IIA"
!Sample_characteristics_ch1\t"tissue: breast tumor adjacent normal tissue"\t"tissue: breast cancer tissue"\t"tissue: breast tumor adjacent normal tissue"\t"tissue: breast cancer tissue"
!Sample_characteristics_ch1\t"patient identifier: 101"\t"patient identifier: 101"\t"patient identifier: 102"\t"patient identifier: 102"
!Sample_platform_id\t"GPL5175"\t"GPL5175"\t"GPL5175"\t"GPL5175"
!Sample_data_processing\t"RMA"\t"RMA"\t"RMA"\t"RMA"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"\t"GSM4"
2315554\t7.31\t7.27\t7.63\t7.38
2315633\t7.24\t7.52\t7.42\t7.37
2315700\t5.0\t9.0\t5.5\t8.5
!series_matrix_table_end
"""

PLATFORM = """^PLATFORM = GPL0
!platform_table_begin
ID\tGB_LIST\tSPOT_ID\tseqname\tRANGE_GB\tRANGE_STRAND\tRANGE_START\tRANGE_STOP\ttotal_probes\tgene_assignment\tmrna_assignment\tcategory
2315554\tx\tchr1:1-2\tchr1\tNC\t+\t1\t2\t20\tNM_1 // ESR1 // estrogen receptor 1 // 6q25.1 // 2099 /// NM_2 // ESR1 // estrogen receptor 1 // 6q25.1 // 2099\tx\tmain
2315633\tx\tchr1:1-2\tchr1\tNC\t+\t1\t2\t20\tNM_3 // AAA // gene a // 1p1 // 1 /// NM_4 // BBB // gene b // 1p1 // 2\tx\tmain
2315700\tx\tchr1:1-2\tchr1\tNC\t+\t1\t2\t20\t---\tx\tnormgene->intron
!platform_table_end
"""


@pytest.fixture
def series_file(tmp_path):
    p = tmp_path / "toy_series_matrix.txt.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write(SERIES)
    return p


def test_parse_series_matrix(series_file):
    sm = parse_series_matrix(series_file)
    assert sm.expression.shape == (3, 4)
    assert list(sm.expression.columns) == ["GSM1", "GSM2", "GSM3", "GSM4"]
    assert sm.expression.index.dtype == object and "2315554" in sm.expression.index
    assert sm.series_meta["geo_accession"] == ["GSE0"]
    assert "char_patient_identifier" in sm.sample_meta_raw.columns


def test_label_and_patient_mapping(series_file):
    meta = build_sample_metadata(parse_series_matrix(series_file).sample_meta_raw)
    assert meta.loc["GSM1", "label"] == 0 and meta.loc["GSM2", "label"] == 1
    assert meta.loc["GSM2", "tissue_type"] == "Tumor"
    assert meta["patient_id"].tolist() == ["101", "101", "102", "102"]
    assert meta["age_years"].tolist() == [40, 40, 51, 51]


def test_title_characteristic_disagreement_raises(series_file):
    raw = parse_series_matrix(series_file).sample_meta_raw
    raw.loc[0, "title"] = "101-Tumor"   # contradicts the tissue characteristic
    with pytest.raises(ValueError, match="disagreement"):
        build_sample_metadata(raw)


def test_unknown_tissue_raises(series_file):
    raw = parse_series_matrix(series_file).sample_meta_raw
    raw.loc[0, "char_tissue"] = "healthy donor"
    with pytest.raises(ValueError, match="Unrecognised"):
        build_sample_metadata(raw)


def test_parse_platform_table(tmp_path):
    p = tmp_path / "gpl.txt"
    p.write_text(PLATFORM)
    ann = parse_platform_table(p)
    assert ann.loc["2315554", "gene_symbol"] == "ESR1"
    assert ann.loc["2315554", "entrez_id"] == "2099"
    assert ann.loc["2315554", "n_gene_symbols"] == 1
    assert ann.loc["2315633", "n_gene_symbols"] == 2
    assert pd.isna(ann.loc["2315700", "gene_symbol"])
    assert ann.loc["2315700", "category"] == "normgene->intron"


def _status(checks, name):
    return next(c for c in checks if c.name == name).status


def test_validation_passes_on_clean_pairs(paired_data, cfg):
    X, meta, _ = paired_data
    c = dict(cfg, dataset=dict(cfg["dataset"], expected_samples=len(meta),
                               expected_patients=meta.patient_id.nunique()))
    checks = validate.run_all(X, meta, c)
    assert not [x for x in checks if x.status == "fail"]


def test_validation_detects_problems(paired_data):
    X, meta, _ = paired_data
    bad = meta.copy()
    bad.iloc[1, bad.columns.get_loc("label")] = 0         # patient P00 now has two normals
    assert _status(validate.check_pairs(bad), "matched_pairs") == "fail"
    bad2 = meta.copy()
    bad2.iloc[0, bad2.columns.get_loc("patient_id")] = ""
    assert _status(validate.check_pairs(bad2), "missing_patient_ids") == "fail"
    dup = pd.concat([meta, meta.iloc[[0]]])
    assert _status(validate.check_duplicate_ids(dup, X), "duplicate_sample_ids") == "fail"
    Xd = X.copy()
    Xd.columns = [X.columns[0]] + list(X.columns[:-1])
    assert _status(validate.check_duplicate_ids(meta, Xd), "duplicate_probe_ids") == "fail"
    assert _status(validate.check_sample_counts(meta, 50, 25), "sample_count") == "fail"


def test_validation_value_checks(paired_data):
    X, _, _ = paired_data
    Xm = X.copy()
    Xm.iloc[0, 0] = np.nan
    Xm.iloc[:, 1] = 5.0
    checks = validate.check_expression_values(Xm, 0.1, 30)
    assert _status(checks, "missing_values") == "warn"
    assert _status(checks, "constant_features") == "warn"
    lin = validate.check_expression_values(2 ** X * 10, 0.1, 30)
    assert _status(lin, "expression_scale") == "warn"


# ----------------------------------------------------------------------------- real data
@pytest.mark.data
def test_real_dataset_integrity(real_dataset):
    ds = real_dataset
    assert ds.X.shape[0] == 50
    assert ds.meta["patient_id"].nunique() == 25
    assert ds.y.value_counts().to_dict() == {0: 25, 1: 25}
    per_patient = ds.meta.groupby("patient_id")["label"].agg(["count", "sum"])
    assert (per_patient["count"] == 2).all() and (per_patient["sum"] == 1).all()
    assert list(ds.X.index) == list(ds.meta.index)
    assert not ds.X.isna().any().any()
    assert ds.X.columns.is_unique and ds.meta.index.is_unique
    assert ds.X.values.max() < 20  # log2 RMA


@pytest.mark.data
def test_real_titles_match_labels(real_dataset):
    m = real_dataset.meta
    kind = m["title"].str.split("-").str[1]
    assert ((kind == "Tumor") == (m["label"] == 1)).all()
    assert (m["title"].str.split("-").str[0] == m["patient_id"]).all()
