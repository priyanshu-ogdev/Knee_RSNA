import numpy as np
import pandas as pd
import pytest

from src.core import config
from src.data.labels import build_labels
from src.data.preprocess import nlp_extractor


def _label_row(uid, value=0.0):
    return {
        "StudyInstanceUID": uid,
        **{target: value for target in config.TARGETS},
    }


def test_build_labels_preserves_gold_and_applies_extractor_confidence(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    train = pd.DataFrame(
        [
            {"StudyInstanceUID": "gold", **{target: np.nan for target in config.TARGETS}},
            {"StudyInstanceUID": "extra", **{target: np.nan for target in config.TARGETS}},
        ]
    )
    train.loc[0, "ACL"] = 1.0
    train.to_csv(data_root / "train.csv", index=False)

    extra = pd.DataFrame(
        [
            {
                **_label_row("gold", 0.0),
                **{f"{target}_weight": 0.5 for target in config.TARGETS},
            },
            {
                **_label_row("extra", 0.25),
                **{f"{target}_weight": 0.5 for target in config.TARGETS},
            },
        ]
    )
    extra_path = tmp_path / "extra.csv"
    extra.to_csv(extra_path, index=False)

    labels = build_labels(str(data_root), str(extra_path), extra_weight=0.5)

    assert labels.loc[0, "ACL"] == 1.0
    assert labels.loc[0, "ACL_weight"] == 1.0
    assert labels.loc[1, "ACL"] == 0.25
    assert labels.loc[1, "ACL_weight"] == 0.25


@pytest.mark.parametrize(
    "bad_value",
    [-0.1, 1.1],
)
def test_build_labels_rejects_out_of_range_extra_targets(tmp_path, bad_value):
    data_root = tmp_path / "data"
    data_root.mkdir()
    pd.DataFrame(
        [{"StudyInstanceUID": "study", **{target: np.nan for target in config.TARGETS}}]
    ).to_csv(data_root / "train.csv", index=False)
    extra_path = tmp_path / "extra.csv"
    pd.DataFrame([_label_row("study", bad_value)]).to_csv(extra_path, index=False)

    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        build_labels(str(data_root), str(extra_path))


def test_extraction_completes_missing_targets_for_partially_gold_studies(tmp_path, monkeypatch):
    targets = nlp_extractor.TARGETS
    data_root = tmp_path / "data"
    data_root.mkdir()
    train = pd.DataFrame(
        [
            {
                "StudyInstanceUID": "gold-study",
                "Report": "gold report",
                **{target: np.nan for target in targets},
            },
            {
                "StudyInstanceUID": "pseudo-study",
                "Report": "report text",
                **{target: np.nan for target in targets},
            },
        ]
    )
    train.loc[0, "ACL"] = 1.0
    train.to_csv(data_root / "train.csv", index=False)

    output = tmp_path / "pseudo.csv"
    cached = _label_row("pseudo-study", 0.0)
    cached["Fracture"] = np.nan
    pd.DataFrame([cached]).to_csv(output, index=False)

    def extract(report, uid):
        return {
            "StudyInstanceUID": uid,
            **{target: 0.0 for target in targets},
            **{f"{target}_weight": 0.5 for target in targets},
        }

    monkeypatch.setattr(nlp_extractor, "extract_by_rules", extract)
    _, stats = nlp_extractor.auto_complete_extraction(
        str(data_root), str(output), engine="rules"
    )

    result = pd.read_csv(output)
    assert stats["status"] == "complete"
    assert stats["total"] == 2
    assert set(result["StudyInstanceUID"]) == {"gold-study", "pseudo-study"}
    assert result[targets].notna().all().all()
