import json

import pandas as pd

from scripts.migrate_qwen_v5_to_v6 import migrate
from src.core import config


def test_qwen_v5_to_v6_migration_repairs_only_low_confidence(tmp_path):
    input_csv = tmp_path / "pseudo.csv"
    output_csv = tmp_path / "pseudo_v6.csv"
    rows = []
    row = {"StudyInstanceUID": "study"}
    for target in config.TARGETS:
        row[target] = 0.85
        row[f"{target}_weight"] = 0.85
    row["ACL"] = 0.15
    row["ACL_weight"] = 0.85
    row.update(
        report_sha256="report",
        extractor_version="clinical-report-labels-v5",
        engine="vllm",
        model_id="Qwen/Qwen2.5-72B-Instruct",
        prompt_sha256="prompt",
    )
    rows.append(row)
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    with open(f"{input_csv}.manifest.json", "w", encoding="utf-8") as stream:
        json.dump(
            {
                "extractor_version": "clinical-report-labels-v5",
                "status": "complete",
                "completed_studies": 1,
                "pseudo_csv_sha256": "old",
            },
            stream,
        )

    result = migrate(str(input_csv), str(output_csv))
    out = pd.read_csv(output_csv)
    assert result["low_confidence_rows_repaired"] == len(config.TARGETS)
    assert out.loc[0, "ACL"] == 0.35
    assert out.loc[0, "ACL_weight"] == 0.25
    assert out.loc[0, "MCL"] == 0.65
    assert out.loc[0, "MCL_weight"] == 0.25
    assert set(out["extractor_version"]) == {"clinical-report-labels-v6"}
