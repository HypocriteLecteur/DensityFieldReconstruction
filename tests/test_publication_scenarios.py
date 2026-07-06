from pathlib import Path

import pytest

from experiments.publication_scenarios import PROFILES, build_specs


def test_table_profiles_expand_scientific_differences(tmp_path):
    table2 = build_specs(
        PROFILES[2], project_root=tmp_path, output_root=Path("outputs")
    )
    table3 = build_specs(
        PROFILES[3], project_root=tmp_path, output_root=Path("outputs")
    )
    table4 = build_specs(
        PROFILES[4], project_root=tmp_path, output_root=Path("outputs")
    )

    assert len(table2) == len(table3) == 12
    assert {spec.training.lr_max_steps for spec in table2} == {100}
    assert {spec.training.lr_max_steps for spec in table3} == {500}
    assert len(table4) == 3
    assert {spec.dataset for spec in table4} == {"starling"}
    assert all(spec.output.workflow == "reconstruction" for spec in table2)


def test_publication_profile_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="not part"):
        build_specs(
            PROFILES[2],
            project_root=tmp_path,
            output_root=Path("outputs"),
            datasets=("unknown",),
        )


def test_table_entry_points_are_thin_and_side_effect_free():
    root = Path(__file__).resolve().parents[1] / "experiments"
    for table in (2, 3, 4):
        source = (root / f"run_scenarios_table_{table}.py").read_text(
            encoding="utf-8"
        )
        assert len(source.splitlines()) < 15
        assert "publication_scenarios import main" in source
        assert f"main({table})" in source
