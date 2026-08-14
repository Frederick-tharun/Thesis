"""Focused tests for the four frozen Chapter 2 thesis figures."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import re

from PIL import Image
import numpy as np
import pytest

from chapter2 import audit_step8 as audit
from chapter2 import plot_thesis_figures as figures
from chapter2.esn_data import file_sha256
from chapter2.esn_optimisation import ORDINARY_BASELINE, PARAMETER_AWARE


@pytest.fixture(scope="module")
def verified_inputs():
    records, evaluation, audit_data = figures.load_verified_results()
    return records, evaluation, audit_data


@pytest.fixture(scope="module")
def generated_set(tmp_path_factory, verified_inputs):
    records, _, _ = verified_inputs
    protected_paths = [
        figures.RAW_RESULTS_PATH,
        figures.AGGREGATE_RESULTS_PATH,
        figures.AUDIT_PATH,
        *[
            figures.PROJECT_ROOT / item["raw_arrays_path"]
            for item in records
        ],
    ]
    before = {path: file_sha256(path) for path in protected_paths}
    output = tmp_path_factory.mktemp("thesis-figures")
    manifest = figures.generate_thesis_figures(output)
    after = {path: file_sha256(path) for path in protected_paths}
    return output, manifest, before, after


def test_seed42_and_first_locked_window_are_deterministic(verified_inputs) -> None:
    records, evaluation, _ = verified_inputs
    first = figures.select_fixed_cases(records, evaluation)
    second = figures.select_fixed_cases(records, evaluation)
    first_ids = [
        item["record_id"]
        for case in first
        for item in (case.aware_record, case.baseline_record)
    ]
    second_ids = [
        item["record_id"]
        for case in second
        for item in (case.aware_record, case.baseline_record)
    ]
    assert first_ids == second_ids
    assert all(case.aware_record["seed"] == 42 for case in first)
    assert all(case.aware_record["window"] == 1 for case in first)
    assert all(case.aware_record["warmup_range"][0] == 70_000 for case in first)
    assert all(case.aware_record["forecast_range"] == [72_000, 80_000] for case in first)


def test_all_known_and_unseen_currents_appear_once_with_correct_labels(
    verified_inputs,
) -> None:
    records, evaluation, _ = verified_inputs
    cases = figures.select_fixed_cases(records, evaluation)
    assert [case.current for case in cases] == [1.67, 3.20, 3.50, 3.29, 3.34]
    assert [case.classification for case in cases] == [
        "used during training",
        "used during training",
        "used during training",
        "unseen current",
        "unseen current",
    ]
    assert len({case.current for case in cases}) == 5


def test_figure1_uses_all_8000_stored_predictions(verified_inputs) -> None:
    records, evaluation, _ = verified_inputs
    for case in figures.select_fixed_cases(records, evaluation):
        for item in (case.aware_record, case.baseline_record):
            arrays = figures.load_record_arrays(item)
            assert arrays["predictions"].shape == (8_000, 3)
            assert arrays["targets"].shape == (8_000, 3)


def test_continuous_boundaries_come_from_stored_current_and_total_four(
    verified_inputs,
) -> None:
    records, _, _ = verified_inputs
    selected = figures.select_continuous_records(records)
    aware = figures.load_record_arrays(selected[PARAMETER_AWARE])
    baseline = figures.load_record_arrays(selected[ORDINARY_BASELINE])
    np.testing.assert_array_equal(aware["current"], baseline["current"])
    expected = np.flatnonzero(aware["current"][1:] != aware["current"][:-1]) + 1
    actual = figures.derive_change_indices(aware["current"])
    np.testing.assert_array_equal(actual, expected)
    assert actual.tolist() == [98_000, 198_000, 298_000, 398_000]


def test_transition_windows_are_exactly_plus_minus_ten_time_units(
    verified_inputs,
) -> None:
    records, _, _ = verified_inputs
    selected = figures.select_continuous_records(records)
    current = figures.load_record_arrays(selected[PARAMETER_AWARE])["current"]
    for start, stop, boundary in figures.transition_slices(current):
        relative = (np.arange(start, stop) - boundary) * figures.DT
        assert relative[0] == pytest.approx(-10.0)
        assert relative[-1] == pytest.approx(10.0)
        assert len(relative) == 2_001


def test_figure4_includes_all_seeds_divergences_and_family_horizons(
    verified_inputs,
) -> None:
    records, _, _ = verified_inputs
    summary = figures.performance_summary(records)
    assert sum(
        summary[family][model]["record_count"]
        for family in figures.FAMILY_ORDER
        for model in figures.MODEL_TYPES
    ) == 210
    assert sum(
        summary[family][model]["divergence_count"]
        for family in figures.FAMILY_ORDER
        for model in figures.MODEL_TYPES
    ) == 22
    for family in figures.FAMILY_ORDER:
        for model in figures.MODEL_TYPES:
            assert summary[family][model]["seed_values"] == [42, 123, 456, 789, 2026]
    assert summary["known_short"][PARAMETER_AWARE]["evaluated_prediction_steps"] == [8_000]
    assert summary["known_long"][PARAMETER_AWARE]["evaluated_prediction_steps"] == [27_999]
    assert summary["continuous"][PARAMETER_AWARE]["evaluated_prediction_steps"] == [497_999]


def test_normalized_vpt_uses_each_records_evaluated_horizon(verified_inputs) -> None:
    records, _, _ = verified_inputs
    summary = figures.performance_summary(records)
    assert summary["known_short"][PARAMETER_AWARE][
        "median_normalized_vpt_percent"
    ] == pytest.approx(100.0)
    assert summary["known_long"][ORDINARY_BASELINE][
        "median_normalized_vpt_percent"
    ] == pytest.approx(4.8966034501232185)
    assert summary["continuous"][ORDINARY_BASELINE][
        "median_normalized_vpt_percent"
    ] == pytest.approx(0.050803314866094108)


def test_plotting_does_not_modify_raw_predictions_or_aggregate_data(
    generated_set,
) -> None:
    _, _, before, after = generated_set
    assert after == before


def test_manifest_is_strict_json_and_records_predetermined_selection(
    generated_set,
) -> None:
    output, manifest, _, _ = generated_set

    def reject(token: str) -> None:
        raise ValueError(token)

    loaded = json.loads(
        (output / "figure_manifest.json").read_text(encoding="utf-8"),
        parse_constant=reject,
    )
    assert loaded == manifest
    assert loaded["selection_policy"]["representative_seed"] == 42
    assert loaded["selection_policy"]["representative_window"] == 1
    assert loaded["selection_policy"]["all_divergent_records_retained_in_figure_4"]


def test_plotter_creates_exact_output_set_and_no_legacy_directory(
    generated_set,
) -> None:
    output, _, _, _ = generated_set
    assert {path.name for path in output.iterdir()} == figures.EXPECTED_OUTPUT_NAMES
    assert not (figures.FINAL_RESULTS / "figures").exists()
    assert not (figures.FINAL_RESULTS / "figures_final").exists()
    source = inspect.getsource(audit.run_audit)
    assert "generate_final_figures" not in source
    assert "FIGURE_MANIFEST_PATH" not in source


def test_all_images_are_nonempty_and_each_pdf_has_one_page(generated_set) -> None:
    output, _, _, _ = generated_set
    for stem in figures.FIGURE_STEMS:
        pdf = output / f"{stem}.pdf"
        png = output / f"{stem}.png"
        assert pdf.stat().st_size > 1_000
        assert png.stat().st_size > 1_000
        contents = pdf.read_bytes()
        assert contents.startswith(b"%PDF-")
        assert contents.rstrip().endswith(b"%%EOF")
        assert len(re.findall(rb"/Type\s*/Page\b", contents)) == 1


def test_png_dimensions_and_resolution_are_thesis_suitable(generated_set) -> None:
    output, _, _, _ = generated_set
    for stem in figures.FIGURE_STEMS:
        with Image.open(output / f"{stem}.png") as image:
            assert image.width >= 1_700
            assert image.height >= 1_200
            dpi = image.info.get("dpi")
            assert dpi is not None
            assert dpi[0] == pytest.approx(300.0, abs=1.0)
            assert dpi[1] == pytest.approx(300.0, abs=1.0)
