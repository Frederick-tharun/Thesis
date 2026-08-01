"""Tests for the isolated Chapter 2 HR diagnostic simulator."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np

from chapter2 import generate_diagnostics as diagnostics
from chapter2.config_ch2 import (
    CONTINUOUS_CURRENT_SEQUENCE,
    DT,
    FIXED_CURRENTS,
    HR_PARAMETERS,
    HRParameters,
    INITIAL_STATE,
    INITIAL_TRANSIENT_STEPS,
    RETAINED_SAMPLES_PER_CURRENT,
)
from chapter2.dynamics_analysis_ch2 import (
    LyapunovAnalysis,
    analyze_spikes_and_bursts,
    preliminary_regime,
)
from chapter2.hr_data_ch2 import (
    HRTrajectory,
    save_trajectory_npz,
    simulate_continuous_currents,
    simulate_fixed_current,
)


OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"
EXPECTED_TRAJECTORY_HASHES = {
    "continuous_switched_currents.npz": "0921cbad321da1830433dc84e58f25ff2b0b6a6d571a31cae769bdfd6dc00a7b",
    "fixed_I_1p67.npz": "cc89af5e9a27d05a9501ea995a2ef361c146d6513ec1b19765238603af9ffb2b",
    "fixed_I_3p20.npz": "c4292a1e0fa5575d08419e7d302f980e2ed085878187f67659aa59decc486ded",
    "fixed_I_3p29.npz": "2ed394f457e5de5f0d14a5dd450611fd9774c02cb13fc68f7b21a2621273b7b4",
    "fixed_I_3p34.npz": "8cad49948c553df9b65a24deacf067dc999325dadb580a4787ded12b6315585d",
    "fixed_I_3p50.npz": "4d042da6adbde026468207c4e9f74443f8587c635dff2d28b69e2b1784d5cbbf",
}


def _manifest(output_root: Path) -> dict[str, object]:
    return json.loads(
        (output_root / "diagnostic_manifest.json").read_text(encoding="utf-8")
    )


def _assert_manifest_consistent(output_root: Path) -> dict[str, object]:
    manifest = _manifest(output_root)
    listed = manifest["files"]
    actual = {
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "diagnostic_manifest.json"
    }
    assert set(listed) == actual
    for relative_path, expected_hash in listed.items():
        path = output_root / relative_path
        assert path.is_file()
        assert sha256(path.read_bytes()).hexdigest() == expected_hash
    assert not any(
        "__pycache__" in name or name.endswith(".pyc") for name in listed
    )
    return manifest


def _trajectory_with_spikes(spike_indices: list[int]) -> HRTrajectory:
    size = max(spike_indices) + 40
    x = np.full(size, -1.0)
    x[spike_indices] = 1.0
    zeros = np.zeros(size)
    return HRTrajectory(np.arange(size) * DT, x, zeros, zeros, zeros)


def _nonpositive_converged_lyapunov() -> LyapunovAnalysis:
    steps = np.arange(100_000, 500_001, 100_000)
    return LyapunovAnalysis(
        0.0,
        "near zero",
        "converged",
        steps,
        np.zeros(len(steps)),
        5.0e-4,
    )


def test_regular_spiking_is_tonic_not_bursting() -> None:
    result = analyze_spikes_and_bursts(
        _trajectory_with_spikes(list(range(40, 481, 40)))
    )

    assert result.burst_structure == "tonic"
    assert result.burst_count == 0
    assert preliminary_regime(result, _nonpositive_converged_lyapunov()) == (
        "periodic spiking"
    )


def test_periodic_bursting_is_detected_despite_high_overall_isi_cv() -> None:
    spikes = [
        40, 70, 100,
        300, 330, 360,
        560, 590, 620,
        820, 850, 880,
    ]
    result = analyze_spikes_and_bursts(_trajectory_with_spikes(spikes))

    assert result.isi_cv > 0.15
    assert result.burst_structure == "bursting"
    assert result.burst_count == 4
    assert result.mean_spikes_per_burst == 3.0
    assert preliminary_regime(result, _nonpositive_converged_lyapunov()) == (
        "periodic bursting"
    )


def test_irregular_bursting_is_not_periodic_bursting() -> None:
    spikes = [
        40, 70,
        300, 330, 360, 390,
        700, 730, 760,
        950, 980,
    ]
    result = analyze_spikes_and_bursts(_trajectory_with_spikes(spikes))

    assert result.burst_structure == "bursting"
    assert result.std_spikes_per_burst > 0.0
    assert result.interburst_interval_cv > 0.15
    assert preliminary_regime(result, _nonpositive_converged_lyapunov()) != (
        "periodic bursting"
    )


def test_unclear_isi_separation_remains_uncertain() -> None:
    intervals = np.array([30, 36, 43, 52, 62, 74, 89])
    spikes = np.concatenate(([40], 40 + np.cumsum(intervals))).tolist()
    result = analyze_spikes_and_bursts(_trajectory_with_spikes(spikes))

    assert result.burst_structure == "uncertain"
    assert result.burst_count is None
    assert np.isnan(result.mean_spikes_per_burst)


def test_stored_trajectory_hashes_are_locked() -> None:
    data_dir = OUTPUT_ROOT / "data"
    actual = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(data_dir.glob("*.npz"))
    }
    assert actual == EXPECTED_TRAJECTORY_HASHES


def test_fixed_output_shape_current_and_finite_values() -> None:
    trajectory = simulate_fixed_current(
        3.29,
        retained_samples=137,
        transient_steps=23,
    )

    assert trajectory.as_matrix().shape == (137, 5)
    assert trajectory.state.shape == (137, 3)
    assert np.all(np.isfinite(trajectory.as_matrix()))
    np.testing.assert_array_equal(trajectory.I, np.full(137, 3.29))


def test_configured_transient_is_one_hundred_thousand_steps() -> None:
    assert INITIAL_TRANSIENT_STEPS == 100_000


def test_continuous_exact_segment_lengths_and_switch_indices() -> None:
    currents = (1.67, 3.29, 3.50, 3.34, 3.20)
    samples_per_segment = 19
    trajectory, switch_indices = simulate_continuous_currents(
        currents,
        samples_per_segment=samples_per_segment,
        transient_steps=11,
    )

    assert trajectory.as_matrix().shape == (95, 5)
    assert np.all(np.isfinite(trajectory.as_matrix()))
    np.testing.assert_array_equal(switch_indices, np.array([19, 38, 57, 76]))
    detected_switches = np.flatnonzero(np.diff(trajectory.I) != 0.0) + 1
    np.testing.assert_array_equal(detected_switches, switch_indices)
    for segment, current in enumerate(currents):
        start = segment * samples_per_segment
        end = start + samples_per_segment
        np.testing.assert_array_equal(
            trajectory.I[start:end],
            np.full(samples_per_segment, current),
        )


def test_switch_preserves_state_instead_of_resetting() -> None:
    trajectory, switches = simulate_continuous_currents(
        (1.67, 3.29),
        samples_per_segment=7,
        transient_steps=5,
    )
    switch = int(switches[0])

    assert not np.array_equal(trajectory.state[switch], np.asarray(INITIAL_STATE))

    first_segment = simulate_fixed_current(
        1.67,
        retained_samples=8,
        transient_steps=5,
    )
    np.testing.assert_array_equal(
        trajectory.state[switch],
        first_segment.state[-1],
    )


def test_deterministic_npz_round_trip_and_bytes(tmp_path) -> None:
    trajectory = simulate_fixed_current(
        3.20,
        retained_samples=31,
        transient_steps=9,
    )
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_trajectory_npz(first, trajectory)
    save_trajectory_npz(second, trajectory)

    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as saved:
        assert saved.files == ["t", "x", "y", "z", "I"]
        np.testing.assert_array_equal(saved["t"], trajectory.t)
        np.testing.assert_array_equal(saved["x"], trajectory.x)
        np.testing.assert_array_equal(saved["y"], trajectory.y)
        np.testing.assert_array_equal(saved["z"], trajectory.z)
        np.testing.assert_array_equal(saved["I"], trajectory.I)


def test_numerical_agreement_with_chapter1_simulator() -> None:
    from data_loader import _rk4_hr

    current = 3.25
    transient_steps = 17
    retained_samples = 101
    params_ch1 = {
        "a": HR_PARAMETERS.a,
        "b": HR_PARAMETERS.b,
        "c": HR_PARAMETERS.c,
        "d": HR_PARAMETERS.d,
        "r": HR_PARAMETERS.r,
        "s": HR_PARAMETERS.s,
        "xr": HR_PARAMETERS.x_r,
        "I": current,
    }
    chapter1 = _rk4_hr(
        INITIAL_STATE,
        transient_steps + retained_samples,
        0.01,
        params_ch1,
    )[transient_steps:]
    chapter2 = simulate_fixed_current(
        current,
        retained_samples=retained_samples,
        transient_steps=transient_steps,
        initial_state=INITIAL_STATE,
        dt=0.01,
        parameters=HRParameters(),
    ).state

    np.testing.assert_array_equal(chapter2, chapter1)


def test_generated_fixed_outputs_and_statistics() -> None:
    output_root = Path(__file__).resolve().parents[1] / "outputs"
    statistics_path = output_root / "fixed_current_statistics.csv"
    with statistics_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
        statistics = {float(row["current"]): row for row in rows}

    assert len(rows) == 5
    assert set(statistics) == set(FIXED_CURRENTS)
    assert "half_window_consistency" in fields
    assert "settled_state_check" not in fields
    assert {row["half_window_consistency"] for row in rows} <= {
        "consistent",
        "inconsistent",
    }
    for current in FIXED_CURRENTS:
        token = f"{current:.2f}".replace(".", "p")
        with np.load(
            output_root / "data" / f"fixed_I_{token}.npz",
            allow_pickle=False,
        ) as saved:
            assert saved.files == ["t", "x", "y", "z", "I"]
            matrix = np.column_stack(tuple(saved[name] for name in saved.files))
            assert matrix.shape == (RETAINED_SAMPLES_PER_CURRENT, 5)
            assert np.all(np.isfinite(matrix))
            np.testing.assert_array_equal(
                saved["I"],
                np.full(RETAINED_SAMPLES_PER_CURRENT, current),
            )
            row = statistics[current]
            assert int(row["n_samples"]) == RETAINED_SAMPLES_PER_CURRENT
            for state_name in ("x", "y", "z"):
                values = saved[state_name]
                assert float(row[f"{state_name}_min"]) == float(np.min(values))
                assert float(row[f"{state_name}_max"]) == float(np.max(values))
                assert float(row[f"{state_name}_mean"]) == float(np.mean(values))
                assert float(row[f"{state_name}_std"]) == float(np.std(values))


def test_generated_continuous_output_segments_and_switches() -> None:
    output_root = Path(__file__).resolve().parents[1] / "outputs"
    with np.load(
        output_root / "data" / "continuous_switched_currents.npz",
        allow_pickle=False,
    ) as saved:
        assert saved.files == ["t", "x", "y", "z", "I"]
        total = len(CONTINUOUS_CURRENT_SEQUENCE) * RETAINED_SAMPLES_PER_CURRENT
        matrix = np.column_stack(tuple(saved[name] for name in saved.files))
        assert matrix.shape == (total, 5)
        assert np.all(np.isfinite(matrix))
        detected_switches = np.flatnonzero(np.diff(saved["I"]) != 0.0) + 1
        np.testing.assert_array_equal(
            detected_switches,
            np.array([100_000, 200_000, 300_000, 400_000]),
        )
        for segment, current in enumerate(CONTINUOUS_CURRENT_SEQUENCE):
            start = segment * RETAINED_SAMPLES_PER_CURRENT
            end = start + RETAINED_SAMPLES_PER_CURRENT
            np.testing.assert_array_equal(
                saved["I"][start:end],
                np.full(RETAINED_SAMPLES_PER_CURRENT, current),
            )


def test_dynamics_summary_has_required_rows_columns_and_finite_lyapunov() -> None:
    output_root = Path(__file__).resolve().parents[1] / "outputs"
    required_columns = {
        "current_I",
        "retained_samples",
        "transient_steps",
        "spike_count",
        "mean_isi",
        "isi_std",
        "isi_cv",
        "burst_structure",
        "burst_count",
        "mean_spikes_per_burst",
        "std_spikes_per_burst",
        "mean_within_burst_isi",
        "within_burst_isi_cv",
        "mean_interburst_interval",
        "interburst_interval_cv",
        "largest_lyapunov_exponent",
        "lyapunov_convergence",
        "lyapunov_classification",
        "half_window_consistency",
        "preliminary_regime",
        "notes",
    }
    with (output_root / "dynamics_summary.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        assert set(reader.fieldnames or ()) == required_columns

    assert len(rows) == 5
    assert [float(row["current_I"]) for row in rows] == list(FIXED_CURRENTS)
    for row in rows:
        assert int(row["retained_samples"]) == 100_000
        assert int(row["transient_steps"]) == 100_000
        assert np.isfinite(float(row["largest_lyapunov_exponent"]))
        assert row["lyapunov_convergence"] in {"converged", "not_converged"}
        assert row["lyapunov_classification"] in {
            "positive",
            "weak positive",
            "near zero",
            "negative",
            "uncertain",
        }



def test_lyapunov_convergence_has_five_finite_checkpoints_per_current() -> None:
    with (OUTPUT_ROOT / "lyapunov_convergence.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 5 * len(FIXED_CURRENTS)
    for current in FIXED_CURRENTS:
        current_rows = [row for row in rows if float(row["current_I"]) == current]
        assert [int(row["evaluation_steps"]) for row in current_rows] == [
            100_000,
            200_000,
            300_000,
            400_000,
            500_000,
        ]
        assert all(np.isfinite(float(row["running_lle"])) for row in current_rows)


def test_figures_and_manifest_are_complete_and_current() -> None:
    expected_figures = {
        "fixed_currents_x_comparison.png",
        "continuous_I_and_x.png",
        "continuous_switches_comparison.png",
        "transient_settling_check.png",
    }
    actual_figures = {
        path.name for path in (OUTPUT_ROOT / "figures").glob("*.png")
    }
    assert actual_figures == expected_figures

    manifest = _assert_manifest_consistent(OUTPUT_ROOT)
    assert manifest["initial_transient_steps"] == 100_000
    assert manifest["transient_steps"] == 100_000
    assert manifest["retained_samples_per_current"] == 100_000
    assert manifest["continuous_samples"] == 500_000
    assert manifest["continuous_switch_indices"] == [
        100_000,
        200_000,
        300_000,
        400_000,
    ]
    assert manifest["switch_indices"] == [
        100_000,
        200_000,
        300_000,
        400_000,
    ]
    assert manifest["lyapunov_evaluation_steps"] == 500_000
    assert manifest["lyapunov_renormalization_interval"] == 10
    assert manifest["lyapunov_checkpoints"] == [
        100_000,
        200_000,
        300_000,
        400_000,
        500_000,
    ]
    assert manifest["lyapunov"]["estimation_steps"] == 500_000
    assert manifest["lyapunov"]["renormalization_steps"] == 10
    assert manifest["lyapunov"]["checkpoint_steps"] == [
        100_000,
        200_000,
        300_000,
        400_000,
        500_000,
    ]
    listed_files = set(manifest["files"])
    assert "lyapunov_convergence.csv" in listed_files
    assert "fixed_current_statistics.csv" in listed_files
    assert {
        path.removeprefix("figures/")
        for path in listed_files
        if path.startswith("figures/")
    } == expected_figures


def test_analysis_only_writers_finish_with_a_current_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "outputs"
    shutil.copytree(OUTPUT_ROOT, output_root)
    before_npz = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in (output_root / "data").glob("*.npz")
    }
    before_figures = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in (output_root / "figures").glob("*.png")
    }

    with (output_root / "dynamics_summary.csv").open(newline="") as file:
        summary = {
            float(row["current_I"]): row for row in csv.DictReader(file)
        }
    with (output_root / "lyapunov_convergence.csv").open(newline="") as file:
        convergence_rows = list(csv.DictReader(file))
    lyapunov = {}
    for current in FIXED_CURRENTS:
        current_rows = [
            row for row in convergence_rows if float(row["current_I"]) == current
        ]
        checkpoint_lle = np.asarray(
            [float(row["running_lle"]) for row in current_rows]
        )
        row = summary[current]
        lyapunov[current] = LyapunovAnalysis(
            float(row["largest_lyapunov_exponent"]),
            row["lyapunov_classification"],
            row["lyapunov_convergence"],
            np.asarray([int(item["evaluation_steps"]) for item in current_rows]),
            checkpoint_lle,
            5.0e-4,
        )

    monkeypatch.setattr(
        diagnostics,
        "estimate_lyapunov",
        lambda current: lyapunov[current],
    )
    generated = diagnostics.regenerate_analysis_outputs(output_root)

    assert [path.name for path in generated] == [
        "fixed_current_statistics.csv",
        "dynamics_summary.csv",
        "dynamics_summary.md",
        "lyapunov_convergence.csv",
        "diagnostic_manifest.json",
    ]
    _assert_manifest_consistent(output_root)
    assert before_npz == {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in (output_root / "data").glob("*.npz")
    }
    assert before_figures == {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in (output_root / "figures").glob("*.png")
    }


def test_python_cache_rules_and_no_cache_artifacts() -> None:
    chapter_root = Path(__file__).resolve().parents[1]
    assert (chapter_root / ".gitignore").read_text(encoding="utf-8") == (
        "__pycache__/\n*.py[cod]\n"
    )
    assert not list(chapter_root.rglob("__pycache__"))
    assert not list(chapter_root.rglob("*.pyc"))
