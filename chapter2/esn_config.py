"""Frozen Chapter 2 protocol constants and ESN construction settings.

No optimisation search spaces, selected thesis hyperparameters, training runs,
or evaluation settings belong here. Ranges are half-open ranges of *transition
indices*: a transition at index ``k`` maps state sample ``k`` to state sample
``k + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CHAPTER2_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CHAPTER2_ROOT.parent
DATA_ROOT = CHAPTER2_ROOT / "outputs" / "data"

TRAIN_CURRENTS = (1.67, 3.20, 3.50)
UNSEEN_CURRENTS = (3.29, 3.34)
CONTINUOUS_SEQUENCE = (1.67, 3.29, 3.50, 3.34, 3.20)
FINAL_SEEDS = (42, 123, 456, 789, 2026)

STATE_DIMENSION = 3
PARAMETER_AWARE_INPUT_DIMENSION = 4
OUTPUT_DIMENSION = 3
REQUIRED_ARRAY_KEYS = ("t", "x", "y", "z", "I")

FITTING_STOP = 40_000
# Step 8 fixed-current benchmark windows. The final window starts at the
# latest possible transition for a 10,000-transition window in a trajectory
# with 100,000 states. Transition 89,999 is scored in window 2 and appears only
# in the unscored warm-up of window 3; scored intervals remain disjoint.
STEP8_FINAL_TRAINING_STOP = 70_000
STEP8_TRAINING_WASHOUT = 2_000
STEP8_WINDOW_STARTS = (70_000, 80_000, 89_999)
STEP8_WARMUP_TRANSITIONS = 2_000
STEP8_FORECAST_TRANSITIONS = 8_000
STEP8_LONG_HORIZON_START = 70_000
STEP8_CONTINUOUS_WARMUP_TRANSITIONS = 2_000

VALIDATION_STOP = 70_000
WARMUP_TRANSITIONS = 2_000
SCORED_ROLLOUT_TRANSITIONS = 8_000
FIXED_STATE_COUNT = 100_000
CONTINUOUS_STATE_COUNT = 500_000
CONTINUOUS_SWITCH_INDICES = (100_000, 200_000, 300_000, 400_000)


@dataclass(frozen=True)
class ESNModelConfig:
    """Validated model-construction settings for Chapter 2 mechanics.

    Defaults are conservative pilot values for unit tests and smoke checks.
    They are not selected thesis hyperparameters. `reservoir_connectivity` is
    the independent Bernoulli probability for each off-diagonal recurrent edge;
    recurrent self-connections are fixed at zero.
    """

    reservoir_size: int = 100
    spectral_radius: float = 0.9
    leak_rate: float = 0.5
    input_scaling: float = 0.5
    bias_scaling: float = 0.1
    reservoir_connectivity: float = 0.1
    ridge_regularisation: float = 1.0e-6
    seed: int = 42
    input_dimension: int = PARAMETER_AWARE_INPUT_DIMENSION
    output_dimension: int = OUTPUT_DIMENSION
    regularise_bias: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.reservoir_size, bool) or not isinstance(
            self.reservoir_size, int
        ):
            raise TypeError("reservoir_size must be an integer")
        if self.reservoir_size < 2:
            raise ValueError("reservoir_size must be at least 2")
        if self.input_dimension not in (
            STATE_DIMENSION,
            PARAMETER_AWARE_INPUT_DIMENSION,
        ):
            raise ValueError("input_dimension must be 3 (baseline) or 4 (parameter-aware)")
        if self.output_dimension != OUTPUT_DIMENSION:
            raise ValueError("output_dimension must be 3")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(self.regularise_bias, bool):
            raise TypeError("regularise_bias must be a boolean")

        finite_fields = {
            "spectral_radius": self.spectral_radius,
            "leak_rate": self.leak_rate,
            "input_scaling": self.input_scaling,
            "bias_scaling": self.bias_scaling,
            "reservoir_connectivity": self.reservoir_connectivity,
            "ridge_regularisation": self.ridge_regularisation,
        }
        for field_name, value in finite_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a real number")
            if not float("-inf") < float(value) < float("inf"):
                raise ValueError(f"{field_name} must be finite")

        if self.spectral_radius <= 0.0:
            raise ValueError("spectral_radius must be positive")
        if not 0.0 < self.leak_rate <= 1.0:
            raise ValueError("leak_rate must be in (0, 1]")
        if self.input_scaling < 0.0:
            raise ValueError("input_scaling must be non-negative")
        if self.bias_scaling < 0.0:
            raise ValueError("bias_scaling must be non-negative")
        if not 0.0 < self.reservoir_connectivity <= 1.0:
            raise ValueError("reservoir_connectivity must be in (0, 1]")
        if self.ridge_regularisation <= 0.0:
            raise ValueError("ridge_regularisation must be positive")


@dataclass(frozen=True, order=True)
class TransitionRange:
    """A non-empty half-open range of one-step transition indices."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or isinstance(self.stop, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.stop, int)
        ):
            raise TypeError("transition boundaries must be integers")
        if self.start < 0:
            raise ValueError("transition range start must be non-negative")
        if self.stop <= self.start:
            raise ValueError("transition range stop must be greater than start")

    def __len__(self) -> int:
        return self.stop - self.start

    def overlaps(self, other: "TransitionRange") -> bool:
        """Return whether this range and ``other`` share a transition."""
        return max(self.start, other.start) < min(self.stop, other.stop)


@dataclass(frozen=True)
class ValidationWindow:
    """One validation range split into true-state warm-up and scoring."""

    number: int
    transitions: TransitionRange
    warmup: TransitionRange
    scored: TransitionRange

    def __post_init__(self) -> None:
        if self.number <= 0:
            raise ValueError("validation window number must be positive")
        if self.warmup.start != self.transitions.start:
            raise ValueError("warm-up must start at the validation window start")
        if self.warmup.stop != self.scored.start:
            raise ValueError("warm-up and scored ranges must be adjacent")
        if self.scored.stop != self.transitions.stop:
            raise ValueError("scored range must end at the validation window stop")


@dataclass(frozen=True)
class TransitionSplit:
    """Chronological fitting, validation, and held-out transition ranges."""

    fitting: TransitionRange
    validation: TransitionRange
    held_out: TransitionRange

    def __post_init__(self) -> None:
        validate_non_overlapping(
            (self.fitting, self.validation, self.held_out),
            label="chronological split",
        )
        if self.fitting.stop != self.validation.start:
            raise ValueError("fitting and validation ranges must be adjacent")
        if self.validation.stop != self.held_out.start:
            raise ValueError("validation and held-out ranges must be adjacent")


@dataclass(frozen=True)
class LockedDataset:
    """Identity and expected structure of one immutable trajectory file."""

    path: Path
    sha256: str
    state_count: int
    current: float | None = None
    current_sequence: tuple[float, ...] = ()
    switch_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (self.current is None) == (not self.current_sequence):
            raise ValueError(
                "dataset must define either one fixed current or a sequence"
            )
        if self.state_count < 2:
            raise ValueError("dataset must contain at least two states")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("dataset SHA-256 must be 64 lowercase hex characters")


FITTING_TRANSITIONS = TransitionRange(0, FITTING_STOP)
VALIDATION_TRANSITIONS = TransitionRange(FITTING_STOP, VALIDATION_STOP)

VALIDATION_WINDOWS = (
    ValidationWindow(
        1,
        TransitionRange(40_000, 50_000),
        TransitionRange(40_000, 42_000),
        TransitionRange(42_000, 50_000),
    ),
    ValidationWindow(
        2,
        TransitionRange(50_000, 60_000),
        TransitionRange(50_000, 52_000),
        TransitionRange(52_000, 60_000),
    ),
    ValidationWindow(
        3,
        TransitionRange(60_000, 70_000),
        TransitionRange(60_000, 62_000),
        TransitionRange(62_000, 70_000),
    ),
)

FIXED_DATASETS = (
    LockedDataset(
        DATA_ROOT / "fixed_I_1p67.npz",
        "cc89af5e9a27d05a9501ea995a2ef361c146d6513ec1b19765238603af9ffb2b",
        FIXED_STATE_COUNT,
        current=1.67,
    ),
    LockedDataset(
        DATA_ROOT / "fixed_I_3p20.npz",
        "c4292a1e0fa5575d08419e7d302f980e2ed085878187f67659aa59decc486ded",
        FIXED_STATE_COUNT,
        current=3.20,
    ),
    LockedDataset(
        DATA_ROOT / "fixed_I_3p29.npz",
        "2ed394f457e5de5f0d14a5dd450611fd9774c02cb13fc68f7b21a2621273b7b4",
        FIXED_STATE_COUNT,
        current=3.29,
    ),
    LockedDataset(
        DATA_ROOT / "fixed_I_3p34.npz",
        "8cad49948c553df9b65a24deacf067dc999325dadb580a4787ded12b6315585d",
        FIXED_STATE_COUNT,
        current=3.34,
    ),
    LockedDataset(
        DATA_ROOT / "fixed_I_3p50.npz",
        "4d042da6adbde026468207c4e9f74443f8587c635dff2d28b69e2b1784d5cbbf",
        FIXED_STATE_COUNT,
        current=3.50,
    ),
)

CONTINUOUS_DATASET = LockedDataset(
    DATA_ROOT / "continuous_switched_currents.npz",
    "0921cbad321da1830433dc84e58f25ff2b0b6a6d571a31cae769bdfd6dc00a7b",
    CONTINUOUS_STATE_COUNT,
    current_sequence=CONTINUOUS_SEQUENCE,
    switch_indices=CONTINUOUS_SWITCH_INDICES,
)
LOCKED_DATASETS = FIXED_DATASETS + (CONTINUOUS_DATASET,)


def validate_non_overlapping(
    ranges: Iterable[TransitionRange],
    *,
    label: str = "transition ranges",
) -> None:
    """Raise when any two half-open ranges share a transition index."""
    ordered = tuple(sorted(ranges))
    for previous, current in zip(ordered, ordered[1:]):
        if previous.overlaps(current):
            raise ValueError(
                f"{label} overlap: [{previous.start}, {previous.stop}) and "
                f"[{current.start}, {current.stop})"
            )


def transition_split(state_count: int) -> TransitionSplit:
    """Return the frozen split for a trajectory containing ``state_count`` states."""
    if isinstance(state_count, bool) or not isinstance(state_count, int):
        raise TypeError("state_count must be an integer")
    final_transition_stop = state_count - 1
    if final_transition_stop <= VALIDATION_STOP:
        raise ValueError(
            f"at least {VALIDATION_STOP + 2} states are required to create a "
            "non-empty held-out split"
        )
    return TransitionSplit(
        FITTING_TRANSITIONS,
        VALIDATION_TRANSITIONS,
        TransitionRange(VALIDATION_STOP, final_transition_stop),
    )


def fixed_dataset(current: float) -> LockedDataset:
    """Return the locked fixed-current dataset record for ``current``."""
    matches = tuple(item for item in FIXED_DATASETS if item.current == current)
    if len(matches) != 1:
        raise ValueError(f"no locked fixed-current dataset for I={current}")
    return matches[0]


def validate_protocol() -> None:
    """Validate the internally frozen allocation, splits, windows, and inventory."""
    if set(TRAIN_CURRENTS) & set(UNSEEN_CURRENTS):
        raise ValueError("training and unseen currents must be disjoint")
    if set(TRAIN_CURRENTS + UNSEEN_CURRENTS) != {
        item.current for item in FIXED_DATASETS
    }:
        raise ValueError("fixed-current allocation does not match locked datasets")
    if len(set(FINAL_SEEDS)) != len(FINAL_SEEDS):
        raise ValueError("final seeds must be unique")
    if (
        STATE_DIMENSION != OUTPUT_DIMENSION
        or PARAMETER_AWARE_INPUT_DIMENSION != STATE_DIMENSION + 1
    ):
        raise ValueError("state, input, and output dimensions are inconsistent")

    split = transition_split(FIXED_STATE_COUNT)
    if (len(split.fitting), len(split.validation), len(split.held_out)) != (
        40_000,
        30_000,
        29_999,
    ):
        raise ValueError("frozen transition counts are inconsistent")

    validate_non_overlapping(
        (window.transitions for window in VALIDATION_WINDOWS),
        label="validation windows",
    )
    if (
        VALIDATION_WINDOWS[0].transitions.start != VALIDATION_TRANSITIONS.start
        or VALIDATION_WINDOWS[-1].transitions.stop != VALIDATION_TRANSITIONS.stop
    ):
        raise ValueError("validation windows must cover the validation split")
    for window in VALIDATION_WINDOWS:
        if len(window.warmup) != WARMUP_TRANSITIONS:
            raise ValueError("validation warm-up length is not frozen at 2,000")
        if len(window.scored) != SCORED_ROLLOUT_TRANSITIONS:
            raise ValueError("validation scored length is not frozen at 8,000")

    paths = tuple(item.path for item in LOCKED_DATASETS)
    if len(paths) != len(set(paths)):
        raise ValueError("locked dataset paths must be unique")


validate_protocol()
