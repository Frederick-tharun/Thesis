"""Design tests: current membership, disjointness, anchors, 3.29 withholding."""
from __future__ import annotations

import json

from chapter2.part2_cross_regime import config, design


def _currents(entries):
    return {e["current"] for e in entries}


def test_frozen_design_exists_and_hash_matches():
    d = json.loads(config.FROZEN_DESIGN_PATH.read_text())
    stored_hash = d.pop("design_hash_sha256")
    recomputed = __import__("hashlib").sha256(
        json.dumps(d, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert stored_hash == recomputed


def test_training_test_disjointness():
    d = json.loads(config.FROZEN_DESIGN_PATH.read_text())
    reg_train = _currents(d["models"]["regular_trained"]["training_currents"])
    cha_train = _currents(d["models"]["chaotic_trained"]["training_currents"])
    rr = _currents(d["models"]["regular_trained"]["rr_test_currents"])
    rc = _currents(d["models"]["regular_trained"]["rc_test_currents"])
    cc = _currents(d["models"]["chaotic_trained"]["cc_test_currents"])
    cr = _currents(d["models"]["chaotic_trained"]["cr_test_currents"])
    assert not (rr & reg_train)
    assert not (rc & reg_train)
    assert not (cc & cha_train)
    assert not (cr & cha_train)


def test_professor_anchors_present():
    d = json.loads(config.FROZEN_DESIGN_PATH.read_text())
    reg_train = _currents(d["models"]["regular_trained"]["training_currents"])
    cha_train = _currents(d["models"]["chaotic_trained"]["training_currents"])
    assert 1.67 in reg_train and 3.50 in reg_train
    assert 3.20 in cha_train and 3.34 in cha_train


def test_i_3_29_withheld_and_used_only_as_cr_test():
    d = json.loads(config.FROZEN_DESIGN_PATH.read_text())
    reg_train = _currents(d["models"]["regular_trained"]["training_currents"])
    cha_train = _currents(d["models"]["chaotic_trained"]["training_currents"])
    cr = _currents(d["models"]["chaotic_trained"]["cr_test_currents"])
    assert 3.29 not in reg_train
    assert 3.29 not in cha_train
    assert 3.29 in cr
    assert d["i_3_29_withheld_from_chaotic_training"] is True


def test_interpolation_extrapolation_labels_present():
    d = json.loads(config.FROZEN_DESIGN_PATH.read_text())
    for entry in d["models"]["chaotic_trained"]["cr_test_currents"]:
        assert "interpolation_status" in entry and entry["interpolation_status"]
