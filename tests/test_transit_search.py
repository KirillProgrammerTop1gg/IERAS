"""Regression tests for iterative transit candidate selection."""

from types import SimpleNamespace

import numpy as np

from ieras_transit_search import (
    Config,
    best_covered_peak,
    classify_alias,
    clearly_invalid_coarse_candidate,
    periodogram_sde,
    pick_peaks,
    refine,
    vet,
)


def test_five_to_one_residual_is_rejected():
    cfg = Config()
    first = {"period": 13.72232764, "duration": 0.192, "t0": 137.84216564}
    period = 68.61231795
    epochs = 137.83316564 + np.arange(20) * period

    alias_of, reason, _ = classify_alias(period, 0.282, epochs, [first], cfg)

    assert alias_of == 1
    assert "5:1" in reason


def test_repeated_period_is_rejected_even_with_shifted_epoch():
    first = {"period": 68.6152, "duration": 0.28, "t0": 137.8}

    alias_of, reason, _ = classify_alias(
        68.6110, 0.28, np.array([160.0, 228.611]), [first], Config()
    )

    assert alias_of == 1
    assert "duplicate period" in reason


def test_one_to_three_subharmonic_is_rejected():
    first = {"period": 13.72232764, "duration": 0.192, "t0": 137.84216564}
    period = first["period"] / 3
    epochs = first["t0"] + np.arange(90) * period

    alias_of, reason, _ = classify_alias(period, 0.19, epochs, [first], Config())

    assert alias_of == 1
    assert "1:3" in reason


def test_one_to_five_subharmonic_is_rejected():
    first = {"period": 18.164, "duration": 0.132, "t0": 144.485}
    period = first["period"] / 5
    epochs = first["t0"] + np.arange(300) * period

    alias_of, reason, _ = classify_alias(period, 0.09, epochs, [first], Config())

    assert alias_of == 1
    assert "1:5" in reason


def test_two_to_five_harmonic_is_rejected():
    first = {"period": 13.7224, "duration": 0.192, "t0": 137.842}
    period = first["period"] * 2 / 5
    epochs = first["t0"] + np.arange(200) * period

    alias_of, reason, _ = classify_alias(period, 0.19, epochs, [first], Config())

    assert alias_of == 1
    assert "2:5" in reason


def test_phase_independent_harmonic_can_be_considered():
    cfg = Config()
    first = {"period": 13.7, "duration": 0.2, "t0": 2.0}
    epochs = 5.0 + np.arange(10) * 68.5

    alias_of, reason, warnings = classify_alias(68.5, 0.3, epochs, [first], cfg)

    assert alias_of is None
    assert reason is None
    assert "independent" in warnings[0]


def test_high_order_rational_match_does_not_hide_independent_period():
    first = {"period": 18.164, "duration": 0.13, "t0": 144.48}
    period = first["period"] * 21 / 31
    epochs = 138.2 + np.arange(110) * period

    alias_of, reason, _ = classify_alias(period, 0.28, epochs, [first], Config())

    assert alias_of is None
    assert reason is None


def test_kepler_62_twelve_day_period_is_not_a_simple_alias():
    accepted = [
        {"period": 18.164, "duration": 0.132, "t0": 144.485},
        {"period": 5.715, "duration": 0.09, "t0": 136.634},
    ]
    epochs = 134.65 + np.arange(108) * 12.4417

    alias_of, _, _ = classify_alias(12.4417, 0.132, epochs, accepted, Config())

    assert alias_of is None


def test_vet_rejects_realistic_five_to_one_alias():
    cfg = Config()
    t = np.arange(0, 400, 0.02)
    first = {"period": 13.7, "duration": 0.2, "t0": 2.0}
    candidate = {"period": 68.5, "duration": 0.3, "t0": 2.0, "depth": 0.001, "snr": 20.0}
    f = np.ones_like(t)
    f[np.abs(((t - 2.0 + 34.25) % 68.5) - 34.25) < 0.15] -= 0.001
    dy = np.full_like(t, 0.0001)

    result = vet(t, f, dy, candidate, [first], cfg)

    assert result["alias_of"] == 1
    assert result["n_transits"] >= 3


def test_strong_signal_tolerates_variable_transit_depths():
    cfg = Config()
    t = np.arange(0, 150, 0.02)
    period = 10.0
    t0 = 2.0
    candidate = {"period": period, "duration": 0.2, "t0": t0, "depth": 0.0005, "snr": 30.0}
    f = np.ones_like(t)
    for epoch in range(15):
        center = t0 + epoch * period
        f[np.abs(t - center) < 0.1] -= 0.0005 + 0.00010 * np.sin(epoch)
    dy = np.full_like(t, 0.0001)

    result = vet(t, f, dy, candidate, [], cfg)

    assert result["n_transits"] >= cfg.strong_signal_transits
    assert result["depth_chi2"] > cfg.max_depth_chi2
    assert result["depth_chi2"] < cfg.max_depth_chi2_strong
    assert result["reasons"] == []

    weak = vet(t, f, dy, {**candidate, "snr": 8.0}, [], cfg)
    assert any("inconsistent" in reason for reason in weak["reasons"])


def test_repeated_signal_allows_modest_depth_scatter():
    t = np.arange(0, 150, 0.02)
    candidate = {"period": 10.0, "duration": 0.2, "t0": 2.0, "depth": 0.0005, "snr": 8.0}
    f = np.ones_like(t)
    for epoch in range(15):
        f[np.abs(t - (2.0 + epoch * 10.0)) < 0.1] -= 0.0005 + 0.00009 * np.sin(epoch)
    dy = np.full_like(t, 0.0001)

    result = vet(t, f, dy, candidate, [], Config())

    assert Config.max_depth_chi2 < result["depth_chi2"] < Config.max_depth_chi2_repeated
    assert result["reasons"] == []


def test_marginal_signal_needs_support_in_both_halves():
    t = np.arange(0, 150, 0.02)
    candidate = {"period": 10.0, "duration": 0.2, "t0": 2.0, "depth": 0.0003, "snr": 8.0}
    f = np.ones_like(t)
    for epoch in range(15):
        depth = 0.0005 if epoch < 8 else 0.00005
        f[np.abs(t - (2.0 + epoch * 10.0)) < 0.1] -= depth
    dy = np.full_like(t, 0.0005)

    result = vet(t, f, dy, candidate, [], Config())

    assert any("half transit SNR" in reason for reason in result["reasons"])


def test_sparse_three_point_events_are_rejected():
    t = np.arange(0, 400, 0.02)
    candidate = {"period": 90.0, "duration": 0.06, "t0": 2.0, "depth": 0.001, "snr": 15.0}
    f = np.ones_like(t)
    for epoch in range(4):
        f[np.abs(t - (2.0 + epoch * 90.0)) < 0.03] -= 0.001
    dy = np.full_like(t, 0.0001)

    result = vet(t, f, dy, candidate, [], Config())

    assert result["n_transits"] == 0
    assert "transits with data" in result["reasons"][0]


def test_strong_repeated_three_point_transits_are_eligible():
    t = np.arange(0, 200, 0.02)
    candidate = {"period": 3.88, "duration": 0.06, "t0": 2.0, "depth": 0.001, "snr": 30.0}
    f = np.ones_like(t)
    for epoch in range(50):
        f[np.abs(t - (2.0 + epoch * 3.88)) < 0.03] -= 0.001
    dy = np.full_like(t, 0.0001)

    result = vet(t, f, dy, candidate, [], Config())

    assert result["n_transits"] >= Config.min_sparse_transits
    assert result["reasons"] == []


def test_local_sde_keeps_a_secondary_peak_visible():
    power = 0.01 * np.arange(10000, dtype=float)
    power[300:320] += 1000
    power[800:810] += 120

    sde = periodogram_sde(power)

    assert sde[805] > 8


def test_peak_selection_skips_alias_and_reaches_weaker_signal():
    cfg = Config()
    first = {"period": 13.7, "duration": 0.2, "t0": 2.0}
    t = np.arange(0, 800, 0.02)
    sc = {
        "period": np.array([60.0, 68.5, 100.0, 242.0, 280.0]),
        "power": np.array([0.0, 15.0, 0.0, 8.0, 0.0]),
        "t0": np.array([2.0, 2.0, 2.0, 7.0, 7.0]),
        "duration": np.full(5, 0.3),
    }

    assert pick_peaks(sc, 1, cfg.peak_separation, t, [first], cfg) == [3]


def test_peak_selection_keeps_close_long_period_signals():
    sc = {
        "period": np.array([126.0, 127.9, 129.0, 129.94, 132.0]),
        "power": np.array([0.0, 20.0, 0.0, 12.0, 0.0]),
    }

    assert pick_peaks(sc, 2, Config().peak_separation) == [1, 3]


def test_coarse_screen_only_rejects_clear_failures():
    cfg = Config()
    valid = {"alias_of": None, "n_transits": 10, "max_event_fraction": 0.43, "depth_chi2": 5.7}
    dominant = {**valid, "max_event_fraction": 0.9}
    inconsistent = {**valid, "depth_chi2": 50.0}

    assert not clearly_invalid_coarse_candidate(valid, cfg)
    assert clearly_invalid_coarse_candidate(dominant, cfg)
    assert clearly_invalid_coarse_candidate(inconsistent, cfg)


def test_refinement_skips_single_observed_event():
    t = np.concatenate((np.arange(0, 3, 0.02), np.arange(10, 13, 0.02), np.arange(20, 23, 0.02)))
    result = SimpleNamespace(
        power=np.array([1.0, 20.0, 1.0, 12.0, 1.0]),
        period=np.array([5.0, 8.5, 5.0, 10.0, 5.0]),
        transit_time=np.array([1.0, 2.0, 1.0, 1.0, 1.0]),
        duration=np.full(5, 0.2),
    )

    assert best_covered_peak(t, result, min_transits=3) == 3


def test_refinement_stays_on_weaker_peak_near_stronger_alias():
    t = np.arange(0, 1500, 0.1)
    f = np.ones_like(t)
    for period, depth in ((242.0, 0.0005), (247.0, 0.001)):
        phase = ((t - 5.0 + period / 2) % period) - period / 2
        f[np.abs(phase) < 0.15] -= depth
    dy = np.full_like(t, 0.0001)
    cfg = Config(durations=(0.3,), max_refine_grid=400)

    candidate = refine(t, f, dy, 242.0, cfg)

    assert abs(candidate["period"] - 242.0) < 0.2
