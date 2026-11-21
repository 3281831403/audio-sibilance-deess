"""Regression tests for the sibilance de-esser contract.

Ten focused tests correspond to the ten known defects.  Five integration
tests cover no-sibilance pass-through, a zero threshold, empty input, a
zero ratio, and successful batch ordering.  The initial stubs make all
fifteen tests fail with ``NotImplementedError``.
"""

import math

import pytest

from deess.deesser import (
    DeessError,
    apply_band_gain,
    band_level,
    band_split,
    deess_batch,
    detect_sibilance,
    gain_reduction,
    smooth_gain,
    validate_params,
)

SAMPLE_RATE = 44100.0
TOL = 1e-6


def _tone(frequency, seconds=0.05, amplitude=0.2, sample_rate=SAMPLE_RATE):
    count = int(round(seconds * sample_rate))
    return [
        amplitude * math.sin(2.0 * math.pi * frequency * i / sample_rate)
        for i in range(count)
    ]


def _amplitude_at(samples, frequency, sample_rate=SAMPLE_RATE):
    if not samples:
        return 0.0
    count = len(samples)
    real = sum(
        value * math.cos(2.0 * math.pi * frequency * index / sample_rate)
        for index, value in enumerate(samples)
    )
    imag = sum(
        value * math.sin(2.0 * math.pi * frequency * index / sample_rate)
        for index, value in enumerate(samples)
    )
    return 2.0 * math.hypot(real, imag) / count


def _config(**overrides):
    values = {
        "sample_rate": SAMPLE_RATE,
        "low": 4000.0,
        "high": 9000.0,
        "threshold_db": -25.0,
        "ratio": 4.0,
        "window": (256, 128),
        "attack_ms": 5.0,
        "release_ms": 80.0,
    }
    values.update(overrides)
    return values


def test_band_split_uses_steep_stopband():
    low_tone = _tone(500.0, seconds=0.2, amplitude=0.2)
    high_tone = _tone(7000.0, seconds=0.2, amplitude=0.2)
    mixed = [low + high for low, high in zip(low_tone, high_tone)]

    band, residual = band_split(mixed, 4000.0, 9000.0, SAMPLE_RATE)

    assert len(band) == len(mixed)
    assert len(residual) == len(mixed)
    assert all(
        abs((in_band + rest) - original) <= TOL
        for original, in_band, rest in zip(mixed, band, residual)
    )
    assert _amplitude_at(band, 7000.0) > 0.7 * 0.2
    assert _amplitude_at(band, 500.0) < 0.1 * 0.2


def test_band_level_localizes_sibilance_window():
    quiet = 0.001
    samples = [quiet] * 200 + [0.5] * 200 + [quiet] * 200

    levels = band_level(samples, (100, 50))

    assert len(levels) == 11
    assert levels[4] == pytest.approx(20.0 * math.log10(0.5), abs=0.05)
    loud = [index for index, level in enumerate(levels) if level > -8.0]
    assert loud == [4, 5, 6, 7, 8]


def test_gain_reduction_honours_ratio():
    gains = gain_reduction([0.0, -10.0, -30.0], -20.0, 2.0)

    expected = [
        10.0 ** (-10.0 / 20.0),
        10.0 ** (-5.0 / 20.0),
        1.0,
    ]
    assert gains == pytest.approx(expected, abs=TOL)


def test_smooth_gain_release_is_not_too_slow():
    target = [1.0] * 10 + [0.25] * 90 + [1.0] * 400

    smoothed = smooth_gain(target, attack_ms=10.0, release_ms=80.0, sample_rate=1000.0)

    assert len(smoothed) == len(target)
    assert smoothed[0] == pytest.approx(1.0, abs=TOL)
    assert smoothed[59] < 0.27
    assert smoothed[100] < 0.95
    assert smoothed[-1] > 0.99
    assert all(0.24 <= value <= 1.0 for value in smoothed)
    assert max(abs(a - b) for a, b in zip(smoothed, smoothed[1:])) < 0.2
    assert all(
        left >= right - 1e-12
        for left, right in zip(smoothed[10:99], smoothed[11:100])
    )
    assert all(
        right >= left - 1e-12
        for left, right in zip(smoothed[100:], smoothed[101:])
    )


def test_apply_band_gain_preserves_residual():
    low_tone = _tone(300.0, seconds=0.1, amplitude=0.2)
    high_tone = _tone(7000.0, seconds=0.1, amplitude=0.2)
    mixed = [low + high for low, high in zip(low_tone, high_tone)]

    output = apply_band_gain(mixed, 0.5, high_tone, SAMPLE_RATE)
    expected = [low + 0.5 * high for low, high in zip(low_tone, high_tone)]

    assert output == pytest.approx(expected, abs=TOL)
    assert apply_band_gain(mixed, 1.0, high_tone, SAMPLE_RATE) == pytest.approx(
        mixed, abs=TOL
    )


def test_detect_sibilance_merges_regions():
    band = [0.0] * 800
    for index in range(200, 600):
        band[index] = 0.0 if 400 <= index < 410 else 0.5
    samples = list(band)

    intervals = detect_sibilance(samples, band, -10.0)

    assert len(intervals) == 1
    start, end = intervals[0]
    assert 50 < start < 250
    assert 550 < end < 750


def test_band_split_rejects_upper_band_above_nyquist():
    with pytest.raises(DeessError):
        band_split([0.0] * 32, 5000.0, 23000.0, 44100.0)

    assert validate_params(5000.0, 23000.0, -20.0)


def test_validate_params_rejects_low_not_below_high():
    assert validate_params(4000.0, 9000.0, -20.0) == []
    assert validate_params(8000.0, 8000.0, -20.0)
    assert validate_params(5000.0, 23000.0, -20.0)
    assert validate_params(4000.0, 9000.0, 3.0)


def test_deess_batch_isolates_track_state():
    first = _tone(6000.0, seconds=0.05, amplitude=0.4)
    second = _tone(7000.0, seconds=0.05, amplitude=0.4)

    alone = deess_batch([("second", second)], _config())["second"]
    together = deess_batch(
        [("first", first), ("second", second)], _config()
    )["second"]

    assert together == pytest.approx(alone, abs=TOL)


def test_gain_reduction_rejects_negative_ratio():
    with pytest.raises(DeessError):
        gain_reduction([-6.0], -20.0, -1.0)


def test_no_sibilance_passthrough_within_1e_9():
    samples = _tone(220.0, seconds=0.05, amplitude=0.25)
    result = deess_batch([("voice", samples)], _config(threshold_db=-30.0, ratio=2.0))

    assert list(result) == ["voice"]
    assert result["voice"] == pytest.approx(samples, abs=1e-9)


def test_threshold_zero_compresses_all_windows():
    levels = [0.5, 3.0, 9.0]

    gains = gain_reduction(levels, 0.0, 4.0)

    expected = [10.0 ** (-(level * 0.75) / 20.0) for level in levels]
    assert gains == pytest.approx(expected, abs=TOL)
    assert all(gain < 1.0 for gain in gains)


def test_empty_input_returns_empty_outputs():
    assert band_split([], 4000.0, 9000.0, SAMPLE_RATE) == ([], [])
    assert band_level([], (256, 128)) == []
    assert apply_band_gain([], 0.5, [], SAMPLE_RATE) == []


def test_ratio_zero_raises_deess_error():
    with pytest.raises(DeessError):
        gain_reduction([-3.0], -20.0, 0.0)


def test_deess_batch_preserves_success_order():
    tracks = [
        ("alpha", _tone(220.0, seconds=0.03, amplitude=0.2)),
        ("beta", _tone(330.0, seconds=0.03, amplitude=0.2)),
        ("gamma", _tone(440.0, seconds=0.03, amplitude=0.2)),
    ]

    result = deess_batch(tracks, _config())

    assert list(result) == ["alpha", "beta", "gamma"]
    assert all(len(result[name]) == len(samples) for name, samples in tracks)
