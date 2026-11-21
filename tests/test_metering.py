"""Red tests for the auto-exposure metering task.

There are exactly fifteen tests here.  Every one exercises the public API in
``exposure.metering``; all of them fail while the implementation is still a
set of ``NotImplementedError`` stubs.
"""

import math

import pytest

from exposure import (
    LOG_LUMA_MAX,
    LOG_LUMA_MIN,
    ExposureError,
    adapt,
    apply_ev,
    exposure_batch,
    histogram,
    luminance,
    meter_ev,
    percentile,
    validate,
)


def _gray(value: float, height: int = 8, width: int = 8) -> list[list[float]]:
    """Build an independent uniform grayscale image."""
    return [[float(value) for _ in range(width)] for _ in range(height)]


def _flatten(img: list[list[float]]) -> list[float]:
    return [sample for row in img for sample in row]


def test_luminance_uses_rec709_weights() -> None:
    """Defect 1: equal-weight averaging gives green the wrong weight."""
    image = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]

    result = luminance(image)

    assert result[0] == pytest.approx([0.2126, 0.7152, 0.0722], abs=1e-6)

    grayscale = [[0.0, 0.25], [0.5, 1.0]]
    assert luminance(grayscale) == grayscale


def test_histogram_uses_log_buckets() -> None:
    """Defect 2: linear buckets crush several stops of shadow detail."""
    image = [[0.02], [0.05], [0.10], [0.20], [0.40]]

    counts = histogram(image, 12)

    expected = [0] * 6 + [1] * 5 + [0]
    assert counts == expected


def test_percentile_interpolates_within_log_bin() -> None:
    """Defect 3: returning a bucket index is not a luminance quantile."""
    hist = [0, 0, 4, 0]
    width = (LOG_LUMA_MAX - LOG_LUMA_MIN) / len(hist)
    expected = 2.0 ** (LOG_LUMA_MIN + (2 + 0.25) * width)

    assert percentile(hist, 0.25) == pytest.approx(expected, abs=1e-6)
    assert expected == pytest.approx(2.0**-5.25, abs=1e-12)


def test_metering_modes_use_distinct_regions() -> None:
    """Defect 4: spot metering must not be identical to global metering."""
    image = _gray(0.02, 21, 21)
    for row in range(9, 12):
        for col in range(9, 12):
            image[row][col] = 0.9

    matrix = meter_ev(image, 0.18, "matrix")
    center = meter_ev(image, 0.18, "center")
    spot = meter_ev(image, 0.18, "spot")

    assert spot < center < matrix
    assert spot < -1.0
    assert matrix > 1.0


def test_adapt_clamps_speed() -> None:
    """Defect 5: an unclamped speed overshoots and makes EV jump."""
    assert adapt(1.0, 3.0, 0.5) == pytest.approx(2.0, abs=1e-6)
    assert adapt(0.0, 4.0, 2.0) == pytest.approx(4.0, abs=1e-6)
    assert adapt(0.0, 4.0, -0.5) == pytest.approx(0.0, abs=1e-6)


def test_apply_ev_multiplies_linear_light() -> None:
    """Defect 6: applying EV is a multiply, not an additive offset."""
    result = apply_ev([[0.2, 0.4]], 1.0)

    assert result[0] == pytest.approx([0.4, 0.8], abs=1e-6)


def test_apply_ev_scales_before_clamping() -> None:
    """Defect 7: clamping first throws away recoverable highlights."""
    result = apply_ev([[2.0, 0.5]], -1.0)

    assert result[0] == pytest.approx([1.0, 0.25], abs=1e-6)


def test_validate_reports_nan() -> None:
    """Defect 8: a missed NaN becomes an all-black rendered frame."""
    assert validate([[0.2, 0.3], [0.4, 0.5]], 8) == []

    problems = validate([[0.2, math.nan]], 8)
    assert problems
    assert any("nan" in problem.lower() for problem in problems)


def test_batch_does_not_share_histogram_between_frames() -> None:
    """Defect 9: each frame must be metered with its own histogram/state."""
    config = {
        "target_p": 0.18,
        "mode": "matrix",
        "bins": 16,
        "speed": 1.0,
        "start_ev": 0.0,
    }
    dark = _gray(0.04)
    bright = _gray(0.64)
    frames = [dark, bright]

    result = exposure_batch(frames, config)

    expected = [
        meter_ev(dark, config["target_p"], config["mode"]),
        meter_ev(bright, config["target_p"], config["mode"]),
    ]
    assert result["evs"] == pytest.approx(expected, abs=1e-6)
    assert result["evs"][0] > 0.0 > result["evs"][1]

    with pytest.raises(ExposureError):
        exposure_batch([dark, []], config)


def test_histogram_rejects_zero_bins() -> None:
    """Defect 10: bins=0 must fail cleanly instead of dividing by zero."""
    with pytest.raises(ExposureError):
        histogram([[0.18]], 0)


def test_metering_mid_gray_has_zero_ev() -> None:
    """A uniform 0.18 linear frame is already at the metering target."""
    ev = meter_ev(_gray(0.18, 16, 16), 0.18, "matrix")

    assert ev == pytest.approx(0.0, abs=1e-6)


def test_apply_ev_zero_is_identity() -> None:
    """A zero-EV adjustment leaves in-range linear samples unchanged."""
    image = [[0.0, 0.18], [0.5, 1.0]]

    result = apply_ev(image, 0.0)

    assert _flatten(result) == pytest.approx(_flatten(image), abs=1e-6)


def test_empty_image_raises_exposure_error() -> None:
    """An empty frame is invalid and must raise the domain exception."""
    with pytest.raises(ExposureError):
        meter_ev([], 0.18, "matrix")


def test_histogram_with_one_bin_has_single_bucket() -> None:
    """One requested bin means every valid sample lands in that one bin."""
    image = [[0.01], [0.10], [0.50], [1.0]]

    assert histogram(image, 1) == [4]


def test_batch_preserves_frame_order() -> None:
    """Successful batches return per-frame results in the input order."""
    frames = [_gray(0.05), _gray(0.18), _gray(0.72)]
    config = {
        "target_p": 0.18,
        "mode": "matrix",
        "bins": 32,
        "speed": 0.5,
        "start_ev": 0.0,
    }

    result = exposure_batch(frames, config)

    expected = [meter_ev(frame, 0.18, "matrix") for frame in frames]
    assert result["evs"] == pytest.approx(expected, abs=1e-6)
    assert result["evs"][0] > result["evs"][1] > result["evs"][2]
    assert len(result["adapted"]) == len(frames)
