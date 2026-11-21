"""Linear-light exposure metering primitives.

Images are plain two-dimensional ``list[list[float]]`` values (rows of
pixels), never NumPy arrays.  Luminance and EV arithmetic is linear in light,
not on gamma-encoded display values.  Floating-point results are expected to
agree with the tests to within an absolute tolerance of ``1e-6``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class ExposureError(ValueError):
    """Raised when an image, histogram, or exposure parameter is invalid."""


# Histograms cover twelve stops in log2 luminance: 2**-12 through 2**0.
LOG_LUMA_MIN = -12.0
LOG_LUMA_MAX = 0.0
MID_GRAY = 0.18


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _image_problems(img: list[list[float]]) -> list[str]:
    """Return human-readable problems with a grayscale image (empty = valid)."""
    problems: list[str] = []
    if not img:
        problems.append("image is empty")
        return problems
    width = None
    for r, row in enumerate(img):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            problems.append(f"row {r} is not a sequence")
            continue
        if width is None:
            width = len(row)
            if width == 0:
                problems.append("image has empty rows")
        elif len(row) != width:
            problems.append(
                f"row {r} is ragged: expected {width} samples, got {len(row)}"
            )
        for c, sample in enumerate(row):
            if not _is_number(sample):
                problems.append(f"sample at ({r}, {c}) is not a number")
                continue
            if math.isnan(sample):
                problems.append(f"NaN sample at ({r}, {c})")
            elif math.isinf(sample):
                problems.append(f"non-finite sample at ({r}, {c})")
            elif sample < 0:
                problems.append(f"negative sample at ({r}, {c})")
    return problems


def _raise_if_invalid(img: list[list[float]]) -> None:
    problems = _image_problems(img)
    if problems:
        raise ExposureError("; ".join(problems))


def luminance(img: list[list[float | Sequence[float]]]) -> list[list[float]]:
    """Return per-pixel linear relative luminance.

    Each outer list is a row and each pixel is either a scalar (an already
    linear grayscale sample) or a three-element RGB sequence.  RGB samples
    use the Rec.709 weights ``0.2126 R + 0.7152 G + 0.0722 B``; an
    equal-weight average is not acceptable.  Grayscale samples are returned
    directly in a new grid.

    Raises ``ExposureError`` for an empty, ragged, negative, or non-finite
    image.  Returned values are linear, are not gamma encoded, and are
    compared by the tests with an absolute tolerance of ``1e-6``.
    """
    if not img:
        raise ExposureError("image is empty")

    result: list[list[float]] = []
    width: int | None = None
    for r, row in enumerate(img):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ExposureError(f"row {r} is not a sequence")
        if width is None:
            width = len(row)
            if width == 0:
                raise ExposureError("image has empty rows")
        elif len(row) != width:
            raise ExposureError(f"row {r} is ragged")

        out_row: list[float] = []
        for c, pixel in enumerate(row):
            if _is_number(pixel):
                value = float(pixel)  # type: ignore[arg-type]
            elif isinstance(pixel, Sequence) and not isinstance(pixel, (str, bytes)):
                if len(pixel) != 3:
                    raise ExposureError(
                        f"RGB pixel at ({r}, {c}) must have 3 channels"
                    )
                channels = []
                for channel in pixel:
                    if not _is_number(channel):
                        raise ExposureError(
                            f"RGB channel at ({r}, {c}) is not a number"
                        )
                    channels.append(float(channel))
                for channel in channels:
                    if not math.isfinite(channel):
                        raise ExposureError(
                            f"non-finite RGB channel at ({r}, {c})"
                        )
                    if channel < 0:
                        raise ExposureError(f"negative RGB channel at ({r}, {c})")
                # Rec.709 luma weights, not an equal-weight average.
                value = (
                    0.2126 * channels[0]
                    + 0.7152 * channels[1]
                    + 0.0722 * channels[2]
                )
                out_row.append(value)
                continue
            else:
                raise ExposureError(f"pixel at ({r}, {c}) is not a number")

            if not math.isfinite(value):
                raise ExposureError(f"non-finite sample at ({r}, {c})")
            if value < 0:
                raise ExposureError(f"negative sample at ({r}, {c})")
            out_row.append(value)
        result.append(out_row)
    return result


def histogram(img: list[list[float]], bins: int) -> list[int]:
    """Build a histogram of log2 luminance.

    The ``bins`` equal-width buckets span ``[LOG_LUMA_MIN, LOG_LUMA_MAX]``
    in log2 space.  A sample ``v <= 0`` goes to bucket 0; otherwise
    ``ev = log2(v)`` is clamped to that range and the bucket is
    ``floor((ev - LOG_LUMA_MIN) / (LOG_LUMA_MAX - LOG_LUMA_MIN) * bins)``,
    capped at ``bins - 1``.  Consequently a stop of shadow detail is not
    compressed into the same bucket as a linear split would compress it.
    The counts sum to the number of pixels.

    ``bins <= 0`` raises ``ExposureError``.  An empty, ragged, negative, or
    non-finite image also raises ``ExposureError``.
    """
    if not isinstance(bins, int) or bins <= 0:
        raise ExposureError(f"bins must be a positive integer, got {bins!r}")
    _raise_if_invalid(img)

    counts = [0] * bins
    span = LOG_LUMA_MAX - LOG_LUMA_MIN
    for row in img:
        for sample in row:
            if sample <= 0:
                counts[0] += 1
                continue
            ev = min(max(math.log2(sample), LOG_LUMA_MIN), LOG_LUMA_MAX)
            bucket = int(math.floor((ev - LOG_LUMA_MIN) / span * bins))
            counts[min(bucket, bins - 1)] += 1
    return counts


def percentile(hist: list[int], p: float) -> float:
    """Return the luminance at quantile ``p`` from a log histogram.

    ``hist`` is a histogram produced by :func:`histogram`; its bin count
    determines the log2 bucket width.  ``p`` must be in ``[0, 1]``.  The
    cumulative distribution is interpolated *inside* the selected bucket:
    the target count is ``p * sum(hist)``, and the returned log2 value moves
    linearly from the bucket's lower edge to its upper edge as that bucket's
    share of the target is consumed.  The result is ``2 ** log_value``, a
    luminance in linear light -- never a bucket index.

    A zero-total histogram or an out-of-range ``p`` raises ``ExposureError``.
    Returned luminance values are compared with an absolute tolerance of
    ``1e-6``.
    """
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ExposureError(f"p must be in [0, 1], got {p!r}")
    total = sum(hist)
    if total <= 0:
        raise ExposureError("histogram has zero total count")

    bins = len(hist)
    width = (LOG_LUMA_MAX - LOG_LUMA_MIN) / bins
    target = p * total

    cumulative = 0
    for i, count in enumerate(hist):
        if count <= 0:
            continue
        if cumulative + count >= target:
            # Interpolate linearly inside this bucket instead of returning
            # the bucket index.
            fraction = (target - cumulative) / count
            fraction = min(max(fraction, 0.0), 1.0)
            log_value = LOG_LUMA_MIN + (i + fraction) * width
            return 2.0 ** log_value
        cumulative += count

    # target == total lands in the last non-empty bucket at its upper edge.
    log_value = LOG_LUMA_MIN + bins * width
    return 2.0 ** log_value


def _region_mean(lum: list[list[float]], mode: str) -> float:
    height = len(lum)
    width = len(lum[0])

    if mode == "matrix":
        rows = range(0, height)
        cols = range(0, width)
    elif mode == "center":
        r0, r1 = height // 4, height - height // 4
        c0, c1 = width // 4, width - width // 4
        rows = range(r0, r1)
        cols = range(c0, c1)
    elif mode == "spot":
        side = max(1, round(0.1 * min(height, width)))
        r0 = (height - side) // 2
        c0 = (width - side) // 2
        rows = range(r0, r0 + side)
        cols = range(c0, c0 + side)
    else:
        raise ExposureError(f"unknown metering mode: {mode!r}")

    total = 0.0
    count = 0
    for r in rows:
        for c in cols:
            total += lum[r][c]
            count += 1
    return total / count


def meter_ev(img: list[list[float]], target_p: float, mode: str) -> float:
    """Meter an image and return its EV offset from the target.

    ``target_p`` is the target linear luminance, normally :data:`MID_GRAY`
    (0.18).  The image is converted to linear luminance and metered with one
    of three regions:

    * ``"matrix"``: mean over every pixel;
    * ``"center"``: mean over the central half of the rows and columns;
    * ``"spot"``: mean over a centered square whose side is
      ``max(1, round(0.1 * min(height, width)))``.

    The result is ``log2(target_p / measured)``, so a uniform 0.18 frame
    metered at ``target_p=0.18`` gives 0.0.  Point/spot metering therefore
    may differ from global/matrix metering.

    Raises ``ExposureError`` for an invalid image, a non-positive or
    non-finite ``target_p``, an unknown mode, or zero measured luminance.
    EV values are compared with an absolute tolerance of ``1e-6``.
    """
    if not math.isfinite(target_p) or target_p <= 0:
        raise ExposureError(f"target_p must be positive and finite, got {target_p!r}")
    _raise_if_invalid(img)

    lum = luminance(img)
    measured = _region_mean(lum, mode)
    if measured <= 0:
        raise ExposureError("measured luminance is zero")
    return math.log2(target_p / measured)


def adapt(prev_ev: float, new_ev: float, speed: float) -> float:
    """Exponentially approach ``new_ev`` from ``prev_ev``.

    ``speed`` is clamped to ``[0, 1]`` before use: 0 keeps ``prev_ev`` and 1
    jumps to ``new_ev``.  The result is
    ``prev_ev + (new_ev - prev_ev) * clamped_speed``, so an out-of-range
    speed can never overshoot the requested EV.  Non-finite inputs raise
    ``ExposureError``.  EV results are compared with an absolute tolerance of
    ``1e-6``.
    """
    for name, value in (("prev_ev", prev_ev), ("new_ev", new_ev), ("speed", speed)):
        if not math.isfinite(value):
            raise ExposureError(f"{name} must be finite, got {value!r}")
    clamped = min(max(speed, 0.0), 1.0)
    return prev_ev + (new_ev - prev_ev) * clamped


def apply_ev(img: list[list[float]], ev: float) -> list[list[float]]:
    """Apply an EV offset to a linear grayscale image.

    Every sample is multiplied by ``2 ** ev`` and only then clamped to
    ``[0, 1]``.  Input samples may exceed 1 (HDR highlights): at -1 EV a
    sample of 2.0 becomes 1.0, not 0.5, because the highlight is scaled
    before it is clipped.  The returned grid is new; the input is not
    mutated.

    ``ev`` must be finite, and an invalid image raises ``ExposureError``.
    Output samples are compared with an absolute tolerance of ``1e-6``.
    """
    if not math.isfinite(ev):
        raise ExposureError(f"ev must be finite, got {ev!r}")
    _raise_if_invalid(img)

    scale = 2.0 ** ev
    return [
        [min(max(sample * scale, 0.0), 1.0) for sample in row]
        for row in img
    ]


def validate(img: list[list[float]], bins: int) -> list[str]:
    """Return a list of human-readable problems; an empty list is valid.

    Validation covers an empty image, ragged/non-rectangular rows, non-finite
    samples (NaN and infinities), negative samples, and ``bins <= 0``.  A NaN
    problem message contains ``"NaN"`` (case-insensitive).  This function
    reports problems by returning strings and does not raise for bad image
    data; callers such as :func:`meter_ev` convert the list into an
    ``ExposureError``.
    """
    problems = _image_problems(img)
    if not isinstance(bins, int) or bins <= 0:
        problems.append(f"bins must be a positive integer, got {bins!r}")
    return problems


def exposure_batch(frames: list[list[list[float]]], config: dict) -> dict:
    """Meter and adapt a batch of frames atomically.

    ``config`` may contain ``target_p`` (default 0.18), ``mode`` (default
    ``"matrix"``), ``speed`` (default 1.0), ``start_ev`` (default 0.0), and
    ``bins`` (default 64).  Each frame is metered independently; no
    histogram or scratch state is shared between frames.

    Every frame is validated before any result is produced.  If even one
    frame is invalid, the call raises ``ExposureError`` and returns nothing,
    so there is no partial batch to roll back.  On success it returns::

        {"evs": [...], "adapted": [...]}

    ``evs`` holds the per-frame :func:`meter_ev` results in input order.
    ``adapted`` applies :func:`adapt` cumulatively in that same order,
    starting from ``start_ev``.  EV values are compared with an absolute
    tolerance of ``1e-6``.
    """
    config = config or {}
    target_p = config.get("target_p", MID_GRAY)
    mode = config.get("mode", "matrix")
    speed = config.get("speed", 1.0)
    start_ev = config.get("start_ev", 0.0)
    bins = config.get("bins", 64)

    # Validate every frame up front so a bad frame aborts the whole batch.
    for i, frame in enumerate(frames):
        problems = validate(frame, bins)
        if problems:
            raise ExposureError(f"frame {i} is invalid: {'; '.join(problems)}")

    # Each frame is metered from its own pixels; no state is shared.
    evs = [meter_ev(frame, target_p, mode) for frame in frames]

    adapted: list[float] = []
    current = start_ev
    for ev in evs:
        current = adapt(current, ev, speed)
        adapted.append(current)

    return {"evs": evs, "adapted": adapted}
