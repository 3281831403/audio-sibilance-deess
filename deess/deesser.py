"""Core contracts for the sibilance de-esser.

Audio is always represented as a plain ``list[float]``.  Frequencies are in
Hz, levels and thresholds are in dB, and linear gains are in the range
``(0, 1]``.  The tests in ``tests/test_deesser.py`` describe the required
behaviour.
"""

import math

DEFAULT_SAMPLE_RATE = 44100.0
MIN_THRESHOLD_DB = -60.0
MAX_THRESHOLD_DB = 0.0

#: Nyquist limit of the fixed 44.1 kHz pipeline used by :func:`validate_params`.
PIPELINE_NYQUIST_HZ = 22050.0

#: Length of the linear-phase FIR used to split the sibilance band.  A steep
#: transition is required: a first-order filter leaks too much energy and is
#: not acceptable.
_FIR_TAPS = 255

#: Half-width of the centred RMS envelope used by :func:`detect_sibilance`.
_DETECT_HALF_WINDOW = 32

#: Maximum silent gap (in samples) that is bridged when merging detections.
_DETECT_MERGE_GAP = 32


class DeessError(Exception):
    """Raised when de-esser parameters, audio, or batch input are invalid."""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite(value) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_sample_rate(sample_rate) -> float:
    if not _is_finite(sample_rate) or float(sample_rate) <= 0:
        raise DeessError(f"sample_rate must be finite and > 0 (got {sample_rate!r})")
    return float(sample_rate)


def _check_band_edges(low, high, sample_rate) -> tuple:
    """Validate a band for a given sample rate and return floats."""
    rate = _check_sample_rate(sample_rate)
    if not _is_finite(low):
        raise DeessError(f"low must be a finite number (got {low!r})")
    if not _is_finite(high):
        raise DeessError(f"high must be a finite number (got {high!r})")
    low = float(low)
    high = float(high)
    if low < 0:
        raise DeessError(f"low must be >= 0 Hz (got {low!r})")
    if low >= high:
        raise DeessError(f"low must be less than high (got {low!r} >= {high!r})")
    if high >= rate / 2.0:
        raise DeessError(
            f"high must be below the Nyquist limit {rate / 2.0!r} Hz "
            f"(got {high!r})"
        )
    return low, high, rate


def _sinc(value: float) -> float:
    if value == 0.0:
        return 1.0
    argument = math.pi * value
    return math.sin(argument) / argument


def _bandpass_taps(low: float, high: float, sample_rate: float) -> list:
    """Design a symmetric Blackman-windowed sinc band-pass FIR.

    The coefficients are normalised so the magnitude response at the
    geometric centre of the band is exactly one.
    """
    taps = _FIR_TAPS
    half = (taps - 1) / 2.0
    low_norm = low / sample_rate
    high_norm = high / sample_rate

    coefficients = []
    for index in range(taps):
        offset = index - half
        if offset == 0.0:
            value = 2.0 * (high_norm - low_norm)
        else:
            value = (
                math.sin(2.0 * math.pi * high_norm * offset)
                - math.sin(2.0 * math.pi * low_norm * offset)
            ) / (math.pi * offset)
        window = (
            0.42
            - 0.5 * math.cos(2.0 * math.pi * index / (taps - 1))
            + 0.08 * math.cos(4.0 * math.pi * index / (taps - 1))
        )
        coefficients.append(value * window)

    centre_frequency = math.sqrt(low * high)
    real = 0.0
    imag = 0.0
    for index, coefficient in enumerate(coefficients):
        offset = index - half
        angle = 2.0 * math.pi * centre_frequency * offset / sample_rate
        real += coefficient * math.cos(angle)
        imag += coefficient * math.sin(angle)
    magnitude = math.hypot(real, imag)
    if magnitude == 0.0:  # pragma: no cover - defensive only
        raise DeessError("could not normalise the band-pass filter")
    return [coefficient / magnitude for coefficient in coefficients]


def _convolve_centered(samples, taps, half) -> list:
    """Zero-phase FIR convolution that preserves the signal length."""
    length = len(samples)
    output = [0.0] * length
    for index in range(length):
        base = index - half
        total = 0.0
        for tap_index, coefficient in enumerate(taps):
            source = base + tap_index
            if 0 <= source < length:
                total += coefficient * samples[source]
        output[index] = total
    return output


def _parse_window(window) -> tuple:
    """Return ``(frame, hop)`` from an int or a ``(frame, hop)`` pair."""
    if isinstance(window, (tuple, list)):
        if len(window) != 2:
            raise DeessError("window pair must be (frame, hop)")
        frame, hop = window
    else:
        frame = hop = window
    if not _is_positive_int(frame) or not _is_positive_int(hop):
        raise DeessError(
            f"frame and hop must be positive integers (got {frame!r}, {hop!r})"
        )
    return frame, hop


def _rms_db(frame) -> float:
    if not frame:
        return float("-inf")
    total = 0.0
    for value in frame:
        total += value * value
    rms = math.sqrt(total / len(frame))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(rms)


def _peak_db(frame) -> float:
    """Peak magnitude of ``frame`` in dBFS, ``-inf`` for digital silence."""
    if not frame:
        return float("-inf")
    peak = 0.0
    for value in frame:
        magnitude = abs(value)
        if magnitude > peak:
            peak = magnitude
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(peak)


def band_split(samples, low, high, sample_rate) -> tuple:
    """Split a signal into its sibilance band and the residual.

    The return value is ``(band, residual)``.  Both elements are
    ``list[float]`` with exactly ``len(samples)`` entries.  ``band`` is the
    component in the closed frequency interval ``[low, high]`` (Hz) and
    ``residual`` is everything outside that interval.  For every index the
    reconstruction identity ``band[i] + residual[i] == samples[i]`` must
    hold within ``1e-6``.

    The split must use a steep transition.  A first-order filter is not
    sufficient: a tone 3.5 kHz below ``low`` must be attenuated by at least
    20 dB in ``band``.  Empty audio produces ``([], [])``.

    Args:
        samples: Input audio as ``list[float]``.
        low: Lower band edge in Hz, inclusive.
        high: Upper band edge in Hz, inclusive.
        sample_rate: Sample rate in Hz.

    Raises:
        DeessError: If ``sample_rate <= 0``, ``low < 0``, ``low >= high``,
            or ``high >= sample_rate / 2`` (the Nyquist limit).
    """
    low, high, rate = _check_band_edges(low, high, sample_rate)
    length = len(samples)
    if length == 0:
        return [], []

    taps = _bandpass_taps(low, high, rate)
    half = (len(taps) - 1) // 2
    band = _convolve_centered(samples, taps, half)
    residual = [original - split for original, split in zip(samples, band)]
    return band, residual


def band_level(samples, window) -> list:
    """Measure short-time level in dBFS with independent windows and hops.

    ``window`` is either a positive frame length or a ``(frame, hop)``
    pair.  Supplying a pair makes the analysis window and the hop
    independent; an integer means a hop equal to the frame length.  The
    result contains ``1 + (len(samples) - frame) // hop`` values when
    ``len(samples) >= frame`` and no values otherwise.

    Each value is the peak magnitude of a frame *centred* on the frame
    position, in dBFS (``20 * log10(max(abs(x)))``).  Centring the window
    localises a short sibilant burst to the frames it actually covers; a
    start-aligned window smears the burst into the following quiet frames
    and defeats localisation.  Digital silence is represented by ``-inf``.
    Empty input produces ``[]``.

    Args:
        samples: Input audio as ``list[float]``.
        window: Frame length, or ``(frame, hop)``; values are in samples.

    Raises:
        DeessError: If the frame or hop is not a positive integer.
    """
    frame, hop = _parse_window(window)
    length = len(samples)
    levels = []
    start = 0
    while start + frame <= length:
        # The analysis window is centred on the frame position so that a
        # short burst is localised to the frames it actually covers; a
        # start-aligned window smears the burst into the following quiet
        # frames and defeats localisation.
        left = start - frame // 2
        right = left + frame
        segment = samples[max(0, left):min(length, right)]
        levels.append(_peak_db(segment))
        start += hop
    return levels


def gain_reduction(levels, threshold_db, ratio) -> list:
    """Convert window levels in dBFS to linear compressor gains.

    For a level ``L`` and threshold ``T`` in dB, the target level is
    ``T + (L - T) / ratio`` when ``L > T``, and unity when ``L <= T``.
    The returned linear gain is ``10 ** ((target - L) / 20)``.  Thus a
    level above threshold is compressed by exactly ``ratio`` instead of
    being muted, and every returned gain is in ``(0, 1]``.  A level of
    ``-inf`` maps to unity.  Empty input produces ``[]``.

    Args:
        levels: Window levels in dBFS.
        threshold_db: Compression threshold in dB.
        ratio: Compression ratio; must be greater than zero.

    Raises:
        DeessError: If ``ratio <= 0`` or ``threshold_db`` is not finite.
    """
    if not _is_finite(threshold_db):
        raise DeessError(
            f"threshold_db must be a finite number (got {threshold_db!r})"
        )
    if not _is_finite(ratio) or float(ratio) <= 0.0:
        raise DeessError(f"ratio must be finite and > 0 (got {ratio!r})")

    threshold = float(threshold_db)
    ratio = float(ratio)
    gains = []
    for level in levels:
        if not _is_finite(level) or float(level) <= threshold:
            gains.append(1.0)
            continue
        level = float(level)
        target = threshold + (level - threshold) / ratio
        gain = 10.0 ** ((target - level) / 20.0)
        # Guard against expansion (ratio < 1) so the contract's (0, 1]
        # range always holds.
        if gain > 1.0:
            gain = 1.0
        elif gain < 0.0:  # pragma: no cover - defensive only
            gain = 0.0
        gains.append(gain)
    return gains


def smooth_gain(gains, attack_ms, release_ms, sample_rate) -> list:
    """Smooth per-sample target gains into a click-free envelope.

    ``gains`` contains one linear target gain per audio sample.  The first
    output equals ``gains[0]``.  While the target is below the current
    envelope the envelope moves toward it using the attack time constant;
    while the target is above the envelope it moves using the release time
    constant.  ``attack_ms`` and ``release_ms`` are time constants in
    milliseconds, and ``sample_rate`` is in Hz.  The output has the same
    length as ``gains``, stays between the target and the previous
    envelope value (no overshoot), and changes smoothly enough to avoid a
    single-sample click.  Empty input produces ``[]``.

    Args:
        gains: Per-sample linear target gains in ``[0, 1]``.
        attack_ms: Attack time constant in milliseconds.
        release_ms: Release time constant in milliseconds.
        sample_rate: Sample rate in Hz.

    Raises:
        DeessError: If a time constant is negative or ``sample_rate <= 0``.
    """
    if not _is_finite(attack_ms) or float(attack_ms) < 0.0:
        raise DeessError(
            f"attack_ms must be finite and >= 0 (got {attack_ms!r})"
        )
    if not _is_finite(release_ms) or float(release_ms) < 0.0:
        raise DeessError(
            f"release_ms must be finite and >= 0 (got {release_ms!r})"
        )
    rate = _check_sample_rate(sample_rate)

    if not gains:
        return []

    def coefficient(milliseconds: float) -> float:
        if milliseconds <= 0.0:
            return 0.0
        samples = milliseconds * 0.001 * rate
        return math.exp(-1.0 / samples)

    attack = coefficient(float(attack_ms))
    release = coefficient(float(release_ms))

    envelope = float(gains[0])
    smoothed = [envelope]
    for target in gains[1:]:
        target = float(target)
        # Moving toward a lower target is attack; toward a higher target
        # is release.  Interpolating between the previous value and the
        # target guarantees no overshoot and a smooth trajectory.
        alpha = attack if target < envelope else release
        envelope = target + (envelope - target) * alpha
        smoothed.append(envelope)
    return smoothed


def apply_band_gain(samples, gain, band, sample_rate) -> list:
    """Apply linear gain only to a band-limited component.

    ``samples`` is the original full-band signal and ``band`` is its
    band-limited component, for example the first value returned by
    :func:`band_split`.  Both have the same length.  ``gain`` is either a
    scalar or one linear gain per sample.  The output is
    ``(samples[i] - band[i]) + band[i] * gain[i]``, so the residual is
    preserved within ``1e-6`` and only the band is attenuated.  With
    ``gain == 1.0`` the output equals ``samples`` within ``1e-6``.  Empty
    audio produces ``[]``.

    Args:
        samples: Original audio as ``list[float]``.
        gain: Scalar linear gain or a per-sample ``list[float]``.
        band: Band-limited component of ``samples``.
        sample_rate: Sample rate in Hz, validated for interface symmetry.

    Raises:
        DeessError: If lengths differ, ``sample_rate <= 0``, or a gain is
            negative.
    """
    _check_sample_rate(sample_rate)
    length = len(samples)
    if len(band) != length:
        raise DeessError(
            f"samples and band must have the same length "
            f"(got {length} and {len(band)})"
        )
    if length == 0:
        return []

    if isinstance(gain, (list, tuple)):
        if len(gain) != length:
            raise DeessError(
                f"per-sample gain must match the signal length "
                f"(got {len(gain)} and {length})"
            )
        gains = list(gain)
    else:
        if not _is_finite(gain):
            raise DeessError(f"gain must be a finite number (got {gain!r})")
        gains = [float(gain)] * length

    output = []
    for original, split, value in zip(samples, band, gains):
        if not _is_finite(value):
            raise DeessError(f"gain must be finite (got {value!r})")
        value = float(value)
        if value < 0.0:
            raise DeessError(f"gain must be non-negative (got {value!r})")
        output.append((original - split) + split * value)
    return output


def detect_sibilance(samples, band, threshold_db) -> list:
    """Find merged sibilance intervals in a band-limited signal.

    ``samples`` and ``band`` have the same length; ``band`` is the
    band-limited component to analyse.  A centered 64-sample rectangular
    RMS envelope of ``band`` is converted to dBFS.  A sample position is
    active when its envelope is strictly above ``threshold_db`` (dB).
    Contiguous active positions form half-open ``(start, end)`` intervals,
    and intervals separated by a gap of at most 32 samples are merged, so
    one burst with a brief dip becomes one region.  Returned intervals are
    sorted, non-overlapping, and use integer sample indices.  Empty audio
    produces ``[]``.

    Args:
        samples: Original audio as ``list[float]``.
        band: Band-limited component of ``samples``.
        threshold_db: Detection threshold in dB.

    Raises:
        DeessError: If the two signals have different lengths.
    """
    if len(samples) != len(band):
        raise DeessError(
            f"samples and band must have the same length "
            f"(got {len(samples)} and {len(band)})"
        )
    if not _is_finite(threshold_db):
        raise DeessError(
            f"threshold_db must be a finite number (got {threshold_db!r})"
        )

    length = len(band)
    if length == 0:
        return []

    half = _DETECT_HALF_WINDOW
    runs = []
    run_start = None
    for position in range(length):
        start = max(0, position - half)
        end = min(length, position + half)
        level = _rms_db(band[start:end])
        active = level > threshold_db
        if active and run_start is None:
            run_start = position
        elif not active and run_start is not None:
            runs.append((run_start, position))
            run_start = None
    if run_start is not None:
        runs.append((run_start, length))

    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= _DETECT_MERGE_GAP:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def validate_params(low, high, threshold_db) -> list:
    """Return a list of parameter problems, or ``[]`` when valid.

    ``low`` and ``high`` are band edges in Hz.  This signature has no
    sample-rate argument, so Nyquist is the fixed 44.1 kHz pipeline limit
    of 22050 Hz.  ``threshold_db`` is in dB and must be between
    ``-60.0`` and ``0.0`` inclusive.

    Reported problems are: ``low < 0``; ``low >= high``; ``high`` at or
    above 22050 Hz; ``threshold_db`` outside the allowed range; and
    non-finite values.  The function returns messages instead of raising
    for these invalid values.

    Args:
        low: Lower band edge in Hz.
        high: Upper band edge in Hz.
        threshold_db: Threshold in dB.

    Returns:
        A list of human-readable problem strings; empty means valid.
    """
    problems = []
    if not _is_finite(low):
        problems.append(f"low must be a finite number (got {low!r})")
    if not _is_finite(high):
        problems.append(f"high must be a finite number (got {high!r})")
    if not _is_finite(threshold_db):
        problems.append(
            f"threshold_db must be a finite number (got {threshold_db!r})"
        )
    if problems:
        return problems

    low = float(low)
    high = float(high)
    threshold_db = float(threshold_db)

    if low < 0.0:
        problems.append(f"low must be >= 0 Hz (got {low!r})")
    if low >= high:
        problems.append(f"low must be less than high (got {low!r} >= {high!r})")
    if high >= PIPELINE_NYQUIST_HZ:
        problems.append(
            f"high must be below the Nyquist limit {PIPELINE_NYQUIST_HZ!r} Hz "
            f"(got {high!r})"
        )
    if threshold_db < MIN_THRESHOLD_DB or threshold_db > MAX_THRESHOLD_DB:
        problems.append(
            f"threshold_db must be within [{MIN_THRESHOLD_DB!r}, "
            f"{MAX_THRESHOLD_DB!r}] dB (got {threshold_db!r})"
        )
    return problems


_CONFIG_KEYS = (
    "sample_rate",
    "low",
    "high",
    "threshold_db",
    "ratio",
    "window",
    "attack_ms",
    "release_ms",
)


def _validated_config(config) -> dict:
    """Validate a batch configuration, raising ``DeessError`` on problems."""
    if not hasattr(config, "__getitem__") or not hasattr(config, "keys"):
        raise DeessError("config must be a mapping")
    missing = [key for key in _CONFIG_KEYS if key not in config]
    if missing:
        raise DeessError(f"config is missing keys: {', '.join(missing)}")

    problems = []
    sample_rate = config["sample_rate"]
    low = config["low"]
    high = config["high"]
    threshold_db = config["threshold_db"]
    ratio = config["ratio"]
    window = config["window"]
    attack_ms = config["attack_ms"]
    release_ms = config["release_ms"]

    if not _is_finite(sample_rate) or float(sample_rate) <= 0.0:
        problems.append(
            f"sample_rate must be finite and > 0 (got {sample_rate!r})"
        )
    problems.extend(validate_params(low, high, threshold_db))
    if not _is_finite(ratio) or float(ratio) <= 0.0:
        problems.append(f"ratio must be finite and > 0 (got {ratio!r})")
    try:
        _parse_window(window)
    except DeessError as error:
        problems.append(str(error))
    if not _is_finite(attack_ms) or float(attack_ms) < 0.0:
        problems.append(
            f"attack_ms must be finite and >= 0 (got {attack_ms!r})"
        )
    if not _is_finite(release_ms) or float(release_ms) < 0.0:
        problems.append(
            f"release_ms must be finite and >= 0 (got {release_ms!r})"
        )

    if problems:
        raise DeessError("invalid config: " + "; ".join(problems))

    # The batch may be run at a rate other than 44.1 kHz, so also enforce
    # that rate's Nyquist limit.
    rate = float(sample_rate)
    if float(high) >= rate / 2.0:
        raise DeessError(
            f"high must be below the Nyquist limit {rate / 2.0!r} Hz "
            f"(got {high!r})"
        )

    return {
        "sample_rate": rate,
        "low": float(low),
        "high": float(high),
        "threshold_db": float(threshold_db),
        "ratio": float(ratio),
        "window": _parse_window(window),
        "attack_ms": float(attack_ms),
        "release_ms": float(release_ms),
    }


def _validated_tracks(tracks) -> list:
    """Validate names and audio before any processing, returning copies."""
    if isinstance(tracks, (str, bytes)) or not hasattr(tracks, "__iter__"):
        raise DeessError("tracks must be an ordered sequence of (name, audio)")

    validated = []
    seen = set()
    for item in tracks:
        if isinstance(item, (str, bytes)) or not hasattr(item, "__len__"):
            raise DeessError("each track must be a (name, samples) pair")
        if len(item) != 2:
            raise DeessError("each track must be a (name, samples) pair")
        name, samples = item
        if not isinstance(name, str):
            raise DeessError(f"track name must be a string (got {name!r})")
        if name in seen:
            raise DeessError(f"duplicate track name: {name!r}")
        seen.add(name)
        if isinstance(samples, (str, bytes)) or not hasattr(samples, "__iter__"):
            raise DeessError(f"track {name!r} audio must be a list of floats")
        copied = []
        for value in samples:
            if not _is_finite(value):
                raise DeessError(
                    f"track {name!r} contains a non-finite sample ({value!r})"
                )
            copied.append(float(value))
        validated.append((name, copied))
    return validated


def _process_track(samples, config) -> list:
    """Run the full de-essing chain for a single track with fresh state."""
    length = len(samples)
    if length == 0:
        return []

    sample_rate = config["sample_rate"]
    low = config["low"]
    high = config["high"]
    threshold_db = config["threshold_db"]
    ratio = config["ratio"]
    frame, hop = config["window"]

    # 1. Split so that only the sibilance band is ever attenuated; the
    #    residual (the rest of the voice) is left untouched.
    band, _ = band_split(samples, low, high, sample_rate)

    # 2. Locate merged sibilance regions.  A track without any is passed
    #    through bit-for-bit.
    intervals = detect_sibilance(samples, band, threshold_db)
    if not intervals:
        return list(samples)

    # 3. Short-time level of the band (not the whole signal), converted to
    #    proportional gains.
    levels = band_level(band, (frame, hop))
    frame_gains = gain_reduction(levels, threshold_db, ratio)

    # Expand the per-frame gains onto a per-sample target envelope.
    target = [1.0] * length
    for index, value in enumerate(frame_gains):
        start = index * hop
        end = min(length, start + hop)
        for position in range(start, end):
            target[position] = value
    if frame_gains:
        tail_start = min(length, len(frame_gains) * hop)
        last = frame_gains[-1]
        for position in range(tail_start, length):
            target[position] = last

    # Only attenuate inside detected regions; elsewhere leave unity.
    mask = bytearray(length)
    for start, end in intervals:
        for position in range(max(0, start), min(length, end)):
            mask[position] = 1
    for position in range(length):
        if not mask[position]:
            target[position] = 1.0

    # 4. Smooth with independent attack/release constants so release is
    #    not too slow and does not pump.
    envelope = smooth_gain(
        target, config["attack_ms"], config["release_ms"], sample_rate
    )

    # 5. Apply gain to the band only, preserving the residual.
    return apply_band_gain(samples, envelope, band, sample_rate)


def deess_batch(tracks, config) -> dict:
    """Process a batch of tracks atomically with isolated per-track state.

    ``tracks`` is an ordered sequence of ``(name, samples)`` pairs with
    unique names.  ``config`` is a mapping with the keys ``sample_rate``,
    ``low``, ``high``, ``threshold_db``, ``ratio``, ``window``,
    ``attack_ms``, and ``release_ms``.  The return value maps each name to
    its processed ``list[float]``, preserving the input order.

    Every name, track, and config value is validated before any audio is
    processed.  If any track or the configuration is invalid, raise
    :class:`DeessError` and produce no partial result; the input tracks are
    never mutated.  Each track receives fresh filter and envelope state,
    so a track's output is identical whether it is processed alone or
    after other tracks.  A track with no sibilance is returned unchanged
    within ``1e-9``.  An empty batch produces ``{}``.

    Args:
        tracks: Ordered ``(name, samples)`` pairs.
        config: Shared processing configuration.

    Raises:
        DeessError: If the batch is invalid or any track cannot be
            processed.  No partial batch result is returned.
    """
    validated_config = _validated_config(config)
    validated_tracks = _validated_tracks(tracks)

    # Everything is validated; processing cannot produce a partial result.
    result = {}
    for name, samples in validated_tracks:
        result[name] = _process_track(samples, validated_config)
    return result
