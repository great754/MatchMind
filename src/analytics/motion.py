"""Smooth positions within observed segments; never bridge missing samples."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


@dataclass
class Motion:
    raw_positions: np.ndarray
    positions: np.ndarray
    velocity: np.ndarray
    instantaneous_speed_mps: np.ndarray
    smoothed_speed_mps: np.ndarray
    distance_m: np.ndarray
    step_distance_m: np.ndarray
    observed: np.ndarray
    valid: np.ndarray
    raw_speed_flag: np.ndarray
    rejected_speed_flag: np.ndarray
    segment_id: np.ndarray


def segments(mask):
    """Yield [start, end) indices for consecutive True runs."""
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def calculate_motion(positions, fps=25.0, smoothing_frames=15, median_frames=3,
                     speed_limit_mps=14.0):
    raw = np.asarray(positions, dtype=float).copy()
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError('positions must have shape (frames, 2)')
    if not np.isfinite([fps,speed_limit_mps]).all() or fps <= 0 or speed_limit_mps <= 0:
        raise ValueError('fps and speed limit must be positive')
    if any(w < 3 or w % 2 == 0 for w in (smoothing_frames, median_frames)):
        raise ValueError('smoothing windows must be odd and >= 3')
    observed = np.isfinite(raw).all(axis=1)
    smooth = np.full_like(raw, np.nan)
    velocity = np.full_like(raw, np.nan)
    instant = np.full(len(raw), np.nan)
    speed = instant.copy()
    steps = instant.copy()
    segment_id = np.full(len(raw), -1, dtype=int)
    for sid, (start, end) in enumerate(segments(observed)):
        segment_id[start:end] = sid
        length = end-start
        if length == 1:
            smooth[start:end] = raw[start:end]
            continue
        instant[start+1:end] = np.linalg.norm(np.diff(raw[start:end], axis=0), axis=1)*fps
        # A coordinate median removes isolated spikes before polynomial smoothing.
        filtered = median_filter(raw[start:end], size=(min(median_frames, length if length % 2 else length-1), 1), mode='nearest')
        window = min(smoothing_frames, length if length % 2 else length-1)
        smooth[start:end] = savgol_filter(filtered, window, min(2, window-1), axis=0, mode='interp') if window >= 3 else filtered
        delta = np.diff(smooth[start:end], axis=0)
        steps[start+1:end] = np.linalg.norm(delta, axis=1)
        speed[start+1:end] = steps[start+1:end]*fps
        velocity[start+1:end] = delta*fps
    rejected = np.isfinite(speed) & (speed > speed_limit_mps)
    # Also discard the following derivative; do not treat a jump as real distance.
    valid_step = np.isfinite(speed) & ~rejected & ~np.r_[False, rejected[:-1]]
    steps[~valid_step] = np.nan
    speed[~valid_step] = np.nan
    velocity[~valid_step] = np.nan
    usable = observed & ~rejected
    smooth[~usable] = np.nan
    # Distances represent covered segments only; a gap adds no fabricated meters.
    distance = np.cumsum(np.where(np.isfinite(steps), steps, 0.0))
    return Motion(raw, smooth, velocity, instant, speed, distance, steps,
                  observed, usable, instant > speed_limit_mps, rejected, segment_id)
