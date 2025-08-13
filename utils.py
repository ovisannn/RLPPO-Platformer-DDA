# file: plot_rewards_with_smoothing.py
"""Plot training rewards with optional smoothing.

Provides moving average (window) and/or exponential moving average (alpha).

Usage example with your variables:
    rewards = list(cb.episode_rewards)
    plot_rewards_with_smoothing(rewards, rewards_png, window=50, alpha=None)

Notes:
- MA is NaN-padded on the left so the curve aligns with the raw series.
- If both window and alpha are provided, both smoothed curves are plotted.
"""
from __future__ import annotations

from typing import Iterable, List, Optional
import math

import numpy as np
import matplotlib.pyplot as plt


def moving_average(data: Iterable[float], window: int) -> np.ndarray:
    """Return a same-length moving average of *data* using a simple window.

    Pads the first (window-1) positions with NaN to keep alignment. This avoids a
    misleading early spike from partial windows.

    Args:
        data: Sequence of numeric values.
        window: Window size (>= 1).
    Returns:
        np.ndarray of shape (len(data),) with NaNs for the first window-1 entries.
    Raises:
        ValueError: If window < 1.
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    x = np.asarray(list(data), dtype=float)
    if x.size == 0:
        return x

    if window == 1:
        return x.copy()

    # Why: use convolution for performance and numerical stability vs manual loops.
    kernel = np.ones(window, dtype=float) / float(window)
    valid = np.convolve(x, kernel, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, valid])


def exponential_moving_average(data: Iterable[float], alpha: float) -> np.ndarray:
    """Compute EMA with smoothing factor alpha in (0, 1].

    Args:
        data: Sequence of numeric values.
        alpha: Smoothing factor (0 < alpha <= 1). Larger -> less smoothing.
    Returns:
        np.ndarray EMA of same length.
    Raises:
        ValueError: If alpha not in (0, 1].
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")

    x = np.asarray(list(data), dtype=float)
    if x.size == 0:
        return x

    ema = np.empty_like(x)
    ema[0] = x[0]
    # Why: recursive definition preserves trend and is O(n).
    for i in range(1, x.size):
        ema[i] = alpha * x[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def plot_rewards_with_smoothing(
    rewards: Iterable[float],
    rewards_png: str,
    *,
    window: Optional[int] = 50,
    alpha: Optional[float] = None,
    title: str = "SB3 (VecNormalize v2) Training Rewards",
) -> None:
    """Plot raw rewards with optional moving average and/or EMA smoothing.

    Args:
        rewards: Episode returns.
        rewards_png: Output image path.
        window: Moving average window (set to None to skip MA).
        alpha: EMA smoothing factor (set to None to skip EMA).
        title: Plot title.
    """
    y = np.asarray(list(rewards), dtype=float)

    plt.figure()
    plt.plot(y, linewidth=1.0, label="raw")

    labels: List[str] = ["raw"]

    if window is not None:
        ma = moving_average(y, int(window))
        plt.plot(ma, linewidth=1.8, label=f"MA (w={int(window)})")
        labels.append(f"MA (w={int(window)})")

    if alpha is not None:
        ema = exponential_moving_average(y, float(alpha))
        plt.plot(ema, linewidth=1.8, label=f"EMA (α={float(alpha):.2f})")
        labels.append(f"EMA (α={float(alpha):.2f})")

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    if len(labels) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(rewards_png, dpi=160)
    plt.close()


# Example integration with your snippet:
# rewards = list(cb.episode_rewards)
# plot_rewards_with_smoothing(rewards, rewards_png, window=50, alpha=None)
