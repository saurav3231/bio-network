"""Artificial retina: encode a pixel image into a spike timetable.

Encoding scheme (M3). A ``Retina`` converts an intensity image (normalized to
``[0, 1]``) into a list of ``(t_ms, pixel_index)`` spikes. Two coding modes are
supported:

- ``latency`` (default): time-to-first-spike coding. A pixel fires exactly once
  at ``t = (1 - intensity) * window_ms``:  brighter pixels spike earlier
  (Masquelier & Thorpe 2007). Pixels darker than ``threshold`` emit no spike.
- ``rate``: Poisson rate coding, each pixel fires at ``intensity *
  max_rate_hz Hz`` for the duration of the window (Diehl & Cook 2015).

Both modes are deterministic given the constructor's ``seed``.
"""

from __future__ import annotations

import numpy as np


class Retina:
    """Encode a pixel array into a ``(t_ms, pixel_index)`` spike timetable."""

    def __init__(
        self,
        image_shape: tuple[int, int] = (28, 28),
        mode: str = "latency",
        window_ms: float = 350.0,
        max_rate_hz: float = 60.0,
        threshold: float = 0.02,
        seed: int = 42,
    ) -> None:
        """Initialize the retina.

        Args:
            image_shape: ``(height, width)`` in pixels.
            mode: ``"latency"`` or ``"rate"`` coding.
            window_ms: presentation window; the retina emits spikes only within
                the first ``window_ms`` of a presentation.
            max_rate_hz: ceiling firing rate for a maximally bright pixel in
                rate mode.
            threshold: pixels with intensity below this value emit no spikes.
            seed: random seed; a given seed always produces the identical
                timetable for a given image and mode.
        """
        self.image_shape = tuple(int(s) for s in image_shape)
        self.mode = mode
        self.window_ms = float(window_ms)
        self.max_rate_hz = float(max_rate_hz)
        self.threshold = float(threshold)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        if mode not in ("latency", "rate"):
            raise ValueError(f"unknown retina mode '{mode}' (expect latency|rate)")

    @property
    def n_pixels(self) -> int:
        """Number of pixels in a full image (``H * W`` flat channels)."""
        return int(np.prod(self.image_shape))

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode one image into a spike timetable.

        Args:
            image: a ``(H, W)`` float array normalized to ``[0, 1]`` (or flat
                with ``H * W`` entries).

        Returns:
            A ``(n_spikes, 2)`` float array whose rows are ``(t_ms,
            pixel_index)``, sorted by spike time.
        """
        img = np.asarray(image, dtype=float).reshape(-1)
        if img.size != self.n_pixels:
            raise ValueError(
                f"image size {img.size} != retina n_pixels {self.n_pixels}"
            )

        if self.mode == "latency":
            return self._encode_latency(img)
        return self._encode_rate(img)

    # -- coding schemes --------------------------------------------------- --
    def _encode_latency(self, flat: np.ndarray) -> np.ndarray:
        """One spike per sufficiently bright pixel, earlier = brighter."""
        on = flat >= self.threshold
        if not on.any():
            return np.empty((0, 2), dtype=float)
        pixels = np.flatnonzero(on)
        intensity = flat[on]
        times = (1.0 - intensity) * self.window_ms
        return np.column_stack([times, pixels.astype(float)])

    def _encode_rate(self, flat: np.ndarray) -> np.ndarray:
        """A Poisson spike train per pixel with rate ``intensity * max_rate_hz``."""
        on = flat >= self.threshold
        if not on.any():
            return np.empty((0, 2), dtype=float)
        pixels = np.flatnonzero(on)
        intensity = flat[on]
        lam = intensity * self.max_rate_hz * (self.window_ms / 1000.0)
        counts = self._rng.poisson(lam)

        times = np.empty(int(counts.sum()))
        pixel_ids = np.empty_like(times)
        cursor = 0
        for pixel, count in zip(pixels, counts):
            if count:
                ts = self._rng.uniform(0.0, self.window_ms, size=int(count))
                times[cursor : cursor + count] = ts
                pixel_ids[cursor : cursor + count] = pixel
                cursor += count
        if cursor == 0:
            return np.empty((0, 2), dtype=float)
        table = np.column_stack([times[:cursor], pixel_ids[:cursor]])
        return table[np.argsort(table[:, 0])]
