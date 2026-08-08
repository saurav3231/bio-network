"""MNIST dataset loading with a raw-IDX fallback downloader.

Loading order (no heavy ML framework required): try ``keras`` first, then
``sklearn``, and finally download the canonical Yann LeCun IDX files directly
over HTTP and parse them with plain ``struct``/``gzip``. The dataset is cached
as a single ``npz`` under ``notebooks/data/`` so it is downloaded once and never
re-fetched (the ``notebooks/data/`` directory is gitignored).
"""

from __future__ import annotations

import gzip
import pathlib
import struct
import urllib.request

import numpy as np

_FALLBACK_DIR = pathlib.Path(__file__).resolve().parents[2] / "notebooks" / "data"

_IMAGES_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
_LABELS_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"
_TEST_IMAGES_URL = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
)
_TEST_LABELS_URL = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
)

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _data_dir() -> pathlib.Path:
    return _FALLBACK_DIR


def _parse_idx_images(path: pathlib.Path) -> np.ndarray:
    """Parse an IDX image file (magic 2051) into ``(n, 28, 28)`` uint8."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic {magic}")
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def _parse_idx_labels(path: pathlib.Path) -> np.ndarray:
    """Parse an IDX label file (magic 2049) into ``(n,)`` uint8."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label magic {magic}")
        return np.frombuffer(f.read(n), dtype=np.uint8)


def _try_keras() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        from keras.datasets import mnist as keras_mnist
    except Exception:  # noqa: BLE001 - optional dependency fallback
        return None
    (x_train, y_train), (x_test, y_test) = keras_mnist.load_data()
    return x_train, y_train, x_test, y_test


def _try_sklearn() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        from sklearn.datasets import fetch_openml
    except Exception:  # noqa: BLE001 - optional dependency fallback
        return None
    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x = data.data.astype(np.uint8).reshape(-1, 28, 28)
    y = data.target.astype(np.int64)
    return x[:60000], y[:60000], x[60000:], y[60000:]


def _download_raw(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Download the four canonical IDX files into ``root`` (cached)."""
    root.mkdir(parents=True, exist_ok=True)
    urls = {
        "train-images-idx3-ubyte.gz": _IMAGES_URL,
        "train-labels-idx1-ubyte.gz": _LABELS_URL,
        "t10k-images-idx3-ubyte.gz": _TEST_IMAGES_URL,
        "t10k-labels-idx1-ubyte.gz": _TEST_LABELS_URL,
    }
    paths = []
    for name, url in urls.items():
        dest = root / name
        if not dest.exists():
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
        paths.append(dest)
    return tuple(paths)


def load_mnist(
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST as ``(x_train, y_train, x_test, y_test)``.

    ``x_*`` are ``uint8`` arrays of shape ``(n, 28, 28)``; ``y_*`` are
    ``(n,)`` integer labels 0-9. Results are cached as ``mnist.npz`` in
    ``notebooks/data/``.

    Args:
        force: re-download / re-parse even if a cached ``mnist.npz`` exists.
    """
    data_dir = _data_dir()
    cache = data_dir / "mnist.npz"
    if cache.exists() and not force:
        with np.load(cache) as data:
            return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

    loaded = _try_keras() or _try_sklearn()
    if loaded is not None:
        x_train, y_train, x_test, y_test = loaded
    else:
        root = data_dir / "raw"
        img_tr, lbl_tr, img_te, lbl_te = _download_raw(root)
        x_train, y_train = _parse_idx_images(img_tr), _parse_idx_labels(lbl_tr)
        x_test, y_test = _parse_idx_images(img_te), _parse_idx_labels(lbl_te)

    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    return x_train, y_train, x_test, y_test


def subsample_mnist(
    x: np.ndarray,
    y: np.ndarray,
    per_class: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic, class-balanced subsample of MNIST images.

    ``per_class`` images are drawn per digit (0-9) using a seeded, shuffled
    per-class ordering, so every class appears exactly ``per_class`` times.
    With 10 classes the returned set has ``10 * per_class`` images.

    Args:
        x: image array ``(n, 28, 28)``.
        y: labels ``(n,)``.
        per_class: number of images per digit class.
        seed: shuffle seed (deterministic across calls).
    """
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    selected_x, selected_y = [], []
    for cls in range(10):
        idx = np.flatnonzero(y == cls)
        idx = rng.permutation(idx)
        take = idx[:per_class]
        selected_x.append(x[take])
        selected_y.append(y[take])
    x_sel = np.concatenate(selected_x, axis=0)
    y_sel = np.concatenate(selected_y, axis=0)
    order = rng.permutation(x_sel.shape[0])
    return x_sel[order], y_sel[order]
