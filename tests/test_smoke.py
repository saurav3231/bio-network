"""Smoke test: the package imports and exposes a version string."""

import bio_network


def test_package_imports() -> None:
    assert bio_network is not None


def test_version_string_exists() -> None:
    assert isinstance(bio_network.__version__, str)
    assert bio_network.__version__ != ""


def test_version_is_semantic() -> None:
    parts = bio_network.__version__.split(".")
    assert len(parts) == 3
    for part in parts:
        assert part.isdigit()