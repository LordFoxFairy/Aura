def test_package_importable_and_has_version() -> None:
    import aura

    assert aura.__version__ == "0.1.0"
