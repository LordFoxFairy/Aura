def test_package_importable_and_has_version() -> None:
    import re

    import aura

    # Don't pin the exact version — the point of this test is that the
    # package imports cleanly AND exposes a SemVer-shaped string. Pinning
    # a literal means every bump costs a useless test edit (and we have
    # hit that exact trap: v0.7.2 tripped over "0.1.0").
    assert isinstance(aura.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-]\S+)?", aura.__version__)
