#!/usr/bin/env python3
"""Exercise replay output ownership and path-containment guards."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from replay_reference import OUTPUT_MARKER_NAME, OUTPUT_ROOT, ROOT, _prepare_output_dir


def _expect_rejected(path: Path) -> None:
    try:
        _prepare_output_dir(path)
    except ValueError:
        return
    raise AssertionError(f"unsafe replay output path was accepted: {path}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    test_root = Path(tempfile.mkdtemp(prefix="replay_output_safety_", dir=OUTPUT_ROOT))
    try:
        _expect_rejected(ROOT / "outside_output")
        _expect_rejected(OUTPUT_ROOT)

        unowned = test_root / "unowned"
        unowned.mkdir()
        sentinel = unowned / "sentinel.txt"
        sentinel.write_text("preserve\n", encoding="utf-8")
        _expect_rejected(unowned)
        if sentinel.read_text(encoding="utf-8") != "preserve\n":
            raise AssertionError("unowned output content was modified")

        owned = _prepare_output_dir(test_root / "owned")
        if not (owned / OUTPUT_MARKER_NAME).is_file():
            raise AssertionError("owned output marker was not created")
        stale = owned / "stale.txt"
        stale.write_text("remove\n", encoding="utf-8")
        if _prepare_output_dir(owned) != owned or stale.exists():
            raise AssertionError("owned replay output was not safely refreshed")

        direct_link = test_root / "direct_link"
        direct_link.symlink_to(owned, target_is_directory=True)
        _expect_rejected(direct_link)

        linked_parent = test_root / "linked_parent"
        linked_parent.symlink_to(owned, target_is_directory=True)
        _expect_rejected(linked_parent / "child")
    finally:
        shutil.rmtree(test_root)

    print("replay output safety validation passed")


if __name__ == "__main__":
    main()
