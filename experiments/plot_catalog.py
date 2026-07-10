"""Read the frozen legacy plotting-function catalog without importing its code.

This module is the supported catalog entry point for the Phase 8 retirement of
``experiments.dfr_plot``. It intentionally depends only on the checked-in
Markdown catalog, so researchers can inspect historical function names after
the legacy plotting implementation is removed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


CATALOG_PATH = Path(__file__).with_name("DFR_PLOT_CATALOG.md")
_SUPPORT_HEADING = "### Supported compatibility wrappers"
_ARCHIVE_HEADING = "### Archive-only public functions"
_OPEN_QUESTIONS_HEADING = "## Open questions for the owner"


def legacy_function_names(catalog_path: Path = CATALOG_PATH) -> tuple[str, ...]:
    """Return the sorted public function names preserved in the frozen catalog.

    Args:
        catalog_path: Markdown catalog to read. The argument primarily supports
            validation tests and does not require importing legacy plot code.

    Raises:
        ValueError: If the catalog does not contain both public-function policy
            sections or lists a function name more than once.
    """
    text = catalog_path.read_text(encoding="utf-8")
    try:
        support_section = text.split(_SUPPORT_HEADING, 1)[1].split(
            _ARCHIVE_HEADING, 1
        )[0]
        archive_section = text.split(_ARCHIVE_HEADING, 1)[1].split(
            _OPEN_QUESTIONS_HEADING, 1
        )[0]
    except IndexError as error:
        raise ValueError(
            f"Catalog {catalog_path} is missing its public-function policy sections."
        ) from error

    names = re.findall(
        r"^- `([A-Za-z0-9_]+)`", support_section + archive_section, re.MULTILINE
    )
    if not names:
        raise ValueError(f"Catalog {catalog_path} does not list public functions.")
    if len(names) != len(set(names)):
        raise ValueError(f"Catalog {catalog_path} lists a public function more than once.")
    return tuple(sorted(names))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone legacy-plot catalog command."""
    parser = argparse.ArgumentParser(
        description="Inspect the frozen Phase 6 legacy plotting-function catalog."
    )
    parser.add_argument(
        "--list-functions",
        action="store_true",
        help="Print the public legacy function names retained in the catalog.",
    )
    parser.add_argument(
        "--show-path",
        action="store_true",
        help="Print the path to the full Markdown catalog.",
    )
    args = parser.parse_args(argv)

    if args.list_functions:
        print("\n".join(legacy_function_names()))
    if args.show_path:
        print(CATALOG_PATH)
    if not args.list_functions and not args.show_path:
        parser.error("Choose --list-functions or --show-path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
