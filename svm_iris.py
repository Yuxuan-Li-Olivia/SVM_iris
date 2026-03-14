#!/usr/bin/env python3
"""Compatibility entrypoint.

This repo is structured as a small project under src/ (see README.md).
You can still run:

  python svm_iris.py --data data/iris.csv

Internally it forwards to iris_svm.train.main.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _bootstrap_src_on_path()
    from iris_svm.train import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
