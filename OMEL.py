#!/usr/bin/env python3

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = {
    "mnist": _HERE / "train_MNIST.py",
    "cifar": _HERE / "train_CIFAR.py",
    "ecny": _HERE / "train_ECNY.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OMEL unified entry: --mode selects data domain, other parameters are passed to the corresponding training script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(_SCRIPTS.keys()),
        help="mnist: Rainbow-MNIST image stream | cifar: cifar-10 image stream | ecny: ECNY transaction stream",
    )
    args, rest = parser.parse_known_args()

    target = _SCRIPTS[args.mode]
    if not target.is_file():
        raise FileNotFoundError(f"Script not found: {target}")

    sys.argv = [str(target)] + rest
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
