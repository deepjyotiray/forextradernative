from __future__ import annotations

import argparse
from pathlib import Path

from engine.debug_bundle import build_debug_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a portable runtime debug bundle.")
    parser.add_argument(
        "--recent-days",
        type=int,
        default=7,
        help="How many recent days of rotated logs to include (default: 7).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    bundle = build_debug_bundle(base_dir, recent_days=max(1, int(args.recent_days)))
    print(f"Created {bundle['bundle_name']}")
    print(bundle["bundle_path"])
    print(f"Size: {bundle['bundle_size_bytes']} bytes")


if __name__ == "__main__":
    main()
