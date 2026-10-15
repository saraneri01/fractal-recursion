"""Create URM dataset views from prepared HRM and TRM data."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrm-data", type=Path, required=True)
    parser.add_argument("--trm-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    hrm = args.hrm_data.resolve(strict=True)
    trm = args.trm_data.resolve(strict=True)
    output = args.output_root.resolve()

    links = {
        output / "arc-aug-1000-trainonly" / "train": hrm / "train",
        output / "arc-aug-1000-eval-v1" / "train": hrm / "train",
        output / "arc-aug-1000-eval-v1" / "test": hrm / "test",
        output / "arc-aug-1000-eval-v1" / "identifiers.json":
            hrm / "identifiers.json",
        output / "arc-aug-1000-eval-v1" / "test_puzzles.json":
            trm / "test_puzzles.json",
    }

    # Check every source and destination before creating any links.
    for destination, source in links.items():
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)

    for destination, source in links.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        relative_source = os.path.relpath(source, destination.parent)
        destination.symlink_to(
            relative_source, target_is_directory=source.is_dir()
        )
        print(f"Created: {destination} -> {relative_source}")


if __name__ == "__main__":
    main()
