"""Apply the two dependency metadata fixes to the exact supported Isaac Lab commit.

No simulation, task, reward, or third-party licence files are changed. This is
idempotent, refuses other revisions, and always runs git apply --check first.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

COMMIT = "28a37cecdd433c22d9eabd6a5954add9f13a8951"
PATCH = Path(__file__).resolve().parents[1] / "patches/isaaclab-sim6-metadata.patch"


def prepare(lab: Path) -> str:
    lab = lab.resolve()
    actual = subprocess.check_output(["git", "-C", str(lab), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError(f"Refusing to patch Isaac Lab {actual}; expected {COMMIT}")
    command = ["git", "-C", str(lab), "apply"]
    if subprocess.run(command + ["--reverse", "--check", str(PATCH)], capture_output=True).returncode == 0:
        return "already applied"
    subprocess.run(command + ["--check", str(PATCH)], check=True)
    subprocess.run(command + [str(PATCH)], check=True)
    return "applied"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isaaclab", type=Path, help="Path to the pinned Isaac Lab source checkout")
    args = parser.parse_args()
    print(f"Isaac Lab {COMMIT}: metadata patch {prepare(args.isaaclab)}")


if __name__ == "__main__":
    main()
