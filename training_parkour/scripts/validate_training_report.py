"""Verify that this launch completed training and saved real checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_report(path: Path, nonce: str) -> dict:
    report = json.loads(path.read_text())
    if report.get("completed") is not True or report.get("status") != "completed":
        raise ValueError("Training did not report completion")
    if report.get("nonce") != nonce:
        raise ValueError("Training completion nonce does not match this launch")
    requested = report.get("requested_iterations")
    start = report.get("start_iteration")
    final = report.get("final_iteration")
    if (not all(type(value) is int for value in (requested, start, final)) or requested <= 0
            or start < 0 or final != start + requested - 1 or report.get("iterations_completed") != requested):
        raise ValueError("Training iteration progress does not match the requested updates")
    started_ns, finished_ns = report.get("started_ns"), report.get("finished_ns")
    if type(started_ns) is not int or type(finished_ns) is not int or finished_ns < started_ns:
        raise ValueError("Invalid training completion timestamps")
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("Training reported no checkpoint")
    for value in checkpoints:
        checkpoint = Path(value["path"])
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError(f"Missing or empty checkpoint: {checkpoint}")
        stat = checkpoint.stat()
        if (stat.st_size != value.get("size") or stat.st_mtime_ns != value.get("mtime_ns")
                or not started_ns <= stat.st_mtime_ns <= finished_ns):
            raise ValueError(f"Checkpoint was not saved during this launch: {checkpoint}")
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != value.get("sha256"):
            raise ValueError(f"Checkpoint hash differs from the completion report: {checkpoint}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("nonce")
    args = parser.parse_args()
    report = validate_report(args.report, args.nonce)
    print("训练完成，日志与 checkpoint：" + report["log_dir"])
