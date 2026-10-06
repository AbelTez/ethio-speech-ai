"""Portable manifests and single-writer experiment records; no ML dependencies."""

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from uuid import uuid4
import zipfile


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(f.read()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Replace a whole file only after a complete write (one writer per directory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def command_output(args, cwd=None):
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    packages = {}
    for name in ("torch", "torchaudio", "transformers", "accelerate", "bitsandbytes", "omnivoice", "datasets", "numpy", "soundfile", "jiwer", "scipy", "tokenizers", "huggingface-hub", "safetensors", "sentencepiece"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages,
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    }


def source_snapshot(repo):
    repo = Path(repo)
    files = [repo / "pyproject.toml"]
    for directory in ("src", "scripts"):
        files.extend((repo / directory).rglob("*.py"))
    hashes = {str(p.relative_to(repo)): digest(p) for p in sorted(files) if p.is_file()}
    commit = command_output(["git", "rev-parse", "HEAD"], repo) if (repo / ".git").exists() else None
    if commit is None and (repo / "SOURCE.json").is_file():
        commit = read_json(repo / "SOURCE.json").get("git_commit")
    return {"git_commit": commit,
            "source_sha256": object_digest(hashes), "files": hashes}


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise ValueError("Identifiers must be 1–100 ASCII letters, numbers, underscores, dots or hyphens.")
    return value


def validate_manifest(manifest, data_root):
    """Audio paths resolve against data_root, never against old Colab absolute paths."""
    root = Path(data_root).resolve()
    ids = set()
    for sample in manifest["samples"]:
        sample_id = safe_name(sample["id"])
        if sample_id in ids:
            raise ValueError(f"Duplicate sample: {sample_id}")
        ids.add(sample_id)
        path = (root / sample["audio_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Audio outside data root: {path}")
        if digest(path) != sample["audio_sha256"]:
            raise ValueError(f"Audio checksum mismatch: {sample_id}")
    if not ids:
        raise ValueError("Manifest must contain samples.")
    return ids


def prepare_run(repo, config_path, manifest_path, data_root, output_root, owner, resume=None):
    config = read_json(config_path)
    manifest = read_json(manifest_path)
    validate_manifest(manifest, data_root)
    safe_name(owner)
    repeats = config.get("planned_repeats")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("planned_repeats must be a positive integer.")
    source = source_snapshot(repo)
    env = environment()
    if config.get("require_gpu") and not env["gpu"]:
        raise RuntimeError("This configuration requires an NVIDIA GPU.")
    identity = {"config": object_digest(config), "dataset": object_digest(manifest),
                "source": source["source_sha256"], "environment": object_digest(env)}
    if resume:
        directory = Path(resume)
        previous = read_json(directory / "run.json")
        if previous["identity"] != identity:
            changed = [k for k in identity if previous["identity"].get(k) != identity[k]]
            raise ValueError(f"Cannot resume: changed {', '.join(changed)}. Create a new run for this environment/configuration.")
        if previous["owner"] != owner:
            raise ValueError("Cannot resume a run owned by another user. Create a new run with your own owner name.")
    else:
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{owner}-{uuid4().hex[:8]}"
        directory = Path(output_root) / run_id
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "config.json", config)
        write_json(directory / "dataset.json", manifest)
        write_json(directory / "run.json", {"schema_version": 1, "run_id": run_id,
                   "owner": owner, "created_at": now(), "identity": identity,
                   "source": source, "environment": env, "stage": config["stage"]})
    write_json(directory / "sessions" / f"{uuid4().hex}.json", {"started_at": now(), "environment": env})
    return directory


def save_result(directory, sample_id, repeat, metrics, status="success", error=None, details=None):
    """Write after measured inference. Failures do not mark a sample complete."""
    directory = Path(directory)
    safe_name(sample_id)
    config = read_json(directory / "config.json")
    ids = {s["id"] for s in read_json(directory / "dataset.json")["samples"]}
    if sample_id not in ids or type(repeat) is not int or not 1 <= repeat <= config["planned_repeats"]:
        raise ValueError("Unknown sample or repeat outside configured range.")
    if status not in ("success", "failed"):
        raise ValueError("status must be success or failed.")
    if not isinstance(metrics, dict) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in metrics.values()):
        raise ValueError("Metrics must contain finite nonnegative numbers; include units in metric names.")
    if status == "success" and not metrics:
        raise ValueError("Success requires measured metrics.")
    if status == "failed" and not error:
        raise ValueError("Failure requires an error description.")
    destination = directory / "results" / f"{sample_id}__{repeat}.json"
    if destination.exists() and read_json(destination)["status"] == "success":
        raise FileExistsError(f"Successful result already exists: {destination}")
    record = {"schema_version": 1, "sample_id": sample_id, "repeat": repeat,
              "status": status, "metrics": metrics, "error": error, "saved_at": now()}
    if details is not None:
        record["details"] = details
    if status == "failed":
        write_json(directory / "failures" / f"{sample_id}__{repeat}__{uuid4().hex}.json", record)
    write_json(destination, record)


def completed_keys(directory):
    return {(r["sample_id"], r["repeat"]) for p in (Path(directory) / "results").glob("*.json")
            if (r := read_json(p))["status"] == "success"}


def export_run(directory, destination):
    directory = Path(directory).resolve()
    destination = Path(destination).resolve()
    if destination.is_relative_to(directory):
        raise ValueError("Write the archive outside the run directory.")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Refusing to export symlink: {path}")
            if path.is_file() and not path.name.startswith(".pending-"):
                archive.write(path, path.relative_to(directory.parent))
    return destination
