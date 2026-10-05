"""Build the private team bundle from this repository; no network or GPU needed."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ["README.md", "COLAB_GUIDE.md", "TEAM_GUIDE.md", "ETHIO-SPEECH-AI-DOC.md", "pyproject.toml", ".gitignore"]
INVENTORY = {
    "src/amharic_s2s": ["*.py", "web/*", "web/vendor/*"],
    "configs": ["*.json"],
    "requirements": ["*.txt"],
    "notebooks": ["ethio_speech_ai_colab.ipynb"],
    "docs": ["bank-demo-test-cases.md"],
    "data/pipeline_public": ["questions_v2.json"],
    "data/pipeline_v2_audio": ["MANIFEST.json", "everyday.wav", "fee.wav", "unknown.wav"],
    "knowledge/bank_v2": ["bank_index.json", "source_audit.json", "build_report.json", "retrieval_preview.json"],
    "tests": ["test_*.py", "test_*.mjs"],
    "scripts": ["package_colab.py"],
}


def inventory():
    paths = {ROOT / name for name in ROOT_FILES}
    for folder, patterns in INVENTORY.items():
        for pattern in patterns:
            matches = {p for p in (ROOT / folder).glob(pattern) if p.is_file()}
            if not matches:
                raise RuntimeError(f"Missing required files: {folder}/{pattern}")
            paths.update(matches)
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"Missing or external bundle file: {path}")
    return sorted(paths)


def validate(payload):
    package = "src/amharic_s2s/"
    for name, content in payload.items():
        if name.endswith(".py"):
            tree = ast.parse(content, filename=name)
            if name.startswith(package):
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                        needed = package + node.module.split(".")[0] + ".py"
                        if needed not in payload:
                            raise RuntimeError(f"{name} requires missing {needed}")
    nb = json.loads(payload["notebooks/ethio_speech_ai_colab.ipynb"])
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 7, "Expected seven numbered code cells"
    for c in cells:
        ast.parse("".join(c["source"]))
        assert not c["outputs"] and c["execution_count"] is None, "Clear notebook outputs before packaging"
    for name, expected in json.loads(payload["data/pipeline_v2_audio/MANIFEST.json"]).items():
        assert hashlib.sha256(payload["data/pipeline_v2_audio/" + name]).hexdigest() == expected, name


def main():
    payload = {str(p.relative_to(ROOT)): p.read_bytes() for p in inventory()}
    validate(payload)
    manifest = {n: hashlib.sha256(v).hexdigest() for n, v in payload.items()}
    payload["BUNDLE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    target = ROOT / "dist/ethio-speech-ai-colab.zip"
    target.parent.mkdir(exist_ok=True)
    # Fixed metadata makes rebuilding unchanged files reproducible.
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in sorted(payload.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 5, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            z.writestr(entry, content)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(payload)
        for name, expected in json.loads(z.read("BUNDLE_MANIFEST.json")).items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".zip.sha256").write_text(f"{checksum}  {target.name}\n")
    print(f"Verified {len(payload)} files: {target}")
    print(f"Size: {target.stat().st_size / 2**20:.2f} MiB; SHA-256: {checksum}")


if __name__ == "__main__":
    main()
