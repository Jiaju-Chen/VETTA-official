"""Create a deterministic source ZIP with no Git history or runtime artifacts."""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                 "wandb", "outputs", "checkpoints", "hf_cache", "ray_results", ".venv"}
EXCLUDED_PATTERNS = ("*.pyc", "*.bak*", "*.log", "*.pt", "*.ckpt", "*.safetensors", "*.parquet",
                     "*.wandb", ".DS_Store", ".env", ".env.*", "*.pem", "*.key")


def source_files(root):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDED_PATTERNS):
            continue
        if path.is_symlink():
            raise ValueError(f"Do not package asset symlinks: {relative}")
        if path.is_file():
            yield path, relative


def package(root, output):
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Place the ZIP outside the source directory")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in source_files(root):
            data = path.read_bytes()
            name = str(Path("VETTA-official") / relative)
            hashes[str(relative)] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    manifest = {"archive": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "file_count": len(hashes), "files": hashes}
    manifest_path = output.with_suffix(".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package(ROOT, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
