"""Create simulator slot records offline; never download unrelated datasets."""
import argparse
import hashlib
import json
from pathlib import Path


def rows(dataset, split, count):
    for index in range(count):
        item = {"data_source": dataset, "prompt": [{"role": "user", "content": ""}],
                "ability": "agent", "extra_info": {"split": split, "index": index}}
        if split != "train":
            field = "game_index" if dataset == "alfworld" else "goal_index"
            item["env_kwargs"] = {field: index}
        yield item


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["alfworld", "webshop"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a new data directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {"train": 128, "monitor": 140 if args.dataset == "alfworld" else 128,
              "test": 140 if args.dataset == "alfworld" else 500}
    manifest = {"dataset": args.dataset, "env_seed": 0, "slots": counts, "files": {}}
    for split, count in counts.items():
        path = args.output / f"{split}.parquet"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        pq.write_table(pa.Table.from_pylist(list(rows(args.dataset, split, count))), path)
        manifest["files"][split] = {"rows": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
