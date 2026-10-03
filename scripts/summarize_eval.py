"""Summarize three completed endpoint runs; never confuse smoke with accuracy."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(paths):
    records = []
    for path in paths:
        launch = json.loads((path / "launch.json").read_text())
        status = json.loads((path / "exit.json").read_text())
        lines = (path / "validation.jsonl").read_text().splitlines()
        if status["returncode"] != 0 or launch["mode"] != "eval" or len(lines) != 1:
            raise ValueError(f"Not a completed endpoint evaluation: {path}")
        row = json.loads(lines[0])
        expected = 140 if launch["dataset"] == "alfworld" else 500
        if row["step"] != 150 or row["metrics"]["val/evaluated_cases"] != expected:
            raise ValueError(f"Expected latest step150 and {expected} cases: {path}")
        records.append({"launch": launch, "row": row, "inputs": json.loads((path / "inputs.json").read_text())})
    if sorted(item["launch"]["decode_seed"] for item in records) != [123, 456, 789]:
        raise ValueError("Require exactly decoding seeds 123, 456, 789")
    signatures = {(item["launch"]["dataset"], str(Path(item["launch"]["resume"]).resolve()),
                   item["inputs"]["validation_data"]["sha256"], item["inputs"]["model_config"]["sha256"]) for item in records}
    if len(signatures) != 1:
        raise ValueError("Use the same dataset, checkpoint, model configuration and task manifest for all seeds")
    weight_signatures = {json.dumps(item["inputs"]["actor_weights"], sort_keys=True) for item in records}
    if len(weight_signatures) != 1 or not records[0]["inputs"]["actor_weights"]:
        raise ValueError("Actor weights changed between evaluations")
    common_keys = set.intersection(*(set(item["row"]["metrics"]) for item in records))
    summary = {}
    for key in sorted(common_keys):
        if "success_rate" not in key:
            continue
        values = [item["row"]["metrics"][key] for item in records]
        summary[key] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "per_seed": {
            str(item["launch"]["decode_seed"]): item["row"]["metrics"][key] for item in records}}
    return {"dataset": records[0]["launch"]["dataset"], "checkpoint_step": 150, "decoding_seeds": [123, 456, 789],
            "std_ddof": 1, "evaluated_cases_per_seed": expected, "metrics": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs=3)
    args = parser.parse_args()
    print(json.dumps(summarize(args.runs), indent=2))
