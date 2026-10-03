import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset,count,key", [("alfworld", 140, "game_index"), ("webshop", 500, "goal_index")])
def test_real_parquet_generation_and_no_overwrite(tmp_path, dataset, count, key):
    output = tmp_path / dataset
    command = [sys.executable, str(ROOT / "scripts/prepare_data.py"), dataset, "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    assert pq.read_table(output / "train.parquet").num_rows == 128
    records = pq.read_table(output / "test.parquet").to_pylist()
    assert [record["env_kwargs"][key] for record in records] == list(range(count))
    before = (output / "test.parquet").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (output / "test.parquet").read_bytes() == before


def make_runs(tmp_path):
    result = []
    for seed, score in [(123, 0.72), (456, 0.74), (789, 0.76)]:
        path = tmp_path / str(seed)
        path.mkdir()
        (path / "launch.json").write_text(json.dumps({"mode": "eval", "dataset": "webshop",
                                                     "resume": "/tmp/ckpts/latest", "decode_seed": seed}))
        (path / "exit.json").write_text(json.dumps({"returncode": 0}))
        (path / "validation.jsonl").write_text(json.dumps({"step": 150, "metrics": {
            "val/evaluated_cases": 500, "val/success_rate": score,
            "val/gigpo_compatible/webshop_task_score (not success_rate)": 0.85}}) + "\n")
        (path / "inputs.json").write_text(json.dumps({"validation_data": {"sha256": "tasks"},
                                                      "model_config": {"sha256": "config"},
                                                      "actor_weights": {"model.pt": "weights"}}))
        result.append(path)
    return result


def test_summary_sample_standard_deviation(tmp_path):
    summary = load("summarize_eval").summarize(make_runs(tmp_path))
    result = summary["metrics"]["val/success_rate"]
    assert result["mean"] == pytest.approx(0.74)
    assert result["sample_std"] == pytest.approx(0.02)


@pytest.mark.parametrize("file,field,value", [("exit.json", "returncode", 1), ("launch.json", "mode", "smoke"),
                                             ("launch.json", "decode_seed", 0), ("validation.jsonl", "step", 145)])
def test_summary_rejects_invalid_results(tmp_path, file, field, value):
    runs = make_runs(tmp_path)
    path = runs[0] / file
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        load("summarize_eval").summarize(runs)


def test_evaluation_rejects_duplicate_goal_ids(tmp_path):
    runner = load("run")
    model, data, ckpts = (tmp_path / part for part in ("model", "data", "ckpts"))
    for directory in (model, data, ckpts / "latest" / "actor"):
        directory.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 1536}))
    (ckpts / "latest_checkpointed_iteration.txt").write_text("150")
    pq.write_table(pa.Table.from_pylist(list(load("prepare_data").rows("webshop", "train", 128))), data / "train.parquet")
    records = list(load("prepare_data").rows("webshop", "test", 500))
    records[-1]["env_kwargs"]["goal_index"] = 0
    pq.write_table(pa.Table.from_pylist(records), data / "test.parquet")
    args = runner.parser().parse_args(["webshop", "eval", "--model", str(model), "--data", str(data),
                                       "--output", str(tmp_path / "out"), "--checkpoints", str(ckpts),
                                       "--resume", str(ckpts / "latest"), "--decode-seed", "123"])
    with pytest.raises(ValueError, match="exactly ordered"):
        runner.validate_inputs(args, runner.build_config(args))
