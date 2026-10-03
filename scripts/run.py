"""Resolve a release profile and launch training without implicit path defaults."""
import argparse
import json
import os
from pathlib import Path
import hashlib
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def validate_args(args):
    if args.gpus not in (2, 4, 8):
        raise ValueError("Supported GPU counts are 2, 4, and 8 (rollout TP=2)")
    mini = 256 if args.dataset == "alfworld" else 64
    for micro in (args.actor_micro, args.critic_micro):
        if micro <= 0 or (mini // args.gpus) % micro:
            raise ValueError("Micro-batch must divide the per-GPU mini-batch")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.val_batch is not None and args.val_batch <= 0:
        raise ValueError("--val-batch must be positive")
    if args.mode == "eval" and args.val_batch is not None:
        expected = 140 if args.dataset == "alfworld" else 100
        if args.val_batch != expected:
            raise ValueError(f"Endpoint evaluation fixes --val-batch to {expected}")


def build_config(args):
    from omegaconf import OmegaConf

    validate_args(args)
    config = OmegaConf.merge(
        OmegaConf.load(ROOT / "verl/trainer/config/ppo_trainer.yaml"),
        OmegaConf.load(ROOT / "configs/release/common.yaml"),
        OmegaConf.load(ROOT / "configs/release" / f"{args.dataset}.yaml"),
    )
    config.actor_rollout_ref.model.path = str(args.model.resolve())
    config.critic.model.path = str(args.model.resolve())
    config.data.train_files = str(args.data.resolve() / "train.parquet")
    split = "test" if args.mode == "eval" else "monitor"
    config.data.val_files = str(args.data.resolve() / f"{split}.parquet")
    config.trainer.experiment_name = args.name
    config.trainer.validation_metrics_path = str(args.output.resolve() / "validation.jsonl")
    config.trainer.default_local_dir = str(args.checkpoints.resolve())
    config.trainer.n_gpus_per_node = args.gpus
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = args.actor_micro
    config.critic.ppo_micro_batch_size_per_gpu = args.critic_micro
    config.critic.forward_micro_batch_size_per_gpu = args.critic_micro
    if args.method != "vetta":
        config.critic.model.num_value_heads = 1
        config.critic.model.override_config.num_hidden_layers = 28
        if args.method == "ppo_star":
            config.algorithm.adv_estimator = "sao_skip_observation"
        elif args.method == "turn_ppo":
            config.algorithm.adv_estimator = "turn_ppo"
            config.algorithm.gamma = config.algorithm.lam = 0.95
            config.actor_rollout_ref.actor.policy_loss.loss_mode = "turn_ppo"
        elif args.method == "gigpo":
            config.algorithm.adv_estimator = "gigpo"
            config.algorithm.gamma = 0.95
            config.data.train_batch_size = 16
            config.env.rollout.n = 8
    if args.mode == "smoke":
        config.data.train_batch_size = args.gpus
        config.env.rollout.n = 1
        config.data.val_batch_size = args.gpus
        config.actor_rollout_ref.actor.ppo_mini_batch_size = args.gpus
        config.critic.ppo_mini_batch_size = args.gpus
        config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
        config.critic.ppo_micro_batch_size_per_gpu = 1
        config.critic.forward_micro_batch_size_per_gpu = 1
        config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
        config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 1
        config.trainer.total_epochs = 2
        config.trainer.total_training_steps = 2
        config.trainer.val_before_train = False
        config.trainer.test_freq = -1
        config.trainer.save_freq = 1
        # Same GAE/alpha/gamma; horizon is deliberately shorter only for a smoke.
        config.env.max_steps = 3
    if args.resume is not None:
        resume = args.resume.resolve()
        if resume.name == "latest":
            if args.checkpoints.resolve() != resume.parent:
                raise ValueError("For latest, --checkpoints must be its parent directory")
            config.trainer.resume_mode = "auto"
        elif re.fullmatch(r"global_step_\d+", resume.name):
            config.trainer.resume_mode = "resume_path"
            config.trainer.resume_from_path = str(resume)
        else:
            raise ValueError("--resume must name latest or global_step_<N>, not best")
    if args.mode == "eval":
        if args.resume is None or args.decode_seed is None:
            raise ValueError("eval requires --resume and --decode-seed")
        # A fixed trained Actor needs no Critic for inference. Avoid loading or
        # allocating a new Critic whose RNG could contaminate the comparison.
        config.algorithm.adv_estimator = "gigpo"
        config.env.rollout.n = 1
        config.trainer.val_only = True
        config.trainer.val_before_train = True
        config.trainer.test_freq = config.trainer.save_freq = -1
        config.data.val_batch_size = 140 if args.dataset == "alfworld" else 100
        config.actor_rollout_ref.rollout.seed = args.decode_seed
        config.trainer.monitor_validation_size = 140 if args.dataset == "alfworld" else 500
    if args.steps is not None:
        config.trainer.total_training_steps = args.steps
    if args.val_batch is not None:
        config.data.val_batch_size = args.val_batch
    config.ray_init._temp_dir = str(args.ray_dir.resolve())
    return config


def validate_inputs(args, config):
    import pyarrow.parquet as pq

    for required in [args.model / "config.json", Path(config.data.train_files), Path(config.data.val_files)]:
        if not required.is_file():
            raise FileNotFoundError(required)
    model_config = json.loads((args.model / "config.json").read_text())
    if (model_config.get("model_type") != "qwen2" or model_config.get("num_hidden_layers") != 28
            or model_config.get("hidden_size") != 1536):
        raise ValueError("These release profiles target Qwen2.5-1.5B-Instruct (28 text layers)")
    if args.mode == "eval":
        key = "game_index" if args.dataset == "alfworld" else "goal_index"
        expected = 140 if args.dataset == "alfworld" else 500
        records = pq.read_table(config.data.val_files).to_pylist()
        indices = [record.get("env_kwargs", {}).get(key) for record in records]
        if indices != list(range(expected)):
            raise ValueError(f"Evaluation manifest must be exactly ordered {key}=0..{expected - 1}")
    if args.resume is not None:
        if not (args.resume / "actor").is_dir():
            raise FileNotFoundError(args.resume / "actor")
        if args.mode != "eval" and args.method != "gigpo" and not (args.resume / "critic").is_dir():
            raise FileNotFoundError(args.resume / "critic")
        if args.resume.name == "latest":
            tracker = args.resume.parent / "latest_checkpointed_iteration.txt"
            if not tracker.is_file() or not tracker.read_text().strip().isdigit():
                raise ValueError("latest requires a valid latest_checkpointed_iteration.txt; never silently restart")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=["alfworld", "webshop"])
    p.add_argument("mode", choices=["config", "smoke", "train", "eval"])
    p.add_argument("--method", choices=["vetta", "ppo_star", "turn_ppo", "gigpo"], default="vetta")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--name", default="vetta-release")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--actor-micro", type=int, default=8)
    p.add_argument("--critic-micro", type=int, default=8)
    p.add_argument("--steps", type=int)
    p.add_argument("--val-batch", type=int)
    p.add_argument("--resume", type=Path)
    p.add_argument("--decode-seed", type=int)
    p.add_argument("--ray-dir", type=Path, default=Path("/tmp/vetta") / uuid.uuid4().hex[:8])
    return p


def main():
    from omegaconf import OmegaConf

    args = parser().parse_args()
    config = build_config(args)
    if args.mode == "config":
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    validate_inputs(args, config)
    if len(str(args.ray_dir)) > 45:
        raise ValueError("Keep --ray-dir short to avoid Ray Unix socket limits")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {args.output}")
    if args.mode != "eval" and args.resume is None and args.checkpoints.exists() and any(args.checkpoints.iterdir()):
        raise FileExistsError(f"Refusing to overwrite checkpoint directory: {args.checkpoints}")
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("GPU processes exist; do not overlap this job with other workloads")
    args.output.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    args.ray_dir.mkdir(parents=True, exist_ok=True)
    path = args.output.resolve() / "resolved.yaml"
    OmegaConf.save(config, path, resolve=True)
    (args.output / "launch.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    inputs = {}
    for name, filename in {"model_config": args.model / "config.json", "train_data": Path(config.data.train_files),
                           "validation_data": Path(config.data.val_files)}.items():
        inputs[name] = {"path": str(filename.resolve()), "sha256": file_sha256(filename)}
    if args.resume is not None:
        actor_files = sorted(path for path in (args.resume / "actor").rglob("*") if path.is_file()
                             and (path.name.startswith("model_world_size_") or path.name.endswith(".safetensors")))
        if not actor_files:
            raise ValueError("No Actor weight files found in the checkpoint")
        inputs["actor_weights"] = {str(path.relative_to(args.resume)): file_sha256(path) for path in actor_files}
    (args.output / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["VERL_ALLOW_MODEL_ONLY_CHECKPOINT"] = "1" if args.mode == "eval" else "0"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(key, "1")
    if args.dataset == "webshop":
        env.setdefault("JAVA_TOOL_OPTIONS", "-Xms32m -Xmx512m -XX:ActiveProcessorCount=1 -XX:+UseSerialGC")
    command = [sys.executable, "-m", "verl.trainer.main_ppo",
               "--config-path", str(path.parent), "--config-name", path.stem]
    print("Launching", command, flush=True)
    with (args.output / "console.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (args.output / "exit.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
    print("Exit", result.returncode, "log", args.output / "console.log", flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
