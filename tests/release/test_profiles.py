import importlib.util
from pathlib import Path

from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run")
data = load("prepare_data")


def args(dataset="alfworld", mode="train", *extra):
    return runner.parser().parse_args([dataset, mode, "--model", "/tmp/model", "--data", "/tmp/data",
                                       "--output", "/tmp/output", "--checkpoints", "/tmp/checkpoints", *extra])


@pytest.mark.parametrize("dataset,alpha,mini,horizon", [("alfworld", 1, 256, 50), ("webshop", 3, 64, 15)])
def test_best_observed_scientific_profiles(dataset, alpha, mini, horizon):
    config = runner.build_config(args(dataset))
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert config.algorithm.adv_estimator == "luna_unified"
    assert config.algorithm.gamma == config.algorithm.lam == 1
    assert config.algorithm.hybrid_advantage.turn_gamma == config.algorithm.hybrid_advantage.turn_lam == 0.95
    assert config.algorithm.hybrid_advantage.token_residual_scale == alpha
    assert config.algorithm.hybrid_advantage.whiten_advantages
    assert config.critic.model.num_value_heads == config.critic.model.override_config.num_hidden_layers == 2
    assert config.env.seed == 0 and config.env.rollout.n == 1 and config.env.max_steps == horizon
    assert config.data.train_batch_size == 128
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == config.critic.ppo_mini_batch_size == mini
    assert config.trainer.total_training_steps == 150 and config.trainer.mask_batch_padding
    assert config.trainer.del_local_ckpt_after_load is False


@pytest.mark.parametrize("dataset,count,key", [("alfworld", 140, "game_index"), ("webshop", 500, "goal_index")])
def test_endpoint_tasks_are_indexed_once(dataset, count, key):
    records = list(data.rows(dataset, "test", count))
    assert [row["env_kwargs"][key] for row in records] == list(range(count))
    assert all("env_kwargs" not in row for row in data.rows(dataset, "train", 128))


@pytest.mark.parametrize("dataset,batch", [("alfworld", 140), ("webshop", 100)])
@pytest.mark.parametrize("seed", [123, 456, 789])
def test_evaluation_is_actor_only_with_explicit_decode_seed(dataset, batch, seed):
    config = runner.build_config(args(dataset, "eval", "--resume", "/tmp/checkpoints/latest", "--decode-seed", str(seed)))
    assert config.trainer.resume_mode == "auto"
    assert config.trainer.val_only and config.trainer.val_before_train
    assert config.algorithm.adv_estimator == "gigpo"
    assert config.actor_rollout_ref.rollout.seed == seed and config.env.seed == 0
    assert config.data.val_batch_size == batch
    assert config.trainer.del_local_ckpt_after_load is False


def test_explicit_global_step_resume():
    config = runner.build_config(args("alfworld", "train", "--resume", "/tmp/checkpoints/global_step_25"))
    assert config.trainer.resume_mode == "resume_path"
    assert config.trainer.resume_from_path.endswith("global_step_25")


@pytest.mark.parametrize("method,estimator,heads,layers", [("ppo_star", "sao_skip_observation", 1, 28),
                                                          ("turn_ppo", "turn_ppo", 1, 28), ("gigpo", "gigpo", 1, 28)])
def test_baseline_architectures(method, estimator, heads, layers):
    config = runner.build_config(args("webshop", "train", "--method", method))
    assert config.algorithm.adv_estimator == estimator
    assert config.critic.model.num_value_heads == heads and config.critic.model.override_config.num_hidden_layers == layers
    if method == "turn_ppo":
        assert config.actor_rollout_ref.actor.policy_loss.loss_mode == "turn_ppo"
    if method == "gigpo":
        assert config.data.train_batch_size * config.env.rollout.n == 128


def test_smoke_is_not_full_protocol():
    config = runner.build_config(args("webshop", "smoke"))
    assert config.trainer.total_training_steps == 2 and config.env.max_steps == 3
    assert config.trainer.save_freq == 1 and not config.trainer.val_before_train
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 8


@pytest.mark.parametrize("extra", [["--actor-micro", "3"], ["--gpus", "1"], ["--steps", "0"]])
def test_invalid_resource_settings_fail_early(extra):
    with pytest.raises(ValueError):
        runner.build_config(args("webshop", "train", *extra))


def test_eval_cannot_silently_change_tasks_or_use_best():
    with pytest.raises(ValueError):
        runner.build_config(args("webshop", "eval", "--resume", "/tmp/checkpoints/latest", "--decode-seed", "123", "--val-batch", "128"))
    with pytest.raises(ValueError):
        runner.build_config(args("alfworld", "eval", "--resume", "/tmp/checkpoints/best", "--decode-seed", "123"))
    with pytest.raises(ValueError):
        runner.build_config(args("alfworld", "eval"))


def test_archive_excludes_history_artifacts_and_is_deterministic(tmp_path):
    packager = load("package_release")
    root = tmp_path / "source"
    root.mkdir()
    (root / "README.md").write_text("VETTA\n")
    (root / "hidden.py.bak").write_text("private backup")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("private remote")
    first = packager.package(root, tmp_path / "first.zip")
    second = packager.package(root, tmp_path / "second.zip")
    assert first["sha256"] == second["sha256"]
    assert first["files"].keys() == {"README.md"}
