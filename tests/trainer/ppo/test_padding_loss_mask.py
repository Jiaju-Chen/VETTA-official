from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from verl import DataProto
from agent_system.multi_turn_rollout.utils import adjust_batch
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_turn_ppo
from verl.trainer.ppo.padding import apply_row_mask, loss_denominator, normalized_loss


def test_adjust_batch_marks_only_copies_without_masking_model_input():
    config = NS(
        trainer=NS(n_gpus_per_node=2, nnodes=1, mask_batch_padding=True),
        algorithm=NS(use_kl_in_reward=False),
        actor_rollout_ref=NS(rollout=NS(log_prob_micro_batch_size_per_gpu=1),
                            actor=NS(use_kl_loss=False, ppo_micro_batch_size_per_gpu=1)),
    )
    data = DataProto.from_dict(
        tensors={"input_ids": torch.ones(3, 2, dtype=torch.long),
                 "attention_mask": torch.ones(3, 2, dtype=torch.long)},
        non_tensors={"traj_uid": np.array(["a", "a", "a"], dtype=object),
                     "step_id": np.array([0, 1, 2])},
    )
    padded = adjust_batch(config, data)
    assert padded.batch["sample_valid_mask"].tolist() == [True, True, True, False]
    assert padded.batch["attention_mask"].sum() == 8
    assert apply_row_mask(padded.batch["attention_mask"], padded.batch).sum() == 6
    again = adjust_batch(config, padded)
    assert again.batch["sample_valid_mask"].tolist() == [True, True, True, False]


@pytest.mark.parametrize("mode", ["token-mean", "seq-mean-token-mean",
                                  "seq-mean-token-sum", "seq-mean-token-sum-norm"])
def test_partitioned_masked_loss_and_gradient_match_unpadded_minibatch(mode):
    reference = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], requires_grad=True)
    mask = torch.tensor([[1., 1., 0.], [1., 0., 0.], [1., 1., 1.]])
    expected = agg_loss(reference.square(), mask, mode)
    expected.backward()
    padded = reference.detach()[[0, 1, 2, 0, 2]].clone().requires_grad_()
    padded_mask = torch.cat([mask, torch.zeros(2, 3)])
    denominator = loss_denominator(padded_mask, mode)
    actual = sum(normalized_loss(agg_loss(padded[part].square(), padded_mask[part], mode),
                                 padded_mask[part], mode, denominator, world_size=1)
                 for part in [slice(0, 1), slice(1, 3), slice(3, 5)])
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(padded.grad[:3], reference.grad)
    assert torch.count_nonzero(padded.grad[3:]) == 0


def test_turn_ppo_all_padding_microbatch_has_finite_zero_gradient():
    log_prob = torch.ones(2, 3, requires_grad=True)
    loss, *metrics = compute_policy_loss_turn_ppo(
        torch.zeros_like(log_prob), log_prob, torch.ones_like(log_prob),
        torch.zeros_like(log_prob), cliprange=0.2,
    )
    loss.backward()
    assert loss.item() == 0
    assert all(torch.isfinite(metric) for metric in metrics)
    assert torch.count_nonzero(log_prob.grad) == 0


@pytest.mark.parametrize("method", ["gae", "sao_skip_observation", "turn_ppo", "luna_unified", "grpo"])
def test_trainer_advantages_ignore_padding_even_when_copy_is_sorted_first(method):
    from verl.trainer.ppo.ray_trainer import compute_advantage

    mask = torch.ones(3, 2)
    data = DataProto.from_dict(
        tensors={"responses": torch.ones(3, 2, dtype=torch.long),
                 "attention_mask": torch.ones(3, 4), "response_mask": mask,
                 "values": torch.zeros(3, 2), "turn_values": torch.zeros(3, 2),
                 "token_level_rewards": torch.tensor([[0., 0.], [0., 0.], [0., 10.]])},
        non_tensors={"traj_uid": np.array(["a", "a", "a"], dtype=object),
                     "uid": np.array(["task"] * 3, dtype=object), "step_id": np.array([0, 1, 2])},
    )
    order = [2, 0, 1, 2]
    padded = data.select_idxs(order)
    padded.batch["sample_valid_mask"] = torch.tensor([False, True, True, True])
    compute_advantage(data, method, gamma=0.95, lam=0.95)
    compute_advantage(padded, method, gamma=0.95, lam=0.95)
    for key in ["advantages", "returns", "turn_returns", "turn_value_mask"]:
        if key in data.batch:
            torch.testing.assert_close(padded.batch[key][1:], data.batch[key])
            assert torch.count_nonzero(padded.batch[key][0]) == 0


def _worker_update(kind, padded, micro, monkeypatch, rank=None, all_padding=False,
                   loss_mode="vanilla", heads=2, multimodal=False):
    from omegaconf import OmegaConf
    from verl.workers.actor import dp_actor
    from verl.workers.critic import dp_critic

    module = dp_actor if kind == "actor" else dp_critic
    monkeypatch.setattr(module, "get_torch_device", lambda: NS(current_device=lambda: "cpu"))
    cls = module.DataParallelPPOActor if kind == "actor" else module.DataParallelPPOCritic
    worker = cls.__new__(cls)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.1)
    weight = model.weight
    if rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)
    worker.config = OmegaConf.create(dict(
        ppo_mini_batch_size=6, ppo_micro_batch_size_per_gpu=micro, ppo_epochs=1,
        use_dynamic_bsz=False, clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None,
        entropy_coeff=0.05, loss_agg_mode="token-mean", use_kl_loss=True,
        kl_loss_type="kl", kl_loss_coef=0.1, policy_loss=dict(loss_mode=loss_mode),
        cliprange_value=0.2,
    ))
    features = torch.tensor([[1., 2.], [3., 0.], [4., 5.]])
    mask = torch.tensor([[1., 1.], [1., 0.], [1., 1.]])
    data = DataProto.from_dict(tensors={
        "input_ids": features, "responses": torch.ones(3, 2, dtype=torch.long),
        "attention_mask": mask, "position_ids": torch.zeros(3, 2, dtype=torch.long),
        "loss_mask": mask, "old_log_probs": features * 0.1,
        "ref_log_prob": torch.zeros_like(features), "advantages": features - 2,
        "values": features * 0.1, "returns": features * 0.3,
        "turn_values": features * 0.2, "turn_returns": features * 0.4,
        "turn_value_mask": torch.tensor([[1., 0.]] * 3), "value_mask": mask,
        "sample_valid_mask": torch.ones(3, dtype=torch.bool),
    }, meta_info={"temperature": 1., "multi_turn": True})
    if padded:
        data = data.select_idxs([0, 1, 2, 0, 1, 2])
        data.batch["sample_valid_mask"][3:] = False
    if rank is not None:
        data = data.select_idxs(list(range(rank * 3, (rank + 1) * 3)))
    if all_padding:
        data.batch["sample_valid_mask"][:] = False
    if multimodal:
        data.non_tensor_batch["multi_modal_inputs"] = np.array([{} for _ in range(len(data))], dtype=object)
        worker.config.ppo_mini_batch_size = len(data)

    def optimizer_step():
        gradient = weight.grad.detach().norm()
        optimizer.step()
        return gradient

    worker._optimizer_step = optimizer_step
    if kind == "actor":
        worker.actor_module, worker.actor_optimizer = model, optimizer

        def forward(micro_batch, temperature, calculate_entropy):
            prediction = model(micro_batch["input_ids"].unsqueeze(-1)).squeeze(-1)
            return prediction.square(), prediction

        worker._forward_micro_batch = forward
        worker.update_policy(data)
    else:
        worker.critic_module, worker.critic_optimizer = model, optimizer
        worker.num_value_heads, worker.use_unified_luna_single_head, worker.turn_loss_coef = heads, False, 1.

        def forward(micro_batch):
            prediction = model(micro_batch["input_ids"].unsqueeze(-1)).squeeze(-1)
            return torch.stack([prediction, prediction * 2], dim=-1) if heads == 2 else prediction

        worker._forward_micro_batch = forward
        worker.update_critic(data)
    return weight.detach().clone()


@pytest.mark.parametrize("kind", ["actor", "critic"])
@pytest.mark.parametrize("micro", [1, 2, 3])
def test_actual_workers_padding_and_microbatch_do_not_change_update(kind, micro, monkeypatch):
    expected = _worker_update(kind, False, 1, monkeypatch)
    actual = _worker_update(kind, True, micro, monkeypatch)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("loss_mode", ["turn_ppo", "gspo"])
def test_other_actor_objectives_ignore_padding(loss_mode, monkeypatch):
    expected = _worker_update("actor", False, 1, monkeypatch, loss_mode=loss_mode)
    actual = _worker_update("actor", True, 2, monkeypatch, loss_mode=loss_mode)
    torch.testing.assert_close(actual, expected)


def test_single_head_critic_ignores_padding(monkeypatch):
    expected = _worker_update("critic", False, 1, monkeypatch, heads=1)
    actual = _worker_update("critic", True, 2, monkeypatch, heads=1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_multimodal_chunking_preserves_padding_mask(kind, monkeypatch):
    expected = _worker_update(kind, False, 1, monkeypatch, multimodal=True)
    actual = _worker_update(kind, True, 2, monkeypatch, multimodal=True)
    torch.testing.assert_close(actual, expected)


def _distributed_worker(rank, store_path, kind, expected):
    import torch.distributed as dist
    from datetime import timedelta

    dist.init_process_group("gloo", init_method="file://" + store_path,
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        with pytest.MonkeyPatch.context() as patch:
            # Rank 1 has only padding; it must still participate in backward.
            actual = _worker_update(kind, True, 1, patch, rank=rank)
            torch.testing.assert_close(actual, expected)
            empty = _worker_update(kind, True, 1, patch, rank=rank, all_padding=True)
            # No optimizer step / weight decay when the whole global mini-batch is empty.
            torch.testing.assert_close(empty, torch.tensor([[0.1]]))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_two_rank_actual_update_with_one_padding_only_rank(kind, tmp_path, monkeypatch):
    expected = _worker_update(kind, False, 1, monkeypatch)
    torch.multiprocessing.spawn(_distributed_worker,
                                args=(str(tmp_path / "gloo_store"), kind, expected),
                                nprocs=2, join=True)
