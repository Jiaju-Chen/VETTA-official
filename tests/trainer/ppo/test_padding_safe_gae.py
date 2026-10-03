from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from verl import DataProto
from agent_system.multi_turn_rollout.utils import adjust_batch
from verl.trainer.ppo.core_algos import (
    compute_dual_critic_hybrid_gae,
    compute_sao_skip_observation_gae,
    compute_turn_ppo_gae,
)


def compute(method, rewards, values, mask, trajectories, steps, whiten):
    common = dict(
        token_level_rewards=rewards, response_mask=mask,
        traj_index=trajectories, step_id=steps, whiten_advantages=whiten,
    )
    if method == "vetta":
        return compute_dual_critic_hybrid_gae(
            **common, token_values=values, turn_values=values * 0.7,
            token_gamma=1.0, token_lam=1.0, turn_gamma=0.95, turn_lam=0.95,
        )
    fn = compute_turn_ppo_gae if method == "turn" else compute_sao_skip_observation_gae
    return fn(**common, values=values, gamma=0.95, lam=0.95)


@pytest.mark.parametrize("method", ["token", "turn", "vetta"])
@pytest.mark.parametrize("whiten", [False, True])
@pytest.mark.parametrize("duplicate", [0, 1, 2])
def test_padding_copies_do_not_change_gae_targets_or_advantages(method, whiten, duplicate):
    rewards = torch.tensor([[0., 0.1, 0.], [0., 0., 0.], [0., 10., 0.], [0., -0.1, 0.]])
    values = torch.tensor([[0.3, 0.5, 0.], [0.7, 0.8, 0.], [1.1, 1.2, 0.], [0.2, 0.4, 0.]])
    mask = torch.tensor([[1., 1., 0.]] * 4)
    trajectories = np.array(["a", "a", "a", "b"], dtype=object)
    steps = np.array([0, 1, 2, 0])
    expected = compute(method, rewards, values, mask, trajectories, steps, whiten)
    # Include a reordered padding copy, not just an appended terminal row.
    order = [duplicate, 3, 2, 0, 1]
    actual = compute(method, rewards[order], values[order], mask[order],
                     trajectories[order], steps[order], whiten)
    for baseline, padded in zip(expected, actual):
        torch.testing.assert_close(padded, baseline[order])


def test_actual_adjust_batch_does_not_double_terminal_reward():
    config = NS(
        trainer=NS(n_gpus_per_node=2, nnodes=1),
        algorithm=NS(use_kl_in_reward=False),
        actor_rollout_ref=NS(
            rollout=NS(log_prob_micro_batch_size_per_gpu=1),
            ref=NS(log_prob_micro_batch_size_per_gpu=1),
            actor=NS(use_kl_loss=True, ppo_micro_batch_size_per_gpu=1, ppo_mini_batch_size=2),
        ),
    )
    data = DataProto.from_dict(
        tensors={"input_ids": torch.zeros(3, 4, dtype=torch.long),
                 "rewards": torch.tensor([[0., 0.], [0., 0.], [0., 10.]]),
                 "values": torch.zeros(3, 2), "mask": torch.ones(3, 2)},
        non_tensors={"traj_uid": np.array(["t"] * 3, dtype=object),
                     "step_id": np.array([0, 1, 2])},
    )
    rng_state = np.random.get_state()
    try:
        for seed in range(100):
            np.random.seed(seed)
            if np.random.choice(3, 1, replace=False)[0] == 2:
                break
        np.random.seed(seed)
        padded = adjust_batch(config, data)
    finally:
        np.random.set_state(rng_state)
    assert padded.non_tensor_batch["step_id"].tolist() == [0, 1, 2, 2]
    _, returns = compute_sao_skip_observation_gae(
        padded.batch["rewards"], padded.batch["values"], padded.batch["mask"],
        padded.non_tensor_batch["traj_uid"], padded.non_tensor_batch["step_id"],
        1., 1., False,
    )
    torch.testing.assert_close(returns, torch.full((4, 2), 10.))
