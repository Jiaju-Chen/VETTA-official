from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.workers.critic.dp_critic import DataParallelPPOCritic


def test_compute_values_masks_response_positions_not_preceding_hidden_states():
    module = torch.nn.Linear(1, 1)
    critic = DataParallelPPOCritic(
        OmegaConf.create({"model": {"use_remove_padding": False},
                          "ulysses_sequence_parallel_size": 1}),
        module, torch.optim.SGD(module.parameters(), lr=0.1))
    critic._forward_micro_batch = lambda _: torch.ones(2, 4)
    data = DataProto.from_dict(tensors={
        "responses": torch.ones(2, 4, dtype=torch.long),
        "input_ids": torch.ones(2, 6, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]]),
        "position_ids": torch.arange(6).expand(2, -1)},
        meta_info={"micro_batch_size": 2, "use_dynamic_bsz": False})
    torch.testing.assert_close(critic.compute_values(data),
                               torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 0.]]))


def test_turn_critic_explained_variance_uses_boundary_mask():
    data = SimpleNamespace(batch={
        "responses": torch.ones(2, 3), "attention_mask": torch.ones(2, 5),
        "token_level_scores": torch.zeros(2, 3), "token_level_rewards": torch.zeros(2, 3),
        "advantages": torch.ones(2, 3),
        "returns": torch.tensor([[1., 0., 0.], [2., 0., 0.]]),
        "values": torch.tensor([[1., 99., 99.], [2., 99., 99.]]),
        "value_mask": torch.tensor([[1., 0., 0.], [1., 0., 0.]])},
        non_tensor_batch={"traj_uid": np.array(["a", "b"]),
                          "episode_rewards": np.zeros(2), "episode_lengths": np.ones(2),
                          "tool_callings": np.zeros(2)})
    metrics = compute_data_metrics(data, use_critic=True)
    assert metrics["critic/vf_explained_var"] == 1.0
    assert metrics["critic/returns/mean"] == 1.5

