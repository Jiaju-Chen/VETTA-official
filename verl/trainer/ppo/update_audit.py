"""Opt-in, read-only observations of a real PPO update."""

import json
from pathlib import Path

import numpy as np
import torch


def write_audit(directory, name, payload):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / name).open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def ratio_summary(log_prob, old_log_prob, mask, advantages):
    with torch.no_grad():
        difference = (log_prob.float() - old_log_prob.float()) * mask
        tokens = difference[mask.bool()]
        valid = mask.sum(-1) > 0
        turns = difference.sum(-1)[valid]
        lengths = mask.sum(-1)[valid]
        turn_adv = ((advantages * mask).sum(-1) / mask.sum(-1).clamp_min(1))[valid]
        beneficial_clip = ((turn_adv > 0) & (turns > np.log(1.2))) | (
            (turn_adv < 0) & (turns < np.log(0.8)))
        return {
            "valid_tokens": int(tokens.numel()), "valid_turns": int(turns.numel()),
            "token_log_diff_abs_mean": float(tokens.abs().mean()) if tokens.numel() else 0.0,
            "token_log_diff_abs_max": float(tokens.abs().max()) if tokens.numel() else 0.0,
            "turn_log_ratio_abs_max": float(turns.abs().max()) if turns.numel() else 0.0,
            "turn_log_ratio_mean": float(turns.mean()) if turns.numel() else 0.0,
            "response_length_mean": float(lengths.float().mean()) if lengths.numel() else 0.0,
            "turn_beneficial_clip_fraction": float(beneficial_clip.float().mean()) if turns.numel() else 0.0,
        }


def audit_credit_batch(data, turn_gamma=0.95, turn_lam=0.95):
    tensors = data.batch
    real = tensors.get("sample_valid_mask", torch.ones(len(data))).bool().cpu()
    mask = tensors["response_mask"].float().cpu()
    rewards = tensors["token_level_rewards"].float().cpu()
    values = tensors.get("turn_values", tensors["values"]).float().cpu()
    targets = tensors.get("turn_returns", tensors["returns"]).float().cpu()
    trajectories = np.asarray(data.non_tensor_batch["traj_uid"])
    steps = np.asarray(data.non_tensor_batch["step_id"])
    groups = {}
    seen = set()
    for row in torch.nonzero(real).flatten().tolist():
        key = (str(trajectories[row]), int(steps[row]))
        if key in seen:
            raise ValueError(f"Duplicate real trajectory step: {key}")
        seen.add(key)
        groups.setdefault(str(trajectories[row]), []).append(row)
    errors = []
    token_errors = []
    samples = []
    for uid, rows in groups.items():
        rows.sort(key=lambda row: int(steps[row]))
        next_value, next_adv = 0.0, 0.0
        token_return = 0.0
        for row in reversed(rows):
            positions = torch.nonzero(mask[row]).flatten()
            if not positions.numel():
                raise ValueError("A real turn has no response tokens")
            start = int(positions[0])
            value = float(values[row, start])
            reward = float((rewards[row] * mask[row]).sum())
            advantage = reward + turn_gamma * next_value - value + turn_gamma * turn_lam * next_adv
            expected = advantage + value
            actual = float(targets[row, start])
            errors.append(abs(expected - actual))
            if len(samples) < 64 or reward > 1:
                samples.append({"traj_uid": uid, "step": int(steps[row]), "reward": reward,
                                "boundary_value": value, "target": actual,
                                "independent_target": expected})
            next_value, next_adv = value, advantage
            if "turn_returns" in tensors:
                # This diagnostic launcher uses token gamma=lambda=1: the target is Monte Carlo.
                for position in reversed(positions.tolist()):
                    token_return += float(rewards[row, position])
                    token_errors.append(abs(token_return - float(tensors["returns"][row, position])))
    finite = all(bool(torch.isfinite(value).all()) for name, value in tensors.items()
                 if name in ("advantages", "returns", "turn_returns", "values", "turn_values"))
    padded_outputs_zero = all(not bool(tensors[name].cpu()[~real].any())
                              for name in ("advantages", "returns", "turn_returns") if name in tensors)
    output = {"real_rows": int(real.sum()), "padding_rows": int((~real).sum()),
              "trajectories": len(groups), "finite": finite,
              "padded_advantages_targets_zero": padded_outputs_zero,
              "turn_target_error_max": max(errors, default=0),
              "turn_target_error_mean": float(np.mean(errors)) if errors else 0.0,
              "token_mc_target_error_max": max(token_errors, default=0.0),
              "token_mc_target_error_mean": float(np.mean(token_errors)) if token_errors else 0.0,
              "reward_min": float(rewards[real].sum(-1).min()),
              "reward_max": float(rewards[real].sum(-1).max()), "samples": samples}
    if "token_residuals" in tensors:
        residual = tensors["token_residuals"].float().cpu() * mask
        output["residual_turn_mean_abs_max"] = float(
            (residual.sum(-1) / mask.sum(-1).clamp_min(1))[real].abs().max())
    if not finite or not padded_outputs_zero:
        raise ValueError(f"Credit audit failed: {output}")
    return output


def critic_replay_summary(data, predictions):
    before = data.batch["values"].float().cpu()
    after = predictions.float().cpu()
    if after.ndim == 3:
        after, after_turn = after[..., 0], after[..., 1]
    else:
        after_turn = after
    mask = data.batch.get("value_mask", data.batch["response_mask"]).float()
    if "sample_valid_mask" in data.batch:
        mask = mask * data.batch["sample_valid_mask"].unsqueeze(-1)
    target = data.batch["returns"].float()
    def mse(prediction, expected, selected):
        return float(((prediction - expected).square() * selected).sum() / selected.sum().clamp_min(1))
    result = {"token_mse_before": mse(before, target, mask), "token_mse_after": mse(after, target, mask)}
    if "turn_returns" in data.batch:
        turn_mask = data.batch["turn_value_mask"].float()
        if "sample_valid_mask" in data.batch:
            turn_mask = turn_mask * data.batch["sample_valid_mask"].unsqueeze(-1)
        result.update(turn_mse_before=mse(data.batch["turn_values"].float(), data.batch["turn_returns"].float(), turn_mask),
                      turn_mse_after=mse(after_turn, data.batch["turn_returns"].float(), turn_mask))
    return result
