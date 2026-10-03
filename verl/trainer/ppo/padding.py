"""Separate model inputs from the training contribution of batch-padding rows."""

import torch
import torch.distributed as dist


def apply_row_mask(mask, batch):
    valid = batch.get("sample_valid_mask")
    if valid is None:
        return mask
    return mask * valid.to(device=mask.device, dtype=mask.dtype).unsqueeze(-1)


def loss_denominator(mask, mode):
    if mode == "token-mean":
        return mask.sum()
    if mode in ("seq-mean-token-mean", "seq-mean-token-sum"):
        return (mask.sum(-1) > 0).sum().to(mask.dtype)
    if mode == "seq-mean-token-sum-norm":
        return mask.new_tensor(1.)
    raise ValueError(f"Unsupported loss aggregation: {mode}")


def minibatch_normalizers(masks, modes, group=None):
    """One collective per mini-batch; never collect per dynamic micro-batch."""
    counts = torch.stack([loss_denominator(mask, mode).float()
                          for mask, mode in zip(masks, modes)])
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group)
        dist.all_reduce(counts, group=group)
    # The fixed-length DrGRPO divisor already occurs in the local loss.
    for index, mode in enumerate(modes):
        if mode == "seq-mean-token-sum-norm":
            counts[index] = 1.
    return counts, world_size


def normalized_loss(local_loss, mask, mode, global_denominator, world_size):
    # FSDP/DDP averages gradients, so undo that average for a global valid mean.
    local_denominator = loss_denominator(mask, mode)
    return local_loss * local_denominator * world_size / global_denominator.clamp_min(1)
