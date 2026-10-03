"""Check real 1.5B weights on CPU, not distributed PPO or benchmark accuracy."""
import argparse
import gc
import json
import os
from pathlib import Path


def check(path):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer

    torch.set_num_threads(2)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type != "qwen2" or config.num_hidden_layers != 28 or config.hidden_size != 1536:
        raise ValueError("Expected the Qwen2.5-1.5B text model")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    inputs = tokenizer("take apple 1 from countertop 2", return_tensors="pt")
    actor = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                               attn_implementation="sdpa", local_files_only=True)
    actor.eval()
    with torch.no_grad():
        logits = actor(**inputs).logits
        assert torch.isfinite(logits).all()
        actor_shape = list(logits.shape)
    del actor, logits
    gc.collect()

    config.num_hidden_layers = 2
    config.num_labels = 2
    config.classifier_dropout = 0.0
    torch.manual_seed(0)
    critic = AutoModelForTokenClassification.from_pretrained(
        path, config=config, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True)
    critic.train()
    values = critic(**inputs).logits
    assert values.shape == (*inputs["input_ids"].shape, 2) and torch.isfinite(values).all()
    loss = values.square().mean()
    loss.backward()
    assert critic.score.weight.grad is not None and torch.isfinite(critic.score.weight.grad).all()
    assert critic.model.layers[0].self_attn.q_proj.weight.grad is not None
    return {"model": "Qwen2.5-1.5B-Instruct", "device": "cpu", "actor_layers": 28,
            "actor_logits_shape": actor_shape, "critic_layers": len(critic.model.layers),
            "critic_values_shape": list(values.shape), "critic_heads": 2,
            "critic_backward": True, "passed": True,
            "scope": "CPU model compatibility only; no rollout, optimizer step or performance claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' for this CPU-only check")
    print(json.dumps(check(args.model), indent=2))
