# VETTA Official Release Candidate

VETTA studies critic-based credit assignment for long-horizon language-agent PPO.
This standalone release candidate provides explicit ALFWorld/WebShop 1.5B
profiles, offline task manifests, guarded training/evaluation entry points,
and padding-safe FP32 credit assignment. It does not contain model weights,
benchmark data, run logs, credentials, or deployment-specific paths.

**Validation scope:** local configuration/package checks are distinct from
GPU training and endpoint reproduction. Full step150 performance with this
candidate is not yet verified. See [Release Acceptance](docs/RELEASE_ACCEPTANCE.md).

## Method

The main critic reuses a shallow Transformer backbone and predicts token-level
and turn-boundary values with separate heads. The actor combines turn-level
advantage with a centered token residual:

```text
A[t,j] = A_turn[t] + alpha * (A_token[t,j] - mean_j(A_token[t,j]))
```

The centered residual lets the turn signal carry cross-turn progress while the
token signal distinguishes decisions within one response. The implementation
also includes single-head/shared-critic ablations and a Turn-PPO baseline used
for controlled comparisons.

## Code lineage

The rollout, environment, PPO, Ray, FSDP, and vLLM infrastructure is derived
from the public GiGPO/`verl-agent` and veRL codebases. Their licenses and notices
are retained in this snapshot. VETTA changes the critic/value path and advantage
composition; it is not a reimplementation of GiGPO's group-in-group objective.

## Benchmarks and assets

The code contains integrations for ALFWorld, WebShop, and SearchQA. Their data,
retrieval indexes/services, model checkpoints, and runtime secrets must be
obtained and configured separately under the corresponding upstream licenses.
Runtime paths, service endpoints, and credentials are supplied through local
configuration rather than embedded in the source.

Start with the [Official Quick Start](docs/OFFICIAL_QUICKSTART_ZH.md) for runtime
setup, asset preparation, VETTA and baseline commands, checkpoint recovery,
and three-seed endpoint evaluation. The old [Getting Started](docs/GETTING_STARTED.md)
contains inherited examples, not the authoritative official-release profiles.

| Qwen2.5-1.5B | Critic | Token gamma/lambda | Turn gamma/lambda | Alpha | Trajectories/update | Mini-batch | Horizon |
|---|---|---|---|---|---|---|---|
| ALFWorld seen | 2 blocks, 2 heads | 1/1 | 0.95/0.95 | 1 | 128 x 1 | 256 | 50 |
| WebShop small | 2 blocks, 2 heads | 1/1 | 0.95/0.95 | 3 | 128 x 1 | 64 | 15 |

The release entry point is `python scripts/run.py DATASET MODE`. Modes are
`config`, `smoke`, `train`, and `eval`. Methods are `vetta`, `ppo_star` (full-depth
cross-observation token PPO), `turn_ppo` (full-depth turn GAE and turn ratio),
and `gigpo` (no Critic, 16 initial tasks x 8 rollouts). Training uses 150 updates;
endpoint evaluation uses latest step150, fixed 140/500 tasks, temperature0.4,
and decoding seeds123/456/789. Engineering defaults target 8x80GB GPUs;
they are not a bitwise reproduction guarantee.

```bash
python scripts/run.py alfworld config --model "$MODEL" --data "$DATA" \
  --output "$RUN" --checkpoints "$CKPTS"
python -m pytest -q tests/release
python scripts/package_release.py --output ../VETTA-official.zip
```

SearchQA and visual-model infrastructure remain available in the inherited
source, but they are not validated by these two text-benchmark release profiles.

## License

The repository retains the upstream Apache-2.0 license and attribution notice.
See `LICENSE` and `Notice.txt`.
