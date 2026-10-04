# Reproduction Checks

Use these checks when preparing a new runtime or reproducing an experiment.
The [Quick Start](QUICKSTART.md) contains the corresponding commands.

## Runtime and Environment

1. Install the constrained Python/CUDA dependencies in an isolated environment.
2. Prepare the model and benchmark assets outside the source checkout.
3. Run `tests/release` and the focused PPO regression tests.
4. Run `check_envs.py` for a fixed reset and a real environment action.
5. Run a two-update GPU smoke, then resume its checkpoint for another update.

A smoke test exercises the training lifecycle. Use the full training profile
and endpoint evaluation for benchmark results; do not initialize a formal
training run from a smoke checkpoint.

## Experiment Profiles

ALFWorld: shared two-block dual-head critic, residual alpha1, token
gamma/lambda1/1, turn gamma/lambda0.95/0.95, final whitening, next-token
value alignment, 128 tasks x1 rollout, PPO mini-batch256, history2, horizon50,
prompt/response4096/512, seen140, and latest step150.

WebShop: the same credit method with alpha3, mini-batch64, horizon15,
the1000-product corpus, fixed goal IDs0..499, and environment seed0.

Both profiles use actor LR1e-6, critic LR1e-5, KL coefficient0.01,
invalid-action penalty0.1, training temperature1, and evaluation
temperature0.4. Record micro-batches, GPU count, and runtime versions
alongside the scientific configuration.

## Credit and Padding

Divisibility-padding rows are excluded from trajectory GAE, whitening,
actor/critic losses, and valid-sample normalization. Credit recursion and
within-response centering use FP32; model forward/backward can use BF16.
Regression tests cover copied/reordered padding rows, all-padding
micro-batches, and distributed valid-sample normalization.

## Checkpoints and Evaluation

Formal training starts from the original pretrained actor checkpoint.
Record the model revision and checksums, benchmark assets, resolved
configuration, and retained checkpoint step.

Evaluate latest step150 on the same fixed task manifest with decoding
seeds123,456,and789. The summary reports every seed, the arithmetic mean,
and sample standard deviation. These are decoding repetitions of one trained
checkpoint, not independently trained models.

Checkpoint reevaluation checks inference and evaluation consistency.
A fresh training run additionally exercises the updated learning procedure.
Keep paper-reported results distinct from measurements of a new run.
