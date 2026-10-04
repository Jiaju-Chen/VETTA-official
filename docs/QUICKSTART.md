# Quick Start

Start with ALFWorld and Qwen2.5-1.5B-Instruct, then use the same launcher for
WebShop or a baseline. Run commands from the repository root in an activated
environment. Keep downloaded assets, logs, and checkpoints outside the source.

## 1. Install

The launch recipes use Linux, Python 3.10, and an idle allocation of 8x80GB
NVIDIA GPUs. Install a CUDA-compatible driver before setting up the runtime.
FlashAttention compilation requires a matching CUDA build toolchain when a
prebuilt wheel is unavailable.

```bash
conda create -n vetta python=3.10 -y
conda activate vetta
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -r requirements.txt -c requirements-reference.txt
python -m pip install vllm==0.8.2
python -m pip install gym==0.24.0 gymnasium==0.29.1 \
  alfworld==0.4.2 textworld==1.7.0
python -m pip install 'huggingface_hub[cli]'
python -m pip install -e . --no-deps
python -m pip check
```

Version constraints are in `requirements-reference.txt`. If installation
reports a conflict, resolve the stated dependency/CUDA mismatch before
launching; keep the constrained Torch, Transformers, and vLLM versions.

## 2. Prepare

Download the model and ALFWorld assets once:

```bash
export VETTA_WORK="$HOME/vetta-work"
export MODEL="$VETTA_WORK/models/Qwen2.5-1.5B-Instruct"
export ALFWORLD_DATA="$VETTA_WORK/alfworld"
export ALF_DATA="$VETTA_WORK/manifests/alfworld"
mkdir -p "$VETTA_WORK"

hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir "$MODEL"
alfworld-download -f
python scripts/prepare_data.py alfworld --output "$ALF_DATA"
CUDA_VISIBLE_DEVICES='' python scripts/check_envs.py alfworld
python -m pytest -q tests/release
```

Existing assets can be reused by setting the variables to their locations.
`ALFWORLD_DATA` must contain `json_2.1.1` and `logic`. The environment check
reports `"passed": true`, 140 valid seen games, and a successful reset/`look`
action. `prepare_data.py` creates simulator slots and fixed evaluation
manifests, not the environment assets themselves.

## 3. Smoke and Train

Check rollout, actor/critic updates, and saving:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SMOKE_ID="$(date +%Y%m%d-%H%M%S)"
export SMOKE_RUN="$VETTA_WORK/runs/alfworld-smoke-$SMOKE_ID"
export SMOKE_CKPTS="$VETTA_WORK/checkpoints/alfworld-smoke-$SMOKE_ID"
python scripts/run.py alfworld smoke \
  --model "$MODEL" --data "$ALF_DATA" \
  --output "$SMOKE_RUN" --checkpoints "$SMOKE_CKPTS"
```

Expected: exit code 0, finite actor/critic update metrics in `console.log`, and
`$SMOKE_CKPTS/latest_checkpointed_iteration.txt` containing `2`. Smoke uses
8 trajectories, a 3-turn limit, and two updates. Start formal training from the
pretrained model, not this shortened smoke checkpoint:

```bash
export DATASET=alfworld
export DATA="$ALF_DATA"
export METHOD=vetta
export ACTOR_MICRO=8
export CRITIC_MICRO=8
export RUN_ID="$(date +%Y%m%d-%H%M%S)"
export RUN="$VETTA_WORK/runs/$DATASET-$METHOD-$RUN_ID"
export CKPTS="$VETTA_WORK/checkpoints/$DATASET-$METHOD-$RUN_ID"
python scripts/run.py "$DATASET" train --method "$METHOD" \
  --model "$MODEL" --data "$DATA" --output "$RUN" --checkpoints "$CKPTS" \
  --actor-micro "$ACTOR_MICRO" --critic-micro "$CRITIC_MICRO"
```

This selects shared 2-block dual-head VETTA with alpha 1, token gamma/lambda 1/1,
turn gamma/lambda 0.95/0.95, 128 independent trajectories/update,
mini-batch 256, and 150 updates.
Remove-padding and gradient checkpointing are enabled.

Each run writes `resolved.yaml`, `launch.json`, `inputs.json`, `console.log`,
`exit.json`, and validation records in `validation.jsonl`. Monitor from a
second terminal using the same `RUN` path:

```bash
tail -f "$RUN/console.log"
```

Validation and latest-only saving occur every 5 updates. Use a new output
directory for each run and place checkpoints on persistent storage with
enough room for full optimizer state. Saving replaces the previous checkpoint
in that run's latest slot. The launcher refuses to overlap existing GPU jobs.
The logger defaults to local console output.

## 4. WebShop

Install the 1000-product environment assets, Java 11, spaCy models, and Lucene
indexes using the bundled setup:

```bash
(cd agent_system/environments/env_package/webshop/webshop && bash setup.sh -d small)
export JAVA_HOME="$CONDA_PREFIX"
export JAVA_LD_LIBRARY_PATH="$JAVA_HOME/lib/server"
export PATH="$JAVA_HOME/bin:$PATH"
javac -version
export WS_DATA="$VETTA_WORK/manifests/webshop"
python scripts/prepare_data.py webshop --output "$WS_DATA"
CUDA_VISIBLE_DEVICES='' python scripts/check_envs.py webshop
python -m pip check
```

Setup installs additional dependencies; recheck the runtime constraints before
training. For an existing installation, link the matching `data` and
`search_engine/indexes*` directories into this checkout. The environment check
reports a successful goal 0 reset/search and a fixed test pool of 500 goals.
The product count and test-goal count are different quantities.

Run a WebShop smoke:

```bash
export SMOKE_ID="$(date +%Y%m%d-%H%M%S)"
python scripts/run.py webshop smoke \
  --model "$MODEL" --data "$WS_DATA" \
  --output "$VETTA_WORK/runs/webshop-smoke-$SMOKE_ID" \
  --checkpoints "$VETTA_WORK/checkpoints/webshop-smoke-$SMOKE_ID"
```

Then train:

```bash
export DATASET=webshop
export DATA="$WS_DATA"
export METHOD=vetta
export ACTOR_MICRO=2
export CRITIC_MICRO=8
export RUN_ID="$(date +%Y%m%d-%H%M%S)"
export RUN="$VETTA_WORK/runs/$DATASET-$METHOD-$RUN_ID"
export CKPTS="$VETTA_WORK/checkpoints/$DATASET-$METHOD-$RUN_ID"
python scripts/run.py "$DATASET" train --method "$METHOD" \
  --model "$MODEL" --data "$DATA" --output "$RUN" --checkpoints "$CKPTS" \
  --actor-micro "$ACTOR_MICRO" --critic-micro "$CRITIC_MICRO"
```

The WebShop profile selects alpha 3, mini-batch 64, and a 15-turn limit. Micro-batch
must divide the per-GPU mini-batch; lower it if peak memory exceeds capacity.
The effective mini-batch remains unchanged. Full-depth baseline critics can
require a smaller critic micro-batch than VETTA.

## 5. Baselines

Set `DATASET`, `DATA`, and the micro-batches for the chosen benchmark above.
Choose one method and create separate run/checkpoint paths:

```bash
export METHOD=ppo_star   # Full-depth, cross-observation token PPO
# export METHOD=turn_ppo # Full-depth, turn GAE + whole-response ratio
# export METHOD=gigpo    # Critic-free, 16 tasks x 8 rollouts
export RUN_ID="$(date +%Y%m%d-%H%M%S)"
export RUN="$VETTA_WORK/runs/$DATASET-$METHOD-$RUN_ID"
export CKPTS="$VETTA_WORK/checkpoints/$DATASET-$METHOD-$RUN_ID"
python scripts/run.py "$DATASET" train --method "$METHOD" \
  --model "$MODEL" --data "$DATA" --output "$RUN" --checkpoints "$CKPTS" \
  --actor-micro "$ACTOR_MICRO" --critic-micro "$CRITIC_MICRO"
```

PPO* and Turn-PPO use a separate full 28-layer scalar critic. Turn-PPO switches
both GAE and the actor ratio. GiGPO produces 128 trajectories from 16 initial
tasks with 8 rollouts each; the critic-based profiles use 128 tasks x 1 rollout.
Use the resolved configuration to record the selected comparison recipe.

## 6. Resume

Keep the checkpoint root of the interrupted run and choose a new log directory:

```bash
export RESUME_ID="$(date +%Y%m%d-%H%M%S)"
python scripts/run.py "$DATASET" train --method "$METHOD" \
  --model "$MODEL" --data "$DATA" \
  --output "$VETTA_WORK/runs/$DATASET-$METHOD-resume-$RESUME_ID" \
  --checkpoints "$CKPTS" --resume "$CKPTS/latest" \
  --actor-micro "$ACTOR_MICRO" --critic-micro "$CRITIC_MICRO"
```

Resume requires a valid latest-step tracker and the necessary actor/critic
checkpoint files. `global_step_N` directories are also supported. The entry
point does not accept `best` or silently restart with missing checkpoint files.

## 7. Evaluate

Use the trained run's `CKPTS`, `DATASET`, `DATA`, and `MODEL` variables. Evaluate
the latest step150 checkpoint with decoding seeds 123, 456, and 789:

```bash
test "$(cat "$CKPTS/latest_checkpointed_iteration.txt")" = 150
export EVAL_ID="$(date +%Y%m%d-%H%M%S)"
export EVAL_ROOT="$VETTA_WORK/eval/$DATASET-$METHOD-$EVAL_ID"
for seed in 123 456 789; do
  python scripts/run.py "$DATASET" eval \
    --model "$MODEL" --data "$DATA" \
    --output "$EVAL_ROOT/seed$seed" --checkpoints "$CKPTS" \
    --resume "$CKPTS/latest" --decode-seed "$seed" || break
done
python scripts/summarize_eval.py \
  "$EVAL_ROOT/seed123" "$EVAL_ROOT/seed456" "$EVAL_ROOT/seed789"
```

Evaluation loads only the trained actor. ALFWorld uses fixed seen140 in one
batch; WebShop uses fixed goal IDs 0..499 in five batches of 100. Environment
seed stays 0, temperature is 0.4 with sampling, and only decoding seed changes.
The summary reports each run, the arithmetic mean, and sample standard deviation
(`ddof=1`). These are decoding seeds for one trained checkpoint, not separate
training seeds. Missing or inconsistent evaluations are rejected.

## Configuration Reference

Ready-to-run profiles are in `configs/release/alfworld.yaml` and
`configs/release/webshop.yaml`, with shared settings in `common.yaml`.
`scripts/run.py` targets Qwen2.5-1.5B-Instruct. To print the complete resolved
configuration without allocating GPUs:

```bash
python scripts/run.py "$DATASET" config --method "$METHOD" \
  --model "$MODEL" --data "$DATA" --output "$RUN" --checkpoints "$CKPTS" \
  --actor-micro "$ACTOR_MICRO" --critic-micro "$CRITIC_MICRO"
```
