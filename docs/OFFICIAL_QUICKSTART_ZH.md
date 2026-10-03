# 官方版启动说明

此入口先支持 Qwen2.5-1.5B-Instruct 的 ALFWorld / WebShop。训练需要 Linux、CUDA 和 NVIDIA GPU；Mac 本地只做配置、数据文件和发布包测试，不代表完成 GPU 复现。

## 1. 环境与资产

建议 Python3.10、8张80GB卡。`requirements-reference.txt` 是已观察到的运行版本约束，不是已经通过全新安装验收的完整 lockfile。先安装匹配 CUDA12.4 的 PyTorch2.6.0 / torchvision0.21.0，再安装：

```bash
python -m pip install -r requirements.txt -c requirements-reference.txt
python -m pip install vllm==0.8.2
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install gym==0.24.0 gymnasium==0.29.1 alfworld==0.4.2 textworld==1.7.0
python -m pip install -e . --no-deps
```

ALFWorld：执行 `alfworld-download -f`，设置 `ALFWORLD_DATA` 指向包含 `json_2.1.1` 和 `logic` 的目录。seen 测试集必须有140个有效 gamefile。

WebShop：按随附环境的 setup 安装数据、Java11、spaCy 模型和 Lucene 索引。setup 会修改 Python 环境，建议在独立 conda 环境执行，完成后检查版本约束。当前 profile 使用1000个商品；测试仍是500个固定 goal，不是500个商品。

```bash
(cd agent_system/environments/env_package/webshop/webshop && bash setup.sh -d small)
export JAVA_HOME="$CONDA_PREFIX"
export PATH="$JAVA_HOME/bin:$PATH"
```

保留已有资产时，可将对应 data 和各 indexes 目录链接进这个新 checkout；不要链接整个旧代码目录。索引位于 `agent_system/environments/env_package/webshop/webshop/search_engine/indexes`。模型下载到源码目录之外：

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir "$MODEL"
```

记录模型 revision、权重 checksum、资产版本；不要把账户、下载缓存、权重或私有路径提交到仓库。模型下载命令需要安装 Hugging Face CLI。

## 2. 固定测试清单

```bash
python scripts/prepare_data.py alfworld --output "$ALF_DATA"
python scripts/prepare_data.py webshop --output "$WS_DATA"
CUDA_VISIBLE_DEVICES='' python scripts/check_envs.py alfworld
CUDA_VISIBLE_DEVICES='' python scripts/check_envs.py webshop
CUDA_VISIBLE_DEVICES='' python scripts/check_model_cpu.py --model "$MODEL"
python -m pytest -q tests/release tests/trainer/ppo
```

prepare_data 仅生成 simulator slot parquet，不下载真实游戏资产。训练128个 slot 每次 reset 采样环境；测试固定 ALFWorld game_index=0..139 / WebShop goal_index=0..499，每题一次。环境 seed 固定0；三个 decoding seed 只改变模型采样。

## 3. VETTA 开跑与续训

下面示例选择 ALFWorld；WebShop 改成 `webshop` 并令 DATA 指向 WS_DATA。所有目录由用户设置；源码不内置服务器路径。

```bash
python scripts/run.py alfworld config --model "$MODEL" --data "$DATA" \
  --output "$RUN" --checkpoints "$CKPTS"
python scripts/run.py alfworld smoke --model "$MODEL" --data "$DATA" \
  --output "$SMOKE_RUN" --checkpoints "$SMOKE_CKPTS"
python scripts/run.py alfworld train --model "$MODEL" --data "$DATA" \
  --output "$RUN" --checkpoints "$CKPTS"
```

smoke 仅8条轨迹、3轮上限、2次更新并保存 checkpoint，不可当作成功率实验。正式训练保持 ALFWorld mini256 / WebShop mini64，150次更新，5步验证和保存；默认保存 latest，旧的同槽内容会被替换。选择全新的输出和 checkpoint 目录，不影响其他运行。

每卡 Actor/Critic micro8/8、remove-padding=True、gradient checkpointing=True 是工程起点，不是最优性能承诺。micro 必须整除每卡 mini-batch；缩小 micro 不改变全局 mini。入口拒绝和已有 GPU 进程重叠，不会自动杀进程。

Ray object store 明确限制为8GiB，避免按大内存服务器的默认比例占满 /dev/shm。checkpoint 应放在空间充足的持久盘；用 /dev/shm 时必须另外备份，重启后文件会丢失。

续训用原 checkpoint 根目录和新的日志目录：

```bash
python scripts/run.py alfworld train --model "$MODEL" --data "$DATA" \
  --output "$RESUME_RUN" --checkpoints "$CKPTS" --resume "$CKPTS/latest"
```

支持 `latest` 加步数 tracker，或 `global_step_N`。缺 tracker/Actor/Critic 时直接失败，不会悄悄从头训练。不要将 smoke checkpoint 用作正式150步训练的初始化；正式训练需从原预训练模型开始。

## 4. Baseline

沿用上面的 train 命令，添加 `--method ppo_star` / `--method turn_ppo` / `--method gigpo`。PPO* 是单独的完整28层 Critic、跨观察 token GAE 和 token ratio；Turn-PPO 是完整28层 Critic、turn GAE0.95/0.95 和 turn ratio；GiGPO 不创建 Critic，16个初始任务每个采样8次，轨迹预算128。

这里的配方用于受控方法对比，不能声称自动复现所有原论文分数。旧 examples 下的启动脚本保留为参考，不是官方配方入口。

## 5. 三种子终点评测

确认 `latest_checkpointed_iteration.txt` 为150。每次用新输出目录：

```bash
for seed in 123 456 789; do
  python scripts/run.py alfworld eval --model "$MODEL" --data "$DATA" \
    --output "$EVAL_ROOT/seed$seed" --checkpoints "$CKPTS" \
    --resume "$CKPTS/latest" --decode-seed "$seed" || break
done
python scripts/summarize_eval.py "$EVAL_ROOT/seed123" "$EVAL_ROOT/seed456" "$EVAL_ROOT/seed789"
```

评测只加载已训练 Actor，内部选择无需 Critic 的路径，并不把 VETTA checkpoint 变成 GiGPO。温度0.4；ALFWorld 单batch140，WebShop五个batch各100；禁止改变终点评测 batch，以免混入不同聚合协议。WebShop总成功/500与等长batch均值相同。ALFWorld六类子任务直接保留环境输出的兼容指标。

每次运行保存 resolved.yaml、launch.json、inputs.json、console.log、exit.json；验证输出 validation.jsonl。汇总工具检查固定三个 seed、相同 checkpoint/模型配置/测试清单、step150、题数，计算样本标准差(ddof=1)。不接受 smoke 结果或未完成的运行。
