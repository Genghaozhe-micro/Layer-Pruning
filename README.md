# V5: DQN 求解 LLM 层剪枝 MDP

本代码是把 LLM 层剪枝建模为 MDP，并用 Double Dueling DQN 学习剪枝顺序的实验代码。包括核心训练、评测入口和项目内依赖。

## 文件

| 文件 | 作用 |
| --- | --- |
| `train_dqn.py` | 主训练入口。多 GPU worker 采集 episode，主进程更新 DQN，并在训练结束后自动评测。 |
| `environment.py` | `LayerEnv`，定义层剪枝 MDP：状态、动作 mask、stop 动作、PPL 奖励和 episode 终止条件。 |
| `dqn_agent.py` | `DQNAgent`、Dueling/Transformer Q 网络、Replay Buffer、Double DQN 更新逻辑。 |
| `evaluation_pipeline.py` | 评测实现：加载数据、rollout 剪枝顺序、分布式 PPL 评测、记录结果。 |
| `eval_experiment.py` | 推荐的独立评测入口。自动读取实验目录 `config.json`，支持 single-budget 和 multi-budget。 |
| `eval_sequence.py` | 给定手写或外部方法剪枝序列，评估不同 prefix 的 PPL。 |
| `llm_pruner/` | 本目录自带的最小本地依赖，只包含数据加载和 LLM wrapper。 |

## MDP 设定

状态维度为 `num_layers * 8 + 4`。

每层 8 维特征：

- `cos_sim`
- `weight_l1`
- `weight_l2`
- `residual_ratio`
- `output_norm_ratio`
- `token_logit_kl`
- `pred_entropy_delta`
- `alive_flag`

全局 4 维 meta 特征：

- 剪枝进度 `total_pruned / max_prune_limit`
- 当前 PPL 值
- 上一个 step 的 PPL 变化
- budget 占总层数比例

动作空间为 `0..num_layers-1` 加一个 stop 动作。层动作表示剪掉对应层，stop 动作表示结束 episode。默认保护第 `0,1` 层；受保护层、已剪层会被 action mask 禁用。

奖励使用相邻 PPL 比值：

```text
reward = -alpha * log(curr_ppl / prev_ppl)
```

正奖励会被 clamp 到 0，避免把 PPL 偶然下降当成剪枝收益。若 PPL 超过 clip 或 ratio threshold，则给 `ppl_penalty` 并终止 episode。

## 训练

进入目录：

```bash
cd my_pure/final
```

单 budget 训练示例：

```bash
python train_dqn.py \
  --model meta-llama/Llama-3.1-8B \
  --budget 8 \
  --num-gpus 4 \
  --num-iters 600 \
  --cal-pool-size 64 \
  --num-cal 4 \
  --cal-seq-len 512 \
  --q-arch transformer \
  --val-interval 100 \
  --ckpt-interval 20 \
  --exp-name final/llama31_budget8
```

多 budget 训练示例：

```bash
python train_dqn.py \
  --model Qwen/Qwen3-8B \
  --budget-candidates "4,8,12,16" \
  --budget 16 \
  --num-gpus 8 \
  --num-iters 800 \
  --cal-pool-size 128 \
  --num-cal 8 \
  --cal-seq-len 512 \
  --val-interval 100 \
  --ckpt-interval 50 \
  --exp-name final/qwen8b_multibudget
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--budget` | `8` | 单 budget 训练时必须剪掉的层数；会同时设置 `max_prune_limit` 和 `min_prune_budget`。 |
| `--budget-candidates` | `None` | 多 budget 训练候选，例如 `"4,8,12,16"`。每个 episode 会随机采样一个 budget。 |
| `--protected-layers` | `"0,1"` | 不允许剪枝的层。设为空字符串可关闭保护。 |
| `--cal-source` | `bookcorpus` | 校准样本来源：`bookcorpus` 或 `wikitext2`。 |
| `--cal-pool-size` | `128` | 用于 episode 采样的校准池大小。 |
| `--num-cal` | `8` | 每个 episode 使用的校准样本数。 |
| `--alpha` | `10.0` | reward 中的奖励缩放超参。 |
| `--ppl-ratio-threshold` | `2.0` | 当前 PPL 相对上一步增长超过 `1 + threshold` 时终止 episode。设为 `0` 可关闭。 |
| `--q-arch` | `mlp` | Q 网络结构：`mlp` 或 `transformer`。 |
| `--warmup-iters` | `100` | 只采集 replay buffer、不更新 DQN 的迭代数。 |
| `--updates-per-iter` | `50` | 每轮采集后执行的 DQN update 次数。 |

训练结束时会自动：保存最终 checkpoint，基于验证集选择最佳 checkpoint，rollout 剪枝顺序，并写出最终评测报告。

## 独立评测

推荐统一使用 `eval_experiment.py`。它会读取 `experiments/<name>/config.json`，自动带上训练时的模型、budget、Q 网络参数和保护层设置。

复评单 budget 实验：

```bash
python eval_experiment.py \
  --exp-dir experiments/final/llama31_budget8 \
  --ckpts 100,200,final \
  --eval-gpus 4
```

评估同一个剪枝顺序的多个 prefix：

```bash
python eval_experiment.py \
  --exp-dir experiments/final/llama31_budget16 \
  --mode single \
  --ckpts 200,400,600 \
  --prune-counts "4,8,12,16" \
  --eval-gpus 4
```

复评多 budget 实验：

```bash
python eval_experiment.py \
  --exp-dir experiments/final/qwen8b_multibudget \
  --mode multi \
  --budgets "4,8,12,16" \
  --ckpts 200,400,final \
  --eval-gpus 8
```

如果 `config.json` 中有 `budget_candidates`，`--mode auto` 会自动走多 budget 评测；否则走单 budget 评测。

评测手写序列或 baseline 序列：

```bash
python eval_sequence.py \
  --model meta-llama/Llama-3.1-8B \
  --order "26,11,22,23,10,12,28,19" \
  --prune-counts "4,8" \
  --gpu 0 \
  --output manual_sequence_eval.json
```

## 输出结构

典型实验目录如下：

```text
experiments/<exp_name>/
├── config.json
├── checkpoints/
│   ├── dqn_iter100.pt
│   ├── dqn_iter200.pt
│   └── dqn_final.pt
├── history.json
├── val_history.json
├── val_results.md
├── pruning_plans.json
├── eval_results.json
├── eval_results.md
└── pipeline_summary.json
```

关键产物：

| 文件 | 内容 |
| --- | --- |
| `history.json` | 训练曲线：reward、PPL、loss、Q 值、epsilon、stop ratio。 |
| `val_history.json` | 每次验证的剪枝顺序、验证 PPL 和各 budget 结果。 |
| `val_results.md` | 验证表格，便于人工选择 checkpoint。 |
| `pruning_plans.json` | 从验证阶段持久化的剪枝顺序，独立评测会优先复用。 |
| `eval_results.json` | 最终评测原始结果。 |
| `eval_results.md` | 最终评测报告。 |
| `pipeline_summary.json` | 最佳 checkpoint、评测报告路径和最终摘要。 |

## 代码边界

核心训练和评测只依赖本目录中的这些项目文件：`train_dqn.py`、`environment.py`、`dqn_agent.py`、`evaluation_pipeline.py`、`eval_experiment.py`、`eval_sequence.py` 和 `llm_pruner/`。

一次性评测脚本、诊断脚本、日志和历史结果没有放进来。新实验优先走 `train_dqn.py`，复评优先走 `eval_experiment.py`，评估人工序列使用 `eval_sequence.py`。
