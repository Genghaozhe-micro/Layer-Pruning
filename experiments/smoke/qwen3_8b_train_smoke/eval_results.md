# Evaluation: smoke/qwen3_8b_train_smoke

- Model: Qwen/Qwen3-8B
- Best checkpoint on validation: **iter 1 / prune 4** (PPL=378.99)

## Baselines

| Dataset | PPL | Tokens |
|---------|-----|--------|
| cal_pool | 35.25 | 127 |
| validation | 345.07 | 8 |
| test | 443.08 | 7 |
| c4_validation | 624.84 | 24 |

## test Results

| Checkpoint | Stop@k | Pruning Order | 4 layers |
|------------|--------|---------------|----------|
| iter 1 | 4 | [28, 13, 2, 21] | 391.02 (-52.06) |

## validation Results

| Checkpoint | Stop@k | Pruning Order | 4 layers |
|------------|--------|---------------|----------|
| iter 1 | 4 | [28, 13, 2, 21] | 378.99 (+33.92) |

## cal_pool Results

| Checkpoint | Stop@k | Pruning Order | 4 layers |
|------------|--------|---------------|----------|
| iter 1 | 4 | [28, 13, 2, 21] | 39.94 (+4.69) |

## Detailed Results

### iter 1

Pruning order: `[28, 13, 2, 21]`
Natural stop: after 4 layers

| Step | Layer | Cal PPL | Delta | Reward |
|------|-------|---------|-------|--------|
| 1 | L28 | 50.1434 | +10.6296 | -2.3824 |
| 2 | L13 | 54.7535 | +4.6101 | -0.8795 |
| 3 | L2 | 55.4388 | +0.6854 | -0.1244 |
| 4 | L21 | 56.6125 | +1.1737 | -0.2095 |

