# Experiments

## Philosophy: What Makes a Good Driving Model

We're training a model to drive in GTA V by predicting 4 binary controls: W (gas), A (left), S (brake), D (right). The goal is **driving intent**, not frame-perfect label prediction.

### Why metrics can be misleading

- **Val_acc** counts a sample correct only if ALL 4 labels match simultaneously. With noisy human-labeled driving data (pressing A one frame late, releasing W briefly mid-turn), this is extremely harsh. A model can be an excellent driver with 45% val_acc.
- **Label noise is fundamental.** A real turn has inconsistent A/D labels frame-to-frame because humans don't press keys with frame-perfect timing. The model should learn "turn here" not "press A on exactly this frame." We want to predict intent, not noise.
- **Label smoothing** intentionally lowers loss/acc metrics. Don't penalize it for that — look at per-control behavior instead.
- **Train acc < val acc** is expected in our proxy setup because training uses 10x augmented features (harder distribution) while eval uses originals.

### What to actually look at

- **A/D recall + specificity** = steering quality. Most important for driving.
  - **Recall > specificity** is preferred: a model that turns too much can self-correct (via the opposing direction), but a model that doesn't turn enough just misses turns entirely. Some balance is still needed — spec shouldn't crater.
- **W recall** must stay high (~95%+), otherwise the model won't drive forward.
- **W specificity** matters — too low means the model never lifts off gas, which hurts cornering.
- **S recall** should be decent (~40-50%+) but is secondary. Don't sacrifice steering quality for brake recall.
- **Overall gestalt**: would this model actually drive well? Think about the driving behavior the metrics imply, not the numbers themselves.

## Proxy Experiment Setup

### What we did
Ran ~90 experiments on **precomputed DINO features** to find the best transformer head configuration before committing to expensive full training runs.

- **Pipeline**: screen frames -> frozen DINOv3 ViT-B16 -> 768-dim features -> frozen Linear(768->128) projection -> saved to disk
- **Dataset**: 10 precomputed feature files (~22GB each): 1 original + 9 augmented versions per frame
- **Model under test**: EfficientTransformer head (RoPE, SwiGLU, RMSNorm) operating on 128-dim token features
- **Why proxy**: avoids recomputing DINO every iteration, enables rapid experimentation

### Limitation of proxy experiments
- The 128-dim projection is a bottleneck — full training has 768-dim DINO output with a learned projection
- Augmentations are precomputed and fixed — full training applies them dynamically
- Proxy "64d" means 128->64 downprojection in the head; full training "64d" means 768->64, a different compression path
- Results transfer directionally but exact hyperparameters may need adjustment

## Key Findings (exp1-91)

### Architecture
- **6 layers** is the sweet spot. 8 layers didn't improve over 6 (exp64 vs exp66). 2 and 4 layers were weaker.
- **64d vs 128d**: Both viable. 128d showed higher peak steering (A recall 90% in exp73) and more consistency across seeds. 64d was more stable in proxy but that may be a bottleneck artifact.
- **96d** underperformed 64d (exp72).

### Schedule (highest-signal finding)
- **Full cosine kills S recall.** Training at high LR all the way to max_iters causes S to collapse from ~50% to <20% late in training. Universal across all full-cosine runs.
- **Fast cosine** (`lr_decay_iters << max_iters`) is essential. The model reaches min_lr early, then the long low-LR tail solidifies minority behaviors (braking, precise steering).
- **cos5000 at 15K** was the best all-rounder: good balance of steering quality and S preservation.
- **cos3000 at 15K** was more stable for S but peaked earlier and stagnated.
- **cos7500 at 15K** gave marginal A specificity gains but lost S recall and overall balance.
- **Warm restarts (SGDR)** did not prevent S collapse.

### Regularization
- **Dropout 0.2** is the sweet spot. 0.1 was too noisy, 0.35 too restrictive. For 128d specifically, dropout 0.3 showed better S preservation (exp78: 63% vs 44%).
- **Weight decay 0.075** is settled. 0.03 caused overfitting (exp65).
- **Label smoothing**: LS=0.05 was too much (D specificity collapsed to 56%, "drunk driving"). LS=0.01 was the sweet spot — better W specificity without hurting S. LS=0.02 was mixed (hurt S in some configs).

### Batch size and LR
- **Effective batch 512** (either real 512 or 256 x grad_acc 2) is clearly better than 256. Smoother gradients, better generalization.
- **LR scaling**: bs256 used LR 5e-5 / muon 0.0015. bs512 used LR 1e-4 / muon 0.003.
- **Important tradeoff**: bs256/lower-LR produced the standout experiments (exp73: A rec 90%, exp66: balanced). bs512/higher-LR was more balanced but with lower peaks. The standout experiments' LR regime may be better for full training.

### Sequence length
- **seq_len=5** (exp85-87) improved D recall significantly (86% vs ~80%) — better "steering lock" for sustained turns. S recall was slightly lower. Promising but needs more coverage.

### Seed variance
- S recall at 5K ranged 45-78% across seeds. Single-seed comparisons with <15pt differences are noise.
- Reproducibility (exp84 vs exp88 exact rerun): metrics matched at 5K but diverged by 15K. Late-stage behavior is stochastic.
- Use `eval_iters >= 50` — eval_iters=12 was too noisy.

## Standout Proxy Experiments

### exp73 (128d, bs256, LR 5e-5, cos5000, drop 0.2)
- A rec **90.2%** / spec 75.7% — best steering of any run
- D rec 76.5% / spec **77.9%** — highest D specificity
- S rec **54.5%** — solid
- W rec 95.2% / spec 67.9%
- Only 15K steps on bs256 — likely undertrained

### exp66 (64d, bs256, LR 5e-5, cos5000, drop 0.2)
- Early peak at 2.5K: A rec 89.6%, D rec 80.4%, S rec 75.6%
- At 15K: A 80.3/82.8, D 84.2/71.2, S 43.6%, W 95.5/72.8
- Very balanced across the board

### exp78 (128d, bs512, LR 1e-4, cos5000, drop 0.3)
- S rec held at **63%** (vs 44% for 128d with drop 0.2)
- Dropout 0.3 specifically helps the larger model

## Full Training Plan

### Differences from proxy
- Real DINOv3 backbone (frozen, but 768-dim features directly available)
- Real-time augmentations (flip, warp, zoom) — not precomputed
- Projection is 768 -> transformer_dim (single step, not 768->128->dim)
- Much slower per iteration (DINO forward pass every step)
- 25K iterations (vs 15K in proxy)

### Settled parameters for all full runs
| Parameter | Value | Source |
|---|---|---|
| num_layers | 6 | Settled in proxy |
| num_heads | 4 | Settled |
| weight_decay | 0.075 | Settled (0.03 caused overfitting) |
| grad_clip | 1.0 | Settled |
| sequence_len | 3 | Default (seq5 promising but untested at full scale) |
| sequence_stride | 5 | Explicitly chosen |
| batch_size | 512 | Fits VRAM, better stability |
| warmup_iters | 200 | Standard |
| eval_iters | 50 | Minimum for reliable metrics |
| FP8 | auto | sm_120 Blackwell support |

### Run 1: 128d baseline (IN PROGRESS)
- `transformer_dim=128`, `dropout=0.2`
- `learning_rate=5e-5`, `muon_lr=0.0015` (the LR regime that produced the standout proxy experiments)
- `lr_decay_iters=5000`, `max_iters=25000`
- `label_smoothing=0.0`
- Rationale: 128d showed highest peak steering quality in proxy. Lower LR produced the best individual experiments. Fast cosine with long tail at min_lr.

### Run 2: 64d baseline
- Same as Run 1 but `transformer_dim=64`
- Rationale: 64d was the most stable proxy config. Direct comparison with 128d at same hyperparameters.

### Run 3: Winner + label smoothing
- Take the better config from Run 1 vs Run 2
- Add `label_smoothing=0.02` (or 0.01)
- Possibly increase dropout to 0.3 if 128d wins (exp78 showed this helps larger models)
- Rationale: label smoothing aligns with our "predict intent not noise" philosophy and improved W specificity in proxy

### Future considerations
- seq_len=5 with stride=5 if steering quality needs improvement
- Dropout 0.3 for 128d if S recall is a problem
- cos3000 if S collapse appears in full training
- Longer training (50K+) if models still look undertrained at 25K
