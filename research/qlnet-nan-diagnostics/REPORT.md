# QLNet NaN investigation — 13 November 2026

**For mathematical collaborators and autonomous agents.** The historical ImageNet failure is not yet reconstructed. This report separates *proved operator bugs*, *controlled reproductions*, and *unverified historical hypotheses*. Every measured result below has a runnable script in the accompanying downloadable reproducibility package.

## Executive conclusion

There are **two distinct, demonstrable routes to broken training** in the originally published architecture:

1. **An exact backpropagation defect** in `hardball`: the unselected branch of `torch.where(norm > R, R*x/norm, x)` is undefined at zero. A finite forward pass can poison gradients with NaNs. This is true even in float64. In a reconstructed 10M-parameter QLNet evaluated on an all-zero tensor, parameter gradients are nonfinite in 153 parameter tensors while the output and input gradient remain finite. A stable radial implementation fixes this failure.
2. **Unbounded bilinear products**: the block multiplies two learned features *before* output clipping. Under FP16, a pair `300*300` becomes infinity; further layers/BatchNorm can create NaNs. Original hardball can also **silently erase** features because its low-precision sum-of-squares overflows: at 512 channels, a vector of all 12s is mapped to zero instead of all 1s. Replacing hardball alone cannot prevent overflow *earlier in the block*.

A small-data training stress test also reproduced the *dynamical* blow-up mechanism in float32: an unclipped four-block network produced ~`1e26` features under a deliberately high SGD learning rate and first became nonfinite at the next bilinear product during validation. The same architecture with stable radial clipping finished the schedule. These findings justify serious concern about rare massive activations, but **do not prove** the actual published run failed for the same reason.

## A. Source and checkpoint provenance: mismatch confirmed at configuration level

Hugging Face model: https://huggingface.co/liuyao/QLNet

- `qlnet-10m.pth.tar`: 40.7 MB, uploaded December 6, 2023.
- `qlnet22-16m.pth.tar`: 64.6 MB.
- Current `qlnet.py`: revised April 29, 2024. The published file currently defaults to 22 blocks `[3,4,12,3]`, a `k=8/4/2` expansion, *interleaved channel pairing* and no stem/final ReLU/radial layer.
- An older indexed revision of `qlnet.py` has `[3,4,6,3]`, `k=4/2`, a depthwise convolution with twice as many output as input channels, split-half pairing, a ReLU stem, and hardball at every residual block (and again before classification).

Reconstructing the older source gives **9,993,640 parameters**, matching the naming/approximate FP32 size of the 10M checkpoint. The newer configuration gives **15,989,160 parameters**. This is **compelling but not final** evidence of which checkpoint belongs to which code: the checkpoint's own tensor shapes must be read. `diagnose_published_qlnet.py --check-only` performs that check, requires exact state-dictionary matching, and refuses unknown architectures.

**Additional packaging defect:** the current Hugging Face `qlnet.py` comments out the import of `create_classifier` but still calls that name in `__init__`. As published, that standalone Python file does not directly instantiate without an external import or patch. Our `historical_qlnet_10m.py` avoids that dependency.

## B. Operator-level test results

Original radial implementation, replicated directly from `qlnet.py`:

```python
norm = torch.sqrt((x*x).sum(dim=1, keepdim=True))
y = torch.where(norm > radius, radius*x/norm, x)
```

| Input/circumstance | Original code | Stable code |
|---|---|---|
| `x=0`, float32 | finite forward, **NaN backward** | finite forward and backward |
| `x=0`, float16 or bfloat16 | finite forward, **NaN backward** | finite forward and backward |
| 512 equal channels of value `12`, float16 | **all zero** | approximately **all one** |
| 8 channels of value `100`, float16 | **all zero** | approximately **all one** |
| `300 * 300` as an FP16 bilinear product | **Inf** | FP32 or BF16 remains finite |
| `1000 * 70` as an FP16 bilinear product | **Inf** | FP32 or BF16 remains finite |

The FP16 maximum finite value is 65,504. In the 512-channel norm example the sum of squares is `512*12*12=73,728`, so the norm overflows even though every input is a small finite FP16 value. The old implementation then takes `R*x/Inf = 0`, destroying the representation silently.

PyTorch gradcheck of stable hardball passed for zero, inside-ball and outside-ball sample inputs in float64. `test_nan_root_cause.py` contains seven passing tests including parameter-count recovery and the whole reconstructed 10M model at zero input.

## C. Directly instrumented QLBlock test — FP16 multiplication is the first failure

`qlblock_stress.py` uses a faithful quadratic residual block, with BatchNorm immediately before interleaved channel products. The experiment **deliberately** sets the BN affine gamma of one partnered pair to 100 or 300. It does not represent observed learned gamma values.

| Normalization gamma of a paired channel | FP32 | FP16 | BF16 |
|---:|---|---|---|
| 1 | finite | finite | finite |
| 30 | finite | finite | finite |
| 100 | finite | **first nonfinite at product** | finite |
| 300 | finite | **first nonfinite at product** | finite |

Changing only hardball to the stable implementation does **not** fix this early product overflow. Computing the entire residual block in a float32 autocast-exclusion region does fix the controlled forward failure. Inserting an additional stable radial clip **before** the bilinear product also fixes this controlled forward failure, even for gamma 1,000.

### Mathematically bounded pre-product operation

Let `z∈R^{2w}` be the pre-product feature and let a stable channelwise `hardball` project `z` onto the ball of radius `R=√(2w)`. Then, for *any* pair of coordinates:

\[
|z_{2j}z_{2j+1}| \leq \tfrac12(z_{2j}^2+z_{2j+1}^2)\leq \tfrac12\|z\|^2\leq w.
\]

The split-half version obeys the same bound. In the published 10M configuration, the largest `w` is on the order of 1,000, comfortably below 65,504. Thus an FP32-implemented radial preclip with a cast back to FP16 **mathematically prevents the product itself from overflowing**, for finite inputs, up to rounding. It does **not** control subsequent convolutions and BN if their weights explode, and it **changes the function** of a pretrained model. It should be treated as a trainable architectural variant rather than a drop-in checkpoint-preserving numerical patch.

### The nonlinearity's differential

For `F(u)=W[(Au)⊙(Bu)]`,

\[
DF(u)[h]=W[(Ah)⊙(Bu)+(Au)⊙(Bh)].
\]

Its operator norm may grow linearly in `||u||`, producing amplification under repeated residual updates `u→u+F(u)`. This is a concrete mechanism; one need not assume the learned system is hyperbolic or interpret the event as a PDE shock.

## D. Genuine SGD stress test on real handwritten digits (not ImageNet)

`digits_training_replay.py` uses 1,797 images from scikit-learn digits, a fixed stratified 70:30 split (`random_state=42`), four quadratic QLNet-like blocks, batch size 64, SGD momentum 0.9, seed 0, no GPU, and float32 except where noted. These images are **not** a claim about the ImageNet architecture's final accuracy.

| LR | Clip after every block? | Outcome |
|---:|---|---|
| 0.1 | yes | 15 epochs, validation accuracy 98.15%, no NaNs |
| 0.1 | no | 15 epochs, validation accuracy 97.59%, no NaNs |
| 0.5 | no | 12 epochs, validation accuracy 91.67%, no NaNs |
| 0.5 | yes | 12 epochs, validation accuracy 93.70%, no NaNs |
| 0.65 | no | 24 epochs, no NaNs; accuracy degraded |
| 0.75 | no | 24 epochs, no NaNs; accuracy degraded |
| 0.85 | no | validation failed at epoch 2 |
| 1.0 | no | **validation failed at epoch 1** |
| 1.0 | yes | **12 epochs completed, 94.07% validation accuracy** |

The separate `trace_training_failure.py` reproduces and localizes the LR=1.0 run (no manual activation injection). Training minibatch outputs and parameter gradients remained finite through epoch 1. **The first failing validation sample was dataset index 1008.** Its trace showed the previous block's post-residual activations of approximately `2.12×10^26`, and the next block's pre-product BN output of approximately `9.70×10^26`. The **first nonfinite tensor** occurred at `layer4.0/BILINEAR_PRODUCT`, where 1,024 output elements were nonfinite. The parameters and BN buffers flagged by the inspection were all finite, though some BN running variances were as large as `9.66×10^8`.

This example is relevant to the paper's description of training minibatches functioning until certain validation images fail; it does **not** identify the historical training epoch or distribution of outliers. The artificially large learning rate is intended as a mechanism stress test only.

## E. Relation to the Zhuang Liu massive-activations observation

The relevant source is Sun, Chen, Kolter and Zhuang Liu, *Massive Activations in Large Language Models*, COLM 2024 (https://arxiv.org/abs/2402.17762). The authors report rare, large-magnitude, sometimes nearly input-independent features functioning as implicit bias terms. The QLNet-specific question is not merely whether `max(abs(z))` is large; it is **whether two partnered factors** are jointly large enough to overflow in FP16, and whether the same channel/spatial positions show recurring outliers across examples.

The scanner records per block, separately at `conv1_BN`, `BILINEAR_PRODUCT`, `bn2`, the residual sum and the radial output:

- `maxabs`, sampled 99th percentile, and max-to-99th-percentile ratio;
- single-channel energy concentration and the coordinates of the maximal absolute activation;
- recurrent top channels and their signed values across inputs;
- estimated float32 product magnitude before multiplication, and count of pairs exceeding the FP16 range;
- the first nonfinite forward/backward operation, including the stage/block, batch and sample;
- BN affine magnitudes and `|gamma| / sqrt(running_var + eps)` gains.

This allows a direct test of the implicit-bias / extreme-outlier analogy on the **actual checkpoint**. The available public checkpoint files could not be downloaded into this execution container. Thus **no claim is made** that they have massive activations or that the published model's values match the synthetic probes.

## F. Reproduce the checkpoint investigation on an Internet-connected machine

```bash
pip install torch torchvision numpy pillow huggingface_hub scikit-learn
hf download liuyao/QLNet qlnet-10m.pth.tar --local-dir .
python diagnose_published_qlnet.py --checkpoint qlnet-10m.pth.tar \
    --check-only --output checkpoint_signature.json
python diagnose_published_qlnet.py --checkpoint qlnet-10m.pth.tar \
    --image-dir /path/to/imagenet/val --limit 128 --precision fp32 \
    --radial legacy --output original_fp32.json
python diagnose_published_qlnet.py --checkpoint qlnet-10m.pth.tar \
    --image-dir /path/to/imagenet/val --limit 128 --precision fp32 \
    --radial stable --output stable_fp32.json
python diagnose_published_qlnet.py --checkpoint qlnet-10m.pth.tar \
    --image-dir /path/to/imagenet/val --limit 128 --precision fp16 \
    --device cuda --radial stable --output stable_fp16.json
python diagnose_published_qlnet.py --checkpoint qlnet-10m.pth.tar \
    --image-dir /path/to/imagenet/val --limit 128 --precision bf16 \
    --device cuda --radial stable --output stable_bf16.json
python analyze_massive_activations.py original_fp32.json \
    --out candidate_bias_channels.json
```

For diagnostics with a frozen checkpoint, use `--mode eval`; to test derivatives, use `--backward` (warning: it currently supplies **dummy class-0 labels**, useful only for NaN detection, not historical loss reproduction). To test which remedy fixes the exact product overflow, use `--guard fp32` or `--guard preclip`. Both are **experimental**; `preclip` alters the model prediction and requires retraining/validation. The code refuses unknown checkpoint shapes rather than silently loading incompatible weights.

To attribute the **historical** event to one cause, recover the **pre-failure model, optimizer, GradScaler, RNG and data-shuffle states**. Replay *the same minibatch*, log first invalid forward/intermediate/backward/update, and repeat with only one intervention at a time. The final best-performing checkpoint by itself cannot determine the training failure's cause.

## G. Recommended fixes and publication claims

**Checkpoint-preserving numerical repairs:** use the corrected radial norm/gradient calculations, retaining the same state-dictionary keys; test first in pure FP32, then BF16. Consider an FP32 region around the complete quadratic block as an ablation to separate numerical range failures from model dynamics. Record first-nonfinite diagnostics and skip an optimizer update if gradients become invalid. Clipping gradients only helps before an intermediate becomes infinite; it cannot repair an already nonfinite forward pass.

**Architecture-level research changes:** add radial normalization **before** the bilinear product, with the analytical product bound above, or add constrained residual steps/weight norms. Compare against post-block radial clipping and BF16/FP32, over multiple training seeds, on an ImageNet-sized task. Check whether an added pre-product radial activation alters the claimed change-of-channel-basis symmetry (the external basis change can keep internal projected features unchanged by transforming incoming weights accordingly). Measure accuracy before claiming a solution.

**Defensible erratum today:** the original radial implementation has an actual undefined-gradient bug and low-precision norm overflow, and the public code/checkpoint revisions are not aligned. The architecture also has a genuine FP16 product-overflow vulnerability, and an unclipped small-scale variant can develop catastrophic activation growth. **Not defensible yet:** the statement that massive activations caused the particular historical two-thirds-of-training failure, or that the 10M checkpoint has been evaluated/retrained here.

## Contents and reproducibility

- `historical_qlnet_10m.py` — 9,993,640-parameter source-recovered architecture, compatible in structure with the older indexed source (actual checkpoint not locally tested).
- `faithful_qlnet.py` — 15,989,160-parameter current-2024-source replica.
- `radial_activations.py` — stable hardball and softball with zero-gradient and FP16 fixes.
- `precision_probes.py`, `precision_results.json` — 111 deterministic arithmetic probes.
- `qlblock_stress.py`, `qlblock_stress_results.json` — first-nonfinite operator under controlled BN amplification.
- `digits_training_replay.py`, `high_lr_results.json`, `lr_sweep_results.json` — dataset and training tests.
- `trace_training_failure.py`, `high_lr_trace.json` — independent first-nonfinite trace of a genuinely trained but deliberately unstable minibatch model.
- `diagnose_published_qlnet.py`, `analyze_massive_activations.py` — image/checkpoint scanner and recurrence analysis.
- `test_nan_root_cause.py` — seven passing PyTorch unit tests including a full 10M-model zero-input backward regression.

**References:** [Original paper, arXiv:2308.01621v4](https://arxiv.org/html/2308.01621v4); [published Hugging Face model](https://huggingface.co/liuyao/QLNet); [Massive Activations (Sun et al., 2024)](https://arxiv.org/abs/2402.17762).