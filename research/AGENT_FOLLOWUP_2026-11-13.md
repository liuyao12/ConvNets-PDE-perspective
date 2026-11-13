# QLNet / PDE-CNN: stability, symmetry and private inference

**Research continuation packet — 2026-11-13.**  For AI agents and mathematical collaborators.  Status: *small CPU experiments and mathematical derivations; not a replication of ImageNet-100; not a cryptographic security proof.*

Original: Yao Liu, Hang Shao, Bing Bai, *A Novel Convolutional Neural Network Architecture with a Continuous Symmetry*, arXiv:2308.01621v4 (2024): https://arxiv.org/html/2308.01621v4. Original code: https://github.com/liuyao12/ConvNets-PDE-perspective (`resnet.py`, `PDEBlock`, `nn_symmetry.ipynb`). Existing proposed fix: https://github.com/liuyao12/ConvNets-PDE-perspective/pull/10.

## Scope, evidence and non-claims

- [OBSERVED] The original `hardball` backward pass produces NaN gradients at the zero channel vector because `torch.where` evaluates the `R*x/norm` branch even when `norm=0`. Forward values remain finite, which hides the defect. This is a bug in the *current code*, not a proven historical cause of the failed activation-free/ReLU run.
- [OBSERVED] Old `hardball` and `softball`, in float16, map input channel vectors with eight entries of `10000` to **zero** due to intermediate square overflow; corrected versions output approximately 1 in each channel. This is silent wrong arithmetic rather than necessarily NaN.
- [HYPOTHESIS] The paper's late-training NaNs without bounding activations could arise from unstable quadratic residual growth, optimizer overshoot, BatchNorm interactions or floating-point overflow; one must replay a historical checkpoint to determine which operation first becomes nonfinite. Do not claim to know why the original run failed at roughly two-thirds of training.
- [MATHEMATICAL] A cubic potential with fully symmetric third-order coefficient tensor gives a symmetric-hyperbolic conservation law. A particular entropy-conservative flux with positive viscosity has a provable **semidiscrete** energy decay; this does NOT by itself prove stable finite-step integration or superior classification.
- [OBSERVED] Orthogonal feature changes of basis can preserve the network function to ~1e-15 in a small float64 quadratic block. A known-model attack reconstructs an orthogonal key to ~1e-15 from public original and transformed weights. Such a transformation must not be described as encryption.
- [NOT EXECUTED] Actual CKKS ciphertext evaluation, cryptographic latency/security benchmarking and any ImageNet-100 replication. No TenSEAL/OpenFHE/Pyfhel installation or GPU was available locally. Polynomial arithmetic was investigated only in plaintext.

## A. Audit and errata candidates

Paper arXiv v4 §3 (lines 108–123 HTML): NaN training and clipping reported; shock formation suggested, not demonstrated. §5.1 explicitly says the network *does not enforce* hyperbolicity. §4 distinguishes GL(n) without radial nonlinearities from O(n) with hardball/softball. Preserve these distinctions.

Source code `resnet.py`:

1. `hardball`: old `torch.where(norm > R, R*x/norm, x)` has an undefined *unselected* branch at zero; backward propagates NaNs. Replace with stable scaled norm and no zero division; test forward/backward and mixed precision. Patch in PR #10.
2. `softball`: direct `x*x` and `(x*x).sum()` in float16 overflows; compute stable channel scales in float32 or higher. Patch in PR #10.
3. `reset_classifier`: `create_classifier(self.num_features, 99, ...)` incorrectly hard-codes 99 rather than `self.num_classes`. Fixed in PR #10; this is separate from NaNs.
4. The file is a **timm `models/resnet.py` replacement** and includes relative imports of timm internals; if keeping an external `radial_activations.py`, copy it into the same timm package directory. Never test a standalone `import resnet` and assume it works.
5. For a change of channel basis at a block boundary, the pre-product 1×1 convolution must be right-multiplied by the inverse input basis, the final 1×1 convolution must be left-multiplied by the output basis, and the skip path must be transformed. BatchNorm on output/skip branches must be fused at inference before applying arbitrary orthogonal changes of basis. `nn_symmetry.ipynb` is a starting point, not a general security proof.
6. The code's `PDEBlock.conv2` is depthwise with independent kernels per hidden channel; its `conv_forward` helper intended for sharing is not called by `forward`. Boundary-gauge changes of basis nevertheless work if internal (post-mixing) hidden channels are not rotated. Do not mistake this for an arbitrary rotation inside a depthwise convolution.

## B. Reproducible CPU experiments

Requirements: Python 3.13, torch 2.10.0+cpu, numpy, scipy, sklearn. Device: CPU, two PyTorch threads. No network access or external dataset.

```
cd qlnet_followup
python experiment_suite.py --mode nontrain --out nontrain_results.json
python experiment_suite.py --mode train --epochs 10 --seeds 0 1 --out digits_10epochs_2seeds.json
python readout_experiment.py
```

`experiment_suite.py` contains an independent simplified quadratic ConvNet, NOT the original full timm/PDEBlock + BatchNorm model. Scikit-learn digits: 1,797 8×8 single-channel images, 1,257 train and 540 test, stratified split `random_state=42`, pixels divided by 16, Adam lr=0.01, batch size 96, ten epochs, seeds 0/1. Four residual blocks, 12 channels, expansion factor 2; no BN, no pretrained weights. Do not extrapolate the accuracy to ImageNet.

| simplified model | test accuracy seed 0 | seed 1 | nonfinite failures |
|---|---:|---:|---:|
| quadratic, no clipping | 0.7111 | 0.5611 | 0/2 |
| quadratic, softball at every block | 0.8333 | 0.8593 | 0/2 |
| quadratic, dt=0.125 + softball | 0.6241 | 0.4204 | 0/2 |
| symmetric flux, dt=0.04 + global average | 0.2167 | 0.2796 | 0/2 |

At the more aggressive Adam lr=0.03, ten epochs, test accuracies were 0.122/0.211 (unclipped); 0.683/0.878 (softball); 0.674/0.444 (dt=0.125+softball); 0.415/0.467 (flux). All runs remained finite; the unclipped models often had optimization problems without numerical NaNs. Small-data exploratory evidence favors the stabilizing activation, not a quantitative comparison of final architectures.

A *critical issue* with the symmetric flux classifier is that periodic spatial flux preserves each channel's spatial mean, so a global-mean readout is blind to the flux dynamics and its parameters receive zero gradients. Verified directly to 5.96e-8 in float32 for the means and exactly zero parameter gradients. In ten-epoch follow-up trials, changing the final readout to flatten the spatial feature map gave test accuracies 0.9759/0.9704 (flux) vs 0.9722/0.9759 (stem-only); **no demonstrated gain from flux**, but the zero-gradient obstruction is removed. A radial weighted mean produced 0.6296/0.6241 for the flux model.

Independent diagnostic results from `nontrain_results.json`:

- Float32 zero vector, old `hardball`: **144 NaN gradient entries out of 144**; patched hardball/softball: finite gradients. Same pathological case observed in fp16/bfloat16.
- Float16 channel vector `[10000]*8`: old hardball and softball both return zero; patched both return approximately `[1]*8`.
- A 6-channel quadratic block reparameterized with distinct orthogonal input and output bases: maximum output covariance error `9.99e-16` in float64.
- Known-weight key recovery: maximum recovered matrix-entry error `5.59e-16` in float64; server is supplied the original and conjugated last projection weights.
- For a periodic 1D flux with random 4-channel state and symmetric rank-7 tensor: measured semidiscrete energy derivative `-18.461655837363917`, matching the independent dissipation formula at displayed precision; orthogonal covariance error `2.22e-16`.
- Scalar quadratic recurrence `u[n+1]=u[n]+dt*u[n]^2` from u0=0.4: dt=1 first exceeds 10^6 after 8 steps; dt=0.05 after 58 steps; either may eventually diverge. Post-update softball keeps it bounded through 100 steps. This is a mechanism illustration, not a historic training reconstruction.

## C. A theorem-level O(n)-covariant symmetric-hyperbolic architecture

For each spatial direction j, choose rank R, vectors `v[j,r] ∈ R^C` and scalars `lambda[j,r]`; define

```
Phi_j(u) = (1/6) sum_r lambda[j,r] (v[j,r] dot u)^3
F_j(u)   = grad Phi_j(u)
         = (1/2) sum_r lambda[j,r] v[j,r] (v[j,r] dot u)^2.
```

The PDE is `u_t + sum_j d_j F_j(u) = 0`. Its principal matrices

```
A_j(u) = D F_j(u) = sum_r lambda[j,r] (v[j,r] dot u) v[j,r] v[j,r]^T
```

are real symmetric, so every directional combination is symmetric with real eigenvalues; the system is symmetric hyperbolic for smooth states. The constant identity matrix is a symmetrizer.

Under `u' = Q u`, `v'[j,r] = Q v[j,r]` with `Q ∈ O(C)`, we have `F'_j(Qu)=Q F_j(u)`. The rank-R parameterization is preserved, as is the Euclidean energy. This simultaneously provides **exact O(C) covariance, symmetric hyperbolicity and O(RC) flux evaluation** (up to spatial work).

### Entropy-dissipative flux proof (periodic one-dimensional grid)

Let `u_i` be the state in cell i. Define the *entropy-conservative* interface flux

```
f_ec(a,b) = integral_0^1 F(a+t(b-a)) dt
          = (1/6) sum_r lambda_r v_r [p_r(a)^2 + p_r(a)p_r(b) + p_r(b)^2],
      p_r(u) = v_r dot u.
```

For quadratic F, this expression is exact. Because `F=grad Phi`, the fundamental theorem of calculus gives

```
(b-a) dot f_ec(a,b) = Phi(b) - Phi(a).
```

Add a dissipative flux `f_num(a,b)=f_ec(a,b) - (alpha/2)(b-a)` with `alpha≥0`. The semidiscrete finite-volume law

```
d u_i/dt = -[f_num(u_i,u_{i+1})-f_num(u_{i-1},u_i)] / dx
```

satisfies, for `E=(dx/2) sum_i ||u_i||^2`, the exact entropy inequality

```
d E/dt = -(alpha/2) sum_i ||u_{i+1}-u_i||^2 <= 0.
```

**This is only semidiscrete.** A fully discrete Euler/RK scheme still requires a step-size constraint or another energy-stable time integrator. The coefficient alpha can be related to `||A_j(u)||`, with an elementary bound `||A_j(u)|| <= ||u|| sum_r |lambda_r| ||v_r||^3`. One must validate a globally meaningful CFL rule and its training gradient before claiming guarantees.

### Homomorphic-friendly optimization

Each projected channel `p_r(u)` is linear. The face-flux polynomial requires only two ciphertext–ciphertext products per rank and direction, using

```
p^2 + pq + q^2 = (3/4)(p+q)^2 + (1/4)(p-q)^2.
```

Thus each explicit flux layer adds **one ciphertext multiplication depth**, ignoring the implementation-dependent relinearization/rescaling/packing details. The original quadratic blocks similarly have degree two, but the radial activations are not native CKKS operations. It is potentially advantageous to train an entropy-stable flux network with **no internal radial activations** and to keep the spatially sensitive classifier head on the client. Whether this retains useful accuracy is an OPEN question.

## D. Orthogonal private inference: functionality is not security

For quadratic block `H(x + W_out K((A x) ⊙ (B x)))` and orthogonal input/output bases `Q_in,Q_out`, transform

```
A' = A Q_in^T       B' = B Q_in^T
W_out' = Q_out W_out
skip' = Q_out Q_in^T
```

(and the output affine biases analogously). Then the transformed block, receiving `Q_in x`, emits exactly `Q_out` times the original output, provided the radial H is O(C)-equivariant and output-path BN is fused. Independent Q per block turns identity skips into dense 1x1 convolutions, potentially losing efficiency; a shared Q across a stage avoids this.

**Known-model attack:** If untransformed `W_out` is public and has full row rank, while the server receives `W_out' = Q_out W_out`, then

```
Q_out = W_out' W_out^T (W_out W_out^T)^{-1}.
```

The test's recovery error was `5.59e-16`. Even without original weights, `Q` preserves feature norms and pairwise dot products. This form of orthogonal obfuscation is not semantic security. A meaningful threat model must specify whether weights, calibration pairs, prompts/images and rotated observations are known and whether fresh keys are used. The recent ConjFormer paper (arXiv:2606.16461) is directly relevant but does not make this scheme cryptographically secure.

## E. CKKS route: polynomial feasibility, not yet ciphertext evidence

For a smooth radial operation, try `x -> x p_d(s)` with `s=||x||^2/R^2` and a polynomial approximation of `(1+s)^(-1/2)`. For any polynomial p, `Qx` obeys exact O(C) equivariance because `||Qx||^2=||x||^2`.

Empirical least-squares Chebyshev fits (uniform grid of 20,001 points) for `s ∈ [0,4]` give the following **non-certified** maximal errors:

| degree d | max sample error on [0,4] | error at s=16 (out of range) | naive Horner multiplication depth including s and x·p |
|---:|---:|---:|---:|
| 3 | 0.021279 | 23.868 | 5 |
| 5 | 0.003103 | 538.208 | 7 |
| 7 | 0.000452 | 13029.593 | 9 |

The absurd out-of-domain errors demonstrate the danger of polynomial activations without certified input ranges. The original sigmoid/softball behavior is globally bounded, but any nonconstant polynomial cannot preserve that global bound. The 2026 *Encrypted Neural Networks without Overflows* paper (arXiv:2605.23096) develops relevant range certification methods.

**Actual CKKS NOT RUN.** TenSEAL/OpenFHE/Pyfhel are unavailable in this execution environment; neither latency nor ciphertext security was measured. A successful future demo must use an actual HE runtime, with a stated estimated security level, ciphertext parameters, ciphertext sizes, packing/rotation keys, latency and measured error against plaintext inference. A plaintext polynomial emulation must never be labeled encrypted inference.

## F. Agent handoff: prioritized investigations and acceptance criteria

**A1 — Historical NaN reproduction (blocking for definitive erratum).** Find checkpoint immediately preceding original late failure, training logs, optimizer state, scaler state, exact code hash, timm/torch versions, RNG seeds, dataset ordering. Re-run deterministic single batch with module hooks (`torch.isfinite`, absolute maximum, gradient norm, `detect_anomaly`) at each ConvBN, bilinear product, skip and radial unit. Mark first nonfinite forward/backward/update operation. Change *one variable at a time* (FP32, lower LR, gradient clipping, stable radial ops, scalar dt, symmetrized flux). ACCEPT: first failure operation found and same transition reproducible on ≥2 executions, or clearly report unavailable checkpoint.

**A2 — Radial patch review.** Run PyTorch gradcheck, near-zero tests, fp16/bfloat16/float32/float64 stress, compare legacy vs patched accuracy, verify O(n) covariance and consistency at clipping radius, check timm import compatibility. ACCEPT: no nonfinite gradients; maximum errors documented; old checkpoint weights still load; test suite passes on supported PyTorch versions.

**A3 — Symmetry/gauge reparameterization.** Formalize affine/BN-fused transformations, residual skip projection, 1×1 and 3×3 channel mixing, stage-shared vs per-block Q. Measure exact output equality before/after (FP64/FP32) on sample data and profile additional FLOPs. ACCEPT: quantified equivalence error and overhead per model size, with explicit counterexamples for unfused BN and unsupported channelwise operations.

**A4 — Symmetric-hyperbolic flux and classifier readout.** Prove semidiscrete entropy formula; derive and test a finite-step condition under a global state range; implement trained low-rank cubic flux network. Account for the conserved-mean/global-average readout obstruction by using a spatially sensitive *client-side* head or other compatible operation. ACCEPT: no zero gradients in flux parameters and meaningful improvement vs stem-only baseline on multiple seeds and at least one more demanding dataset. Do not report small-digits performance as solved.

**A5 — Privacy attack benchmark.** Implement known-public-weights key recovery, known input/rotated feature pairs, Gram-matrix leakage, noisy observations, pseudorandom per-query Q and model-knowledge scenarios; measure reconstruction/attack accuracy, bandwidth, and rekeying expense. ACCEPT: stated threat model, attack success, and clear boundary between obfuscation and cryptographic encryption.

**A6 — CKKS proof of concept.** Start with a learned ≤4-block flux network whose operations are quadratic and whose readout runs on client. Implement in TenSEAL/OpenFHE or equivalent; align same plaintext model and encrypt inputs; measure output error, run time, ciphertext traffic, multiplicative depth, key material, estimated bit security. Compare with an optional polynomial radial layer using mathematically bounded approximation intervals. ACCEPT: actual ciphertext processing with no secret-key exposure to server, accuracy/error numbers and reproducible code.

**A7 — Formal mathematics (Lean optional).** Formalize `(b-a)·f_ec(a,b)=Phi(b)-Phi(a)`, entropy sum identity on a cyclic grid, orthogonal covariance of low-rank cubic flux and of `x p(||x||²)`, and a counterexample showing orthogonal obfuscation preserves Gram matrices. ACCEPT: proof without admitted axioms beyond core library facts, with programmatic numerical tests as cross-checks.

## References

- Liu–Shao–Bai (2024 version): https://arxiv.org/html/2308.01621v4
- Yukhimchuk–Shulga–Kolar–Takáč, *Privacy from Symmetry*, 2026: https://arxiv.org/abs/2606.16461
- Kern et al., *Encrypted Neural Networks without Overflows*, 2026: https://arxiv.org/abs/2605.23096
- Source repository: https://github.com/liuyao12/ConvNets-PDE-perspective