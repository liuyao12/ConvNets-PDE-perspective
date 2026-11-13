# QLNet NaN failure investigation (13 November 2026)

This is an **evidence-based, reproducible** erratum research note, not a confirmed reconstruction of the original ImageNet-100 training failure.

Read [REPORT.md](REPORT.md) for two independent demonstrated NaN mechanisms, the recovered 10M vs 16M source discrepancy, finite-forward/NaN-gradient bugs, fp16 bilinear overflow, and an independently traced small-data SGD failure.

Run the **minimal, self-contained regression probes**:

```bash
python research/qlnet-nan-diagnostics/quickcheck.py
```

The complete companion package provided in the associated research conversation includes: recovered checkpoint-compatible 10M/16M architectures, strict checkpoint-key/shape checker, forward/backward activation tracer, massive-activation recurrence analyzer, 7 regression tests, and exact JSON from the digits experiments.

**Evidence boundary:** We could access the public Hugging Face model listing and source but not download the published checkpoint weights into this runner. The exact historical failure cause is therefore unresolved. The original 2023 checkpoint should be paired with its exact historical source and tested in both FP32 and BF16 with the scanner; attributing the failure to particular outlier channels requires measured checkpoints and input images.

The original source is at https://huggingface.co/liuyao/QLNet .
