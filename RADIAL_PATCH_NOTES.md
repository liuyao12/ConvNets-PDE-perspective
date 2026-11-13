# Radial-activation NaN patch: installation and validation

This patch targets the `resnet.py` implementation used as a replacement for the
`timm.models.resnet` module. It does not make `resnet.py` a standalone package:
the file already imports `._builder`, `._manipulate` and `._registry`.

**When replacing the timm model module, copy both `resnet.py` and
`radial_activations.py` into the *same* `timm/models/` package directory.**
The additional relative import `from .radial_activations import hardball, softball`
requires the second file.

For standalone numerical tests, from the repository root run:

```sh
python -m unittest discover -s tests -p 'test_radial_activations.py'
```

The test suite is deliberately independent of the full timm training environment.
The `reset_classifier` fix in this branch also removes an unrelated hard-coded
99-class target.

**Scope:** The patch fixes real numerical errors in radial activations. It does not
explain the historical no-activation/ReLU training failures described in
arXiv:2308.01621v4: those experiments did not necessarily use these radial
operators. The research replication plan and distinct hypotheses are in the
agent-oriented follow-up research PR.

Tests should additionally be run with exactly the original torch/timm versions
and checkpoint loading scheme before merging into the original training setup.
