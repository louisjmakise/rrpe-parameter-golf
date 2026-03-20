# R-RPE Micro-LM for Parameter Golf

R-RPE (identity-projected reward prediction error) is a framework where prediction error emerges from divergence between internal representations rather than external reward.

This repository explores whether internal trajectory alignment can improve language modeling under extreme parameter constraints (≤16MB), where behavioral compression may be more efficient than scaling model capacity.

**Key result:** >60% reduction in hedging on 2B-class models (preprint + aggregate results).

## Why this matters

Classical reward prediction error assumes learning is driven by the gap between expected and obtained reward after interaction with reality.

R-RPE starts from a different premise:

> a system can adjust behavior based on anticipated divergence between internal states, even without external feedback.

This reframes learning as internal alignment rather than external correction.

## Core hypothesis

Under extreme parameter constraints, aligning internal representations can be more efficient than increasing model capacity.

Rather than adding more weights, R-RPE aims to compress behavior by constraining generation toward a more coherent internal trajectory.

## Minimal mechanism

At token step `t`:

- the model produces a token distribution `p_t`
- an internal reference distribution `I_t` is derived from hidden states through a lightweight projection head

A minimal objective is:

`L = CE + λ · KL(p_t || I_t)`

Possible implementation paths:

- decode-time mixing between model logits and reference logits
- training-time regularization via KL divergence

The intended effect is to reduce hedging, sharpen commitment in token selection, and improve directional consistency without materially increasing parameter count.

## Results

Prior 2B-scale runs (preprint + aggregate results) show **>60% hedging reduction**.

See:
- `results/results.csv`
- `METHOD.md`

## Preprint

Preprint: https://doi.org/10.5281/zenodo.17118156

## Repository scope

This repository is minimal by design for Parameter Golf: concept, method note, and aggregate results only.

The >60% hedging reduction comes from experiments conducted outside the challenge; the Parameter Golf objective is to adapt and test this mechanism in a ≤16MB micro-LM setting.

## Repository contents

- `METHOD.md` — high-level method note
- `results/results.csv` — aggregate result summary
- `CITATION.cff` — citation metadata
- `LICENSE` — MIT

## Citation

If you use this work, please cite the preprint metadata in `CITATION.cff`.

## Contact

louis.j@claritism.org