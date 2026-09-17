# Publication validation — 2026-09-17

## Provenance and scope

Original source: `Sprint 11 - Álgebra Linear.ipynb`. SHA-256: `bf77e8c68d37fd87ad08f1433f1090ba1820503fb6c4f4a606c97ae9a5e17b6d`.

This is a substantive revision of the implementation, not a relabeling of old outputs. No original-data training or analysis run is claimed. No empirical business result from the original notebook is presented as a result of this corrected code.

## Corrections

- Removed the square root incorrectly applied to R² and fixed execution-order dependencies.
- Moved supervised scaling inside the training pipeline and stopped selecting k from test performance.
- Replaced inverse-based regression with stable least squares; made full-rank assumptions explicit.
- Demonstrated reversal explicitly and removed any implication that a linear transform provides cryptographic protection.
- Removed exact duplicate feature/target records before the split; this choice is reported in the result.

## Executed checks

- `python -m unittest discover -s tests -v`: **4 tests passed**.
- Python modules parsed and imported successfully in the verification environment.
- All notebook code cells ran sequentially in an isolated Python process with their default synthetic example and explicit original-data skip; captured output is included. The environment blocked the socket-based Jupyter kernel, so this is a Python execution check rather than a Jupyter-kernel run. No notebook magics or widget execution are used.
- `reports/synthetic_demo.json` contains the generated example results, labeled as synthetic.

## Verification environment

Python 3.12. Direct library versions observed during checks:

- pandas 2.2.3
- numpy 2.3.5
- scikit-learn 1.8.0

## Remaining empirical work

Rerun the original dataset with the corrected pipeline, report baseline comparisons and inspect sensitivity to split design and duplicate handling.

The transformation is reversible: this is not encryption or anonymization. Full-rank training design and ordinary unregularized least squares are required for the held-out proof used here. Arbitrary transformations can change kNN distances and regularized models. The random split is educational; real applications need an evaluation design matched to deployment and careful assessment of sensitive features.
