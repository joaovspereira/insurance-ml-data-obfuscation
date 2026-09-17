![Insurance ML & Data Obfuscation](assets/banner.svg)

**English** · [Português](README.pt-BR.md) · [Portfolio](https://github.com/joaovspereira)

# Insurance ML & Data Obfuscation

**Decision question:** Can an invertible transformation of features preserve ordinary least-squares predictions, and how should insurance models be evaluated?

## Published scope

A linear-algebra demonstration and corrected modeling pipeline based on the original insurance project. The numerical proof executes with synthetic data; the original insurance dataset must be supplied separately for empirical model evaluation.

## What this project demonstrates

- Prove OLS invariance under an invertible feature transform with an unchanged intercept.
- Use least-squares solvers instead of explicitly inverting the normal equations.
- Fit scaling inside training cross-validation, select kNN hyperparameters on training folds and report a held-out test.
- Report F1, RMSE and R² correctly with dummy baselines and one shared train/test split.

## Verified example

The executable example uses **180 synthetic training rows and 70 test rows**. It checks held-out OLS prediction equality at tolerance **1e-8** under a non-orthogonal invertible matrix and demonstrates recovery of the original synthetic features. See [the proof](docs/LINEAR_ALGEBRA.md) and [saved numerical output](reports/synthetic_demo.json).

## Improvements over the original project

- Removed the square root incorrectly applied to R² and fixed execution-order dependencies.
- Moved supervised scaling inside the training pipeline and stopped selecting k from test performance.
- Replaced inverse-based regression with stable least squares; made full-rank assumptions explicit.
- Demonstrated reversal explicitly and removed any implication that a linear transform provides cryptographic protection.
- Removed exact duplicate feature/target records before the split; this choice is reported in the result.

## Tools

Python · NumPy · pandas · scikit-learn · linear algebra · kNN · OLS · cross-validation

## Run locally

Use Python 3.12. From this repository's root:

```bash
python -m venv .venv
# Activate: source .venv/bin/activate (macOS/Linux)
# Activate: .venv\Scripts\Activate.ps1 (Windows PowerShell)
python -m pip install -r requirements.txt
python insurance.py
python -m unittest discover -s tests -v
python -m notebook notebooks/analysis.ipynb
```

Run all included checks with `python -m unittest discover -s tests -v`.

[Notebook](notebooks/analysis.ipynb) · [Implementation](insurance.py) · [Synthetic output](reports/synthetic_demo.json) · [Data requirements](data/README.md) · [Validation record](VALIDATION.md)

## Interpretation and limitations

The transformation is reversible: this is not encryption or anonymization. Full-rank training design and ordinary unregularized least squares are required for the held-out proof used here. Arbitrary transformations can change kNN distances and regularized models. The random split is educational; real applications need an evaluation design matched to deployment and careful assessment of sensitive features.

## Learning and next improvement

The public revision makes assumptions, units, denominators and validation boundaries explicit, so another analyst can inspect how results would be produced.

Rerun the original dataset with the corrected pipeline, report baseline comparisons and inspect sensitivity to split design and duplicate handling.

Educational portfolio project derived from work in the TripleTen Data Science Bootcamp, revised for public use in September 2026. Dataset files are not redistributed. Direct dependency versions are documented from the verification environment; a complete historical environment lock is not available.

[João Vitor Pereira](https://github.com/joaovspereira) · [Contact](mailto:joaovitorsouza20pereira@gmail.com)
