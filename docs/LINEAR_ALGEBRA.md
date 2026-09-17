# Why ordinary least-squares predictions can stay the same

Let X be the feature matrix, P an invertible square matrix, w the feature
coefficients and b the intercept. Transform features to Z = XP.

For any w, set v = P⁻¹w. Then Zv + b = XPP⁻¹w + b = Xw + b.
The transformation therefore preserves the set of possible prediction vectors
and the minimum achievable least-squares objective.

For the held-out numerical check, the augmented training matrix [1, X] has full
column rank. Its unique optimum maps to the unique optimum for [1, XP], with
the same intercept. This makes predictions equal on new points transformed
with the same P, up to floating-point error.

With rank deficiency, independent minimum-norm solutions can extrapolate
differently. Regularization also changes the objective: arbitrary invertible
transformations do not generally preserve ridge/lasso or kNN behavior.

The implementation uses `numpy.linalg.lstsq`, not the explicit inverse of XᵀX.
A QR-derived rotation combined with unequal scale factors gives a well-conditioned,
invertible **non-orthogonal** test matrix. The intercept is never multiplied by P.

## Reversibility

X can be recovered from Z and P by solving XP = Z. This is not encryption,
anonymization, differential privacy or a guarantee that customer information is
protected. The example explicitly demonstrates recovery with synthetic records.
