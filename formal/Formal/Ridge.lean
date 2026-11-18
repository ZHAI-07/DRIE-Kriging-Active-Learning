/-
  Kriging / kernel-ridge algebra.

  Claim in the project: `code/revision/validate_models.py` and the kernel-ridge
  exploration report that linear kernel ridge regression reproduces the ridge
  control to ~1e-15. That is not a numerical accident, it is an algebraic
  identity: the dual (kernel) coefficients and the primal (ridge) coefficients
  are related by `A⁻¹ T = T B⁻¹` whenever `A T = T B`, with
  `A = XᵀX + λI`, `B = XXᵀ + λI`, `T = Xᵀ`.

  Formalising it means no future agent can "rediscover" a different formula and
  present it as equivalent without the checker rejecting it.
-/
import Mathlib

open Matrix

namespace DRIE

/-- The structural identity `(XᵀX + λI) Xᵀ = Xᵀ (XXᵀ + λI)`.

This is the fact that makes the primal and dual ridge problems the same
problem, and it holds for every matrix `X` and every scalar `λ`. -/
theorem ridge_key {n m : Type*} [Fintype n] [Fintype m] [DecidableEq n] [DecidableEq m]
    (X : Matrix n m ℝ) (lam : ℝ) :
    (Xᵀ * X + lam • (1 : Matrix m m ℝ)) * Xᵀ
      = Xᵀ * (X * Xᵀ + lam • (1 : Matrix n n ℝ)) := by
  rw [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.one_mul,
      Matrix.mul_smul, Matrix.mul_one, Matrix.mul_assoc]

/-- Cancellation form of the primal/dual equivalence.

If `Ainv` is a left inverse of `A`, `Binv` a right inverse of `B`, and
`A T = T B`, then `Ainv T = T Binv`. With `A`, `B`, `T` as in ridge regression
this says the dual (kernel-ridge) coefficient operator equals the primal
(ridge) coefficient operator. -/
theorem krr_dual_eq_primal {n m : Type*} [Fintype n] [Fintype m]
    [DecidableEq n] [DecidableEq m]
    (A : Matrix m m ℝ) (B : Matrix n n ℝ) (T : Matrix m n ℝ)
    (Ainv : Matrix m m ℝ) (Binv : Matrix n n ℝ)
    (hkey : A * T = T * B) (hA : Ainv * A = 1) (hB : B * Binv = 1) :
    Ainv * T = T * Binv := by
  calc Ainv * T = Ainv * T * (B * Binv) := by rw [hB, Matrix.mul_one]
    _ = Ainv * (T * B) * Binv := by
          rw [← Matrix.mul_assoc (Ainv * T) B Binv, Matrix.mul_assoc Ainv T B]
    _ = Ainv * (A * T) * Binv := by rw [← hkey]
    _ = (Ainv * A) * T * Binv := by rw [← Matrix.mul_assoc Ainv A T]
    _ = T * Binv := by rw [hA, Matrix.one_mul]

/-- Scalar ridge objective: completing the square.

`(xb - y)² + λb² - λy²/(x²+λ) = (x²+λ)(b - xy/(x²+λ))²`.

This is the mechanism behind "ridge shrinks toward zero with strength
λ/(x²+λ)", quoted in the shrinkage reports. -/
theorem ridge_scalar_completion (x y lam b : ℝ) (h : 0 < x ^ 2 + lam) :
    (x * b - y) ^ 2 + lam * b ^ 2 - y ^ 2 * lam / (x ^ 2 + lam)
      = (x ^ 2 + lam) * (b - x * y / (x ^ 2 + lam)) ^ 2 := by
  field_simp
  ring

/-- Hence the ridge optimum `xy/(x²+λ)` is a global minimiser. -/
theorem ridge_scalar_min (x y lam b : ℝ) (h : 0 < x ^ 2 + lam) :
    y ^ 2 * lam / (x ^ 2 + lam) ≤ (x * b - y) ^ 2 + lam * b ^ 2 := by
  have hc := ridge_scalar_completion x y lam b h
  have hnonneg : 0 ≤ (x ^ 2 + lam) * (b - x * y / (x ^ 2 + lam)) ^ 2 :=
    mul_nonneg (le_of_lt h) (sq_nonneg _)
  linarith

end DRIE
