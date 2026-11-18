/-
  Formal statements for the DRIE-Kriging-Active-Learning project.

  Scope, stated honestly:

  Lean can machine-check the *mathematics* that the code relies on. It cannot
  check that a particular numerical experiment on 23 records produced a
  particular R^2, because that would require formalising the floating-point
  evaluation itself. So the theorems below pin down the algebraic and
  combinatorial facts behind the project's claims; the empirical numbers stay
  the responsibility of the reproducible Python code.
-/
import Mathlib

open Finset BigOperators

namespace DRIE

/-! ### 1. Normalisation used by the acquisition function

`code/research/core.py:unit_range` maps a vector to [0,1] by subtracting the
minimum and dividing by the range, returning all-ones when the range is
numerically zero. The theorem below is the `extent > 0` branch: the rescaled
value of any member of the interval `[lo, hi]` lies in `[0,1]`, and the
endpoints map exactly to 0 and 1. -/

theorem unit_range_mem {lo hi x : ℝ} (h : lo < hi) (hx : lo ≤ x) (hx' : x ≤ hi) :
    0 ≤ (x - lo) / (hi - lo) ∧ (x - lo) / (hi - lo) ≤ 1 := by
  have hpos : 0 < hi - lo := by linarith
  constructor
  · exact div_nonneg (by linarith) (le_of_lt hpos)
  · rw [div_le_one hpos]; linarith

theorem unit_range_left {lo hi : ℝ} (_h : lo < hi) :
    (lo - lo) / (hi - lo) = 0 := by
  simp

theorem unit_range_right {lo hi : ℝ} (h : lo < hi) :
    (hi - lo) / (hi - lo) = 1 := by
  field_simp

/-! ### 2. The `beats_mean` screening rule

`code/research/optimization.py:optimization_readiness` marks a response as
acceptable only when the model's RMSE is below the training-mean baseline. The
saved metric is R², so the screen is equivalent to `R² > 0`. This theorem makes
that equivalence explicit, which is what justifies reading the saved `r2`
column as a "better than the mean predictor" test. -/

theorem r2_pos_iff_better_than_mean {sse sst : ℝ} (h : 0 < sst) :
    0 < 1 - sse / sst ↔ sse < sst := by
  rw [sub_pos, div_lt_iff₀ h, one_mul]

theorem r2_nonpos_iff_not_better {sse sst : ℝ} (h : 0 < sst) :
    1 - sse / sst ≤ 0 ↔ sst ≤ sse := by
  rw [sub_nonpos, le_div_iff₀ h, one_mul]

/-! ### 3. Why the mean minimises squared error

The `Mean` baseline in `basic_evaluation` predicts the training mean, and the
NRMSE is normalised by the total sum of squares `SST`. This theorem is the
foundation of both: over a finite non-empty index set, the sum of squared
deviations is minimised exactly at the arithmetic mean. Consequently no model
can have `R² > 1`, and `R² ≤ 0` really does mean "no better than the mean". -/

theorem mean_minimizes_sse {ι : Type*} [Fintype ι] [Nonempty ι]
    (y : ι → ℝ) (c : ℝ) :
    ∑ i, (y i - (∑ j, y j) / (Fintype.card ι : ℝ)) ^ 2
      ≤ ∑ i, (y i - c) ^ 2 := by
  classical
  set μ : ℝ := (∑ j, y j) / (Fintype.card ι : ℝ) with hμ
  have hcard_pos : (0 : ℝ) < (Fintype.card ι : ℝ) := by
    exact_mod_cast Fintype.card_pos
  have hcard_ne : (Fintype.card ι : ℝ) ≠ 0 := ne_of_gt hcard_pos
  have hsum : ∑ i, (y i - μ) = 0 := by
    rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, hμ, Finset.card_univ]
    field_simp
    ring
  have hdecomp : ∑ i, (y i - c) ^ 2
      = ∑ i, (y i - μ) ^ 2 + (Fintype.card ι : ℝ) * (μ - c) ^ 2 := by
    have hexpand : ∀ i, (y i - c) ^ 2
        = (y i - μ) ^ 2 + 2 * (μ - c) * (y i - μ) + (μ - c) ^ 2 := by
      intro i; ring
    calc ∑ i, (y i - c) ^ 2
        = ∑ i, ((y i - μ) ^ 2 + 2 * (μ - c) * (y i - μ) + (μ - c) ^ 2) := by
          exact Finset.sum_congr rfl fun i _ => hexpand i
      _ = ∑ i, (y i - μ) ^ 2 + ∑ i, 2 * (μ - c) * (y i - μ)
            + ∑ i, (μ - c) ^ 2 := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
      _ = ∑ i, (y i - μ) ^ 2 + 2 * (μ - c) * (∑ i, (y i - μ))
            + (Fintype.card ι : ℝ) * (μ - c) ^ 2 := by
          rw [← Finset.mul_sum, Finset.sum_const, nsmul_eq_mul, Finset.card_univ]
      _ = ∑ i, (y i - μ) ^ 2 + (Fintype.card ι : ℝ) * (μ - c) ^ 2 := by
          rw [hsum]; ring
  rw [hdecomp]
  have hnonneg : 0 ≤ (Fintype.card ι : ℝ) * (μ - c) ^ 2 :=
    mul_nonneg (le_of_lt hcard_pos) (sq_nonneg _)
  linarith

/-! ### 4. Squared-error expansion

`experiments._metric_rows` reports RMSE against held-out truths and normalises
by the initial-training standard deviation. The identity below is the algebraic
core of every R² and NRMSE: the squared residual expands into the three terms
that the ANOVA decomposition and the bias-variance split are built from. It is
stated for an arbitrary predictor so it needs no distributional assumption. -/

theorem sst_decomposition {ι : Type*} [Fintype ι] (y f : ι → ℝ) :
    ∑ i, (y i - f i) ^ 2
      = ∑ i, (y i) ^ 2 - 2 * ∑ i, (y i * f i) + ∑ i, (f i) ^ 2 := by
  have hterm : ∀ i, (y i - f i) ^ 2 = y i ^ 2 - 2 * (y i * f i) + f i ^ 2 := by
    intro i; ring
  calc ∑ i, (y i - f i) ^ 2
      = ∑ i, (y i ^ 2 - 2 * (y i * f i) + f i ^ 2) :=
          Finset.sum_congr rfl fun i _ => hterm i
    _ = ∑ i, y i ^ 2 - ∑ i, 2 * (y i * f i) + ∑ i, f i ^ 2 := by
          rw [Finset.sum_add_distrib, Finset.sum_sub_distrib]
    _ = ∑ i, y i ^ 2 - 2 * ∑ i, (y i * f i) + ∑ i, f i ^ 2 := by
          rw [← Finset.mul_sum]

end DRIE
