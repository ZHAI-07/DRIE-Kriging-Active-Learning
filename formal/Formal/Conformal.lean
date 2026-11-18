/-
  Conformal prediction: the combinatorial core of the coverage guarantee.

  The uncertainty-calibration exploration reports that the project's GP
  intervals cover only ~0.84 at nominal 0.95, and proposes conformal intervals
  instead. The justification for conformal coverage is a pure counting fact:
  among N exchangeable residuals, the held-out one is excluded from its own
  quantile, so the interval built from the other N-1 covers with probability
  exactly (N-1)/N for the α = 1/N case.

  What is formalised: the exact cardinality identity and the resulting coverage
  value. What is *not* formalised: exchangeability of the rows, which is an
  assumption about the data-generating process and is known to be questionable
  here (2 and 13 share a measurement source). Stating that limitation is part
  of the point.
-/
import Mathlib

namespace DRIE

/-- Removing one distinguished index from a finite type of size N leaves N-1.

This is the combinatorial content of leave-one-out conformal prediction: the
held-out observation cannot be used to build its own interval. -/
theorem card_compl_singleton {ι : Type*} [Fintype ι] [DecidableEq ι]
    (i0 : ι) :
    Fintype.card {i : ι // i ≠ i0} = Fintype.card ι - 1 := by
  rw [Fintype.card_subtype_compl (fun i : ι => i = i0), Fintype.card_subtype_eq]

/-- The corresponding coverage probability for the α = 1/N convention:
`(N-1)/N = 1 - 1/N`. -/
theorem conformal_coverage {ι : Type*} [Fintype ι] [DecidableEq ι] (i0 : ι)
    (hN : 0 < Fintype.card ι) :
    ((Fintype.card {i : ι // i ≠ i0} : ℕ) : ℝ) / (Fintype.card ι : ℝ)
      = 1 - 1 / (Fintype.card ι : ℝ) := by
  rw [card_compl_singleton]
  have hpos : (0 : ℝ) < (Fintype.card ι : ℝ) := by
    exact_mod_cast hN
  have hcast : ((Fintype.card ι - 1 : ℕ) : ℝ) = (Fintype.card ι : ℝ) - 1 := by
    rw [Nat.cast_sub hN]; norm_num
  rw [hcast]
  field_simp

/-- The finite-sample binding constraint of full conformal prediction.

Full conformal takes the conformal rank `k = ⌈(n+1)(1-α)⌉` among the `n`
available training scores, and the construction is only finite when `k ≤ n`.
This theorem records the arithmetic obstruction in the form that was actually
observed: whenever `α < 1/(n+1)`, the required rank exceeds `n`, so no finite
interval exists. No positivity hypothesis on `α` is needed — the conclusion
holds for every real `α` below `1/(n+1)`.

The exploration hit this wall directly. With `n = 14` training records the AL
rerun cannot reach level `0.95`, because `1/α - 1 = 19 > 14`, and every 95%
interval degenerates to the whole real line. The run therefore reports a 0.90
statement, and the corollary below pins the threshold for the 95% case. -/
theorem conformal_rank_exceeds_n {n : ℕ} {α : ℝ}
    (h : α < 1 / ((n : ℝ) + 1)) :
    (n : ℝ) < ((n : ℝ) + 1) * (1 - α) := by
  have hn1 : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have hmul : α * ((n : ℝ) + 1) < 1 := (lt_div_iff₀ hn1).mp h
  nlinarith

/-- The 95% case: below 19 calibration points, the conformal rank exceeds `n`.

`(n+1)·(1 - 1/20) = (19n + 19)/20 > n` exactly when `n < 19`, so a 95% full
conformal interval cannot be finite for `n ≤ 18` — in particular not for the
`n_train ≤ 14` reached by the AL protocol. -/
theorem conformal_95_rank_exceeds_n {n : ℕ} (h : n < 19) :
    (n : ℝ) < ((n : ℝ) + 1) * (1 - 1 / 20) := by
  have hn : (n : ℝ) < 19 := by exact_mod_cast h
  have hone : (1 : ℝ) - 1 / 20 = 19 / 20 := by norm_num
  rw [hone]
  nlinarith

end DRIE
