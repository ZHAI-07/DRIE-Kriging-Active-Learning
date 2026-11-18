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

end DRIE
