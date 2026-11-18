/-
  Kernel facts used by the four-dimensional generalisation study.

  The 4D experiment compares a kernel on (Coil, Platen) against one on
  (Coil, Platen, SF6, pressure) with the two extra axes given a very long
  length scale. Both kernels are Matern 5/2 and differ only in the standardised
  distance that feeds them:

      s = sqrt 5 * d / tau,     f(s) = (1 + s + s^2/3) * exp(-s).

  To say what the 4D kernel *is* in 2D terms, the study solves for the length
  scale tau' that reproduces each pairwise correlation exactly:

      tau'(i,j) = d_2D(i,j) / d_4D(i,j).

  That inversion is only well posed if f is injective on the relevant range,
  which follows from f being strictly decreasing on [0, infinity). The theorem
  below is that fact. Everything downstream depends on the equivalence being
  unique: the study finds that a global rescaling of the 2D kernel reproduces
  the incumbent (+0.210 vs +0.209) and therefore cannot explain the 4D gain, so
  "there is exactly one equivalent length scale per pair" is load-bearing rather
  than decorative.

  Derivative, for the record:

      f'(s) = -(s/3) * (1 + s) * exp(-s) < 0    for s > 0.
-/
import Mathlib

namespace DRIE

/-- The Matern 5/2 correlation function, as a function of the scaled distance. -/
noncomputable def matern52 (s : ℝ) : ℝ := (1 + s + s ^ 2 / 3) * Real.exp (-s)

/-- Closed form of the derivative of `f(s) = (1 + s + s^2/3) * exp(-s)`. -/
theorem deriv_matern52_lambda (x : ℝ) :
    deriv (fun s : ℝ => (1 + s + s ^ 2 / 3) * Real.exp (-s)) x
      = -(x / 3) * (1 + x) * Real.exp (-x) := by
  have hpoly : HasDerivAt (fun t : ℝ => 1 + t + t ^ 2 / 3) (1 + 2 * x / 3) x := by
    have hsq : HasDerivAt (fun t : ℝ => t ^ 2 / 3) (2 * x / 3) x := by
      simpa using (hasDerivAt_pow 2 x).div_const 3
    have hsum := ((hasDerivAt_const x (1 : ℝ)).add (hasDerivAt_id x)).add hsq
    convert hsum using 1
    ext t
    simp [Pi.add_apply]
    ring
  have hexp : HasDerivAt (fun t : ℝ => Real.exp (-t)) (-Real.exp (-x)) x := by
    have hcomp := (Real.hasDerivAt_exp (-x)).comp x (hasDerivAt_id x).neg
    convert hcomp using 1
    · ext t
      simp
    · ring
  have hprod := hpoly.mul hexp
  convert hprod.deriv using 1
  ring

/-- The Matern 5/2 correlation is strictly decreasing on `[0, ∞)`.

Injectivity of `s ↦ f s` on this range is what makes the equivalent-length-scale
map `tau' = d_2D / d_4D` well defined and unique. -/
theorem matern52_strictAntiOn : StrictAntiOn matern52 (Set.Ici 0) := by
  have key : StrictAntiOn (fun s : ℝ => (1 + s + s ^ 2 / 3) * Real.exp (-s))
      (Set.Ici 0) := by
    apply strictAntiOn_of_deriv_neg (convex_Ici 0)
    · fun_prop
    · intro x hx
      rw [interior_Ici] at hx
      have hxpos : 0 < x := hx
      rw [deriv_matern52_lambda x]
      have h1 : 0 < x / 3 := by positivity
      have h2 : 0 < 1 + x := by linarith
      have h3 : 0 < Real.exp (-x) := Real.exp_pos _
      have hprod_pos : 0 < (x / 3) * (1 + x) * Real.exp (-x) :=
        mul_pos (mul_pos h1 h2) h3
      nlinarith
  change StrictAntiOn (fun s : ℝ => (1 + s + s ^ 2 / 3) * Real.exp (-s)) (Set.Ici 0)
  exact key

end DRIE
