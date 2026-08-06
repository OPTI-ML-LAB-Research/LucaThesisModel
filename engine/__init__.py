"""Engine — deterministic, non-learned symbolic modules.

This package is the *interpretability backbone* of the project.
It contains:
  * the bond-mapping database (peak wavenumber -> chemical bond/mode)
  * peak extraction (scipy.find_peaks + Voigt fitting)
  * symbolic mapping logic
  * novelty detection for unmatched peaks
  * physics-constraint utilities (e.g., Beer-Lambert linearity check)

Nothing in this package is trained. Every output is inspectable and
modifiable by the user. This is the differentiator vs. black-box DL.
"""
