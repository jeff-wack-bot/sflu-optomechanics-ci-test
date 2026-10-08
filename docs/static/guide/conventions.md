# Conventions

Every quantum-noise paper picks its own conventions, and most disagreements
between a model and a paper turn out to be about a sign, a factor of two, or
a single- vs double-sided spectrum. These are the ones this package uses. The
code that defines them is `sflu.readout`.

## Fields and quadratures

* Each spatial mode carries a two-component vector `[amplitude, phase]`. These
  are the two-photon quadratures $(a_1, a_2)$ of Caves & Schumaker, with
  $a_1$ along the carrier. With `nhom` higher-order modes the vector has
  `2 (nhom + 1)` components.
* Carrier fields are in $\sqrt{\mathrm{W}}$, and the power is $|E|^2$. A power
  fluctuation is then $\delta P = 2\sqrt{P}\,\delta a_1$, which is what the
  radiation-pressure edges assume.
* **Homodyne angle** $\zeta$: the readout is $b_1\cos\zeta + b_2\sin\zeta$, so
  $\zeta=\pi/2$ reads the phase quadrature (`readout.quadrature`).
* **Squeeze angle** $\lambda$: the squeezed quadrature is
  $a_1\cos\lambda + a_2\sin\lambda$, so $\lambda=\pi/2$ is phase squeezing
  (`readout.squeezed`).
* `MatrixLib.LO(phi)` predates this and uses `phi = pi/2 - zeta`. New code
  should use `readout.quadrature`.

## Spectra

* All PSDs are **single-sided**. The vacuum spectral density of one
  quadrature is $\hbar\omega_0/2$ in these units (`readout.vacuum_psd`).
  This is KLMTV's normalisation (their Eq. 22); a Braginsky–Khalili-style
  double-sided spectrum is half of it.
* `readout.referred_psd(T, row, signal)` divides the readout noise by the
  response to `signal`. With `signal="ETM.pos.exc"` that gives a
  displacement PSD in m²/Hz.

## Sign of the radiation-pressure coupling

Here radiation pressure enters a tuned cavity's output as
$b_2 = a_2 + \mathcal K a_1$. KLMTV and most papers since write
$b_2 = a_2 - \mathcal K a_1$. The two differ only in the sign of the phase
quadrature, so every angle taken from such a paper is used with its sign
flipped (`sflu.papers.klmtv2001.sflu_angle`). Each paper page states the
convention it needed and how it was checked.

## Detuning

`LinkEdge(detune_rad=...)` is applied on every pass through a link.
`topologies.detuning_rad(detune_Hz, L)` converts a cavity detuning to that
phase. Positive `detune_Hz` puts the carrier *above* the cavity resonance
($\omega_0 - \omega_\mathrm{res} > 0$), which is KLMTV's $\xi > 0$. This was
verified by reproducing KLMTV's filter-cavity readout.

## Bandwidths: exact poles, not first order

Papers usually define a cavity half-width as $\gamma = Tc/4L$, which is first
order in $T$. `topologies.half_bandwidth_T` instead returns the transmission
whose *exact* pole is at $\gamma$. The difference is about 1% for an
aLIGO-like arm. That sounds negligible, but configurations that cancel
back-action (variational readout, for example) subtract two numbers of size
$\mathcal K \sim 10^3$, and there 1% becomes a factor of 30. For the same
reason, models set the arm power from the paper's definition with
`solve.scale_dc`, rather than inferring it from the cavity gain $4/T$.

## Reproducibility

`SFLU.reduce_auto()` eliminates graph nodes in Python set-iteration order, and
that order depends on `PYTHONHASHSEED`. The choice of elimination order moves
the reference budgets by up to $10^{-3}$. Every `make` target and the CI pin
the seed to 0. The seed cannot be pinned from inside Python, so `conftest.py`
warns when a run is unpinned. See `REFACTOR_PLAN.md`, Finding 1.
