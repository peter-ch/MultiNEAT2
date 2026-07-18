# Changelog

## Unreleased

- Added `python demos.py`, a standard-library Tk launcher for all 20 examples
  with automatic runtime selection, immediate launch, safe smoke modes,
  process control, live logs, and output-folder access.
- Added bounded CLI smoke modes for XOR and Asteroids.
- Replaced duplicated Box2D and MuJoCo scripts with a reusable Gymnasium NEAT
  trainer and preserved every existing script name as a compatible entry
  point.
- Expanded the physics suite to seven Box2D configurations and all eleven
  Gymnasium MuJoCo v5 tasks.
- Added runtime space inference, bounded continuous/discrete action adapters,
  pooled-pixel CarRacing inputs, fair episode seeding, persistent parallel
  workers, checkpoint/resume, JSONL metrics, plots, video, and smoke modes.
- Added simulator dependency files, a suite runner, focused integration tests,
  and a complete demo guide.

## Unreleased

### Evolution algorithms

- Completed the previously declared selection-mode interface with explicit
  truncation, roulette, linear rank, exponential rank, tournament, stochastic
  acceptance, and Boltzmann selection.
- Added single-point, BLX-alpha, and simulated-binary crossover while retaining
  multipoint and average crossover.
- Added uniform, Gaussian, Cauchy, and bounded polynomial weight perturbation.
- Added validation and complete persistence for every algorithm control.
- Exposed all operators through the C++ and Python APIs.
- Added random and compatibility-medoid species representatives with bounded
  candidate sampling.
- Added exact niche floors, top-species protection, and stochastic-remainder
  offspring allocation.
- Added proportional compatibility-threshold control for generational and
  steady-state evolution.
- Added expected multi-operator mutation budgets with optional
  stagnation-driven adaptation.
- Added shifted, tie-aware rank, sigma, and Boltzmann population fitness
  scaling with overflow-safe allocation across the complete finite range.
- Made stagnation penalties configurable and added opt-in strict evaluation
  and finite-fitness enforcement.
- Corrected multi-elite reproduction to retain distinct top genomes.

### Performance

- Made `ActivateFast` a fused unchecked phenotype hot path.
- Added `ActivateSteps` to amortize validation and language-boundary overhead.
- Replaced ordered endpoint lookups with reserved hash maps during phenotype
  construction and compatibility calculations.
- Removed repeated linear endpoint/neuron scans from crossover, skipped
  inactive compatibility components, and made mean-complexity calculation
  linear in population size.
- Reduced feed-forward cycle screening during add-link mutation from one graph
  traversal per candidate pair to one traversal per target neuron.
- Added an opt-in repeatable core benchmark.
- Indexed historical innovations by type and endpoint while retaining the
  public innovation vector and its first/last/all lookup semantics.
- Added batched independent inference and an `O(neurons * connections)`
  sparse RTRL path with exact derivatives for all differentiable activations.

### Visualization

- Rebuilt `neattools.py` around topology-aware layered layouts with crossing
  reduction and recurrent-cycle handling.
- Added richer static rendering, interactive Plotly graphs, population
  dashboards, evolution tracking, genome comparison, topology summaries, and
  multi-format export.
- Added interactive population/evolution dashboards, entropy-based effective
  species diagnostics, detailed species health summaries, and JSON/CSV
  evolution-history export.
- Kept the original visualization helper names available.
- Added three-panel structural genome diffs plus depth, density, self-loop,
  strongly-connected-component, and cyclic-component diagnostics.

### Verification and compatibility

- Added deterministic C++ and Python coverage for every new operator.
- Added optional headless visualization smoke tests.
- Kept legacy selection switches, crossover entry points, misspelled
  compatibility aliases, defaults, and serialization loading functional.
