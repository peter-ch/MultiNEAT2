# Changelog

## Unreleased

### Second audit (2026-10-24)

Correctness:

- McCulloch-Pitts neurons were vetoed forever by a single inhibitory spike:
  decaying synaptic currents stalled at a subnormal value. Decayed currents and
  STDP/rate traces are now flushed to zero below `1e-12`.
- LIF and leaky rate neurons use exact exponential integration; explicit Euler
  overshot for time steps longer than the membrane time constant, so purely
  inhibitory drive could fire a neuron. e-prop eligibility now matches the
  solver, uses an exact memoryless path for McCulloch-Pitts targets, the
  decayed adaptation for adaptive LIF, and the transmitted analog signal.
- e-prop random/uniform feedback mixed other outputs' errors into each output.
- Poisson-rate inputs fell to 0 or 1 Hz after one step of an `Activate()` loop.
- `NeuralNetwork.Save()` dropped in-flight delayed spikes and printed values in
  fixed point (losing small magnitudes); corrupt files threw instead of failing
  to load. `Serialize()` output containing `inf`/`nan` could not be read back,
  and oversized counts raised `length_error` instead of `runtime_error`.
- FS-NEAT seeds contained duplicate bias links and failed validation.
- `Genome::Save(FILE*)` (and therefore `Population.Save()`) lost traits,
  McCulloch-Pitts vetoes, coordinates, and evaluation state.
- Removing a neuron between recurrent links produced a non-recurrent cycle;
  dead-end detection removed healthy outputs that had an outgoing recurrent
  link; clone detection ignored the inhibitory veto gene.
- Real-time evolution (`Tick`) selected parents uniformly at random (adjusted
  fitness was never computed) and legacy truncation picked the worst survivor.
- `NoveltySearchTick` never marked offspring as evaluated, so novelty runs
  failed after about one population's worth of ticks.
- Copying or moving a `Population` aliased the source's novelty archive.
- One non-finite fitness in legacy mode made every later checkpoint fail.
- Lamarckian improvement ran once per evaluation request instead of once per
  genome per generation.
- Python: container and variant properties (`m_NeuronGenes`, trait maps,
  `m_Details`, evaluation samples, QD archive, ...) returned references that
  dangled after reallocation; they now return copies. `m_PhenotypeBehavior`
  is read-only. `Parameters.Save()` no longer truncates the file when
  serialization fails, `Load()` keeps custom constraint callbacks, scalar
  parsing is strict, and `RNG.Seed()` accepts 64-bit seeds.
- Custom HyperNEAT connectivity, genome/network/parameter deserializers, and
  the demo executable's command line validate untrusted input.

Performance:

- Genes, genomes, traits, and species regained move semantics (a declared copy
  assignment had disabled them), and crossover aligns genes through pointers:
  multipoint crossover is about 1.8x faster.
- Spike-history trimming is amortized, and `neattools.SpikingRecorder` reads
  events lazily instead of rebuilding them on every step.

Demos, packaging, and CI:

- Every demo is a standalone file that depends only on `pymultineat` and
  third-party packages; the shared `gymnasium_neat.py` and `spiking_neat.py`
  modules are gone. Numba is optional for the Asteroids demo.
- `pip install .` builds a wheel through scikit-build-core and installs
  `pymultineat` (with `__version__`) and `neattools`.
- CI uses current actions, pip caching, Python 3.10/3.12/3.13, a Box2D/MuJoCo
  smoke job, and a wheel job. New native suites cover networks and
  populations.

### Audit fixes

- Made bootstrap summaries and stored evidence independent of worker completion
  order, including checkpoint resume; avoided intermediate overflow for large
  finite scores and descriptors.
- Avoided full sorting of bootstrap replicates and skipped resampling for
  constant evidence; added reliability aggregation microbenchmarks.
- Fixed duplicate QD archive insertion after reevaluation, preserving the newly
  collected archive evidence.
- Made evidence batch replacement transactional and safe when its input aliases
  existing samples; reject invalid descriptors before consuming requests.
- Kept population and species owners alive for borrowed Python genome references.
- Preserved running networks and RNG state on failed loads, and fixed a loop on
  unopened network streams and uninitialized fields in truncated network records.
- Validate spiking inputs, learning signals, and full training batch dimensions
  before advancing runtime state or applying partial training.
- Consolidated Python test registration, added reliability and runtime safety
  suites and test timeouts, execute the installed-package consumer in CI, and
  corrected the minimum CMake version to 3.18.
- Fixed Windows shared-library exports and added a shared build to Windows CI.

### Spiking neural networks

- Added evolvable McCulloch-Pitts neurons with binary threshold firing,
  canonical absolute inhibitory vetoes, refractory timing, mutation,
  compatibility distance, persistence, bindings, and optional STDP/e-prop.
- Added `--mcculloch-pitts` variants to every spiking-capable example and the
  graphical launcher.
- Added physical 3D HyperNEAT neuron coordinates, stored axon lengths,
  maximum-length pruning, distance-derived conduction delays, and a 3D
  phenotype renderer.
- Extended ES-HyperNEAT from quadtree-only 2D discovery to bounded octree 3D
  discovery and added a runnable 3D CPPN-evolution example.
- Added an opt-in e-prop learner with local eligibility traces, adaptive-neuron
  temporal credit, fast-sigmoid/triangular/arctangent surrogate derivatives,
  random/symmetric/uniform feedback, supervised and externally modulated
  learning paths, and MSE/Huber temporal losses.
- Added online AdamW and SGD, update accumulation, global gradient clipping,
  weight decay and bounds, connection-class masks, topology guards, explicit
  STDP coexistence control, and deterministic checkpoint/resume of feedback,
  eligibility, gradients, and optimizer state.
- Added an e-prop convergence demo and deterministic C++/Python learning,
  binding, persistence, and safety regression tests.
- Added evolvable LIF, adaptive-LIF, and Izhikevich activation modes that can
  coexist with rate neurons.
- Added deterministic synchronous simulation with continuous, binary-spike,
  and seeded Poisson-rate input encoders plus spike, cumulative-rate,
  filtered-rate, and membrane output decoders.
- Added delayed current-based exponential synapses and optional bounded
  pair-based STDP.
- Added built-in mutation, crossover persistence, initialization, validation,
  and compatibility-distance support for all spiking neuron and synapse
  parameters.
- Added complete live-state checkpointing for membrane/recovery/adaptation
  state, refractory timers, pending events, synaptic currents, STDP traces,
  RNG state, spike counters, and recorded events.
- Added static, interactive, and real-time animated phenotype, raster,
  membrane, and firing-rate visualizations with NPZ/JSON/CSV recording export.
- Added temporal-pattern and built-in Cart-Pole spiking evolution demos with
  synchronized task and network animation.
- Added opt-in spiking variants to XOR, Asteroids, and every existing Box2D
  and MuJoCo demo while preserving their original rate-network defaults.
- Added a shared seeded-Poisson policy bridge with configurable solver steps,
  time step, rate ranges, filtered-rate action decoding, and recurrent
  adaptive-LIF/LIF evolution presets.
- Added combined environment/fitness/phenotype/raster summaries, aligned NPZ
  traces beside plots and videos, a live Asteroids policy panel, and a
  graphical-launcher spiking variant switch.
- Added deterministic C++, Python, visualization, persistence, evolution, and
  end-to-end demo regression tests.

- Added `python demos.py`, a standard-library Tk launcher for all 24 examples
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
