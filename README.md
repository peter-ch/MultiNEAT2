# MultiNEAT2

MultiNEAT2 is a C++17 implementation of NEAT (NeuroEvolution of
Augmenting Topologies) with optional Python bindings. It includes
speciation, structural and trait mutation, recurrent networks, phased
search, novelty search, HyperNEAT substrates, and ES-HyperNEAT phenotype
generation.

This repository continues the original MultiNEAT codebase while retaining
its established C++ and Python names wherever possible. Compatibility
spellings such as `SetInputOutputDimentions`, `GetConnectionLenght`, and
`Elitism` remain available alongside corrected names.

The additions are source- and checkpoint-compatible. As with any C++ library
whose public value types gain fields, applications linking a precompiled
MultiNEAT binary must rebuild when upgrading to 2.1.

## Build

Requirements:

- CMake 3.15 or newer
- A C++17 compiler (recent GCC, Clang, or MSVC)
- Python development files and pybind11 when building `pymultineat`

Configure, build, and test:

```sh
python -m pip install pybind11
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Useful CMake options:

| Option | Default | Purpose |
| --- | --- | --- |
| `MULTINEAT_BUILD_PYTHON` | `ON` | Build the `pymultineat` extension |
| `MULTINEAT_BUILD_DEMO` | `ON` | Build the C++ XOR demo |
| `MULTINEAT_BUILD_BENCHMARKS` | `OFF` | Build repeatable core microbenchmarks |
| `BUILD_TESTING` | `ON` | Build and register regression tests |
| `MULTINEAT_WARNINGS_AS_ERRORS` | `OFF` | Treat compiler warnings as errors |
| `MULTINEAT_ENABLE_SANITIZERS` | `OFF` | Enable address and undefined-behavior sanitizers |

For a C++-only build:

```sh
cmake -S . -B build -DMULTINEAT_BUILD_PYTHON=OFF
cmake --build build --config Release --parallel
```

The Python extension is created in the selected build configuration
directory. Add that directory to `PYTHONPATH`, install the CMake project,
or copy the extension into your environment before importing it.

## Install and consume from CMake

```sh
cmake --install build --config Release --prefix /your/install/prefix
```

Downstream projects can use the exported target:

```cmake
find_package(MultiNEAT 2 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE MultiNEAT::multineat)
```

Point `CMAKE_PREFIX_PATH` at the chosen install prefix if it is not a
standard system location.

## Python example

```python
import time
import pymultineat as neat

parameters = neat.Parameters()
parameters.PopulationSize = 150

initial = neat.GenomeInitStruct()
initial.NumInputs = 3       # two inputs plus the bias neuron
initial.NumOutputs = 1
initial.OutputActType = neat.UNSIGNED_SIGMOID

seed = neat.Genome(parameters, initial)
population = neat.Population(seed, parameters, True, 1.0, int(time.time()))

for _ in range(100):
    for species in population.m_Species:
        for genome in species.m_Individuals:
            network = neat.NeuralNetwork()
            genome.BuildPhenotype(network)
            # Evaluate the network here. Negative fitness values are supported.
            genome.SetFitness(evaluate(network))
            genome.SetEvaluated()
    population.Epoch()
```

Run the self-contained XOR example with:

```sh
python demos/xor.py
```

The physics-control suite covers seven Box2D configurations and all eleven
Gymnasium MuJoCo tasks through a shared trainer with reproducible seeding,
parallel evaluation, checkpoint/resume, metrics, plots, video recording, and
real three-step smoke modes. Existing demo filenames remain runnable.

```sh
python -m pip install -r requirements-box2d.txt
python demos/box2d/lunar_lander_box2d.py --smoke

python -m pip install -r requirements-mujoco.txt
python demos/run_gymnasium_suite.py --family mujoco --inspect
```

See [demos/README.md](demos/README.md) for the complete task catalog and CLI
guide. Simulator dependencies are optional and are not required by the core
library.

## Advanced evolutionary operators

All new operators are opt-in. Default parameters preserve the historical
MultiNEAT selection, crossover, and uniform weight-mutation behavior.

Parent selection can be selected explicitly with
`Parameters.ParentSelectionMode`:

| Mode | Behavior |
| --- | --- |
| `LEGACY_SELECTION` | Existing truncation/roulette/tournament booleans |
| `TRUNCATION` | Uniform sampling from the best `SurvivalRate` fraction |
| `ROULETTE` | Shifted fitness-proportionate selection |
| `RANK_LINEAR` | Baker linear ranking with configurable pressure |
| `RANK_EXP` | Exponentially decaying rank selection |
| `TOURNAMENT` | Best of `TournamentSize` random draws |
| `STOCHASTIC` | Fitness-proportionate stochastic acceptance |
| `BOLTZMANN` | Numerically stable softmax selection |

Reproduction supports multipoint, average, single-point, BLX-alpha, and
simulated-binary crossover. `MultipointCrossoverRate` keeps its original
meaning; the new rate fields are additive, and remaining probability uses
average crossover.

Weight perturbation can use uniform, Gaussian, Cauchy, or bounded polynomial
mutation:

```python
parameters = neat.Parameters()
parameters.ParentSelectionMode = neat.RANK_EXP
parameters.RankSelectionExponent = 3.0

parameters.MultipointCrossoverRate = 0.4
parameters.SinglePointCrossoverRate = 0.2
parameters.BlendCrossoverRate = 0.2
parameters.SimulatedBinaryCrossoverRate = 0.1

parameters.WeightMutationDistribution = neat.GAUSSIAN_MUTATION
parameters.WeightMutationSigma = 0.5

# Protect viable niches and choose a central species representative.
parameters.MinSpeciesSize = 2
parameters.SpeciesElitism = 2
parameters.SpeciesRepresentativeSelection = neat.MEDOID_REPRESENTATIVE
parameters.RepresentativeSelectionCandidates = 32
parameters.OffspringAllocation = neat.STOCHASTIC_REMAINDER

# Smoothly target eight species.
parameters.CompatibilityThresholdControl = (
    neat.PROPORTIONAL_COMPATIBILITY_THRESHOLD
)
parameters.TargetSpecies = 8
parameters.CompatibilityThresholdGain = 0.25

# Increase the mutation budget after sustained global stagnation.
parameters.MutationOperatorsPerOffspring = 1.0
parameters.AdaptiveMutationStart = 20
parameters.AdaptiveMutationRate = 0.05
parameters.AdaptiveMutationMaxFactor = 3.0

# Stabilize species allocation against outliers without changing the
# within-species parent selector.
parameters.FitnessScaling = neat.SIGMA_FITNESS_SCALING
parameters.FitnessSigmaScale = 2.0
```

Direct experiments can use
`Genome.MateWithMode(..., neat.SIMULATED_BINARY, ...)`; the historical
`Genome.Mate(..., average_mating, ...)` signature remains unchanged.
Strict experiments can enable `RequireEvaluatedGenomes` and
`RejectNonFiniteFitness` to catch incomplete evaluation batches before
selection. These checks are opt-in for compatibility.
See [docs/ALGORITHMS.md](docs/ALGORITHMS.md) for formulas, tuning guidance,
and compatibility details.

## Visualization and analysis

Install the optional visualization stack with:

```sh
python -m pip install -r requirements-visualization.txt
```

`neattools.py` retains its original helpers and now provides:

- topology-, split-, coordinate-, spring-, and Kamada-Kawai layouts;
- weight sign/magnitude encoding and curved recurrent connections;
- activation and trait labels;
- genome comparison and machine-readable topology statistics;
- three-panel structural diffs for inspecting crossover and mutation;
- population/species dashboards and an `EvolutionTracker`;
- population health summaries with entropy-based effective diversity;
- interactive population and evolution dashboards;
- static DOT, GraphML, GEXF, JSON, SVG, PNG, and PDF export;
- optional interactive Plotly HTML graphs with rich hover data.

```python
from neattools import (
    DrawGenome,
    DrawGenomeComparison,
    DrawPopulation,
    InteractiveGenome,
    InteractivePopulation,
    population_summary,
)

DrawGenome(genome, layout="topology", show_activation=True)
DrawGenomeComparison(parent, offspring)
DrawPopulation(population)
InteractiveGenome(genome).write_html("genome.html")
InteractivePopulation(population).write_html("population.html")
print(population_summary(population))
```

## Persistence

`Parameters`, `Genome`, `Species`, `InnovationDatabase`, `NeuralNetwork`,
and `Population` provide round-trippable serialization and Python pickle
support.

For a complete resumable population checkpoint:

```python
population.SaveState("experiment.state")
population = neat.Population("experiment.state")
```

The older `Population.Save()` method deliberately retains its historical
parameters/innovations/genomes file format for existing applications.
Use `SaveState()` when generation counters, RNG state, species state,
archives, and all trait data must be preserved exactly.

## Algorithm and safety notes

- `Parameters.Validate()` checks probability distributions, numeric ranges,
  mutation limits, and trait schemas. Population construction and evolution
  fail early with a descriptive exception when parameters are invalid.
- Structural mutation exhaustively samples valid candidates, respects
  recurrent-link split flags, prevents accidental feed-forward cycles, and
  enforces `MaxLinks` and `MaxNeurons`.
- Offspring are apportioned exactly. Stochastic-remainder allocation removes
  deterministic tie bias, while minimum species sizes and species elitism can
  protect viable niches. Negative fitness is supported by parent and removal
  selection.
- Population-wide shifted, linear-rank, sigma, and Boltzmann fitness scaling
  are available for offspring allocation. Every mode remains finite across
  the complete `double` range; shifted scaling is the compatibility default.
- Elitism copies distinct top-ranked genomes rather than repeating one
  champion to fill a multi-elite quota.
- Medoid representatives, proportional compatibility-threshold control, and
  stagnation-adaptive mutation budgets are available without changing legacy
  defaults.
- `BuildESHyperNEATPhenotype()` implements quadtree division, variance and band
  pruning, optional LEO expression, iterative hidden discovery, deterministic
  node indexing, link deduplication, and reachability pruning. Quadtree depth
  is bounded to prevent accidental exponential allocation.
- RTRL supports multiple outputs and configurable learning rates. Exact
  derivatives are implemented for every differentiable activation. The
  additive sparse RTRL path stores `O(neurons * connections)` sensitivities
  instead of the legacy cubic matrix while preserving recurrent state.
- `ActivateFast()` is the unchecked phenotype hot path and fuses signal
  calculation with accumulation. `Activate()` remains the topology-validating
  entry point. `ActivateSteps()` validates once (when requested) and advances
  recurrent networks without repeated API crossings. `ActivateBatch()`
  evaluates independent samples in one C++/Python call.

Corrected convenience names such as `SetInputOutputDimensions()` and
`GetConnectionLength()` are additive; historical misspellings remain
available for downstream source compatibility.

## Benchmarks

Build and run the opt-in benchmark with:

```sh
cmake -S . -B build -DMULTINEAT_BUILD_BENCHMARKS=ON
cmake --build build --config Release --parallel
./build/multineat_benchmarks
```

It compares checked activation with the fused phenotype hot path on a dense
network and reports compatibility-distance and crossover throughput on a
complex genome. Results depend on compiler, topology, and CPU.

## Project layout

- `src/` — C++ library and pybind11 bindings
- `tests/` — C++ and Python regression tests plus an installed-package test
- `demos/` — XOR, Box2D, MuJoCo, and visualization examples
- `neattools.py` — genome visualization helpers

## License

MultiNEAT2 is distributed under the GNU Lesser General Public License,
version 3 or (at your option) any later version, matching the license
notices in the inherited MultiNEAT source. See [LICENSE](LICENSE).
