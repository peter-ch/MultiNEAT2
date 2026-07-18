# Evolutionary operator reference

MultiNEAT2 keeps historical behavior as its default. Every operator in this
document is additive and is serialized in `Parameters`, exposed to Python,
validated before evolution, and covered by deterministic regression tests.

## Parent selection

Set `ParentSelectionMode` to `LEGACY_SELECTION` to use the original
`TruncationSelection`, `RouletteWheelSelection`, and `TournamentSelection`
switches. Explicit modes are mutually exclusive and ignore those legacy
switches.

All explicit selectors operate only on evaluated individuals and support
negative fitness. Fitness-proportionate selectors shift the candidate set by
its minimum when necessary.

### Truncation

`TRUNCATION` samples uniformly from the best
`max(1, floor(SurvivalRate * n))` candidates. A survival rate of one includes
the complete evaluated population.

### Roulette and stochastic acceptance

`ROULETTE` performs cumulative fitness-proportionate sampling.
`STOCHASTIC` uses stochastic acceptance: it samples a candidate uniformly and
accepts it with probability `weight / max_weight`. It falls back to cumulative
roulette after a bounded number of attempts, so very skewed distributions
cannot stall.

### Linear and exponential ranking

`RANK_LINEAR` uses Baker's linear-ranking probability for zero-based rank
`r`, population size `n`, and pressure `s` in `[1, 2]`:

```text
p(r) = (2 - s) / n + 2 (n - r - 1) (s - 1) / (n (n - 1))
```

`RANK_EXP` assigns the unnormalized weight:

```text
w(r) = exp(-RankSelectionExponent * r / (n - 1))
```

Rank selection is invariant to fitness scale and is useful when outliers make
fitness-proportionate selection too aggressive.

### Tournament

`TOURNAMENT` draws `TournamentSize` candidates with replacement and returns
the fittest. It is insensitive to fitness scale and costs `O(TournamentSize)`
per parent.

### Boltzmann

`BOLTZMANN` applies stable softmax sampling:

```text
w(i) = exp((fitness(i) - max_fitness) / BoltzmannTemperature)
```

High temperature approaches uniform selection; low temperature becomes
greedier. Subtracting the maximum prevents exponential overflow.

## Crossover

The four explicit rate fields are checked to sum to at most one:

```text
MultipointCrossoverRate
+ SinglePointCrossoverRate
+ BlendCrossoverRate
+ SimulatedBinaryCrossoverRate
```

The remaining probability selects `AVERAGE`. Disjoint and excess genes retain
NEAT's fitter-parent rule (or the historical interspecies behavior). Matching
innovations use the selected operator.

- `MULTIPOINT` independently inherits matching genes, honoring
  `PreferFitterParentRate`.
- `AVERAGE` averages matching link weights and mates their traits.
- `SINGLE_POINT` selects one position in the ordered matching-innovation
  sequence and switches parents at that position.
- `BLEND` implements BLX-alpha. For parental weights `lo` and `hi`, it samples
  uniformly from
  `[lo - alpha * (hi - lo), hi + alpha * (hi - lo)]`.
- `SIMULATED_BINARY` implements SBX with distribution index
  `CrossoverSBXEta`. Larger values keep children nearer their parents.

BLX-alpha and SBX results are clamped to `MinWeight` and `MaxWeight`.
`Genome::MateWithMode` exposes each operator directly. `Genome::Mate` remains
source-compatible and maps its Boolean argument to `MULTIPOINT` or `AVERAGE`.

## Weight mutation

Replacement and severe mutation retain their established behavior. The
`WeightMutationDistribution` field controls ordinary perturbation:

- `UNIFORM_MUTATION` is the exact historical
  `[-WeightMutationMaxPower, +WeightMutationMaxPower]` perturbation.
- `GAUSSIAN_MUTATION` uses standard deviation
  `WeightMutationSigma * WeightMutationMaxPower`.
- `CAUCHY_MUTATION` uses scale
  `WeightMutationCauchyScale * WeightMutationMaxPower`. Its heavy tails make
  occasional long jumps and can help escape plateaus.
- `POLYNOMIAL_MUTATION` is a bounded Deb-style mutation controlled by
  `WeightMutationPolynomialEta`. Higher values concentrate changes near the
  current value. Its maximum step also respects `WeightMutationMaxPower`.

All results are finite and clamped to the configured weight range.

## Runtime paths

`NeuralNetwork::Activate()` validates every connection endpoint before an
activation. It is the safe choice for manually assembled networks.

Phenotypes produced by `Genome::BuildPhenotype()` are already validated.
`ActivateFast()` therefore skips the repeated topology scan and fuses signal
calculation with target accumulation. `ActivateSteps(steps, fast)` amortizes
the Python/C++ boundary and can validate once before multiple recurrent steps.

The `multineat_benchmarks` executable provides a repeatable checked-versus-fast
activation benchmark. Performance depends on topology, compiler, and CPU, so
measure on the target deployment.

## Species representatives

`SpeciesRepresentativeSelection` controls the genome retained temporarily as
the species representative during generational reproduction:

- `FIRST_REPRESENTATIVE` preserves the historical best-first representative.
- `BEST_REPRESENTATIVE` explicitly selects the current evaluated leader.
- `RANDOM_REPRESENTATIVE` samples uniformly from the species.
- `MEDOID_REPRESENTATIVE` selects the candidate with the smallest sum of
  compatibility distances to all species members.

The medoid is more robust than a historical founder or an unusually fit
outlier when a species spans several topology clusters. Its exact cost is
quadratic in species size. `RepresentativeSelectionCandidates` bounds the
number of uniformly sampled medoid candidates while still measuring each
candidate against the complete species. Zero examines every candidate.

## Offspring apportionment and niche protection

Fractional species quotas are converted to an exact population total before
reproduction. `LARGEST_REMAINDER` retains deterministic Hamilton
apportionment. `STOCHASTIC_REMAINDER` samples the remaining seats in
proportion to their fractional remainders, without replacement, eliminating
persistent species-order bias while preserving the expected quota.

`MinSpeciesSize` reserves a floor for every species with a positive quota that
fits in the population. When all requested floors cannot fit, species are
retained best-first. `SpeciesElitism` additionally reserves a quota for the
best N species. The protected allocation is validated so it cannot exceed
`PopulationSize`.

Within a species, `EliteFraction` now copies distinct top-ranked genomes. The
old behavior copied the same champion repeatedly whenever the elite count was
greater than one.

`StagnationPenalty` replaces the historical hard-coded `1e-7` multiplier for
non-champion species beyond `SpeciesMaxStagnation`. Its default remains
`1e-7`.

## Compatibility-threshold control

The legacy controller adds or subtracts `CompatTresholdModifier` whenever the
species count leaves the configured range. This is retained as
`LEGACY_COMPATIBILITY_THRESHOLD`.

`PROPORTIONAL_COMPATIBILITY_THRESHOLD` performs a scale-aware multiplicative
update:

```text
error = (observed_species - target_species) / target_species
threshold_next = threshold * exp(gain * error)
```

`TargetSpecies == 0` uses the midpoint of `MinSpecies` and `MaxSpecies`.
`CompatibilityThresholdGain` controls response speed, and the result is
clamped to `[MinCompatTreshold, MaxCompatTreshold]`. The same controller is
used by generational and steady-state evolution.

## Adaptive mutation budgets

`MutationOperatorsPerOffspring` is the expected number of mutation operators
applied whenever an offspring is selected for mutation. Fractional values use
stochastic rounding. A value of one exactly preserves the historical
single-operator path.

After `AdaptiveMutationStart` stagnant generations, the budget is multiplied
by:

```text
min(
    AdaptiveMutationMaxFactor,
    1 + AdaptiveMutationRate * stagnant_generations
)
```

An `AdaptiveMutationRate` of zero disables adaptation. Mutation availability
is recomputed after every operator, so structural limits and phased-search
constraints remain enforced as the genome changes.

## Evaluation integrity

Historical `Epoch()` calls mark every genome evaluated. This remains the
default. `RequireEvaluatedGenomes` instead rejects incomplete batches, and
`RejectNonFiniteFitness` rejects NaN or infinite results before selection.
Even in compatibility mode, non-finite values no longer become champions or
violate sorting requirements.

## Population-wide fitness scaling

`FitnessScaling` transforms objective values before species age modifiers,
stagnation penalties, and explicit fitness sharing. It controls species
offspring allocation and is intentionally independent from
`ParentSelectionMode`, which chooses parents within a species.

- `SHIFTED_FITNESS_SCALING` preserves historical minimum-shifted allocation.
- `LINEAR_RANK_FITNESS_SCALING` applies tie-aware Baker linear ranking using
  `FitnessRankPressure` in `[1, 2]`.
- `SIGMA_FITNESS_SCALING` uses
  `max(epsilon, 1 + (fitness - mean) / (scale * deviation))`.
- `BOLTZMANN_FITNESS_SCALING` applies a stable population softmax controlled
  by `FitnessBoltzmannTemperature`.

All arithmetic is normalized before summation. This keeps finite objectives
finite even when experiments use values near `DBL_MAX`, and common scaling
does not alter categorical allocation probabilities.

## Sparse online RTRL

`InitRTRLMatrix()` and the historical RTRL methods retain their public cubic
sensitivity matrix. Large recurrent phenotypes can instead call
`InitSparseRTRLMatrix()`, `RTRL_update_gradients_sparse()`, and
`RTRL_update_error_sparse()`. The sparse state is indexed by neuron and actual
connection, reducing storage from `O(neurons^3)` to
`O(neurons * connections)`.

The runtime retains each neuron's last pre-activation so sigmoid, tanh,
cubic-tanh, Gaussian, absolute, sine, linear, ReLU, and softplus derivatives
are exact. Sparse RTRL state is included in neural-network checkpoints.

## Evolution hot paths

Compatibility distance skips trait and neuron work when their coefficients
and schemas are inactive, and uses a linear merge for canonically sorted
neuron genes. Crossover maintains hashed child endpoint and neuron sets,
removing repeated linear scans as a child grows. Mean population complexity
is accumulated directly in one pass instead of repeatedly resolving flat
indexes through every species.

Innovation lookup is indexed by endpoint and innovation type, avoiding a
linear scan through the complete historical database. Because
`m_Innovations` remains public for compatibility, callers that directly
replace its elements should call `RebuildIndex()`.

The benchmark executable reports activation, compatibility-distance,
crossover, innovation lookup, and dense-versus-sparse RTRL throughput.
