# Reliability-first evolution

MultiNEAT 3 separates environment execution from evolutionary decisions. The
population issues `EvaluationRequest` values; the caller evaluates each
requested genome on the exact `case_id` and submits an `EvaluationSample`.
Requests can be evaluated in any order or process. Case IDs are shared across
genomes and the scheduler, samples, archive, operator state, and emitter state
are all checkpointed for deterministic resume.

Bootstrap draws and stored evidence use a canonical case ordering. Worker
completion order therefore does not affect selection or resumed evolution.
Serialize calls to `SubmitEvaluation` on the coordinating thread; the population
itself is not a concurrent result queue. `GetEvaluationGenome(request)` returns
a copy suitable for evaluation workers. Python's `AccessGenomeByIndex` and
`AccessGenomeByID` return borrowed genomes and retain their population owner,
but population mutations such as `Epoch` and `Tick` can invalidate those references.

## Training protocol

Stochastic objectives use 2, 4, 8, and 16 cumulative cases. After each stage,
the top half of each species advances, ordered by the deterministic 90%
bootstrap lower confidence bound, then mean, then lower link count. Automatic
epsilon-lexicase selection operates on shared cases with a median absolute
deviation epsilon. Species quotas use linear ranks, so a single extreme score
cannot capture a generation.

```python
requests = population.GetEvaluationRequests()
while requests:
    results = evaluate_in_parallel(requests)
    for request, result in zip(requests, results):
        population.SubmitEvaluation(
            request.genome_id,
            request.case_id,
            result.score,
            result.has_success,
            result.success,
            result.behavior_descriptor,
        )
    requests = population.GetEvaluationRequests()

assert population.ReadyForEpoch()
provisional = population.GetProvisionalChampion()
population.Epoch()
```

Scores and descriptor values must be finite. Fitness is maximized. Set
`DeterministicEvaluation = true` for a one-case deterministic task.
Nonempty descriptors for one genome must have the same dimension, and QD
descriptors must match the configured bins. Rejected scores or descriptors
leave the request pending so it can be corrected and submitted again.

## Champion certification

Training, validation, and certification IDs occupy disjoint streams. The
standard flow selects five provisional candidates on 32 validation cases,
then evaluates the validation winner once on 64 unseen certification cases.

```python
requests = population.ValidateCandidates(5)
while requests:
    # Evaluate and SubmitEvaluation exactly as above.
    ...

champion = population.GetValidatedChampion()
if champion.validated:
    deploy(champion.genome)
```

`validated` can become true only when every certification sample contains a
success flag and the one-sided 95% Wilson lower bound meets
`ReliabilityTarget`. A high mean return alone is never certification.

## Quality diversity and adaptation

Set `EvolutionModeSetting = ROBUST_QD_NEAT` and provide descriptor minimums,
maximums, and bin counts to enable the uncertainty-aware archive. It retains
two candidates per cell by default, re-evaluates 10% of occupied entries each
generation, and replaces equal-evidence incumbents only on a higher lower
confidence bound. Half of QD-mode parent draws come from uniformly sampled
compatible archive cells.

Each species maintains decayed Thompson posteriors for weight mutation,
structural addition/removal, parameter mutation, crossover, and its weight
emitter. Archive replacement earns full credit and robust parent improvement
earns half credit. After three generations with the same ordered topology, a
separable normalized continuous-parameter emitter may use one batch, but only
when the species quota leaves at least half of its offspring to ordinary NEAT.

## Optional local improvement

A caller can attach an e-prop, RTRL, or other task-specific improvement pass to
the genome returned for evaluation. Baldwinian mode is the default and changes
only the evaluated copy; Lamarckian mode writes the improved parameters back to
the population.

```python
population.SetLocalImprovementHook(improve_genome, lamarckian=False)
genome = population.GetEvaluationGenome(request)
```

The hook is disabled by default and is intentionally not serialized because it
is caller-owned executable code. Reattach it after loading a checkpoint.

## Compatibility

`Parameters.ConfigureLegacyV2()` restores v2 defaults. A v2 population
checkpoint loads in legacy mode automatically. V3 checkpoints additionally
persist evaluation evidence, scheduler position, validation state, QD archive,
lineage, operator posteriors, and emitter state. `SetFitness()` remains a
supported deterministic one-case fallback for existing integrations.
