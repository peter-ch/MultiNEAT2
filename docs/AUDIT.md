# Project audit — 2026-10-24

A second full review covered the C++ core (genomes, innovations,
populations, species, reliability, networks, spiking solver, e-prop),
parameters and serialization, the Python bindings, `neattools.py`, the demos,
the launcher, packaging, and CI. Every finding below was reproduced with a
scratch program or test before it was fixed, and each fix has a regression
test (`tests/network_tests.cpp`, `tests/population_tests.cpp`, new cases in
`tests/core_tests.cpp`, `tests/test_runtime_safety.py`, and
`tests/test_neattools.py`).

| Area | Reproduced problem | Result |
| --- | --- | --- |
| McCulloch-Pitts veto | One inhibitory spike vetoed a neuron for 100,000+ steps: decay stalled at a subnormal current. | Decayed currents and traces flush to zero below `1e-12`. |
| LIF / leaky integration | With dt > tau, inhibitory drive alone produced spikes (explicit Euler overshoot). | Exact exponential integration; e-prop eligibility updated to match. |
| e-prop | Cross-output feedback leaked errors between outputs; MCP targets used LIF eligibility; adaptive and analog traces were one step stale. | Diagonal output feedback, exact MCP eligibility, decayed adaptation, transmitted analog signal. |
| Poisson input | `Activate()` loops ran at 0/1 Hz after the first step (1 spike vs 128 expected). | The requested rate is kept separately from the sampled spike. |
| Network files | Save dropped in-flight spikes, lost values below 1e-18, and corrupt counts threw from `Load`; `inf`/`nan` could not be deserialized. | `%.17g`, live-state records, bounded counts, round-trip parsing facet. |
| Genome | Duplicate FS-NEAT bias links; legacy save lost traits/vetoes/state; recurrent merge created a non-recurrent cycle; dead-end detection deleted healthy links; clone check ignored the veto; unchecked custom HyperNEAT indices; unbounded deserializer counts; signed overflow in trait arithmetic. | Fixed; seeded `Genome(..., RNG&)` constructor added. |
| Population | `Tick` selected parents uniformly (983/10000 picks for a 1000x fitter genome) and truncation picked the worst survivor; novelty ticks never marked babies evaluated; copies aliased the novelty archive; one NaN made all later checkpoints fail; Lamarckian hooks ran once per request. | Fixed with real-time adjusted fitness, sorting, flags, sanitized lineage scores, and per-generation tracking. |
| Python ownership | References into vectors, maps, and variants dangled after reallocation (garbage reads, one segfault); `m_PhenotypeBehavior` held a raw pointer to a collected object. | Copy semantics for container properties; read-only behavior pointer. |
| Parameters | A failing save truncated the file to 0 bytes; `Load` discarded constraint callbacks; lossy scalar parsing; int/float trait mismatches broke mutation. | Serialize-then-write, preserved callbacks, strict parsing, numeric coercion. |
| Performance | A declared copy assignment disabled moves for genes, genomes, traits, and species; `SpikingRecorder` was quadratic. | Implicit moves restored (crossover ~1.8x faster); lazy events. |
| Demos | Box2D/MuJoCo scripts were wrappers around a shared trainer; others imported `neattools`/`spiking_neat`; numba was mandatory. | Every demo is a standalone file; numba is optional. |
| Packaging / CI | No Python packaging; Node-20 actions; no physics or multi-version coverage. | scikit-build-core wheel, updated CI matrix, physics and wheel jobs. |

Validation (Windows x64, MSVC 19.44, CMake 4.2.3, Python 3.12.10): a fresh
Release build with warnings as errors passes all 11 CTest suites, including
the Box2D/MuJoCo smoke tests; the native suites pass under MSVC
AddressSanitizer; the installed CMake package consumer builds and runs; the
wheel installs into a clean virtual environment and imports; Ruff is clean.
Linux/macOS, GCC/Clang sanitizers, and Python 3.10/3.13 are covered by CI but
were not run locally.

Compatibility: LIF and leaky-rate trajectories change slightly (exact
integration), e-prop feedback consumes the RNG differently, Python container
properties now return copies (assign the whole value back), and
`RNG.Seed()` values of 2^32 or more no longer match older Linux runs.

# Project audit — 2026-09-06

Reviewed the C++ evolution and simulation APIs, reliability scheduler, checkpoint
handling, Python ownership, demos and visualization tests, CMake packaging, CI,
and build documentation. Changes focus on reproduced failures and regression
coverage. Existing edits to demo rendering and its tests were preserved.

## Findings and fixes

| Area | Reproduced problem | Result |
| --- | --- | --- |
| Asynchronous evolution | Reversing completion order changed bootstrap bounds and which genomes survived racing. | Canonical evidence ordering makes summaries, racing, and checkpoint resume independent of submission order. |
| QD archive | Four candidates became eight entries when archive reevaluation finished. | Archive work completes without reinserting training candidates; reevaluation counts and evidence are retained. |
| Numeric stability | Repeated maximum finite scores produced an infinite mean; representable standard deviations overflowed during squaring. | Normalized accumulation avoids intermediate overflow in score and descriptor statistics; Wilson bounds also handle extreme finite confidence multipliers. |
| Evidence updates | Invalid batches discarded prior evidence; passing a genome's own evidence vector cleared it. | Batch replacement commits only after validation and handles aliased input. |
| Request validation | Nonfinite validation descriptors were accepted, and inconsistent training descriptors failed only after requests were consumed. | Invalid values and dimensions are rejected before consuming requests, allowing corrected submissions. |
| Python ownership | Genome references could outlive their population or species and access freed storage. | Borrowed results retain their owners using `reference_internal`; population mutation can still invalidate them, as documented. |
| Network persistence | Failed loads cleared an existing network; an unopened stream could loop indefinitely; truncated records read uninitialized fields. | Parsing occurs in a temporary network, extraction failure ends scanning, and enum fields are checked before use. |
| RNG persistence | Failed state loading partially changed the generator; trailing garbage was accepted. | Loading validates a temporary generator before committing it. |
| Online learning | Bad later input channels consumed RNG state; malformed signals or later batch rows advanced simulation or partially trained weights. | Inputs, signals, and batch dimensions are checked before the affected operations change state. |
| Windows shared libraries | `BUILD_SHARED_LIBS=ON` produced no import library, failing executable links. | CMake exports DLL symbols; Windows CI now builds and tests this configuration. |
| Build and CI | CMake 3.15 was advertised despite using `Development.Module`, introduced in 3.18. Python test registration was duplicated, and the installed consumer was linked but never executed. | Corrected the minimum version, consolidated registration, added timeouts and two regression suites, and execute the consumer in CI. |

## Validation

Local environment: Windows x64, MSVC 19.44, CMake 4.2.3, Python 3.12.10.
Builds used `MULTINEAT_WARNINGS_AS_ERRORS=ON`.

- Baseline: all seven original CTest suites passed before the changes; new
  regressions reproduced the missing coverage.
- Updated Release build: all nine CTest suites passed, with no suites skipped.
  This includes the core, bindings, reliability, runtime safety, visualization,
  Gymnasium demo, launcher, spiking demo, and native demo suites.
- Native RelWithDebInfo build: the core suite passed with MSVC AddressSanitizer.
- Windows shared Release build: both native suites passed.
- Installed static and shared packages: downstream CMake consumers configured,
  linked, and ran successfully. The shared consumer used the installed `bin`
  directory on its process `PATH`.
- Python Ruff checks and `git diff --check` passed.
- Core microbenchmarks completed, including the added evidence workloads.

The new Python suites include 17 test methods, with further subcases for invalid
inputs and reference ownership. Native regressions also cover aliased evidence,
failed loads, stream termination, and large-score statistics.

## Aggregation performance

A local Release microbenchmark compared `src/Reliability.cpp` from baseline
commit `62983e3` with the updated implementation, using the same driver and
compiler: 2,000 summaries, 16 scores per summary, 2,048 bootstrap replicates.

| Workload | Baseline | Updated |
| --- | ---: | ---: |
| Variable scores (`i * i % 17`) | 0.422478 s | 0.357745 s |
| Constant scores (`1.0`) | 0.344566 s | 0.000237 s |

Variable-score aggregation used about 15% less time in this run. Selecting the
required bootstrap quantile avoids sorting every replicate; constant scores
skip bootstrap work entirely. These are local microbenchmark measurements,
not end-to-end evolution speed guarantees. The repository benchmark now
includes both workloads for repeatable measurement.

## Scope and compatibility

Public method names and checkpoint format versions remain unchanged. Corrected
numerical calculations can change last-bit results and subsequent selection
relative to an older binary. Deterministic resume is tested within the updated
implementation, including reversed worker completion order.

Validation here covers Windows and the installed optional test dependencies.
Linux/macOS builds and GCC/Clang undefined-behavior sanitizers remain covered by
the CI configuration but were not run locally. Interactive native GUI behavior
and long physics training campaigns were not manually exercised. This audit
does not establish convergence or certification performance on every task.
