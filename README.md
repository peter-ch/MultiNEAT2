# MultiNEAT2

MultiNEAT2 is a C++17 implementation of NEAT (NeuroEvolution of
Augmenting Topologies) with optional Python bindings. It includes
speciation, structural and trait mutation, recurrent networks, phased
search, novelty search, HyperNEAT substrates, and ES-HyperNEAT parameters.

This repository continues the original MultiNEAT codebase while retaining
its established C++ and Python names wherever possible. Compatibility
spellings such as `SetInputOutputDimentions`, `GetConnectionLenght`, and
`Elitism` remain available alongside corrected names.

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
| `BUILD_TESTING` | `ON` | Build and register regression tests |
| `MULTINEAT_WARNINGS_AS_ERRORS` | `OFF` | Treat compiler warnings as errors |

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

The Box2D and MuJoCo examples under `demos/` require their corresponding
Gymnasium extras and visualization dependencies. They are not required by
the core library.

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

## Project layout

- `src/` — C++ library and pybind11 bindings
- `tests/` — C++ and Python regression tests plus an installed-package test
- `demos/` — XOR, Box2D, MuJoCo, and visualization examples
- `neattools.py` — genome visualization helpers

## License

MultiNEAT2 is distributed under the GNU Lesser General Public License,
version 3 or (at your option) any later version, matching the license
notices in the inherited MultiNEAT source. See [LICENSE](LICENSE).
