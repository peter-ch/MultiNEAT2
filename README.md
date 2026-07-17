# MultiNEAT2

MultiNEAT2 is a powerful, open-source C++ implementation of NEAT (NeuroEvolution of Augmenting Topologies) with Python bindings. MultiNEAT2 evolves neural networks by gradually complexifying their topology while optimizing connection weights. The project features a full C++ library (with modern C++17 features) plus a Python module (via pybind11) that lets you experiment with neuroevolution, run training simulations (for example on the XOR problem), and visualize results using networkx and matplotlib.

I named this MultiNEAT2 to keep the original MultiNEAT intact. Development of the original library is shutting down and work continues on this new project.

---

## Key Features

- **Complete NEAT Implementation**: Full implementation of the NEAT algorithm including speciation, dynamic compatibility threshold adjustment, genome mating and mutation operators, and innovation tracking.
- **Advanced Evolution Strategies**: Support for Phased Searching, Novelty Search, and ES-HyperNEAT extensions.
- **Modular C++ Architecture**: Clean separation of concerns with dedicated modules for Genome, Population, Species, NeuralNetwork, Innovation, Traits, and Utilities.
- **High-Performance Python Bindings**: Seamless Python integration via pybind11 with the `pymultineat` module for experimentation and visualization.
- **Physics Environment Integration**: Pre-built demos for Box2D and MuJoCo physics environments including Lunar Lander, Bipedal Walker, and Humanoid control.
- **Comprehensive Visualization**: Advanced network visualization tools with `neattools.py` for analyzing genome structure and evolution progress.
- **Parallel Evaluation**: Built-in support for multiprocessing to accelerate fitness evaluation in computationally intensive environments.

---

## Project Structure

### Core C++ Components
- **`src/`** - Main C++ source code directory
  - *Genome.cpp/h*: Genome representation and manipulation
  - *Species.cpp/h*: Speciation algorithms and species management
  - *Population.cpp/h*: Population-level evolutionary operations
  - *NeuralNetwork.cpp/h*: Phenotype construction and activation
  - *Innovation.cpp/h*: Innovation tracking system
  - *Traits.cpp/h*: Trait parameter management
  - *Utils.cpp/h*: Utility functions and mathematical operations
  - *Random.cpp/h*: Random number generation
  - *Substrate.cpp/h*: HyperNEAT substrate definitions
  - *Bindings.cpp*: pybind11 interface definitions
  - *Assert.h*: Debug assertion macros
  - *Main.cpp*: Entry point for C++ demonstrations

### Python Interface
- **`pymultineat`**: Python module exposing core C++ classes via pybind11
- **`neattools.py`**: Comprehensive utility library for genome analysis and visualization

### Demonstration Scripts
- **`demos/xor.py`**: Classic XOR problem demonstration
- **`demos/box2d/`**: Physics-based environments using Box2D
  - *lunar_lander_box2d.py*: Lunar Lander control
  - *bipedal_walker_box2d.py*: Bipedal locomotion
  - *car_racing_box2d.py*: Car racing challenge
- **`demos/mujoco/`**: Advanced physics environments using MuJoCo
  - *humanoid_mujoco.py*: Humanoid control
  - *ant_mujoco.py*: Quadrupedal locomotion
  - *halfcheetah_mujoco.py*: Fast bipedal running

---

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **C++ Compiler**: Compiler supporting C++17 (GCC 7+, Clang 6+, or MSVC 2017+)
- **CMake**: Version 3.10 or later
- **Python**: Python 3.7+ with pip

### Python Dependencies
```bash
pip install pybind11 networkx matplotlib gymnasium pygame tqdm numpy
```

### Physics Engine Dependencies (Optional)
For Box2D and MuJoCo demos:
```bash
# Box2D environments
pip install Box2D

# MuJoCo environments (requires MuJoCo license)
pip install mujoco
```

---

## Build Instructions

### Windows (Visual Studio)
```bash
git clone https://github.com/peter-ch/MultiNEAT2
cd MultiNEAT2

# Create build directory
mkdir build
cd build

# Configure with Visual Studio generator
cmake .. -G "Visual Studio 17 2022" -A x64

# Build using Visual Studio or command line
cmake --build . --config Release
```

### Windows (MinGW/Ninja)
```bash
git clone https://github.com/peter-ch/MultiNEAT2
cd MultiNEAT2

# Create build directory
mkdir build
cd build

# Configure with Ninja generator
cmake .. -G "Ninja" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

# Build
cmake --build .
```

### Linux/macOS
```bash
git clone https://github.com/peter-ch/MultiNEAT2
cd MultiNEAT2

# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
make -j$(nproc)
```

---

## Parameter Reference

The `Parameters` class controls all aspects of the NEAT algorithm. Below are key parameters organized by category:

### Population & Speciation
| Parameter | Default | Description |
|---------|--------|------------|
| `PopulationSize` | 150 | Number of genomes in the population |
| `Speciation` | true | Enable species formation |
| `DynamicCompatibility` | true | Automatically adjust compatibility threshold |
| `MinSpecies` | 2 | Minimum number of species |
| `MaxSpecies` | 10 | Maximum number of species |
| `CompatTreshold` | 2.0 | Compatibility threshold for speciation |
| `SurvivalRate` | 0.2 | Fraction of best individuals that reproduce |

### Mutation Operators
| Parameter | Default | Description |
|---------|--------|------------|
| `MutateAddNeuronProb` | 0.01 | Probability of adding a neuron |
| `MutateAddLinkProb` | 0.1 | Probability of adding a link |
| `MutateRemLinkProb` | 0.0 | Probability of removing a link |
| `RecurrentProb` | 0.0 | Probability of creating recurrent connections |
| `OverallMutationRate` | 0.3 | Probability of mutation after crossover |
| `WeightMutationRate` | 0.85 | Probability of mutating a weight |
| `WeightMutationMaxPower` | 0.5 | Maximum weight perturbation |

### Compatibility Calculation
| Parameter | Default | Weight |
|---------|--------|--------|
| `ExcessCoeff` | 1.0 | Excess genes importance |
| `DisjointCoeff` | 1.0 | Disjoint genes importance |
| `WeightDiffCoeff` | 0.4 | Weight difference importance |
| `ActivationFunctionDiffCoeff` | 0.0 | Activation function difference |

### Age-Based Parameters
| Parameter | Default | Description |
|---------|--------|------------|
| `YoungAgeTreshold` | 15 | Age threshold for young species |
| `YoungAgeFitnessBoost` | 1.0 | Fitness boost for young species |
| `OldAgeTreshold` | 35 | Age threshold for old species |
| `OldAgePenalty` | 0.0 | Penalty multiplier for old species |
| `SpeciesMaxStagnation` | 15 | Generations without improvement before penalty |

### Activation Functions
| Function | Probability | Description |
|---------|------------|------------|
| `UNSIGNED_SIGMOID` | 1.0 | Standard sigmoid (0-1) |
| `TANH` | 0.0 | Hyperbolic tangent (-1 to 1) |
| `SIGNED_SIGMOID` | 0.0 | Sigmoid (-1 to 1) |
| `LINEAR` | 0.0 | Linear activation |
| `RELU` | 0.0 | Rectified Linear Unit |

---

## Usage Examples

### Running Demos

#### XOR Problem (Basic)
```bash
python demos/xor.py
```

#### Lunar Lander (Box2D)
```bash
# Serial evaluation (slower, easier to debug)
python demos/box2d/lunar_lander_box2d.py --serial

# Parallel evaluation (faster)
python demos/box2d/lunar_lander_box2d.py
```

#### Humanoid Control (MuJoCo)
```bash
python demos/mujoco/humanoid_mujoco.py --serial
```

### Custom Implementation

```python
import pymultineat as pnt
from neattools import DrawGenome, print_genome_summary
import time

def custom_fitness_function(genome):
    # Create neural network phenotype
    nn = pnt.NeuralNetwork()
    genome.BuildPhenotype(nn)
    
    # Evaluate on your custom task
    total_error = 0.0
    # ... your evaluation logic here ...
    
    # Return non-negative fitness
    fitness = 1.0 / (1.0 + total_error)
    return fitness

def main():
    # Configure NEAT parameters
    params = pnt.Parameters()
    params.PopulationSize = 200
    params.MutateAddNeuronProb = 0.02
    params.MutateAddLinkProb = 0.15
    params.RecurrentProb = 0.3
    
    # Initialize genome structure
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 8  # Adjust for your problem
    init_struct.NumOutputs = 2
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.OutputActType = pnt.TANH
    
    # Create population
    genome_prototype = pnt.Genome(params, init_struct)
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))
    
    # Evolution loop
    for generation in range(1000):
        # Evaluate all genomes
        for species in pop.m_Species:
            for individual in species.m_Individuals:
                fitness = custom_fitness_function(individual)
                individual.SetFitness(fitness)
        
        # Get best genome
        best_genome = pop.GetBestGenome()
        print(f"Generation {generation}: {best_genome.GetFitness()}")
        
        # Advance generation
        pop.Epoch()
    
    # Visualize result
    DrawGenome(best_genome)
    print_genome_summary(best_genome)

if __name__ == "__main__":
    main()
```

---

## Advanced Features

### Parallel Evaluation
MultiNEAT2 supports multiprocessing for faster fitness evaluation:

```python
import multiprocessing
from tqdm import tqdm

def init_worker():
    global worker_env
    worker_env = gym.make('LunarLander-v3')

def evaluate_genome_parallel(genome):
    return evaluate_genome(genome, env=worker_env)

# In your main loop:
with multiprocessing.Pool(processes=8, initializer=init_worker) as pool:
    fitnesses = pool.map(evaluate_genome_parallel, genomes)
```

### Network Visualization
The `neattools.py` module provides comprehensive visualization:

```python
from neattools import DrawGenome, DrawGenomes, get_layered_nodes

# Draw single genome
DrawGenome(best_genome)

# Draw multiple genomes in grid
DrawGenomes([genome1, genome2, genome3])

# Analyze network structure
layers = get_layered_nodes(best_genome)
print("Network layers:", layers)
```

### Genome Analysis
Detailed genome inspection tools:

```python
from neattools import print_genome_summary, narrate_traits

# Print basic statistics
print_genome_summary(genome)

# Detailed trait analysis
narrate_traits(genome)

# Export to Graphviz format
export_genome_graph(genome, "network.dot")
```

---

## Troubleshooting

### Common Build Issues
- **"pybind11 not found"**: Install via `pip install pybind11` and ensure CMake can locate it
- **Compiler errors**: Verify your compiler supports C++17 features
- **Linking errors**: Ensure Python development headers are installed

### Runtime Issues
- **Import errors**: Verify the `pymultineat` module is in your Python path
- **Segmentation faults**: Check for memory access violations in custom code
- **Slow performance**: Consider reducing population size or enabling parallel evaluation

---

## License

Apache 2.0
