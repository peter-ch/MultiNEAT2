# MultiNEAT

MultiNEAT is a powerful, open–source C++ implementation of NEAT (NeuroEvolution of Augmenting Topologies) with Python bindings. MultiNEAT evolves neural networks by gradually complexifying their topology while optimizing connection weights. The project features a full C++ library (with modern C++17 features) plus a Python module (via pybind11) that lets you experiment with neuroevolution, run training simulations (for example on the XOR problem), and visualize results using networkx and matplotlib.

---

## Features

- **NEAT Evolution:** Implements the full NEAT algorithm including speciation, dynamic compatibility threshold adjustment, genome mating and mutation operators, and innovation tracking.
- **Modular Design:** The project is split into several source modules (Genome, Population, Species, NeuralNetwork, Innovation, Traits, and Utilities) for clearer organization and easy extension.
- **Python Bindings:** Use the provided pybind11 bindings (the module is named `pymultineat`) to experiment with MultiNEAT from Python. The included helper module `neattools.py` provides utility functions such as converting genomes to networkx graphs and drawing neural network diagrams.
- **Demonstration Scripts:** An example Python script (`xor.py`) demonstrates how to set up a simple XOR task by creating the required parameters, initializing a population and evolving solution candidates.

---

## Project Structure

- **CMakeLists.txt:** The build script used with CMake to compile the C++ library and executable.
- **src/**  
  - *Genome.cpp / Genome.h:* Classes and methods for representing and manipulating genomes.
  - *Species.cpp / Species.h:* Classes for speciation and grouping genomes by similarity.
  - *Population.cpp / Population.h:* The high–level population class that drives the evolutionary cycle.
  - *NeuralNetwork.cpp / NeuralNetwork.h:* Code for converting a genome into a fast neural network phenotype and activation routines.
  - *Innovation.cpp / Innovation.h:* Classes for tracking innovations (new neurons and links) using the NEAT method.
  - *Traits.cpp / Traits.h:* Definitions for traits and parameters controlling trait mutations.
  - *Utils.cpp / Utils.h:* Miscellaneous helper functions (scaling, clamping, rounding, etc.).
  - *Random.cpp / Random.h:* Random number generator functions built on top of std::mt19937.
  - *Substrate.cpp / Substrate.h:* (for HyperNEAT extensions) Defines the phenotype substrate of neural networks.
  - *Bindings.cpp:* Contains the pybind11 module definition that exposes MultiNEAT classes and functions to Python.
  - *Assert.h:* Custom assertion and debug macros.
  - *Main.cpp:* Contains the main() function for running a NEAT evolution simulation (for example, training on XOR).

- **Python Scripts:**  
  - *neattools.py:* Provides helper functions for converting genomes to networkx graphs, visualizing networks with matplotlib, and additional utility functions.
  - *xor.py:* A complete example that sets up a NEAT evolutionary process on the XOR problem using the Python bindings.

---

## Requirements

- **C++ Compiler:** A compiler that supports C++17.
- **CMake:** Version 3.10 or later.
- **Python:** Python 3.x; required for the pybind11 module and the demo scripts.
- **pybind11:** The Python bindings library. (Installable via pip: `pip install pybind11` or via your distribution’s package manager.)
- **Python Packages:**  
  - networkx  
  - matplotlib  
  - (Optional) pydot (if you wish to export graphs in DOT format)
  
*Tip:* Make sure you have a recent C++ compiler (e.g., gcc 7+, clang 6+ or MSVC 2017+) installed.

---

## Building the Project

1. **Clone the repository:**

   git clone https://github.com/yourusername/multineat.git  
   cd multineat

2. **Create a build directory and run CMake:**

   mkdir build  
   cd build  
   cmake ..  
   make

This will compile both the executable (typically named `multineat_exe`) and the Python module (e.g. `pymultineat` shared library). Refer to the generated build files for your platform.

---

## Usage

### Running the C++ Demo

After building, you can run the compiled executable:

    ./multineat_exe

This executable uses the Main.cpp code to perform an evolution simulation on the XOR problem (or another demo task if modified).

### Using Python Bindings

The Python module `pymultineat` exposes the core classes (Genome, Population, NeuralNetwork, etc.). In addition, `neattools.py` provides additional functionality (conversion to networkx graphs, drawing functions, etc.). For example, you can run the XOR demo by executing:

    python xor.py

This script demonstrates how to:
- Set up MultiNEAT parameters and a GenomeInitStruct.
- Create a population.
- Evaluate genomes on an XOR task.
- Evolve the population for a specified number of generations.
- Visualize the best evolved neural network.

---

## Example: Evolving an XOR Network

Below is a brief outline of the steps used in the `xor.py` script:

1. **Import Modules:**

   ```python
   import pymultineat as pnt
   from neattools import DrawGenome, DrawGenomes
   import time
   ```

2. **Define Training Data:**  
   (XOR input/output pairs with bias input included.)

3. **Define Fitness Function:**  
   The function constructs a neural network phenotype from a genome, feeds the XOR inputs, activates the network, and computes an error. The fitness is defined to be higher when the error is lower.

4. **Set Parameters and Initialize:**  
   Create a `Parameters` instance and a `GenomeInitStruct` (with 3 inputs, 1 output, 0 hidden nodes, using PERCEPTRON seed) and then instantiate a prototype genome and a `Population`.

5. **Evolution Loop:**  
   For a fixed number of generations, evaluate each genome’s fitness on XOR and call `Epoch()` to produce the next generation.

6. **Display Result:**  
   After evolution completes, the best genome is drawn via matplotlib.
