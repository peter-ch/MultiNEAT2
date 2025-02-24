#!/usr/bin/env python
# python_xor_demo.py

import time
import pymultineat as pnt
from neattools import DrawGenome, DrawGenomes

# Define the XOR training data.
# Note that the genome is initialized with three inputs: two (the XOR inputs)
# plus an extra bias input (set to 1.0). Hence each input vector has three values.
xor_data = [
    ([0.0, 0.0, 1.0], 0.0),
    ([0.0, 1.0, 1.0], 1.0),
    ([1.0, 0.0, 1.0], 1.0),
    ([1.0, 1.0, 1.0], 0.0)
]

# Define a function to evaluate the fitness of a genome on XOR.
def xor_test(genome):
    # Create a neural network object.
    nn = pnt.NeuralNetwork()
    # ‘BuildPhenotype’ constructs the network from the genome.
    genome.BuildPhenotype(nn)
    total_error = 0.0

    # For each XOR sample, clear the network, provide the inputs and propagate.
    for inputs, expected in xor_data:
        nn.Flush()              # reset activations
        nn.Input(inputs)        # the genome expects 3 inputs (2 data + bias)
        # Calling Activate twice (as in the C++ example) allows the signal to propagate through the network.
        nn.Activate()
        nn.Activate()
        output = nn.Output()[0]
        total_error += abs(expected - output)

    # In this example the fitness is defined as the square of (4 – total error)
    fitness = (4.0 - total_error) ** 2
    return fitness


def main():
    # Create and customize MultiNEAT parameters.
    params = pnt.Parameters()
    params.PopulationSize = 150
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = False
    params.WeightDiffCoeff = 0.1
    params.CompatTreshold = 2.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 15
    params.OldAgeTreshold = 35
    params.MinSpecies = 2
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.0
    params.OverallMutationRate = 0.3
    params.ArchiveEnforcement = False
    params.MutateWeightsProb = 0.25
    params.WeightMutationMaxPower = 0.5
    params.WeightReplacementMaxPower = 8.0
    params.MutateWeightsSevereProb = 0.0
    params.WeightMutationRate = 0.85
    params.WeightReplacementRate = 0.2
    params.MaxWeight = 8.0
    params.MutateAddNeuronProb = 0.01
    params.MutateAddLinkProb = 0.1
    params.MutateRemLinkProb = 0.0
    params.MutateRemSimpleNeuronProb = 0.0
    params.NeuronTries = 64
    params.MutateAddLinkFromBiasProb = 0.0
    params.CrossoverRate = 0.0
    params.MultipointCrossoverRate = 0.0
    params.SurvivalRate = 0.2
    params.MutateNeuronTraitsProb = 0.0
    params.MutateLinkTraitsProb = 0.0

    params.AllowLoops = False
    params.AllowClones = False

    # Create a GenomeInitStruct.
    # Here we specify 3 inputs (2 XOR plus bias), 1 output, no hidden nodes,
    # and use the PERCEPTRON seed type with UNSIGNED_SIGMOID activation functions.
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 3
    init_struct.NumOutputs = 1
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.HiddenActType = pnt.UNSIGNED_SIGMOID
    init_struct.OutputActType = pnt.UNSIGNED_SIGMOID

    # Create a prototype genome using the parameters and initialization struct.
    genome_prototype = pnt.Genome(params, init_struct)

    # Create the initial population.
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))

    generations = 100  # total number of generations
    for gen in range(generations):
        # Evaluate all genomes in every species on the XOR task.
        for species in pop.m_Species:
            for i in range(len(species.m_Individuals)):
                genome = species.m_Individuals[i]
                fitness = xor_test(genome)
                genome.SetFitness(fitness)
                genome.SetEvaluated()

        # Get the best genome in the population.
        bestGenome = pop.GetBestGenome()
        bestFitness = bestGenome.GetFitness()
        print("Generation: {}, Best Fitness: {}".format(gen, bestFitness), flush=True)

        # Advance one generation.
        pop.Epoch()

    print("Simulation completed.", flush=True)

    # DrawGenome(bestGenome)
    gs = [x.m_Individuals[0] for x in pop.m_Species]
    DrawGenomes(gs)

    import pickle as pkl 
    pkl.dump(gs, open('shit.pkl','wb'))

if __name__ == "__main__":
    main()
