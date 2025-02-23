#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>
#include "Genome.h"
#include "Population.h"
#include "NeuralNetwork.h"
#include "Parameters.h"

using namespace NEAT;

// XOR input and output pairs
std::vector<std::pair<std::vector<double>, double>> xor_data = {
    {{0, 0}, 0},
    {{0, 1}, 1},
    {{1, 0}, 1},
    {{1, 1}, 0}
};

// Function to evaluate the XOR fitness of a genome
double xortest(Genome& g) {
    NeuralNetwork nn;
    g.BuildPhenotype(nn);

    double total_error = 0.0;
    for (auto& pair : xor_data) {
        std::vector<double>& inputs = pair.first;
        double expected_output = pair.second;

        nn.Flush();
        nn.Input(inputs);
        nn.Activate();
        nn.Activate();
        double output = nn.Output()[0];

        total_error += std::fabs(expected_output - output);
    }

    // Calculate fitness based on error (smaller error translates to higher fitness)
    double fitness = (4.0 - total_error)*(4.0 - total_error); // Maximum total_error is 16

    return fitness;
}

int main() {
    Parameters params;

    // Setting essential NEAT parameters
    params.PopulationSize = 150;
    params.DynamicCompatibility = true;
    params.NormalizeGenomeSize = false;
    params.WeightDiffCoeff = 0.1;
    params.CompatTreshold = 2.0;
    params.YoungAgeTreshold = 15;
    params.SpeciesMaxStagnation = 15;
    params.OldAgeTreshold = 35;
    params.MinSpecies = 2;
    params.MaxSpecies = 10;
    params.RouletteWheelSelection = false;
    params.RecurrentProb = 0.0;
    params.OverallMutationRate = 0.3;

    params.ArchiveEnforcement = false;

    params.MutateWeightsProb = 0.25;

    params.WeightMutationMaxPower = 0.5;
    params.WeightReplacementMaxPower = 8.0;
    params.MutateWeightsSevereProb = 0.0;
    params.WeightMutationRate = 0.85;
    params.WeightReplacementRate = 0.2;

    params.MaxWeight = 8;

    params.MutateAddNeuronProb = 0.001;
    params.MutateAddLinkProb = 0.03;
    params.MutateRemLinkProb = 0.0;

    params.MinActivationA = 4.9;
    params.MaxActivationA = 4.9;

    params.ActivationFunction_SignedSigmoid_Prob = 0.0;
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0;
    params.ActivationFunction_Tanh_Prob = 0.0;
    params.ActivationFunction_SignedStep_Prob = 0.0;

    params.CrossoverRate = 0.0;
    params.MultipointCrossoverRate = 0.0;
    params.SurvivalRate = 0.2;

    params.MutateNeuronTraitsProb = 0;
    params.MutateLinkTraitsProb = 0;

    params.AllowLoops = false;
    params.AllowClones = false;

    GenomeInitStruct ints;
    ints.NumInputs = 3;
    ints.NumOutputs = 1;
    ints.NumHidden = 0;
    ints.SeedType = PERCEPTRON;
    ints.HiddenActType = UNSIGNED_SIGMOID;
    ints.OutputActType = UNSIGNED_SIGMOID;

    Genome genomePrototype(params, ints);
    Population pop(genomePrototype, params, true, 1.0, time(0));

    const int generations = 1000;
    for (int gen = 0; gen < generations; ++gen) {
        for (auto& species : pop.m_Species) {
            for (auto& individual : species.m_Individuals) {
                double fitness = xortest(individual);
                individual.SetFitness(fitness);
                individual.SetEvaluated();
            }
        }

        auto bestGenome = pop.GetBestGenome();
        double bestFitness = bestGenome.GetFitness();
        printf("Generation: %d, Best Fitness: %3.5f\n", gen, bestFitness);

        pop.Epoch();
    }

    std::cout << "\nSimulation completed.\n";
    return 0;
}

