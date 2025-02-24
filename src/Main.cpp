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
    { {0, 0, 1}, 0 },
    { {0, 1, 1}, 1 },
    { {1, 0, 1}, 1 },
    { {1, 1, 1}, 0 }
};

// Function to evaluate the XOR fitness of a genome.
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
    // Fitness is higher when error is lower.
    double fitness = (4.0 - total_error) * (4.0 - total_error);
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
    params.MutateAddNeuronProb = 0.01;
    params.MutateAddLinkProb = 0.1;
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

    GenomeInitStruct init;
    init.NumInputs = 3;
    init.NumOutputs = 1;
    init.NumHidden = 0;
    init.SeedType = PERCEPTRON;
    init.HiddenActType = UNSIGNED_SIGMOID;
    init.OutputActType = UNSIGNED_SIGMOID;

    Genome genomePrototype(params, init);
    Population pop(genomePrototype, params, true, 1.0, static_cast<int>(time(nullptr)));

    const int generations = 1000;
    for (int gen = 0; gen < generations; ++gen) {
        for (auto& species : pop.m_Species) {
            for (auto& individual : species.m_Individuals) {
                double fitness = xortest(individual);
                individual.SetFitness(fitness);
                individual.SetEvaluated();
            }
        }
        Genome bestGenome = pop.GetBestGenome();
        double bestFitness = bestGenome.GetFitness();
        std::cout << "Generation: " << gen << ", Best Fitness: " << bestFitness << std::endl;
        pop.Epoch();
    }
    std::cout << "\nSimulation completed.\n";
    return 0;
}
