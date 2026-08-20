#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
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
double xortest(Genome& g, bool spiking, int spiking_steps) {
    NeuralNetwork nn;
    g.BuildPhenotype(nn);

    if (spiking) {
        nn.SetSpikingInputMode(BINARY_SPIKE_INPUT);
        nn.SetSpikingOutputMode(SPIKE_OUTPUT);
        nn.SetSpikingTimeStep(0.001);
    }

    double total_error = 0.0;
    for (auto& pair : xor_data) {
        std::vector<double>& inputs = pair.first;
        double expected_output = pair.second;

        nn.Flush();
        double output = 0.0;
        if (spiking) {
            for (int step = 0; step < spiking_steps; ++step) {
                output += nn.StepSpiking(inputs)[0];
            }
            output /= static_cast<double>(spiking_steps);
        } else {
            nn.Input(inputs);
            nn.Activate();
            nn.Activate();
            output = nn.Output()[0];
        }
        total_error += std::fabs(expected_output - output);
    }
    // Fitness is higher when error is lower.
    double fitness = (4.0 - total_error) * (4.0 - total_error);
    return fitness;
}

int main(int argc, char** argv) {
    bool spiking = false;
    bool mcculloch_pitts = false;
    int generations = 1000;
    int population_size = 150;
    int spiking_steps = 16;
    int seed = static_cast<int>(time(nullptr));
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&](const char* option) {
            if (index + 1 >= argc) {
                std::cerr << option << " requires an integer value\n";
                std::exit(2);
            }
            return std::stoi(argv[++index]);
        };
        if (argument == "--spiking") {
            spiking = true;
        } else if (argument == "--mcculloch-pitts") {
            spiking = true;
            mcculloch_pitts = true;
        } else if (argument == "--smoke") {
            generations = 1;
            population_size = 10;
            spiking_steps = 4;
        } else if (argument == "--generations") {
            generations = value("--generations");
        } else if (argument == "--population") {
            population_size = value("--population");
        } else if (argument == "--seed") {
            seed = value("--seed");
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: multineat_exe [--spiking|--mcculloch-pitts] "
                   "[--smoke] [--generations N] [--population N] "
                   "[--seed N]\n";
            return 0;
        } else {
            std::cerr << "Unknown option: " << argument << '\n';
            return 2;
        }
    }
    if (generations < 1 || population_size < 2) {
        std::cerr << "Generations must be >= 1 and population must be >= 2\n";
        return 2;
    }

    Parameters params;

    // Setting essential NEAT parameters
    params.PopulationSize = population_size;
    params.DynamicCompatibility = true;
    params.NormalizeGenomeSize = false;
    params.WeightDiffCoeff = 0.1;
    params.CompatTreshold = 2.0;
    params.YoungAgeTreshold = 15;
    params.SpeciesMaxStagnation = 15;
    params.OldAgeTreshold = 35;
    params.MinSpecies = 2;
    params.MaxSpecies = 10;
    params.RouletteWheelSelection = true;
    params.TournamentSelection = true;
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

    if (mcculloch_pitts) {
        params.ConfigureMcCullochPitts(true, false);
    } else if (spiking) {
        params.ConfigureSpiking(false);
    }
    if (spiking) {
        params.RecurrentProb = 0.0;
        params.AllowLoops = false;
        params.MinSpikeThreshold = 0.25;
        params.MaxSpikeThreshold = 1.5;
    }

    GenomeInitStruct init;
    init.NumInputs = 3;
    init.NumOutputs = 1;
    init.NumHidden = 0;
    init.SeedType = PERCEPTRON;
    init.HiddenActType = mcculloch_pitts
        ? MCCULLOCH_PITTS
        : (spiking ? SPIKING_ADAPTIVE_LIF : UNSIGNED_SIGMOID);
    init.OutputActType = mcculloch_pitts
        ? MCCULLOCH_PITTS
        : (spiking ? SPIKING_LIF : UNSIGNED_SIGMOID);

    Genome genomePrototype(params, init);
    Population pop(genomePrototype, params, true, 1.0, seed);

    double bestFitness = 0.0;
    for (int gen = 0; gen < generations; ++gen) {
        for (auto& species : pop.m_Species) {
            for (auto& individual : species.m_Individuals) {
                double fitness = xortest(
                    individual, spiking, spiking_steps);
                individual.SetFitness(fitness);
                individual.SetEvaluated();
            }
        }
        Genome bestGenome = pop.GetBestGenome();
        bestFitness = bestGenome.GetFitness();
        std::cout << "Generation: " << gen << ", Best Fitness: " << bestFitness << std::endl;
        if (gen + 1 < generations) {
            pop.Epoch();
        }
    }
    std::cout
        << "{\"demo\":\"cpp_xor\",\"policy\":\""
        << (mcculloch_pitts ? "mcculloch-pitts" :
            (spiking ? "lif" : "rate"))
        << "\",\"generations\":" << generations
        << ",\"population\":" << population_size
        << ",\"best_fitness\":" << bestFitness << "}\n";

    // test the rng here
    /*RNG rng;
    std::vector<double> probs;
    probs.push_back(10.0);
    probs.push_back(3.0);
    probs.push_back(0.0);
    probs.push_back(11.0);
    probs.push_back(1.0);
    probs.push_back(100.0);
    //for (int gen = 0; gen < 1000; ++gen)
    //{
    //    std::cout << rng.Roulette(probs) << "\n";
    //}

    // Map to track the frequency of each choice
    std::map<int, int> stats;

    for (int gen = 0; gen < 10000; ++gen) {
        int choice = rng.Roulette(probs);
        std::cout << choice << "\n";
        stats[choice]++; // Increment the count for the chosen index
    }

    // Print statistics
    std::cout << "\nStatistics:\n";
    for (const auto& pair : stats) {
        std::cout << "Choice " << pair.first << ": " << pair.second << " times ("
            << (pair.second / 100.0) << "%)\n";
    }*/


    return 0;
}
