#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Genome.h"
#include "Innovation.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Random.h"

namespace
{
using Clock = std::chrono::steady_clock;
volatile double benchmark_sink = 0.0;

double Seconds(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}
}

int main()
{
    constexpr unsigned int input_count = 20;
    constexpr unsigned int output_count = 10;
    constexpr unsigned int neuron_count = 120;
    constexpr unsigned int steps = 10000;

    NEAT::NeuralNetwork network;
    for (unsigned int index = 0; index < neuron_count; ++index)
    {
        NEAT::Neuron neuron;
        neuron.m_type =
            index < input_count ? NEAT::INPUT : NEAT::HIDDEN;
        neuron.m_activation_function_type = NEAT::TANH;
        network.AddNeuron(neuron);
    }
    for (unsigned int source = 0; source < neuron_count; ++source)
    {
        for (unsigned int target = input_count;
             target < neuron_count;
             ++target)
        {
            NEAT::Connection connection;
            connection.m_source_neuron_idx =
                static_cast<int>(source);
            connection.m_target_neuron_idx =
                static_cast<int>(target);
            connection.m_weight =
                (static_cast<double>(
                     (source * 17U + target * 13U) % 31U) -
                 15.0) /
                100.0;
            network.AddConnection(connection);
        }
    }
    network.SetInputOutputDimensions(input_count, output_count);
    std::vector<double> inputs(input_count, 0.1);
    network.Input(inputs);

    const auto checked_start = Clock::now();
    for (unsigned int step = 0; step < steps; ++step)
        network.Activate();
    const auto checked_end = Clock::now();

    network.Flush();
    network.Input(inputs);
    const auto fast_start = Clock::now();
    network.ActivateSteps(steps, true);
    const auto fast_end = Clock::now();

    const double checked_seconds =
        Seconds(checked_start, checked_end);
    const double fast_seconds = Seconds(fast_start, fast_end);
    std::cout << std::fixed << std::setprecision(4)
              << "neurons=" << neuron_count
              << " connections=" << network.m_connections.size()
              << " steps=" << steps << '\n'
              << "Activate:     " << checked_seconds << " s\n"
              << "ActivateFast: " << fast_seconds << " s\n"
              << "speedup:      "
              << checked_seconds / fast_seconds << "x\n";

    NEAT::Parameters parameters;
    NEAT::GenomeInitStruct init;
    init.NumInputs = static_cast<int>(input_count + 1U);
    init.NumOutputs = static_cast<int>(output_count);
    NEAT::Genome left(parameters, init);
    left.SetID(1);
    NEAT::InnovationDatabase innovations;
    innovations.Init(left);
    NEAT::RNG rng;
    rng.Seed(12345);
    for (int mutation = 0; mutation < 32; ++mutation)
        left.Mutate_AddNeuron(innovations, parameters, rng);
    NEAT::Genome right = left;
    right.SetID(2);
    for (std::size_t index = 0;
         index < right.m_LinkGenes.size();
         ++index)
    {
        right.m_LinkGenes[index].SetWeight(
            static_cast<double>(index % 17U) / 8.0 - 1.0);
    }
    left.SetFitness(2.0);
    right.SetFitness(1.0);

    constexpr unsigned int compatibility_steps = 100000;
    const auto compatibility_start = Clock::now();
    for (unsigned int step = 0;
         step < compatibility_steps;
         ++step)
    {
        benchmark_sink +=
            left.CompatibilityDistance(right, parameters);
    }
    const auto compatibility_end = Clock::now();
    const double compatibility_seconds =
        Seconds(compatibility_start, compatibility_end);

    constexpr unsigned int crossover_steps = 10000;
    const auto crossover_start = Clock::now();
    for (unsigned int step = 0; step < crossover_steps; ++step)
    {
        const NEAT::Genome child = left.MateWithMode(
            right,
            NEAT::MULTIPOINT,
            false,
            rng,
            parameters);
        benchmark_sink += static_cast<double>(child.NumLinks());
    }
    const auto crossover_end = Clock::now();
    const double crossover_seconds =
        Seconds(crossover_start, crossover_end);
    std::cout << "\nEvolution hot paths: neurons="
              << left.NumNeurons()
              << " links=" << left.NumLinks() << '\n'
              << "CompatibilityDistance: "
              << compatibility_seconds << " s ("
              << static_cast<double>(compatibility_steps) /
                     compatibility_seconds
              << " ops/s)\n"
              << "Multipoint crossover:  "
              << crossover_seconds << " s ("
              << static_cast<double>(crossover_steps) /
                     crossover_seconds
              << " ops/s)\n";

    NEAT::InnovationDatabase indexed_innovations;
    constexpr int innovation_count = 100000;
    for (int index = 0; index < innovation_count; ++index)
    {
        indexed_innovations.AddLinkInnovation(
            1 + index % 997,
            1 + (index * 31) % 991);
    }
    constexpr unsigned int innovation_queries = 1000000;
    const auto innovation_start = Clock::now();
    for (unsigned int query = 0;
         query < innovation_queries;
         ++query)
    {
        benchmark_sink += indexed_innovations.CheckInnovation(
            1 + static_cast<int>(query % 997U),
            1 + static_cast<int>((query * 31U) % 991U),
            NEAT::NEW_LINK);
    }
    const auto innovation_end = Clock::now();
    const double innovation_seconds =
        Seconds(innovation_start, innovation_end);
    std::cout << "Indexed innovation lookup: "
              << innovation_seconds << " s ("
              << static_cast<double>(innovation_queries) /
                     innovation_seconds
              << " ops/s over " << innovation_count
              << " historical innovations)\n";

    NEAT::NeuralNetwork online_template;
    constexpr unsigned int online_inputs = 4;
    constexpr unsigned int online_neurons = 32;
    for (unsigned int index = 0;
         index < online_neurons;
         ++index)
    {
        NEAT::Neuron neuron;
        neuron.m_type =
            index < online_inputs ? NEAT::INPUT : NEAT::HIDDEN;
        neuron.m_activation_function_type = NEAT::TANH;
        online_template.AddNeuron(neuron);
    }
    for (unsigned int target = online_inputs;
         target < online_neurons;
         ++target)
    {
        for (const unsigned int source :
             {target - 1U, target % online_inputs})
        {
            NEAT::Connection online_connection;
            online_connection.m_source_neuron_idx =
                static_cast<int>(source);
            online_connection.m_target_neuron_idx =
                static_cast<int>(target);
            online_connection.m_weight = 0.1;
            online_template.AddConnection(online_connection);
        }
    }
    online_template.SetInputOutputDimensions(online_inputs, 2);
    std::vector<double> online_values(online_inputs, 0.25);
    constexpr unsigned int online_steps = 1000;

    NEAT::NeuralNetwork dense_rtrl = online_template;
    dense_rtrl.InitRTRLMatrix();
    dense_rtrl.Input(online_values);
    const auto dense_rtrl_start = Clock::now();
    for (unsigned int step = 0; step < online_steps; ++step)
    {
        dense_rtrl.ActivateFast();
        dense_rtrl.RTRL_update_gradients();
    }
    const auto dense_rtrl_end = Clock::now();

    NEAT::NeuralNetwork sparse_rtrl = online_template;
    sparse_rtrl.InitSparseRTRLMatrix();
    sparse_rtrl.Input(online_values);
    const auto sparse_rtrl_start = Clock::now();
    for (unsigned int step = 0; step < online_steps; ++step)
    {
        sparse_rtrl.ActivateFast();
        sparse_rtrl.RTRL_update_gradients_sparse();
    }
    const auto sparse_rtrl_end = Clock::now();
    const std::size_t dense_state =
        static_cast<std::size_t>(online_neurons) *
        online_neurons * online_neurons;
    std::cout << "Dense RTRL:   "
              << Seconds(dense_rtrl_start, dense_rtrl_end)
              << " s, " << dense_state
              << " sensitivity values\n"
              << "Sparse RTRL:  "
              << Seconds(sparse_rtrl_start, sparse_rtrl_end)
              << " s, " << sparse_rtrl.SparseRTRLStateSize()
              << " sensitivity values\n";
    return 0;
}
