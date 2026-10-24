// Regression tests for the rate and spiking network solvers and their
// persistence formats.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "NeuralNetwork.h"
#include "SpikingLearning.h"

namespace
{
int failures = 0;

void Check(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

using namespace NEAT;

Neuron InputNeuron()
{
    Neuron neuron;
    neuron.m_type = INPUT;
    return neuron;
}

Connection Synapse(int source, int target, double weight)
{
    Connection connection;
    connection.m_source_neuron_idx = source;
    connection.m_target_neuron_idx = target;
    connection.m_weight = weight;
    return connection;
}

std::filesystem::path TemporaryPath(const char* name)
{
    return std::filesystem::temp_directory_path() / name;
}

void TestInhibitoryVetoExpires()
{
    // Exponential decay used to stall at a subnormal negative current, so a
    // single inhibitory spike vetoed a McCulloch-Pitts neuron forever.
    NeuralNetwork network;
    network.AddNeuron(InputNeuron());
    network.AddNeuron(InputNeuron());
    Neuron output;
    output.m_type = OUTPUT;
    output.m_activation_function_type = MCCULLOCH_PITTS;
    output.m_spike_threshold = 1.0;
    output.m_refractory_period = 0.0;
    network.AddNeuron(output);
    Connection inhibitory = Synapse(0, 2, -1.0);
    Connection excitatory = Synapse(1, 2, 2.0);
    excitatory.m_synaptic_time_constant = 0.0001;
    network.AddConnection(inhibitory);
    network.AddConnection(excitatory);
    network.SetInputOutputDimensions(2, 1);
    network.SetSpikingInputMode(BINARY_SPIKE_INPUT);
    network.SetSpikingTimeStep(0.001);
    network.Flush();

    network.StepSpiking({1.0, 0.0});
    bool fired = false;
    for (int step = 0; step < 5000 && !fired; ++step)
        fired = network.StepSpiking({0.0, 1.0}).front() == 1.0;
    Check(fired, "an inhibitory McCulloch-Pitts veto eventually expires");
    Check(
        network.m_connections[0].m_synaptic_current == 0.0,
        "decayed synaptic current is flushed to exactly zero");
}

void TestLifStableForLargeSteps()
{
    // Explicit Euler overshot for dt > tau, so inhibitory drive alone could
    // make a LIF neuron fire.
    NeuralNetwork network;
    network.AddNeuron(InputNeuron());
    Neuron output;
    output.m_type = OUTPUT;
    output.m_activation_function_type = SPIKING_LIF;
    output.m_timeconst = 0.005;
    output.m_spike_threshold = 1.0;
    output.m_reset_potential = 0.0;
    output.m_resting_potential = 0.0;
    output.m_refractory_period = 0.0;
    network.AddNeuron(output);
    network.AddConnection(Synapse(0, 1, -1.0));
    network.SetInputOutputDimensions(1, 1);
    network.SetSpikingInputMode(CURRENT_INPUT);
    network.SetSpikingTimeStep(0.02);
    network.Flush();
    bool fired = false;
    for (int step = 0; step < 50; ++step)
        fired = fired || network.StepSpiking({1.0}).front() == 1.0;
    Check(!fired, "purely inhibitory drive never fires a LIF neuron");
    Check(
        network.m_neurons[1].m_membrane_potential < 0.0 &&
            network.m_neurons[1].m_membrane_potential >= -1.0,
        "LIF membrane converges monotonically toward its steady state");
}

void TestLeakyRateStableForLargeSteps()
{
    NeuralNetwork network;
    network.AddNeuron(InputNeuron());
    Neuron output;
    output.m_type = OUTPUT;
    output.m_activation_function_type = LINEAR;
    output.m_timeconst = 0.01;
    network.AddNeuron(output);
    network.AddConnection(Synapse(0, 1, 1.0));
    network.SetInputOutputDimensions(1, 1);
    network.Flush();
    std::vector<double> input{1.0};
    network.Input(input);
    for (int step = 0; step < 20; ++step)
        network.ActivateLeaky(0.05);
    const double value = network.m_neurons[1].m_membrane_potential;
    Check(
        std::isfinite(value) && value > 0.99 && value <= 1.0 + 1e-12,
        "leaky integration converges for steps longer than tau");
}

void TestPoissonRateSurvivesActivate()
{
    // Activate() rebuilt its inputs from activations, which Poisson sampling
    // overwrites with 0/1, so the requested rate applied for one step only.
    NeuralNetwork network;
    network.AddNeuron(InputNeuron());
    Neuron output;
    output.m_type = OUTPUT;
    output.m_activation_function_type = SPIKING_LIF;
    network.AddNeuron(output);
    network.AddConnection(Synapse(0, 1, 0.0));
    network.SetInputOutputDimensions(1, 1);
    network.SetSpikingInputMode(POISSON_RATE_INPUT);
    network.SetSpikingTimeStep(0.001);
    network.SeedSpiking(7);
    network.Flush();
    std::vector<double> rate{1000.0};
    network.Input(rate);
    for (int step = 0; step < 200; ++step)
        network.Activate();
    Check(
        network.m_neurons[0].m_spike_count > 80,
        "Poisson input keeps its rate across repeated Activate() calls");
}

void TestInputSumsDoNotAccumulate()
{
    NeuralNetwork network;
    network.AddNeuron(InputNeuron());
    Neuron output;
    output.m_type = OUTPUT;
    output.m_activation_function_type = LINEAR;
    network.AddNeuron(output);
    network.AddConnection(Synapse(0, 1, 1.0));
    network.AddConnection(Synapse(1, 0, 1.0));
    network.SetInputOutputDimensions(1, 1);
    network.Flush();
    std::vector<double> input{1.0};
    network.Input(input);
    for (int step = 0; step < 10; ++step)
        network.Activate();
    Check(
        network.m_neurons[0].m_activesum == 0.0,
        "connections into input neurons do not accumulate sums");
}

NeuralNetwork DelayedSpikingNetwork()
{
    NeuralNetwork network;
    network.AddNeuron(InputNeuron());
    Neuron output;
    output.m_type = OUTPUT;
    output.m_activation_function_type = MCCULLOCH_PITTS;
    output.m_spike_threshold = 1.0;
    network.AddNeuron(output);
    Connection connection = Synapse(0, 1, 2.0);
    connection.m_synaptic_delay = 0.005;
    connection.m_synaptic_time_constant = 0.001;
    network.AddConnection(connection);
    network.SetInputOutputDimensions(1, 1);
    network.SetSpikingInputMode(BINARY_SPIKE_INPUT);
    network.SetSpikingTimeStep(0.001);
    network.Flush();
    return network;
}

int FirstSpike(NeuralNetwork& network, int limit)
{
    for (int step = 0; step < limit; ++step)
    {
        if (network.StepSpiking({0.0}).front() == 1.0)
            return step;
    }
    return -1;
}

void TestFileSavePreservesLiveState()
{
    NeuralNetwork network = DelayedSpikingNetwork();
    network.m_connections[0].m_weight = 2.0 + 3e-19;
    network.m_neurons[1].m_bias = 3e-19;
    network.StepSpiking({1.0});

    const auto path = TemporaryPath("multineat_network_state.txt");
    network.Save(path.string().c_str());
    NeuralNetwork loaded;
    Check(loaded.Load(path.string().c_str()), "saved network loads");
    std::filesystem::remove(path);

    Check(
        loaded.m_connections.size() == 1 &&
            loaded.m_connections[0].m_pending_events.size() == 1,
        "in-flight delayed spikes survive a file round trip");
    Check(
        loaded.m_neurons.size() == 2 &&
            loaded.m_neurons[1].m_bias == 3e-19,
        "file persistence keeps full double precision");
    NeuralNetwork original = network;
    Check(
        FirstSpike(original, 20) == FirstSpike(loaded, 20) &&
            FirstSpike(network, 20) != -1,
        "a reloaded network continues exactly like the original");
}

void TestCorruptFileLoadFails()
{
    NeuralNetwork network = DelayedSpikingNetwork();
    const auto path = TemporaryPath("multineat_network_corrupt.txt");
    network.Save(path.string().c_str());
    std::ifstream original_file(path);
    std::string text(
        (std::istreambuf_iterator<char>(original_file)),
        std::istreambuf_iterator<char>());
    original_file.close();
    const auto marker = text.find("spatial_neuron");
    Check(marker != std::string::npos, "saved network has spatial records");
    // Replace the coordinate count of the first neuron with -1.
    const auto line_end = text.find('\n', marker);
    const auto count_start = text.rfind(' ', line_end) + 1;
    text.replace(count_start, line_end - count_start, "-1");
    {
        std::ofstream corrupt(path);
        corrupt << text;
    }
    NeuralNetwork target = DelayedSpikingNetwork();
    bool loaded = true;
    try
    {
        loaded = target.Load(path.string().c_str());
    }
    catch (const std::exception&)
    {
        Check(false, "corrupt network files do not throw from Load");
    }
    std::filesystem::remove(path);
    Check(!loaded, "corrupt network files are rejected");
    Check(
        target.m_neurons.size() == 2,
        "a failed load leaves the existing network unchanged");
}

void TestSerializeNonFiniteState()
{
    NeuralNetwork network;
    network.AddNeuron(InputNeuron());
    Neuron output;
    output.m_type = OUTPUT;
    output.m_activation_function_type = LINEAR;
    network.AddNeuron(output);
    network.AddConnection(Synapse(0, 1, 10.0));
    network.SetInputOutputDimensions(1, 1);
    network.Flush();
    network.m_neurons[1].m_activation =
        std::numeric_limits<double>::infinity();
    network.m_neurons[1].m_last_input =
        -std::numeric_limits<double>::infinity();
    network.m_neurons[1].m_membrane_potential =
        std::numeric_limits<double>::quiet_NaN();
    bool restored = false;
    try
    {
        const NeuralNetwork copy =
            NeuralNetwork::Deserialize(network.Serialize());
        restored =
            std::isinf(copy.m_neurons[1].m_activation) &&
            copy.m_neurons[1].m_last_input < 0.0 &&
            std::isnan(copy.m_neurons[1].m_membrane_potential);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
    }
    Check(restored, "non-finite runtime values survive Serialize()");

    bool rejected = false;
    try
    {
        NeuralNetwork::Deserialize(
            "NeuralNetworkFormat 7\nState 1 1 0 0 0.001 0 0 0 0 0\n"
            "TotalWeightChange 18446744073709551615\n");
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    catch (const std::exception&)
    {
    }
    Check(rejected, "oversized serialized counts raise runtime_error");
}

void TestEpropOutputFeedbackIsDiagonal()
{
    NeuralNetwork network;
    network.AddNeuron(InputNeuron());
    for (int output = 0; output < 2; ++output)
    {
        Neuron neuron;
        neuron.m_type = OUTPUT;
        neuron.m_activation_function_type = SPIKING_LIF;
        network.AddNeuron(neuron);
        network.AddConnection(Synapse(0, 1 + output, 1.0));
    }
    network.SetInputOutputDimensions(1, 2);
    network.SetSpikingInputMode(BINARY_SPIKE_INPUT);
    network.SetSpikingOutputMode(MEMBRANE_POTENTIAL_OUTPUT);
    network.Flush();
    for (const EPropFeedbackMode mode :
         {EPROP_RANDOM_FEEDBACK, EPROP_UNIFORM_FEEDBACK})
    {
        EPropConfig config;
        config.feedback_mode = mode;
        EPropLearner learner(config);
        learner.Initialize(network);
        // Row-major [neuron][output] matrix.
        const std::vector<double>& feedback = learner.FeedbackMatrix();
        bool diagonal = feedback.size() == 3 * 2;
        for (std::size_t neuron = 1; diagonal && neuron < 3; ++neuron)
        {
            for (std::size_t output = 0; output < 2; ++output)
            {
                const double expected =
                    neuron - 1 == output ? 1.0 : 0.0;
                diagonal = diagonal &&
                           feedback[neuron * 2 + output] == expected;
            }
        }
        Check(
            diagonal,
            "each e-prop output receives only its own loss derivative");
    }
}
} // namespace

int main()
{
    TestInhibitoryVetoExpires();
    TestLifStableForLargeSteps();
    TestLeakyRateStableForLargeSteps();
    TestPoissonRateSurvivesActivate();
    TestInputSumsDoNotAccumulate();
    TestFileSavePreservesLiveState();
    TestCorruptFileLoadFails();
    TestSerializeNonFiniteState();
    TestEpropOutputFeedbackIsDiagonal();
    if (failures != 0)
    {
        std::cerr << failures << " network test(s) failed\n";
        return 1;
    }
    std::cout << "All network tests passed\n";
    return 0;
}
