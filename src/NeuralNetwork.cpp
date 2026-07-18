#include <math.h>
#include <float.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include "NeuralNetwork.h"
#include "FileIO.h"
#include "Assert.h"
#include "Utils.h"

#define sqr(x) ((x)*(x))
#define LEARNING_RATE 0.0001

namespace NEAT
{
    namespace
    {
        void ValidateNetworkTopology(const NeuralNetwork& network)
        {
            const std::size_t neuron_count = network.m_neurons.size();
            if (network.m_num_inputs > neuron_count ||
                network.m_num_outputs > neuron_count - network.m_num_inputs)
            {
                throw std::runtime_error(
                    "Neural network input/output dimensions exceed its neuron count");
            }
            for (const auto &connection : network.m_connections)
            {
                if (connection.m_source_neuron_idx < 0 ||
                    connection.m_target_neuron_idx < 0 ||
                    static_cast<std::size_t>(connection.m_source_neuron_idx) >=
                        neuron_count ||
                    static_cast<std::size_t>(connection.m_target_neuron_idx) >=
                        neuron_count)
                {
                    throw std::runtime_error(
                        "Neural network connection index is out of range");
                }
            }
        }
    }

    inline double af_sigmoid_unsigned(double aX, double aSlope, double aShift)
    {
        return 1.0 / (1.0 + exp(-aSlope * aX - aShift));
    }

    inline double af_sigmoid_signed(double aX, double aSlope, double aShift)
    {
        double tY = af_sigmoid_unsigned(aX, aSlope, aShift);
        return (tY - 0.5) * 2.0;
    }

    inline double af_tanh(double aX, double aSlope, double aShift)
    {
        return tanh(aX * aSlope + aShift);
    }

    inline double af_tanh_cubic(double aX, double aSlope, double aShift)
    {
        return tanh(aX * aX * aX * aSlope + aShift);
    }

    inline double af_step_signed(double aX, double aShift)
    {
        return (aX > aShift) ? 1.0 : -1.0;
    }

    inline double af_step_unsigned(double aX, double aShift)
    {
        return (aX > (0.5 + aShift)) ? 1.0 : 0.0;
    }

    inline double af_gauss_signed(double aX, double aSlope, double aShift)
    {
        double tY = exp(-aSlope * aX * aX + aShift);
        return (tY - 0.5) * 2.0;
    }

    inline double af_gauss_unsigned(double aX, double aSlope, double aShift)
    {
        return exp(-aSlope * aX * aX + aShift);
    }

    inline double af_abs(double aX, double aShift)
    {
        return (aX + aShift < 0.0) ? -(aX + aShift) : (aX + aShift);
    }

    inline double af_sine_signed(double aX, double aFreq, double aShift)
    {
        return sin(aX * aFreq + aShift);
    }

    inline double af_sine_unsigned(double aX, double aFreq, double aShift)
    {
        double tY = sin(aX * aFreq + aShift);
        return (tY + 1.0) / 2.0;
    }

    inline double af_linear(double aX, double aShift)
    {
        return aX + aShift;
    }

    inline double af_relu(double aX)
    {
        return (aX > 0) ? aX : 0;
    }

    inline double af_softplus(double aX)
    {
        return std::max(aX, 0.0) + log1p(exp(-std::abs(aX)));
    }

    double unsigned_sigmoid_derivative(double x)
    {
        return x * (1 - x);
    }

    double tanh_derivative(double x)
    {
        return 1 - x * x;
    }

    double activation_derivative(const Neuron& neuron)
    {
        const double output = neuron.m_activation;
        const double input = neuron.m_last_input;
        switch (neuron.m_activation_function_type)
        {
        case SIGNED_SIGMOID:
            return neuron.m_a * (1.0 - output * output) * 0.5;
        case UNSIGNED_SIGMOID:
            return neuron.m_a * output * (1.0 - output);
        case TANH:
            return neuron.m_a * (1.0 - output * output);
        case TANH_CUBIC:
            return neuron.m_a * 3.0 * input * input *
                   (1.0 - output * output);
        case SIGNED_GAUSS:
            return -2.0 * neuron.m_a * input * (output + 1.0);
        case UNSIGNED_GAUSS:
            return -2.0 * neuron.m_a * input * output;
        case ABS:
        {
            const double shifted = input + neuron.m_b;
            return shifted > 0.0 ? 1.0 : (shifted < 0.0 ? -1.0 : 0.0);
        }
        case SIGNED_SINE:
            return neuron.m_a *
                   std::cos(input * neuron.m_a + neuron.m_b);
        case UNSIGNED_SINE:
            return 0.5 * neuron.m_a *
                   std::cos(input * neuron.m_a + neuron.m_b);
        case LINEAR:
            return 1.0;
        case RELU:
            return input > 0.0 ? 1.0 : 0.0;
        case SOFTPLUS:
            if (input >= 0.0)
                return 1.0 / (1.0 + std::exp(-input));
            {
                const double exponential = std::exp(input);
                return exponential / (1.0 + exponential);
            }
        default:
            // Step functions are non-differentiable.
            return 0.0;
        }
    }

    inline double EvaluateActivation(Neuron& neuron, double input)
    {
        neuron.m_last_input = input;
        switch (neuron.m_activation_function_type)
        {
        case SIGNED_SIGMOID:
            return af_sigmoid_signed(input, neuron.m_a, neuron.m_b);
        case UNSIGNED_SIGMOID:
            return af_sigmoid_unsigned(input, neuron.m_a, neuron.m_b);
        case TANH:
            return af_tanh(input, neuron.m_a, neuron.m_b);
        case TANH_CUBIC:
            return af_tanh_cubic(input, neuron.m_a, neuron.m_b);
        case SIGNED_STEP:
            return af_step_signed(input, neuron.m_b);
        case UNSIGNED_STEP:
            return af_step_unsigned(input, neuron.m_b);
        case SIGNED_GAUSS:
            return af_gauss_signed(input, neuron.m_a, neuron.m_b);
        case UNSIGNED_GAUSS:
            return af_gauss_unsigned(input, neuron.m_a, neuron.m_b);
        case ABS:
            return af_abs(input, neuron.m_b);
        case SIGNED_SINE:
            return af_sine_signed(input, neuron.m_a, neuron.m_b);
        case UNSIGNED_SINE:
            return af_sine_unsigned(input, neuron.m_a, neuron.m_b);
        case LINEAR:
            return af_linear(input, neuron.m_b);
        case RELU:
            return af_relu(input);
        case SOFTPLUS:
            return af_softplus(input);
        default:
            return af_sigmoid_unsigned(input, neuron.m_a, neuron.m_b);
        }
    }

    NeuralNetwork::NeuralNetwork(bool a_Minimal)
    {
        if (!a_Minimal)
        {
            m_neurons.resize(5);
            m_num_inputs = 3;
            m_num_outputs = 1;

            const int endpoints[][2] = {
                {0, 3}, {1, 3}, {2, 3},
                {0, 4}, {1, 4}, {2, 4},
                {4, 3}
            };
            for (const auto &endpoint : endpoints)
            {
                Connection connection;
                connection.m_source_neuron_idx = endpoint[0];
                connection.m_target_neuron_idx = endpoint[1];
                connection.m_weight =
                    static_cast<double>(std::rand()) /
                        static_cast<double>(RAND_MAX) -
                    0.5;
                m_connections.push_back(connection);
            }
            InitRTRLMatrix();
        }
        else
        {
            m_num_inputs = m_num_outputs = 0;
            m_total_error = 0;
            Clear();
        }
    }

    NeuralNetwork::NeuralNetwork()
    {
        m_num_inputs = m_num_outputs = 0;
        m_total_error = 0;
        Clear();
    }

    void NeuralNetwork::InitRTRLMatrix()
    {
        if (IsSpiking())
        {
            throw std::invalid_argument(
                "RTRL is not defined for non-differentiable spiking "
                "activations; use evolution or STDP");
        }
        m_sparse_rtrl_sensitivities.clear();
        for (auto &neuron : m_neurons)
        {
            neuron.m_sensitivity_matrix.assign(
                m_neurons.size(),
                std::vector<double>(m_neurons.size(), 0.0));
        }
        FlushCube();
        m_total_error = 0;
        m_total_weight_change.assign(m_connections.size(), 0.0);
    }

    void NeuralNetwork::InitSparseRTRLMatrix()
    {
        ValidateNetworkTopology(*this);
        if (IsSpiking())
        {
            throw std::invalid_argument(
                "RTRL is not defined for non-differentiable spiking "
                "activations; use evolution or STDP");
        }
        for (auto& neuron : m_neurons)
            neuron.m_sensitivity_matrix.clear();
        m_sparse_rtrl_sensitivities.assign(
            m_neurons.size(),
            std::vector<double>(m_connections.size(), 0.0));
        m_total_error = 0.0;
        m_total_weight_change.assign(m_connections.size(), 0.0);
    }

    void NeuralNetwork::ActivateFast()
    {
        if (IsSpiking())
        {
            ValidateNetworkTopology(*this);
            std::vector<double> inputs;
            inputs.reserve(m_num_inputs);
            for (unsigned int i = 0; i < m_num_inputs; ++i)
                inputs.push_back(m_neurons[i].m_activation);
            StepSpiking(inputs);
            return;
        }
        // The phenotype builder guarantees valid endpoint indexes. This is the
        // intentionally unchecked hot path; Activate() remains the validating
        // entry point for networks assembled through public vectors.
        for (auto &conn : m_connections)
        {
            conn.m_source_activation =
                m_neurons[conn.m_source_neuron_idx].m_activation;
            conn.m_signal = conn.m_source_activation * conn.m_weight;
            m_neurons[conn.m_target_neuron_idx].m_activesum += conn.m_signal;
        }
        for (size_t i = m_num_inputs; i < m_neurons.size(); i++)
        {
            Neuron& neuron = m_neurons[i];
            const double input = neuron.m_activesum;
            neuron.m_activesum = 0;
            neuron.m_activation =
                EvaluateActivation(neuron, input);
        }
    }

    void NeuralNetwork::Activate()
    {
        ValidateNetworkTopology(*this);
        ActivateFast();
    }

    void NeuralNetwork::ActivateUseInternalBias()
    {
        ValidateNetworkTopology(*this);
        if (IsSpiking())
        {
            throw std::invalid_argument(
                "Use StepSpiking for spiking networks; neuron bias is "
                "already included as tonic current");
        }
        for (auto &conn : m_connections)
        {
            conn.m_source_activation =
                m_neurons[conn.m_source_neuron_idx].m_activation;
            conn.m_signal = conn.m_source_activation * conn.m_weight;
        }
        for (auto &conn : m_connections)
            m_neurons[conn.m_target_neuron_idx].m_activesum += conn.m_signal;
        for (size_t i = m_num_inputs; i < m_neurons.size(); i++)
        {
            Neuron& neuron = m_neurons[i];
            const double input =
                neuron.m_activesum + neuron.m_bias;
            neuron.m_activesum = 0;
            neuron.m_activation =
                EvaluateActivation(neuron, input);
        }
    }

    void NeuralNetwork::ActivateLeaky(double a_dtime)
    {
        ValidateNetworkTopology(*this);
        if (IsSpiking())
        {
            throw std::invalid_argument(
                "Use StepSpiking to advance stateful spiking "
                "activations");
        }
        if (!std::isfinite(a_dtime) || a_dtime < 0.0)
        {
            throw std::invalid_argument(
                "Leaky activation time step must be finite and non-negative");
        }
        for (std::size_t i = m_num_inputs; i < m_neurons.size(); ++i)
        {
            if (!std::isfinite(m_neurons[i].m_timeconst) ||
                m_neurons[i].m_timeconst <= 0.0)
            {
                throw std::domain_error(
                    "Leaky activation requires positive, finite neuron time constants");
            }
        }
        for (auto &conn : m_connections)
        {
            conn.m_source_activation =
                m_neurons[conn.m_source_neuron_idx].m_activation;
            conn.m_signal = conn.m_source_activation * conn.m_weight;
        }
        for (auto &conn : m_connections)
            m_neurons[conn.m_target_neuron_idx].m_activesum += conn.m_signal;
        for (size_t i = m_num_inputs; i < m_neurons.size(); i++)
        {
            double t_const = a_dtime / m_neurons[i].m_timeconst;
            m_neurons[i].m_membrane_potential = (1.0 - t_const) * m_neurons[i].m_membrane_potential + t_const * m_neurons[i].m_activesum;
        }
        for (size_t i = m_num_inputs; i < m_neurons.size(); i++)
        {
            Neuron& neuron = m_neurons[i];
            const double input =
                neuron.m_membrane_potential + neuron.m_bias;
            neuron.m_activesum = 0;
            neuron.m_activation =
                EvaluateActivation(neuron, input);
        }
    }

    void NeuralNetwork::Flush()
    {
        for (auto &neuron : m_neurons)
        {
            neuron.m_activation = 0;
            neuron.m_activesum = 0;
            neuron.m_membrane_potential =
                neuron.m_activation_function_type ==
                        SPIKING_IZHIKEVICH
                    ? neuron.m_izhikevich_c
                    : neuron.m_resting_potential;
            neuron.m_last_input = 0;
            neuron.m_refractory_remaining = 0.0;
            neuron.m_adaptation = 0.0;
            neuron.m_izhikevich_recovery =
                neuron.m_izhikevich_b *
                neuron.m_membrane_potential;
            neuron.m_spike = false;
            neuron.m_spike_count = 0;
            neuron.m_last_spike_time = -1.0;
            neuron.m_rate_trace = 0.0;
        }
        for (auto& connection : m_connections)
        {
            connection.m_signal = 0.0;
            connection.m_source_activation = 0.0;
            connection.m_synaptic_current = 0.0;
            connection.m_presynaptic_signal = 0.0;
            connection.m_stdp_pre_trace = 0.0;
            connection.m_stdp_post_trace = 0.0;
            connection.m_pending_events.clear();
        }
        m_spiking_time = 0.0;
        m_spike_history.clear();
    }

    void NeuralNetwork::FlushCube()
    {
        for (auto &neuron : m_neurons)
            for (auto &row : neuron.m_sensitivity_matrix)
                std::fill(row.begin(), row.end(), 0.0);
    }

    void NeuralNetwork::Input(std::vector<double> &a_Inputs)
    {
        if (m_num_inputs > m_neurons.size())
        {
            throw std::runtime_error(
                "Neural network input count exceeds its neuron count");
        }
        const size_t mx = std::min(a_Inputs.size(),
                                   static_cast<size_t>(m_num_inputs));
        for (size_t i = 0; i < mx; i++)
            m_neurons[i].m_activation = a_Inputs[i];
    }

    void NeuralNetwork::InputExact(const std::vector<double>& a_Inputs)
    {
        if (a_Inputs.size() != m_num_inputs)
        {
            throw std::invalid_argument(
                "Neural network input count must match exactly");
        }
        if (m_num_inputs > m_neurons.size())
        {
            throw std::runtime_error(
                "Neural network input count exceeds its neuron count");
        }
        for (std::size_t i = 0; i < a_Inputs.size(); ++i)
            m_neurons[i].m_activation = a_Inputs[i];
    }

    void NeuralNetwork::ActivateSteps(unsigned int steps, bool fast)
    {
        if (!fast)
            ValidateNetworkTopology(*this);
        if (IsSpiking())
        {
            ValidateNetworkTopology(*this);
            std::vector<double> inputs;
            inputs.reserve(m_num_inputs);
            for (unsigned int i = 0; i < m_num_inputs; ++i)
                inputs.push_back(m_neurons[i].m_activation);
            for (unsigned int step = 0; step < steps; ++step)
                StepSpiking(inputs);
        }
        else
        {
            for (unsigned int step = 0; step < steps; ++step)
                ActivateFast();
        }
    }

    std::vector<std::vector<double>> NeuralNetwork::ActivateBatch(
        const std::vector<std::vector<double>>& inputs,
        unsigned int steps,
        bool use_internal_bias)
    {
        ValidateNetworkTopology(*this);
        if (IsSpiking())
        {
            throw std::invalid_argument(
                "ActivateBatch is for independent rate-network samples; "
                "use SimulateSpiking for temporal spiking inputs");
        }
        std::vector<std::vector<double>> outputs;
        outputs.reserve(inputs.size());
        for (const auto& sample : inputs)
        {
            Flush();
            InputExact(sample);
            for (unsigned int step = 0; step < steps; ++step)
            {
                if (use_internal_bias)
                    ActivateUseInternalBias();
                else
                    ActivateFast();
            }
            std::vector<double> output;
            output.reserve(m_num_outputs);
            for (unsigned int i = 0; i < m_num_outputs; ++i)
            {
                output.push_back(
                    m_neurons[m_num_inputs + i].m_activation);
            }
            outputs.push_back(std::move(output));
        }
        return outputs;
    }

    bool NeuralNetwork::IsSpiking() const
    {
        return std::any_of(
            m_neurons.begin(),
            m_neurons.end(),
            [](const Neuron& neuron)
            {
                return IsSpikingActivation(
                    neuron.m_activation_function_type);
            });
    }

    void NeuralNetwork::SetSpikingTimeStep(double time_step)
    {
        if (!std::isfinite(time_step) || time_step <= 0.0)
        {
            throw std::invalid_argument(
                "Spiking time step must be finite and positive");
        }
        m_spiking_time_step = time_step;
    }

    void NeuralNetwork::SetSpikingInputMode(SpikingInputMode mode)
    {
        if (mode < CURRENT_INPUT || mode > POISSON_RATE_INPUT)
            throw std::invalid_argument("Unsupported spiking input mode");
        m_spiking_input_mode = mode;
    }

    void NeuralNetwork::SetSpikingOutputMode(SpikingOutputMode mode)
    {
        if (mode < SPIKE_OUTPUT ||
            mode > MEMBRANE_POTENTIAL_OUTPUT)
        {
            throw std::invalid_argument(
                "Unsupported spiking output mode");
        }
        m_spiking_output_mode = mode;
    }

    void NeuralNetwork::SeedSpiking(std::uint64_t seed)
    {
        // Xorshift generators cannot advance from an all-zero state.
        m_spiking_rng_state =
            seed == 0 ? UINT64_C(0x9e3779b97f4a7c15) : seed;
    }

    void NeuralNetwork::EnableSpikeRecording(
        bool enabled,
        std::size_t max_events)
    {
        m_record_spikes = enabled;
        m_max_recorded_spikes = max_events;
        if (!enabled)
            m_spike_history.clear();
        else if (max_events > 0 &&
                 m_spike_history.size() > max_events)
        {
            m_spike_history.erase(
                m_spike_history.begin(),
                m_spike_history.end() -
                    static_cast<std::ptrdiff_t>(max_events));
        }
    }

    void NeuralNetwork::EnableSTDP(bool enabled)
    {
        for (auto& connection : m_connections)
            connection.m_stdp_enabled = enabled;
    }

    std::vector<double> NeuralNetwork::OutputSpikes() const
    {
        if (m_num_inputs + m_num_outputs > m_neurons.size())
            throw std::runtime_error("Invalid network output dimensions");
        std::vector<double> result;
        result.reserve(m_num_outputs);
        for (unsigned int i = 0; i < m_num_outputs; ++i)
        {
            result.push_back(
                m_neurons[m_num_inputs + i].m_spike ? 1.0 : 0.0);
        }
        return result;
    }

    std::vector<double> NeuralNetwork::OutputRates() const
    {
        if (m_num_inputs + m_num_outputs > m_neurons.size())
            throw std::runtime_error("Invalid network output dimensions");
        std::vector<double> result;
        result.reserve(m_num_outputs);
        for (unsigned int i = 0; i < m_num_outputs; ++i)
        {
            const Neuron& neuron = m_neurons[m_num_inputs + i];
            result.push_back(
                m_spiking_time > 0.0
                    ? static_cast<double>(neuron.m_spike_count) /
                          m_spiking_time
                    : 0.0);
        }
        return result;
    }

    std::vector<double> NeuralNetwork::OutputFilteredSpikes() const
    {
        if (m_num_inputs + m_num_outputs > m_neurons.size())
            throw std::runtime_error("Invalid network output dimensions");
        std::vector<double> result;
        result.reserve(m_num_outputs);
        for (unsigned int i = 0; i < m_num_outputs; ++i)
            result.push_back(
                m_neurons[m_num_inputs + i].m_rate_trace);
        return result;
    }

    std::vector<double> NeuralNetwork::OutputMembranePotentials() const
    {
        if (m_num_inputs + m_num_outputs > m_neurons.size())
            throw std::runtime_error("Invalid network output dimensions");
        std::vector<double> result;
        result.reserve(m_num_outputs);
        for (unsigned int i = 0; i < m_num_outputs; ++i)
        {
            result.push_back(
                m_neurons[m_num_inputs + i].m_membrane_potential);
        }
        return result;
    }

    std::vector<double> NeuralNetwork::OutputDecoded() const
    {
        switch (m_spiking_output_mode)
        {
        case SPIKE_OUTPUT:
            return OutputSpikes();
        case FIRING_RATE_OUTPUT:
            return OutputRates();
        case FILTERED_SPIKE_OUTPUT:
            return OutputFilteredSpikes();
        case MEMBRANE_POTENTIAL_OUTPUT:
            return OutputMembranePotentials();
        default:
            throw std::logic_error("Invalid spiking output mode");
        }
    }

    std::vector<double> NeuralNetwork::StepSpiking(
        const std::vector<double>& inputs,
        double time_step)
    {
        ValidateNetworkTopology(*this);
        if (inputs.size() != m_num_inputs)
        {
            throw std::invalid_argument(
                "Spiking input count must match exactly");
        }
        const double dt =
            time_step < 0.0 ? m_spiking_time_step : time_step;
        if (!std::isfinite(dt) || dt <= 0.0)
        {
            throw std::invalid_argument(
                "Spiking time step must be finite and positive");
        }

        const double step_start = m_spiking_time;
        const double step_end = step_start + dt;
        std::vector<bool> source_spikes(m_neurons.size(), false);
        std::vector<double> source_amplitudes(
            m_neurons.size(), 1.0);

        const auto random_unit = [this]()
        {
            std::uint64_t x = m_spiking_rng_state;
            x ^= x >> 12U;
            x ^= x << 25U;
            x ^= x >> 27U;
            m_spiking_rng_state = x;
            const std::uint64_t value =
                x * UINT64_C(2685821657736338717);
            return static_cast<double>(value >> 11U) *
                   (1.0 / 9007199254740992.0);
        };
        const auto record =
            [this](const SpikeEvent& event)
        {
            if (!m_record_spikes)
                return;
            m_spike_history.push_back(event);
            if (m_max_recorded_spikes > 0 &&
                m_spike_history.size() > m_max_recorded_spikes)
            {
                const std::size_t excess =
                    m_spike_history.size() -
                    m_max_recorded_spikes;
                m_spike_history.erase(
                    m_spike_history.begin(),
                    m_spike_history.begin() +
                        static_cast<std::ptrdiff_t>(excess));
            }
        };

        for (std::size_t i = 0; i < m_neurons.size(); ++i)
        {
            Neuron& neuron = m_neurons[i];
            neuron.m_activesum = 0.0;
            if (i < m_num_inputs)
            {
                const double value = inputs[i];
                if (!std::isfinite(value))
                {
                    throw std::invalid_argument(
                        "Spiking inputs must be finite");
                }
                bool spike = false;
                double amplitude = 1.0;
                switch (m_spiking_input_mode)
                {
                case CURRENT_INPUT:
                    neuron.m_activation = value;
                    break;
                case BINARY_SPIKE_INPUT:
                    spike = value > 0.0;
                    amplitude = value;
                    neuron.m_activation = spike ? amplitude : 0.0;
                    break;
                case POISSON_RATE_INPUT:
                {
                    if (value < 0.0)
                    {
                        throw std::invalid_argument(
                            "Poisson input rates cannot be negative");
                    }
                    const double probability =
                        -std::expm1(-value * dt);
                    spike = random_unit() < probability;
                    neuron.m_activation = spike ? 1.0 : 0.0;
                    break;
                }
                default:
                    throw std::logic_error(
                        "Invalid spiking input mode");
                }
                neuron.m_spike = spike;
                source_spikes[i] = spike;
                source_amplitudes[i] = amplitude;
                if (spike)
                {
                    ++neuron.m_spike_count;
                    neuron.m_last_spike_time = step_start;
                    record(SpikeEvent{
                        step_start,
                        static_cast<int>(i),
                        amplitude,
                        true});
                }
            }
            else
            {
                // Non-input spikes were produced at the end of the previous
                // synchronous step and are transmitted now.
                source_spikes[i] = neuron.m_spike;
                source_amplitudes[i] = 1.0;
            }
        }

        const double delivery_epsilon =
            std::numeric_limits<double>::epsilon() *
            std::max(1.0, std::abs(step_end)) * 8.0;
        for (auto& connection : m_connections)
        {
            if (!std::isfinite(connection.m_synaptic_delay) ||
                connection.m_synaptic_delay < 0.0 ||
                !std::isfinite(connection.m_synaptic_time_constant) ||
                connection.m_synaptic_time_constant <= 0.0 ||
                !std::isfinite(connection.m_weight))
            {
                throw std::domain_error(
                    "Spiking synapses require a non-negative finite delay "
                    "a positive finite time constant, and a finite weight");
            }
            connection.m_synaptic_current *=
                std::exp(-dt /
                         connection.m_synaptic_time_constant);
            connection.m_signal = 0.0;
            connection.m_presynaptic_signal = 0.0;

            const std::size_t source =
                static_cast<std::size_t>(
                    connection.m_source_neuron_idx);
            const Neuron& source_neuron = m_neurons[source];
            const bool event_source =
                IsSpikingActivation(
                    source_neuron.m_activation_function_type) ||
                (source < m_num_inputs &&
                 m_spiking_input_mode != CURRENT_INPUT);
            if (event_source && source_spikes[source])
            {
                PendingSynapticEvent event;
                event.delivery_time =
                    step_start + connection.m_synaptic_delay;
                event.amplitude =
                    source_amplitudes[source] *
                    connection.m_weight;
                event.source_amplitude =
                    source_amplitudes[source];
                connection.m_pending_events.push_back(event);
            }

            auto pending = connection.m_pending_events.begin();
            while (pending != connection.m_pending_events.end())
            {
                if (!std::isfinite(pending->delivery_time) ||
                    !std::isfinite(pending->amplitude) ||
                    !std::isfinite(
                        pending->source_amplitude))
                {
                    throw std::domain_error(
                        "Pending synaptic events must be finite");
                }
                if (pending->delivery_time <=
                    step_end + delivery_epsilon)
                {
                    connection.m_synaptic_current +=
                        pending->amplitude;
                    connection.m_signal += pending->amplitude;
                    connection.m_presynaptic_signal +=
                        pending->source_amplitude;
                    pending =
                        connection.m_pending_events.erase(pending);
                }
                else
                {
                    ++pending;
                }
            }

            Neuron& target = m_neurons[
                static_cast<std::size_t>(
                    connection.m_target_neuron_idx)];
            target.m_activesum +=
                connection.m_synaptic_current;
            if (!event_source)
            {
                const double analog =
                    source_neuron.m_activation *
                    connection.m_weight;
                connection.m_signal += analog;
                connection.m_presynaptic_signal =
                    source_neuron.m_activation;
                target.m_activesum += analog;
            }
        }

        for (std::size_t i = m_num_inputs;
             i < m_neurons.size();
             ++i)
        {
            Neuron& neuron = m_neurons[i];
            const double current =
                neuron.m_activesum + neuron.m_bias;
            if (!std::isfinite(current))
            {
                throw std::domain_error(
                    "Spiking neuron input current must remain finite");
            }
            neuron.m_last_input = current;
            neuron.m_spike = false;
            neuron.m_activation = 0.0;
            if (!std::isfinite(neuron.m_rate_time_constant) ||
                neuron.m_rate_time_constant <= 0.0)
            {
                throw std::domain_error(
                    "Spike-rate filters require a positive finite "
                    "time constant");
            }
            neuron.m_rate_trace *=
                std::exp(-dt / neuron.m_rate_time_constant);

            if (!IsSpikingActivation(
                    neuron.m_activation_function_type))
            {
                neuron.m_activation =
                    EvaluateActivation(neuron, current);
                neuron.m_activesum = 0.0;
                continue;
            }

            if (neuron.m_activation_function_type ==
                SPIKING_IZHIKEVICH)
            {
                const double dt_ms = dt * 1000.0;
                double& voltage = neuron.m_membrane_potential;
                double& recovery = neuron.m_izhikevich_recovery;
                if (!std::isfinite(voltage) ||
                    !std::isfinite(recovery) ||
                    !std::isfinite(neuron.m_spike_threshold) ||
                    !std::isfinite(neuron.m_izhikevich_a) ||
                    !std::isfinite(neuron.m_izhikevich_b) ||
                    !std::isfinite(neuron.m_izhikevich_c) ||
                    !std::isfinite(neuron.m_izhikevich_d))
                {
                    throw std::domain_error(
                        "Izhikevich state must be finite");
                }
                const auto derivative =
                    [&](double v)
                {
                    return 0.04 * v * v + 5.0 * v +
                           140.0 - recovery + current;
                };
                voltage +=
                    0.5 * dt_ms * derivative(voltage);
                voltage +=
                    0.5 * dt_ms * derivative(voltage);
                recovery +=
                    dt_ms * neuron.m_izhikevich_a *
                    (neuron.m_izhikevich_b * voltage -
                     recovery);
                if (voltage >= neuron.m_spike_threshold)
                {
                    neuron.m_spike = true;
                    voltage = neuron.m_izhikevich_c;
                    recovery += neuron.m_izhikevich_d;
                }
            }
            else
            {
                if (!std::isfinite(neuron.m_timeconst) ||
                    neuron.m_timeconst <= 0.0 ||
                    !std::isfinite(neuron.m_spike_threshold) ||
                    !std::isfinite(neuron.m_reset_potential) ||
                    !std::isfinite(neuron.m_resting_potential) ||
                    !std::isfinite(neuron.m_refractory_period) ||
                    neuron.m_refractory_period < 0.0 ||
                    !std::isfinite(neuron.m_membrane_resistance))
                {
                    throw std::domain_error(
                        "LIF neurons require finite parameters and a "
                        "positive membrane time constant");
                }
                if (neuron.m_activation_function_type ==
                    SPIKING_ADAPTIVE_LIF)
                {
                    if (!std::isfinite(
                            neuron.m_adaptation_time_constant) ||
                        neuron.m_adaptation_time_constant <= 0.0 ||
                        !std::isfinite(
                            neuron.m_adaptation_increment))
                    {
                        throw std::domain_error(
                            "Adaptive LIF neurons require finite "
                            "adaptation parameters");
                    }
                    neuron.m_adaptation *= std::exp(
                        -dt /
                        neuron.m_adaptation_time_constant);
                }
                else
                {
                    neuron.m_adaptation = 0.0;
                }

                if (neuron.m_refractory_remaining > 0.0)
                {
                    neuron.m_refractory_remaining =
                        std::max(
                            0.0,
                            neuron.m_refractory_remaining - dt);
                    neuron.m_membrane_potential =
                        neuron.m_reset_potential;
                }
                else
                {
                    neuron.m_membrane_potential +=
                        (dt / neuron.m_timeconst) *
                        (neuron.m_resting_potential -
                         neuron.m_membrane_potential +
                         neuron.m_membrane_resistance * current -
                         neuron.m_adaptation);
                    if (neuron.m_membrane_potential >=
                        neuron.m_spike_threshold)
                    {
                        neuron.m_spike = true;
                        neuron.m_membrane_potential =
                            neuron.m_reset_potential;
                        neuron.m_refractory_remaining =
                            neuron.m_refractory_period;
                        if (neuron.m_activation_function_type ==
                            SPIKING_ADAPTIVE_LIF)
                        {
                            neuron.m_adaptation +=
                                neuron.m_adaptation_increment;
                        }
                    }
                }
            }

            if (neuron.m_spike)
            {
                neuron.m_activation = 1.0;
                ++neuron.m_spike_count;
                neuron.m_last_spike_time = step_end;
                neuron.m_rate_trace +=
                    1.0 / neuron.m_rate_time_constant;
                record(SpikeEvent{
                    step_end,
                    static_cast<int>(i),
                    1.0,
                    false});
            }
            neuron.m_activesum = 0.0;
        }

        for (auto& connection : m_connections)
        {
            if (!connection.m_stdp_enabled)
                continue;
            if (!std::isfinite(connection.m_stdp_tau_plus) ||
                connection.m_stdp_tau_plus <= 0.0 ||
                !std::isfinite(connection.m_stdp_tau_minus) ||
                connection.m_stdp_tau_minus <= 0.0 ||
                !std::isfinite(connection.m_stdp_plus) ||
                connection.m_stdp_plus < 0.0 ||
                !std::isfinite(connection.m_stdp_minus) ||
                connection.m_stdp_minus < 0.0 ||
                !std::isfinite(connection.m_stdp_min_weight) ||
                !std::isfinite(connection.m_stdp_max_weight) ||
                connection.m_stdp_min_weight >
                    connection.m_stdp_max_weight)
            {
                throw std::domain_error(
                    "STDP requires positive trace time constants and "
                    "ordered finite weight bounds");
            }
            connection.m_stdp_pre_trace *=
                std::exp(-dt / connection.m_stdp_tau_plus);
            connection.m_stdp_post_trace *=
                std::exp(-dt / connection.m_stdp_tau_minus);
            const bool pre =
                m_neurons[static_cast<std::size_t>(
                    connection.m_source_neuron_idx)].m_spike;
            const bool post =
                m_neurons[static_cast<std::size_t>(
                    connection.m_target_neuron_idx)].m_spike;
            if (pre)
            {
                connection.m_weight -=
                    connection.m_stdp_minus *
                    connection.m_stdp_post_trace;
                connection.m_stdp_pre_trace += 1.0;
            }
            if (post)
            {
                connection.m_weight +=
                    connection.m_stdp_plus *
                    connection.m_stdp_pre_trace;
                connection.m_stdp_post_trace += 1.0;
            }
            Clamp(
                connection.m_weight,
                connection.m_stdp_min_weight,
                connection.m_stdp_max_weight);
        }

        m_spiking_time = step_end;
        return OutputDecoded();
    }

    std::vector<std::vector<double>> NeuralNetwork::SimulateSpiking(
        const std::vector<std::vector<double>>& inputs,
        double time_step,
        bool reset)
    {
        if (reset)
            Flush();
        std::vector<std::vector<double>> outputs;
        outputs.reserve(inputs.size());
        for (const auto& sample : inputs)
            outputs.push_back(StepSpiking(sample, time_step));
        return outputs;
    }

    std::size_t NeuralNetwork::SparseRTRLStateSize() const
    {
        std::size_t size = 0;
        for (const auto& row : m_sparse_rtrl_sensitivities)
            size += row.size();
        return size;
    }

    std::vector<double> NeuralNetwork::Output()
    {
        ValidateNetworkTopology(*this);
        std::vector<double> t_output;
        t_output.reserve(m_num_outputs);
        for (unsigned int i = 0; i < m_num_outputs; i++)
            t_output.push_back(m_neurons[i + m_num_inputs].m_activation);
        return t_output;
    }

    double NeuralNetwork::GetConnectionLenght(Neuron source, Neuron target)
    {
        const double dx = target.m_x - source.m_x;
        const double dy = target.m_y - source.m_y;
        const double dz = target.m_z - source.m_z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    double NeuralNetwork::GetTotalConnectionLength()
    {
        ValidateNetworkTopology(*this);
        double total = 0.0;
        for (const auto &connection : m_connections)
        {
            total += GetConnectionLenght(
                m_neurons[static_cast<std::size_t>(
                    connection.m_source_neuron_idx)],
                m_neurons[static_cast<std::size_t>(
                    connection.m_target_neuron_idx)]);
        }
        return total;
    }

    void NeuralNetwork::Adapt(Parameters &a_Parameters)
    {
        ValidateNetworkTopology(*this);
        if (!std::isfinite(a_Parameters.MinWeight) ||
            !std::isfinite(a_Parameters.MaxWeight) ||
            a_Parameters.MinWeight > a_Parameters.MaxWeight)
        {
            throw std::invalid_argument(
                "Adapt requires a valid, finite weight range");
        }
        double maximum_weight = 0.0;
        for (const auto &connection : m_connections)
        {
            maximum_weight =
                std::max(maximum_weight, std::abs(connection.m_weight));
        }

        for (auto &connection : m_connections)
        {
            const double input = m_neurons[static_cast<std::size_t>(
                connection.m_source_neuron_idx)].m_activation;
            const double output = m_neurons[static_cast<std::size_t>(
                connection.m_target_neuron_idx)].m_activation;
            if (connection.m_weight > 0.0)
            {
                const double delta =
                    connection.m_hebb_rate *
                        (maximum_weight - connection.m_weight) *
                        input * output +
                    connection.m_hebb_pre_rate * maximum_weight *
                        input * (output - 1.0);
                connection.m_weight += delta;
            }
            else if (connection.m_weight < 0.0)
            {
                double magnitude = -connection.m_weight;
                const double delta =
                    connection.m_hebb_pre_rate *
                        (maximum_weight - magnitude) *
                        input * (1.0 - output) -
                    connection.m_hebb_rate * maximum_weight * input * output;
                magnitude = std::max(0.0, magnitude + delta);
                connection.m_weight = -magnitude;
            }
            Clamp(connection.m_weight,
                  a_Parameters.MinWeight,
                  a_Parameters.MaxWeight);
        }
    }

    int NeuralNetwork::ConnectionExists(int a_to, int a_from)
    {
        for (std::size_t i = 0; i < m_connections.size(); ++i)
        {
            if (m_connections[i].m_source_neuron_idx == a_from &&
                m_connections[i].m_target_neuron_idx == a_to)
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    void NeuralNetwork::RTRL_update_gradients()
    {
        ValidateNetworkTopology(*this);
        const std::size_t neuron_count = m_neurons.size();
        bool initialized = true;
        for (const auto &neuron : m_neurons)
        {
            if (neuron.m_sensitivity_matrix.size() != neuron_count)
            {
                initialized = false;
                break;
            }
            for (const auto &row : neuron.m_sensitivity_matrix)
            {
                if (row.size() != neuron_count)
                {
                    initialized = false;
                    break;
                }
            }
        }
        if (!initialized)
        {
            InitRTRLMatrix();
        }

        std::vector<std::vector<std::vector<double>>>
            previous_sensitivities;
        previous_sensitivities.reserve(neuron_count);
        for (const auto& neuron : m_neurons)
            previous_sensitivities.push_back(neuron.m_sensitivity_matrix);

        // Index the sparse topology once. The historical implementation
        // repeatedly linearly searched every connection from inside four
        // nested loops, making a single RTRL step prohibitively expensive
        // even for modest recurrent networks.
        std::vector<std::vector<int>> connection_indices(
            neuron_count, std::vector<int>(neuron_count, -1));
        std::vector<std::vector<std::pair<std::size_t, double>>>
            incoming(neuron_count);
        for (std::size_t index = 0; index < m_connections.size(); ++index)
        {
            const Connection& connection = m_connections[index];
            const std::size_t source = static_cast<std::size_t>(
                connection.m_source_neuron_idx);
            const std::size_t target = static_cast<std::size_t>(
                connection.m_target_neuron_idx);
            if (connection_indices[target][source] >= 0)
            {
                throw std::invalid_argument(
                    "RTRL requires unique connection endpoints");
            }
            connection_indices[target][source] =
                static_cast<int>(index);
            incoming[target].emplace_back(
                source, connection.m_weight);
        }

        for (std::size_t k = m_num_inputs; k < neuron_count; ++k)
        {
            const double derivative = activation_derivative(m_neurons[k]);
            for (auto& row : m_neurons[k].m_sensitivity_matrix)
                std::fill(row.begin(), row.end(), 0.0);

            // Sensitivities exist only for weights that actually occur in the
            // network, so iterate those sparse parameters directly.
            for (const Connection& parameter : m_connections)
            {
                const std::size_t i = static_cast<std::size_t>(
                    parameter.m_target_neuron_idx);
                const std::size_t j = static_cast<std::size_t>(
                    parameter.m_source_neuron_idx);
                if (i < m_num_inputs)
                    continue;

                double sum = 0.0;
                for (const auto& recurrent : incoming[k])
                {
                    sum += recurrent.second *
                           previous_sensitivities[recurrent.first][i][j];
                }
                if (i == k)
                    sum += parameter.m_source_activation;
                m_neurons[k].m_sensitivity_matrix[i][j] =
                    derivative * sum;
            }
        }
    }

    void NeuralNetwork::RTRL_update_gradients_sparse()
    {
        ValidateNetworkTopology(*this);
        const std::size_t neuron_count = m_neurons.size();
        const std::size_t connection_count = m_connections.size();
        bool initialized =
            m_sparse_rtrl_sensitivities.size() == neuron_count;
        if (initialized)
        {
            initialized = std::all_of(
                m_sparse_rtrl_sensitivities.begin(),
                m_sparse_rtrl_sensitivities.end(),
                [connection_count](const std::vector<double>& row)
                {
                    return row.size() == connection_count;
                });
        }
        if (!initialized)
            InitSparseRTRLMatrix();

        const auto previous = m_sparse_rtrl_sensitivities;
        std::vector<std::vector<std::pair<std::size_t, double>>>
            incoming(neuron_count);
        for (const Connection& connection : m_connections)
        {
            incoming[static_cast<std::size_t>(
                         connection.m_target_neuron_idx)]
                .emplace_back(
                    static_cast<std::size_t>(
                        connection.m_source_neuron_idx),
                    connection.m_weight);
        }

        for (std::size_t neuron = 0; neuron < neuron_count; ++neuron)
        {
            auto& sensitivities =
                m_sparse_rtrl_sensitivities[neuron];
            std::fill(
                sensitivities.begin(), sensitivities.end(), 0.0);
            if (neuron < m_num_inputs)
                continue;

            const double derivative =
                activation_derivative(m_neurons[neuron]);
            for (std::size_t parameter = 0;
                 parameter < connection_count;
                 ++parameter)
            {
                double sensitivity =
                    static_cast<std::size_t>(
                        m_connections[parameter]
                            .m_target_neuron_idx) == neuron
                        ? m_connections[parameter]
                              .m_source_activation
                        : 0.0;
                for (const auto& recurrent : incoming[neuron])
                {
                    sensitivity +=
                        recurrent.second *
                        previous[recurrent.first][parameter];
                }
                sensitivities[parameter] =
                    derivative * sensitivity;
            }
        }
    }

    void NeuralNetwork::RTRL_update_error(double a_target)
    {
        std::vector<double> targets = Output();
        if (targets.empty())
        {
            throw std::runtime_error(
                "RTRL error update requires at least one output neuron");
        }
        targets.front() = a_target;
        RTRL_update_error(targets, LEARNING_RATE);
    }

    void NeuralNetwork::RTRL_update_error(
        const std::vector<double>& targets,
        double learning_rate)
    {
        ValidateNetworkTopology(*this);
        if (m_num_outputs == 0)
        {
            throw std::runtime_error(
                "RTRL error update requires at least one output neuron");
        }
        if (targets.size() != m_num_outputs)
        {
            throw std::invalid_argument(
                "RTRL target count must match the network output count");
        }
        if (!std::isfinite(learning_rate) || learning_rate < 0.0)
        {
            throw std::invalid_argument(
                "RTRL learning rate must be finite and non-negative");
        }
        if (m_total_weight_change.size() != m_connections.size())
        {
            m_total_weight_change.assign(m_connections.size(), 0.0);
        }
        for (unsigned int output = 0; output < m_num_outputs; ++output)
        {
            const auto& matrix =
                m_neurons[m_num_inputs + output].m_sensitivity_matrix;
            if (matrix.size() != m_neurons.size())
            {
                throw std::runtime_error(
                    "RTRL gradients must be initialized before updating error");
            }
            for (const auto& row : matrix)
            {
                if (row.size() != m_neurons.size())
                {
                    throw std::runtime_error(
                        "RTRL sensitivity matrix has invalid dimensions");
                }
            }
        }

        const std::vector<double> outputs = Output();
        std::vector<double> errors(m_num_outputs, 0.0);
        m_total_error = 0.0;
        for (unsigned int output = 0; output < m_num_outputs; ++output)
        {
            errors[output] = targets[output] - outputs[output];
            m_total_error += errors[output];
        }

        for (std::size_t connection_index = 0;
             connection_index < m_connections.size();
             ++connection_index)
        {
            const Connection& connection =
                m_connections[connection_index];
            const std::size_t target =
                static_cast<std::size_t>(
                    connection.m_target_neuron_idx);
            const std::size_t source =
                static_cast<std::size_t>(
                    connection.m_source_neuron_idx);
            double gradient = 0.0;
            for (unsigned int output = 0;
                 output < m_num_outputs;
                 ++output)
            {
                gradient +=
                    errors[output] *
                    m_neurons[m_num_inputs + output]
                        .m_sensitivity_matrix[target][source];
            }
            m_total_weight_change[connection_index] +=
                gradient * learning_rate;
        }
    }

    void NeuralNetwork::RTRL_update_error_sparse(
        double target,
        double learning_rate)
    {
        if (m_num_outputs != 1)
        {
            throw std::invalid_argument(
                "Scalar sparse RTRL targets require exactly one output");
        }
        RTRL_update_error_sparse(
            std::vector<double>{target}, learning_rate);
    }

    void NeuralNetwork::RTRL_update_error_sparse(
        const std::vector<double>& targets,
        double learning_rate)
    {
        ValidateNetworkTopology(*this);
        if (targets.size() != m_num_outputs)
        {
            throw std::invalid_argument(
                "RTRL target count must match the network output count");
        }
        if (!std::isfinite(learning_rate) || learning_rate < 0.0)
        {
            throw std::invalid_argument(
                "RTRL learning rate must be finite and non-negative");
        }
        if (m_sparse_rtrl_sensitivities.size() != m_neurons.size() ||
            std::any_of(
                m_sparse_rtrl_sensitivities.begin(),
                m_sparse_rtrl_sensitivities.end(),
                [this](const std::vector<double>& row)
                {
                    return row.size() != m_connections.size();
                }))
        {
            throw std::runtime_error(
                "Sparse RTRL gradients must be initialized before "
                "updating error");
        }
        if (m_total_weight_change.size() != m_connections.size())
            m_total_weight_change.assign(m_connections.size(), 0.0);

        const std::vector<double> outputs = Output();
        m_total_error = 0.0;
        for (std::size_t parameter = 0;
             parameter < m_connections.size();
             ++parameter)
        {
            double gradient = 0.0;
            for (unsigned int output = 0;
                 output < m_num_outputs;
                 ++output)
            {
                const double error =
                    targets[output] - outputs[output];
                gradient +=
                    error *
                    m_sparse_rtrl_sensitivities[
                        m_num_inputs + output][parameter];
                if (parameter == 0)
                    m_total_error += error;
            }
            m_total_weight_change[parameter] +=
                learning_rate * gradient;
        }
        if (m_connections.empty())
        {
            for (unsigned int output = 0;
                 output < m_num_outputs;
                 ++output)
            {
                m_total_error += targets[output] - outputs[output];
            }
        }
    }

    void NeuralNetwork::RTRL_update_weights()
    {
        if (m_total_weight_change.size() != m_connections.size())
        {
            throw std::runtime_error(
                "RTRL weight state does not match the network connections");
        }
        for (std::size_t i = 0; i < m_connections.size(); ++i)
        {
            m_connections[i].m_weight += m_total_weight_change[i];
            m_total_weight_change[i] = 0.0;
        }
        m_total_error = 0.0;
    }

    void NeuralNetwork::Save(const char* a_filename)
    {
        if (a_filename == nullptr)
            throw std::invalid_argument(
                "NeuralNetwork::Save: filename is null.");
        FILE *fil = detail::OpenFile(a_filename, "w");
        if (fil == nullptr)
        {
            throw std::runtime_error("Cannot open neural network file for writing");
        }
        Save(fil);
        if (fclose(fil) != 0)
            throw std::runtime_error(
                "NeuralNetwork::Save: failed to close output file.");
    }

    void NeuralNetwork::Save(FILE *a_file)
    {
        if (a_file == nullptr)
            throw std::invalid_argument(
                "NeuralNetwork::Save: file is null.");
        ValidateNetworkTopology(*this);
        fprintf(a_file, "NNstart\n");
        fprintf(a_file, "%u %u\n", m_num_inputs, m_num_outputs);
        fprintf(
            a_file,
            "spiking_state %3.18f %3.18f %d %d %d %zu %llu\n",
            m_spiking_time,
            m_spiking_time_step,
            static_cast<int>(m_spiking_input_mode),
            static_cast<int>(m_spiking_output_mode),
            static_cast<int>(m_record_spikes),
            m_max_recorded_spikes,
            static_cast<unsigned long long>(m_spiking_rng_state));
        for (const auto &neuron : m_neurons)
        {
            fprintf(a_file, "neuron %d %3.18f %3.18f %3.18f %3.18f %d %3.18f\n",
                    static_cast<int>(neuron.m_type), neuron.m_a,
                    neuron.m_b, neuron.m_timeconst, neuron.m_bias,
                    static_cast<int>(neuron.m_activation_function_type),
                    neuron.m_split_y);
            fprintf(
                a_file,
                "spiking_neuron %3.18f %3.18f %3.18f %3.18f "
                "%3.18f %3.18f %3.18f %3.18f %3.18f %3.18f "
                "%3.18f %3.18f %3.18f %3.18f %3.18f %d %llu "
                "%3.18f %3.18f %3.18f\n",
                neuron.m_spike_threshold,
                neuron.m_reset_potential,
                neuron.m_resting_potential,
                neuron.m_refractory_period,
                neuron.m_refractory_remaining,
                neuron.m_membrane_resistance,
                neuron.m_adaptation_time_constant,
                neuron.m_adaptation_increment,
                neuron.m_adaptation,
                neuron.m_izhikevich_a,
                neuron.m_izhikevich_b,
                neuron.m_izhikevich_c,
                neuron.m_izhikevich_d,
                neuron.m_izhikevich_recovery,
                neuron.m_rate_trace,
                static_cast<int>(neuron.m_spike),
                static_cast<unsigned long long>(
                    neuron.m_spike_count),
                neuron.m_last_spike_time,
                neuron.m_rate_time_constant,
                neuron.m_membrane_potential);
        }
        for (const auto &conn : m_connections)
        {
            fprintf(a_file, "connection %d %d %3.18f %d %3.18f %3.18f\n",
                    conn.m_source_neuron_idx,
                    conn.m_target_neuron_idx, conn.m_weight,
                    static_cast<int>(conn.m_recur_flag),
                    conn.m_hebb_rate, conn.m_hebb_pre_rate);
            fprintf(
                a_file,
                "spiking_connection %3.18f %3.18f %3.18f %d "
                "%3.18f %3.18f %3.18f %3.18f %3.18f %3.18f "
                "%3.18f %3.18f\n",
                conn.m_synaptic_delay,
                conn.m_synaptic_time_constant,
                conn.m_synaptic_current,
                static_cast<int>(conn.m_stdp_enabled),
                conn.m_stdp_plus,
                conn.m_stdp_minus,
                conn.m_stdp_tau_plus,
                conn.m_stdp_tau_minus,
                conn.m_stdp_pre_trace,
                conn.m_stdp_post_trace,
                conn.m_stdp_min_weight,
                conn.m_stdp_max_weight);
        }
        fprintf(a_file, "NNend\n\n");
    }

    bool NeuralNetwork::Load(std::ifstream &a_DataFile)
    {
        std::string t_str;
        do { a_DataFile >> t_str; } while (t_str != "NNstart" && !a_DataFile.eof());
        if (a_DataFile.eof()) return false;
        Clear();
        a_DataFile >> m_num_inputs >> m_num_outputs;
        int last_neuron = -1;
        int last_connection = -1;
        while (a_DataFile >> t_str && t_str != "NNend")
        {
            if (t_str == "spiking_state")
            {
                int input_mode = 0;
                int output_mode = 0;
                int record = 0;
                unsigned long long rng_state = 0;
                a_DataFile >> m_spiking_time
                           >> m_spiking_time_step
                           >> input_mode
                           >> output_mode
                           >> record
                           >> m_max_recorded_spikes
                           >> rng_state;
                m_spiking_input_mode =
                    static_cast<SpikingInputMode>(input_mode);
                m_spiking_output_mode =
                    static_cast<SpikingOutputMode>(output_mode);
                m_record_spikes = record != 0;
                m_spiking_rng_state =
                    static_cast<std::uint64_t>(rng_state);
            }
            else if (t_str == "neuron")
            {
                Neuron t_n;
                int t_type, t_aftype;
                a_DataFile >> t_type >> t_n.m_a >> t_n.m_b
                           >> t_n.m_timeconst >> t_n.m_bias;
                a_DataFile >> t_aftype >> t_n.m_split_y;
                t_n.m_type = static_cast<NeuronType>(t_type);
                t_n.m_activation_function_type = static_cast<ActivationFunction>(t_aftype);
                m_neurons.push_back(t_n);
                last_neuron =
                    static_cast<int>(m_neurons.size()) - 1;
            }
            else if (t_str == "spiking_neuron")
            {
                if (last_neuron < 0)
                {
                    Clear();
                    return false;
                }
                Neuron& neuron =
                    m_neurons[static_cast<std::size_t>(
                        last_neuron)];
                int spike = 0;
                unsigned long long spike_count = 0;
                a_DataFile >> neuron.m_spike_threshold
                           >> neuron.m_reset_potential
                           >> neuron.m_resting_potential
                           >> neuron.m_refractory_period
                           >> neuron.m_refractory_remaining
                           >> neuron.m_membrane_resistance
                           >> neuron.m_adaptation_time_constant
                           >> neuron.m_adaptation_increment
                           >> neuron.m_adaptation
                           >> neuron.m_izhikevich_a
                           >> neuron.m_izhikevich_b
                           >> neuron.m_izhikevich_c
                           >> neuron.m_izhikevich_d
                           >> neuron.m_izhikevich_recovery
                           >> neuron.m_rate_trace
                           >> spike
                           >> spike_count
                           >> neuron.m_last_spike_time
                           >> neuron.m_rate_time_constant
                           >> neuron.m_membrane_potential;
                neuron.m_spike = spike != 0;
                neuron.m_spike_count =
                    static_cast<std::uint64_t>(spike_count);
            }
            else if (t_str == "connection")
            {
                Connection t_c;
                int t_isrecur;
                a_DataFile >> t_c.m_source_neuron_idx >> t_c.m_target_neuron_idx >> t_c.m_weight >> t_isrecur >> t_c.m_hebb_rate >> t_c.m_hebb_pre_rate;
                t_c.m_recur_flag = static_cast<bool>(t_isrecur);
                m_connections.push_back(t_c);
                last_connection =
                    static_cast<int>(m_connections.size()) - 1;
            }
            else if (t_str == "spiking_connection")
            {
                if (last_connection < 0)
                {
                    Clear();
                    return false;
                }
                Connection& connection =
                    m_connections[static_cast<std::size_t>(
                        last_connection)];
                int stdp = 0;
                a_DataFile >> connection.m_synaptic_delay
                           >> connection.m_synaptic_time_constant
                           >> connection.m_synaptic_current
                           >> stdp
                           >> connection.m_stdp_plus
                           >> connection.m_stdp_minus
                           >> connection.m_stdp_tau_plus
                           >> connection.m_stdp_tau_minus
                           >> connection.m_stdp_pre_trace
                           >> connection.m_stdp_post_trace
                           >> connection.m_stdp_min_weight
                           >> connection.m_stdp_max_weight;
                connection.m_stdp_enabled = stdp != 0;
            }
        }
        if (!a_DataFile || t_str != "NNend")
        {
            Clear();
            return false;
        }
        try
        {
            ValidateNetworkTopology(*this);
        }
        catch (const std::exception&)
        {
            Clear();
            return false;
        }
        return true;
    }

    bool NeuralNetwork::Load(const char *a_filename)
    {
        if (a_filename == nullptr)
            return false;
        std::ifstream t_DataFile(a_filename);
        if (!t_DataFile)
            return false;
        return Load(t_DataFile);
    }

    std::string NeuralNetwork::Serialize() const
    {
        ValidateNetworkTopology(*this);
        std::ostringstream output;
        output << std::setprecision(
            std::numeric_limits<double>::max_digits10);
        output << "NeuralNetworkFormat 6\n";
        output << "State " << m_num_inputs << ' ' << m_num_outputs << ' '
               << m_total_error << ' ' << m_spiking_time << ' '
               << m_spiking_time_step << ' '
               << static_cast<int>(m_spiking_input_mode) << ' '
               << static_cast<int>(m_spiking_output_mode) << ' '
               << static_cast<int>(m_record_spikes) << ' '
               << m_max_recorded_spikes << ' '
               << m_spiking_rng_state << '\n';
        output << "TotalWeightChange " << m_total_weight_change.size();
        for (double value : m_total_weight_change)
            output << ' ' << value;
        output << '\n';
        output << "Neurons " << m_neurons.size() << '\n';
        for (const auto &neuron : m_neurons)
        {
            output << "Neuron " << neuron.m_activesum << ' '
                   << neuron.m_activation << ' ' << neuron.m_a << ' '
                   << neuron.m_b << ' ' << neuron.m_timeconst << ' '
                   << neuron.m_bias << ' ' << neuron.m_membrane_potential
                   << ' ' << neuron.m_last_input << ' '
                   << static_cast<int>(neuron.m_activation_function_type)
                   << ' ' << neuron.m_x << ' ' << neuron.m_y << ' '
                   << neuron.m_z << ' ' << neuron.m_sx << ' ' << neuron.m_sy
                   << ' ' << neuron.m_sz << ' ' << neuron.m_split_y << ' '
                   << static_cast<int>(neuron.m_type) << ' '
                   << neuron.m_spike_threshold << ' '
                   << neuron.m_reset_potential << ' '
                   << neuron.m_resting_potential << ' '
                   << neuron.m_refractory_period << ' '
                   << neuron.m_refractory_remaining << ' '
                   << neuron.m_membrane_resistance << ' '
                   << neuron.m_adaptation_time_constant << ' '
                   << neuron.m_adaptation_increment << ' '
                   << neuron.m_adaptation << ' '
                   << neuron.m_izhikevich_a << ' '
                   << neuron.m_izhikevich_b << ' '
                   << neuron.m_izhikevich_c << ' '
                   << neuron.m_izhikevich_d << ' '
                   << neuron.m_izhikevich_recovery << ' '
                   << static_cast<int>(neuron.m_spike) << ' '
                   << neuron.m_spike_count << ' '
                   << neuron.m_last_spike_time << ' '
                   << neuron.m_rate_trace << ' '
                   << neuron.m_rate_time_constant << '\n';
            output << "SubstrateCoordinates "
                   << neuron.m_substrate_coords.size();
            for (double coordinate : neuron.m_substrate_coords)
                output << ' ' << coordinate;
            output << '\n';
            output << "Sensitivity " << neuron.m_sensitivity_matrix.size()
                   << '\n';
            for (const auto& row : neuron.m_sensitivity_matrix)
            {
                output << "SensitivityRow " << row.size();
                for (double value : row)
                    output << ' ' << value;
                output << '\n';
            }
        }
        output << "Connections " << m_connections.size() << '\n';
        for (const auto &connection : m_connections)
        {
            output << "Connection " << connection.m_source_neuron_idx << ' '
                   << connection.m_target_neuron_idx << ' '
                   << connection.m_weight << ' ' << connection.m_signal << ' '
                   << connection.m_source_activation << ' '
                   << static_cast<int>(connection.m_recur_flag) << ' '
                   << connection.m_hebb_rate << ' '
                   << connection.m_hebb_pre_rate << ' '
                   << connection.m_synaptic_delay << ' '
                   << connection.m_synaptic_time_constant << ' '
                   << connection.m_synaptic_current << ' '
                   << connection.m_presynaptic_signal << ' '
                   << static_cast<int>(connection.m_stdp_enabled) << ' '
                   << connection.m_stdp_plus << ' '
                   << connection.m_stdp_minus << ' '
                   << connection.m_stdp_tau_plus << ' '
                   << connection.m_stdp_tau_minus << ' '
                   << connection.m_stdp_pre_trace << ' '
                   << connection.m_stdp_post_trace << ' '
                   << connection.m_stdp_min_weight << ' '
                   << connection.m_stdp_max_weight << '\n';
            output << "PendingEvents "
                   << connection.m_pending_events.size();
            for (const auto& event : connection.m_pending_events)
            {
                output << ' ' << event.delivery_time << ' '
                       << event.amplitude << ' '
                       << event.source_amplitude;
            }
            output << '\n';
        }
        output << "SparseRTRL "
               << m_sparse_rtrl_sensitivities.size() << '\n';
        for (const auto& row : m_sparse_rtrl_sensitivities)
        {
            output << "SparseRTRLRow " << row.size();
            for (double value : row)
                output << ' ' << value;
            output << '\n';
        }
        output << "SpikeHistory " << m_spike_history.size() << '\n';
        for (const auto& event : m_spike_history)
        {
            output << "SpikeEvent " << event.time << ' '
                   << event.neuron_index << ' ' << event.amplitude << ' '
                   << static_cast<int>(event.input) << '\n';
        }
        output << "NeuralNetworkEnd\n";
        return output.str();
    }

    NeuralNetwork NeuralNetwork::Deserialize(const std::string &data)
    {
        NeuralNetwork network;
        std::istringstream input(data);
        std::string token;
        input >> token;
        if (token != "NeuralNetworkFormat")
        {
            // Legacy pickle format.
            try
            {
                network.m_num_inputs =
                    static_cast<unsigned int>(std::stoul(token));
            }
            catch (const std::exception&)
            {
                throw std::runtime_error(
                    "NeuralNetwork::Deserialize: malformed header.");
            }
            input >> network.m_num_outputs;
            std::size_t neuron_count = 0;
            input >> neuron_count;
            for (std::size_t i = 0; i < neuron_count; ++i)
            {
                Neuron neuron;
                int type, activation;
                input >> type >> neuron.m_a >> neuron.m_b
                      >> neuron.m_timeconst >> neuron.m_bias >> activation
                      >> neuron.m_split_y;
                neuron.m_type = static_cast<NeuronType>(type);
                neuron.m_activation_function_type =
                    static_cast<ActivationFunction>(activation);
                network.m_neurons.push_back(neuron);
            }
            std::size_t connection_count = 0;
            input >> connection_count;
            for (std::size_t i = 0; i < connection_count; ++i)
            {
                Connection connection;
                int recurrent = 0;
                input >> connection.m_source_neuron_idx
                      >> connection.m_target_neuron_idx >> connection.m_weight
                      >> recurrent >> connection.m_hebb_rate
                      >> connection.m_hebb_pre_rate;
                connection.m_recur_flag = recurrent != 0;
                network.m_connections.push_back(connection);
            }
            if (!input)
                throw std::runtime_error(
                    "NeuralNetwork::Deserialize: malformed legacy data.");
            ValidateNetworkTopology(network);
            return network;
        }

        int version = 0;
        input >> version;
        if (version < 2 || version > 6)
            throw std::runtime_error(
                "NeuralNetwork::Deserialize: unsupported format.");
        input >> token;
        if (token != "State")
            throw std::runtime_error(
                "NeuralNetwork::Deserialize: missing State marker.");
        input >> network.m_num_inputs >> network.m_num_outputs
              >> network.m_total_error;
        if (version >= 5)
        {
            int input_mode = 0;
            int output_mode = 0;
            int record_spikes = 0;
            input >> network.m_spiking_time
                  >> network.m_spiking_time_step
                  >> input_mode
                  >> output_mode
                  >> record_spikes
                  >> network.m_max_recorded_spikes
                  >> network.m_spiking_rng_state;
            network.m_spiking_input_mode =
                static_cast<SpikingInputMode>(input_mode);
            network.m_spiking_output_mode =
                static_cast<SpikingOutputMode>(output_mode);
            network.m_record_spikes = record_spikes != 0;
            if (network.m_spiking_input_mode < CURRENT_INPUT ||
                network.m_spiking_input_mode > POISSON_RATE_INPUT ||
                network.m_spiking_output_mode < SPIKE_OUTPUT ||
                network.m_spiking_output_mode >
                    MEMBRANE_POTENTIAL_OUTPUT)
            {
                throw std::runtime_error(
                    "NeuralNetwork::Deserialize: invalid spiking mode.");
            }
        }

        std::size_t count = 0;
        input >> token >> count;
        if (token != "TotalWeightChange")
            throw std::runtime_error(
                "NeuralNetwork::Deserialize: missing weight-change state.");
        network.m_total_weight_change.resize(count);
        for (double& value : network.m_total_weight_change)
            input >> value;

        input >> token >> count;
        if (token != "Neurons")
            throw std::runtime_error(
                "NeuralNetwork::Deserialize: missing Neurons marker.");
        network.m_neurons.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            input >> token;
            if (token != "Neuron")
                throw std::runtime_error(
                    "NeuralNetwork::Deserialize: missing Neuron marker.");
            Neuron neuron;
            int activation = 0;
            int type = 0;
            input >> neuron.m_activesum >> neuron.m_activation >> neuron.m_a
                  >> neuron.m_b >> neuron.m_timeconst >> neuron.m_bias
                  >> neuron.m_membrane_potential;
            if (version >= 4)
                input >> neuron.m_last_input;
            input >> activation >> neuron.m_x >> neuron.m_y >> neuron.m_z
                  >> neuron.m_sx >> neuron.m_sy >> neuron.m_sz
                  >> neuron.m_split_y >> type;
            neuron.m_activation_function_type =
                static_cast<ActivationFunction>(activation);
            neuron.m_type = static_cast<NeuronType>(type);
            if (version >= 5)
            {
                int spike = 0;
                input >> neuron.m_spike_threshold
                      >> neuron.m_reset_potential
                      >> neuron.m_resting_potential
                      >> neuron.m_refractory_period
                      >> neuron.m_refractory_remaining
                      >> neuron.m_membrane_resistance
                      >> neuron.m_adaptation_time_constant
                      >> neuron.m_adaptation_increment
                      >> neuron.m_adaptation
                      >> neuron.m_izhikevich_a
                      >> neuron.m_izhikevich_b
                      >> neuron.m_izhikevich_c
                      >> neuron.m_izhikevich_d
                      >> neuron.m_izhikevich_recovery
                      >> spike
                      >> neuron.m_spike_count
                      >> neuron.m_last_spike_time
                      >> neuron.m_rate_trace
                      >> neuron.m_rate_time_constant;
                neuron.m_spike = spike != 0;
            }

            std::size_t coordinates = 0;
            input >> token >> coordinates;
            if (token != "SubstrateCoordinates")
                throw std::runtime_error(
                    "NeuralNetwork::Deserialize: missing coordinates.");
            neuron.m_substrate_coords.resize(coordinates);
            for (double& coordinate : neuron.m_substrate_coords)
                input >> coordinate;

            std::size_t rows = 0;
            input >> token >> rows;
            if (token != "Sensitivity")
                throw std::runtime_error(
                    "NeuralNetwork::Deserialize: missing sensitivity state.");
            neuron.m_sensitivity_matrix.resize(rows);
            for (auto& row : neuron.m_sensitivity_matrix)
            {
                std::size_t columns = 0;
                input >> token >> columns;
                if (token != "SensitivityRow")
                    throw std::runtime_error(
                        "NeuralNetwork::Deserialize: missing sensitivity row.");
                row.resize(columns);
                for (double& value : row)
                    input >> value;
            }
            network.m_neurons.push_back(std::move(neuron));
        }

        input >> token >> count;
        if (token != "Connections")
            throw std::runtime_error(
                "NeuralNetwork::Deserialize: missing Connections marker.");
        network.m_connections.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            input >> token;
            if (token != "Connection")
                throw std::runtime_error(
                    "NeuralNetwork::Deserialize: missing Connection marker.");
            Connection connection;
            int recurrent = 0;
            input >> connection.m_source_neuron_idx
                  >> connection.m_target_neuron_idx >> connection.m_weight
                  >> connection.m_signal;
            if (version >= 3)
                input >> connection.m_source_activation;
            input >> recurrent >> connection.m_hebb_rate
                  >> connection.m_hebb_pre_rate;
            connection.m_recur_flag = recurrent != 0;
            if (version >= 5)
            {
                int stdp = 0;
                input >> connection.m_synaptic_delay
                      >> connection.m_synaptic_time_constant
                      >> connection.m_synaptic_current;
                if (version >= 6)
                    input >> connection.m_presynaptic_signal;
                input >> stdp
                      >> connection.m_stdp_plus
                      >> connection.m_stdp_minus
                      >> connection.m_stdp_tau_plus
                      >> connection.m_stdp_tau_minus
                      >> connection.m_stdp_pre_trace
                      >> connection.m_stdp_post_trace
                      >> connection.m_stdp_min_weight
                      >> connection.m_stdp_max_weight;
                connection.m_stdp_enabled = stdp != 0;
                std::size_t pending_count = 0;
                input >> token >> pending_count;
                if (token != "PendingEvents")
                {
                    throw std::runtime_error(
                        "NeuralNetwork::Deserialize: missing pending "
                        "synaptic events.");
                }
                connection.m_pending_events.resize(pending_count);
                for (auto& event : connection.m_pending_events)
                {
                    input >> event.delivery_time >> event.amplitude;
                    if (version >= 6)
                    {
                        input >> event.source_amplitude;
                    }
                    else if (std::abs(connection.m_weight) >
                             std::numeric_limits<double>::epsilon())
                    {
                        event.source_amplitude =
                            event.amplitude /
                            connection.m_weight;
                    }
                }
            }
            network.m_connections.push_back(connection);
        }
        if (version >= 4)
        {
            std::size_t rows = 0;
            input >> token >> rows;
            if (token != "SparseRTRL")
                throw std::runtime_error(
                    "NeuralNetwork::Deserialize: missing sparse RTRL state.");
            network.m_sparse_rtrl_sensitivities.resize(rows);
            for (auto& row : network.m_sparse_rtrl_sensitivities)
            {
                std::size_t columns = 0;
                input >> token >> columns;
                if (token != "SparseRTRLRow")
                    throw std::runtime_error(
                        "NeuralNetwork::Deserialize: missing sparse RTRL row.");
                row.resize(columns);
                for (double& value : row)
                    input >> value;
            }
            if (!network.m_sparse_rtrl_sensitivities.empty() &&
                (network.m_sparse_rtrl_sensitivities.size() !=
                     network.m_neurons.size() ||
                 std::any_of(
                     network.m_sparse_rtrl_sensitivities.begin(),
                     network.m_sparse_rtrl_sensitivities.end(),
                     [&network](const std::vector<double>& row)
                     {
                         return row.size() !=
                                network.m_connections.size();
                     })))
            {
                throw std::runtime_error(
                    "NeuralNetwork::Deserialize: invalid sparse RTRL state.");
            }
        }
        if (version >= 5)
        {
            std::size_t event_count = 0;
            input >> token >> event_count;
            if (token != "SpikeHistory")
            {
                throw std::runtime_error(
                    "NeuralNetwork::Deserialize: missing spike history.");
            }
            network.m_spike_history.resize(event_count);
            for (auto& event : network.m_spike_history)
            {
                int is_input = 0;
                input >> token >> event.time >> event.neuron_index
                      >> event.amplitude >> is_input;
                if (token != "SpikeEvent")
                {
                    throw std::runtime_error(
                        "NeuralNetwork::Deserialize: malformed spike "
                        "history.");
                }
                event.input = is_input != 0;
            }
        }
        input >> token;
        if (token != "NeuralNetworkEnd" || !input)
            throw std::runtime_error(
                "NeuralNetwork::Deserialize: malformed network data.");
        ValidateNetworkTopology(network);
        return network;
    }
}
