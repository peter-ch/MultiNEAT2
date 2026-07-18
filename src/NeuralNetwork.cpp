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
        switch (neuron.m_activation_function_type)
        {
        case SIGNED_SIGMOID:
            return neuron.m_a * (1.0 - output * output) * 0.5;
        case UNSIGNED_SIGMOID:
            return neuron.m_a * output * (1.0 - output);
        case TANH:
            return neuron.m_a * (1.0 - output * output);
        case LINEAR:
            return 1.0;
        case RELU:
            return output > 0.0 ? 1.0 : 0.0;
        case SOFTPLUS:
            return 1.0 - std::exp(-output);
        default:
            // Step functions are non-differentiable, while the pre-activation
            // needed by the remaining functions is not part of the historical
            // runtime state. Treat those derivatives as zero.
            return 0.0;
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

    void NeuralNetwork::ActivateFast()
    {
        ValidateNetworkTopology(*this);
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
            double x = m_neurons[i].m_activesum;
            m_neurons[i].m_activesum = 0;
            double y = 0.0;
            switch (m_neurons[i].m_activation_function_type)
            {
                case SIGNED_SIGMOID:    y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_SIGMOID:  y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case TANH:              y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case TANH_CUBIC:        y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case SIGNED_STEP:       y = af_step_signed(x, m_neurons[i].m_b); break;
                case UNSIGNED_STEP:     y = af_step_unsigned(x, m_neurons[i].m_b); break;
                case SIGNED_GAUSS:      y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_GAUSS:    y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case ABS:               y = af_abs(x, m_neurons[i].m_b); break;
                case SIGNED_SINE:       y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_SINE:     y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case LINEAR:            y = af_linear(x, m_neurons[i].m_b); break;
                case RELU:              y = af_relu(x); break;
                case SOFTPLUS:          y = af_softplus(x); break;
                default:                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
            }
            m_neurons[i].m_activation = y;
        }
    }

    void NeuralNetwork::Activate()
    {
        ValidateNetworkTopology(*this);
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
            double x = m_neurons[i].m_activesum;
            m_neurons[i].m_activesum = 0;
            double y = 0.0;
            switch (m_neurons[i].m_activation_function_type)
            {
                case SIGNED_SIGMOID:    y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_SIGMOID:  y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case TANH:              y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case TANH_CUBIC:        y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case SIGNED_STEP:       y = af_step_signed(x, m_neurons[i].m_b); break;
                case UNSIGNED_STEP:     y = af_step_unsigned(x, m_neurons[i].m_b); break;
                case SIGNED_GAUSS:      y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_GAUSS:    y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case ABS:               y = af_abs(x, m_neurons[i].m_b); break;
                case SIGNED_SINE:       y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_SINE:     y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case LINEAR:            y = af_linear(x, m_neurons[i].m_b); break;
                case RELU:              y = af_relu(x); break;
                case SOFTPLUS:          y = af_softplus(x); break;
                default:                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
            }
            m_neurons[i].m_activation = y;
        }
    }

    void NeuralNetwork::ActivateUseInternalBias()
    {
        ValidateNetworkTopology(*this);
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
            double x = m_neurons[i].m_activesum + m_neurons[i].m_bias;
            m_neurons[i].m_activesum = 0;
            double y = 0.0;
            switch (m_neurons[i].m_activation_function_type)
            {
                case SIGNED_SIGMOID:    y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_SIGMOID:  y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case TANH:              y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case TANH_CUBIC:        y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case SIGNED_STEP:       y = af_step_signed(x, m_neurons[i].m_b); break;
                case UNSIGNED_STEP:     y = af_step_unsigned(x, m_neurons[i].m_b); break;
                case SIGNED_GAUSS:      y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_GAUSS:    y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case ABS:               y = af_abs(x, m_neurons[i].m_b); break;
                case SIGNED_SINE:       y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_SINE:     y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case LINEAR:            y = af_linear(x, m_neurons[i].m_b); break;
                case RELU:              y = af_relu(x); break;
                case SOFTPLUS:          y = af_softplus(x); break;
                default:                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
            }
            m_neurons[i].m_activation = y;
        }
    }

    void NeuralNetwork::ActivateLeaky(double a_dtime)
    {
        ValidateNetworkTopology(*this);
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
            double x = m_neurons[i].m_membrane_potential + m_neurons[i].m_bias;
            m_neurons[i].m_activesum = 0;
            double y = 0.0;
            switch (m_neurons[i].m_activation_function_type)
            {
                case SIGNED_SIGMOID:    y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_SIGMOID:  y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case TANH:              y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case TANH_CUBIC:        y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case SIGNED_STEP:       y = af_step_signed(x, m_neurons[i].m_b); break;
                case UNSIGNED_STEP:     y = af_step_unsigned(x, m_neurons[i].m_b); break;
                case SIGNED_GAUSS:      y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_GAUSS:    y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case ABS:               y = af_abs(x, m_neurons[i].m_b); break;
                case SIGNED_SINE:       y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case UNSIGNED_SINE:     y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
                case LINEAR:            y = af_linear(x, m_neurons[i].m_b); break;
                case RELU:              y = af_relu(x); break;
                case SOFTPLUS:          y = af_softplus(x); break;
                default:                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b); break;
            }
            m_neurons[i].m_activation = y;
        }
    }

    void NeuralNetwork::Flush()
    {
        for (auto &neuron : m_neurons)
        {
            neuron.m_activation = 0;
            neuron.m_activesum = 0;
            neuron.m_membrane_potential = 0;
        }
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
        for (const auto &neuron : m_neurons)
        {
            fprintf(a_file, "neuron %d %3.18f %3.18f %3.18f %3.18f %d %3.18f\n",
                    static_cast<int>(neuron.m_type), neuron.m_a,
                    neuron.m_b, neuron.m_timeconst, neuron.m_bias,
                    static_cast<int>(neuron.m_activation_function_type),
                    neuron.m_split_y);
        }
        for (const auto &conn : m_connections)
        {
            fprintf(a_file, "connection %d %d %3.18f %d %3.18f %3.18f\n",
                    conn.m_source_neuron_idx,
                    conn.m_target_neuron_idx, conn.m_weight,
                    static_cast<int>(conn.m_recur_flag),
                    conn.m_hebb_rate, conn.m_hebb_pre_rate);
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
        while (a_DataFile >> t_str && t_str != "NNend")
        {
            if (t_str == "neuron")
            {
                Neuron t_n;
                int t_type, t_aftype;
                a_DataFile >> t_type >> t_n.m_a >> t_n.m_b
                           >> t_n.m_timeconst >> t_n.m_bias;
                a_DataFile >> t_aftype >> t_n.m_split_y;
                t_n.m_type = static_cast<NeuronType>(t_type);
                t_n.m_activation_function_type = static_cast<ActivationFunction>(t_aftype);
                m_neurons.push_back(t_n);
            }
            else if (t_str == "connection")
            {
                Connection t_c;
                int t_isrecur;
                a_DataFile >> t_c.m_source_neuron_idx >> t_c.m_target_neuron_idx >> t_c.m_weight >> t_isrecur >> t_c.m_hebb_rate >> t_c.m_hebb_pre_rate;
                t_c.m_recur_flag = static_cast<bool>(t_isrecur);
                m_connections.push_back(t_c);
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
        output << "NeuralNetworkFormat 3\n";
        output << "State " << m_num_inputs << ' ' << m_num_outputs << ' '
               << m_total_error << '\n';
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
                   << ' '
                   << static_cast<int>(neuron.m_activation_function_type)
                   << ' ' << neuron.m_x << ' ' << neuron.m_y << ' '
                   << neuron.m_z << ' ' << neuron.m_sx << ' ' << neuron.m_sy
                   << ' ' << neuron.m_sz << ' ' << neuron.m_split_y << ' '
                   << static_cast<int>(neuron.m_type) << '\n';
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
                   << connection.m_hebb_pre_rate << '\n';
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
        if (version < 2 || version > 3)
            throw std::runtime_error(
                "NeuralNetwork::Deserialize: unsupported format.");
        input >> token;
        if (token != "State")
            throw std::runtime_error(
                "NeuralNetwork::Deserialize: missing State marker.");
        input >> network.m_num_inputs >> network.m_num_outputs
              >> network.m_total_error;

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
                  >> neuron.m_membrane_potential >> activation >> neuron.m_x
                  >> neuron.m_y >> neuron.m_z >> neuron.m_sx >> neuron.m_sy
                  >> neuron.m_sz >> neuron.m_split_y >> type;
            neuron.m_activation_function_type =
                static_cast<ActivationFunction>(activation);
            neuron.m_type = static_cast<NeuronType>(type);

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
            network.m_connections.push_back(connection);
        }
        input >> token;
        if (token != "NeuralNetworkEnd" || !input)
            throw std::runtime_error(
                "NeuralNetwork::Deserialize: malformed network data.");
        ValidateNetworkTopology(network);
        return network;
    }
}
