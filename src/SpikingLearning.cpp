#include "SpikingLearning.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace
{
    double FiniteTimeStep(
        const NEAT::NeuralNetwork& network,
        double requested)
    {
        const double result =
            requested < 0.0
                ? network.SpikingTimeStep()
                : requested;
        if (!std::isfinite(result) || result <= 0.0)
        {
            throw std::invalid_argument(
                "e-prop time step must be finite and positive");
        }
        return result;
    }

    bool ReadMarker(
        std::istream& input,
        const char* expected)
    {
        std::string marker;
        return static_cast<bool>(input >> marker) &&
               marker == expected;
    }
}

namespace NEAT
{
    void EPropLearner::ValidateConfig() const
    {
        const auto positive_finite =
            [](double value)
        {
            return std::isfinite(value) && value > 0.0;
        };
        if (!positive_finite(m_config.learning_rate) ||
            !positive_finite(m_config.surrogate_scale) ||
            !positive_finite(m_config.surrogate_dampening) ||
            !std::isfinite(m_config.gradient_clip_norm) ||
            m_config.gradient_clip_norm < 0.0 ||
            !std::isfinite(m_config.weight_decay) ||
            m_config.weight_decay < 0.0 ||
            !std::isfinite(m_config.min_weight) ||
            !std::isfinite(m_config.max_weight) ||
            m_config.min_weight > m_config.max_weight ||
            m_config.update_interval == 0 ||
            !positive_finite(m_config.huber_delta))
        {
            throw std::invalid_argument(
                "e-prop configuration contains invalid ranges");
        }
        if (m_config.optimizer < EPROP_ADAMW ||
            m_config.optimizer > EPROP_SGD ||
            m_config.feedback_mode < EPROP_RANDOM_FEEDBACK ||
            m_config.feedback_mode > EPROP_UNIFORM_FEEDBACK ||
            m_config.surrogate < EPROP_FAST_SIGMOID ||
            m_config.surrogate > EPROP_ARCTAN ||
            m_config.loss < EPROP_MEAN_SQUARED_ERROR ||
            m_config.loss > EPROP_HUBER_LOSS)
        {
            throw std::invalid_argument(
                "e-prop configuration contains an unsupported mode");
        }
        if (!std::isfinite(m_config.adam_beta1) ||
            m_config.adam_beta1 < 0.0 ||
            m_config.adam_beta1 >= 1.0 ||
            !std::isfinite(m_config.adam_beta2) ||
            m_config.adam_beta2 < 0.0 ||
            m_config.adam_beta2 >= 1.0 ||
            !positive_finite(m_config.adam_epsilon))
        {
            throw std::invalid_argument(
                "e-prop AdamW parameters are invalid");
        }
    }

    void EPropLearner::ValidateTopology(
        const NeuralNetwork& network) const
    {
        if (!IsInitialized())
        {
            throw std::logic_error(
                "e-prop learner is not initialized");
        }
        if (network.m_neurons.size() != m_neuron_count ||
            network.NumInputs() != m_input_count ||
            network.NumOutputs() != m_output_count ||
            network.m_connections.size() !=
                m_connection_state.size())
        {
            throw std::logic_error(
                "e-prop topology changed; call Initialize again");
        }
        if (network.NumInputs() + network.NumOutputs() >
            network.m_neurons.size())
        {
            throw std::invalid_argument(
                "e-prop network dimensions are invalid");
        }
        for (std::size_t i = 0; i < m_sources.size(); ++i)
        {
            const Connection& connection =
                network.m_connections[i];
            if (connection.m_source_neuron_idx != m_sources[i] ||
                connection.m_target_neuron_idx != m_targets[i])
            {
                throw std::logic_error(
                    "e-prop connection ordering changed; "
                    "call Initialize again");
            }
        }
        for (std::size_t i = 0; i < m_neuron_count; ++i)
        {
            if (static_cast<int>(
                    network.m_neurons[i].m_type) !=
                    m_neuron_types[i] ||
                static_cast<int>(
                    network.m_neurons[i]
                        .m_activation_function_type) !=
                    m_activation_types[i])
            {
                throw std::logic_error(
                    "e-prop neuron roles or activation modes changed; "
                    "call Initialize again");
            }
        }
        if (!m_config.allow_stdp)
        {
            const bool stdp_enabled = std::any_of(
                network.m_connections.begin(),
                network.m_connections.end(),
                [](const Connection& connection)
                {
                    return connection.m_stdp_enabled;
                });
            if (stdp_enabled)
            {
                throw std::logic_error(
                    "e-prop and STDP are both enabled; disable STDP or "
                    "set EPropConfig.allow_stdp explicitly");
            }
        }
    }

    bool EPropLearner::IsTrainable(
        const NeuralNetwork& network,
        std::size_t connection_index) const
    {
        const Connection& connection =
            network.m_connections[connection_index];
        if (connection.m_source_neuron_idx < 0 ||
            connection.m_target_neuron_idx < 0)
        {
            return false;
        }
        const std::size_t source =
            static_cast<std::size_t>(
                connection.m_source_neuron_idx);
        const std::size_t target =
            static_cast<std::size_t>(
                connection.m_target_neuron_idx);
        if (source >= network.m_neurons.size() ||
            target >= network.m_neurons.size() ||
            target < network.NumInputs() ||
            !IsSpikingActivation(
                network.m_neurons[target]
                    .m_activation_function_type))
        {
            return false;
        }
        if (connection.m_recur_flag &&
            !m_config.train_recurrent_connections)
        {
            return false;
        }
        if (source < network.NumInputs() &&
            !m_config.train_input_connections)
        {
            return false;
        }
        const std::size_t output_begin = network.NumInputs();
        const std::size_t output_end =
            output_begin + network.NumOutputs();
        if (target >= output_begin && target < output_end)
            return m_config.train_output_connections;
        return m_config.train_hidden_connections;
    }

    double EPropLearner::SurrogateDerivative(
        const Neuron& neuron) const
    {
        if (neuron.m_refractory_remaining > 0.0 &&
            !neuron.m_spike)
        {
            return 0.0;
        }
        const double voltage_scale = std::max(
            {
                std::abs(
                    neuron.m_spike_threshold -
                    neuron.m_reset_potential),
                std::abs(
                    neuron.m_spike_threshold -
                    neuron.m_resting_potential),
                1.0e-6});
        const double relative_voltage =
            neuron.m_spike
                ? 0.0
                : (neuron.m_membrane_potential -
                   neuron.m_spike_threshold) /
                      voltage_scale;
        const double x =
            m_config.surrogate_scale * relative_voltage;
        double shape = 0.0;
        switch (m_config.surrogate)
        {
        case EPROP_FAST_SIGMOID:
            shape =
                1.0 /
                ((1.0 + std::abs(x)) *
                 (1.0 + std::abs(x)));
            break;
        case EPROP_TRIANGULAR:
            shape = std::max(0.0, 1.0 - std::abs(x));
            break;
        case EPROP_ARCTAN:
            shape = 1.0 / (1.0 + x * x);
            break;
        default:
            throw std::logic_error(
                "Unsupported e-prop surrogate");
        }
        return m_config.surrogate_dampening *
               shape / voltage_scale;
    }

    void EPropLearner::Initialize(
        const NeuralNetwork& network)
    {
        ValidateConfig();
        if (!network.IsSpiking())
        {
            throw std::invalid_argument(
                "e-prop requires a spiking neural network");
        }
        if (network.NumInputs() + network.NumOutputs() >
            network.m_neurons.size() ||
            network.NumOutputs() == 0)
        {
            throw std::invalid_argument(
                "e-prop requires valid non-empty outputs");
        }
        m_neuron_count = network.m_neurons.size();
        m_input_count = network.NumInputs();
        m_output_count = network.NumOutputs();
        m_connection_state.assign(
            network.m_connections.size(),
            EPropConnectionState{});
        m_sources.clear();
        m_targets.clear();
        m_neuron_types.clear();
        m_activation_types.clear();
        m_sources.reserve(network.m_connections.size());
        m_targets.reserve(network.m_connections.size());
        m_neuron_types.reserve(m_neuron_count);
        m_activation_types.reserve(m_neuron_count);
        for (const Neuron& neuron : network.m_neurons)
        {
            m_neuron_types.push_back(
                static_cast<int>(neuron.m_type));
            m_activation_types.push_back(
                static_cast<int>(
                    neuron.m_activation_function_type));
        }
        for (const Connection& connection :
             network.m_connections)
        {
            if (connection.m_source_neuron_idx < 0 ||
                connection.m_target_neuron_idx < 0 ||
                static_cast<std::size_t>(
                    connection.m_source_neuron_idx) >=
                    m_neuron_count ||
                static_cast<std::size_t>(
                    connection.m_target_neuron_idx) >=
                    m_neuron_count)
            {
                throw std::invalid_argument(
                    "e-prop connection endpoint is invalid");
            }
            m_sources.push_back(
                connection.m_source_neuron_idx);
            m_targets.push_back(
                connection.m_target_neuron_idx);
        }
        m_accumulated_steps = 0;
        m_optimizer_step = 0;
        m_last_gradient_norm = 0.0;
        m_last_updated_connections = 0;
        RefreshFeedback(network);
        ValidateTopology(network);
    }

    void EPropLearner::RefreshFeedback(
        const NeuralNetwork& network)
    {
        if (network.m_neurons.size() != m_neuron_count ||
            network.NumInputs() != m_input_count ||
            network.NumOutputs() != m_output_count)
        {
            throw std::logic_error(
                "e-prop feedback topology does not match learner");
        }
        m_feedback.assign(
            m_neuron_count * m_output_count,
            0.0);
        const std::size_t output_begin = network.NumInputs();
        const double scale =
            1.0 / std::sqrt(
                static_cast<double>(
                    std::max<std::size_t>(1, m_output_count)));
        std::uint64_t random_state =
            m_config.random_seed == 0
                ? UINT64_C(0x6a09e667f3bcc909)
                : m_config.random_seed;
        const auto random_signed =
            [&random_state]()
        {
            random_state ^= random_state >> 12U;
            random_state ^= random_state << 25U;
            random_state ^= random_state >> 27U;
            const std::uint64_t value =
                random_state *
                UINT64_C(2685821657736338717);
            const double unit =
                static_cast<double>(value >> 11U) *
                (1.0 / 9007199254740992.0);
            return 2.0 * unit - 1.0;
        };

        for (std::size_t neuron = network.NumInputs();
             neuron < m_neuron_count;
             ++neuron)
        {
            for (std::size_t output = 0;
                 output < m_output_count;
                 ++output)
            {
                double value = 0.0;
                if (neuron == output_begin + output)
                {
                    value = 1.0;
                }
                else if (
                    m_config.feedback_mode ==
                    EPROP_RANDOM_FEEDBACK)
                {
                    value = random_signed() * scale;
                }
                else if (
                    m_config.feedback_mode ==
                    EPROP_UNIFORM_FEEDBACK)
                {
                    value = scale;
                }
                else
                {
                    for (const Connection& connection :
                         network.m_connections)
                    {
                        if (connection.m_source_neuron_idx ==
                                static_cast<int>(neuron) &&
                            connection.m_target_neuron_idx ==
                                static_cast<int>(
                                    output_begin + output))
                        {
                            value += connection.m_weight;
                        }
                    }
                }
                m_feedback[
                    neuron * m_output_count + output] = value;
            }
        }
    }

    void EPropLearner::ResetEligibility()
    {
        for (auto& state : m_connection_state)
        {
            state.synaptic_trace = 0.0;
            state.voltage_eligibility = 0.0;
            state.adaptation_eligibility = 0.0;
            state.readout_eligibility = 0.0;
        }
    }

    void EPropLearner::ZeroGradients()
    {
        for (auto& state : m_connection_state)
            state.gradient = 0.0;
        m_accumulated_steps = 0;
        m_last_gradient_norm = 0.0;
        m_last_updated_connections = 0;
    }

    void EPropLearner::ResetOptimizer()
    {
        for (auto& state : m_connection_state)
        {
            state.first_moment = 0.0;
            state.second_moment = 0.0;
        }
        m_optimizer_step = 0;
        ZeroGradients();
    }

    std::vector<double>
    EPropLearner::BroadcastOutputSignals(
        const std::vector<double>& output_signals) const
    {
        if (output_signals.size() != m_output_count)
        {
            throw std::invalid_argument(
                "e-prop output learning-signal count must match "
                "network outputs");
        }
        std::vector<double> neuron_signals(
            m_neuron_count,
            0.0);
        for (std::size_t neuron = 0;
             neuron < m_neuron_count;
             ++neuron)
        {
            for (std::size_t output = 0;
                 output < m_output_count;
                 ++output)
            {
                neuron_signals[neuron] +=
                    m_feedback[
                        neuron * m_output_count + output] *
                    output_signals[output];
            }
        }
        return neuron_signals;
    }

    void EPropLearner::AccumulateDirectSignals(
        NeuralNetwork& network,
        const std::vector<double>& neuron_signals,
        double time_step)
    {
        ValidateConfig();
        ValidateTopology(network);
        if (neuron_signals.size() != m_neuron_count)
        {
            throw std::invalid_argument(
                "e-prop direct learning signals must match neurons");
        }
        const double dt = FiniteTimeStep(network, time_step);
        for (double signal : neuron_signals)
        {
            if (!std::isfinite(signal))
            {
                throw std::invalid_argument(
                    "e-prop learning signals must be finite");
            }
        }

        for (std::size_t index = 0;
             index < network.m_connections.size();
             ++index)
        {
            Connection& connection =
                network.m_connections[index];
            EPropConnectionState& state =
                m_connection_state[index];
            const std::size_t source =
                static_cast<std::size_t>(
                    connection.m_source_neuron_idx);
            const std::size_t target =
                static_cast<std::size_t>(
                    connection.m_target_neuron_idx);
            const Neuron& source_neuron =
                network.m_neurons[source];
            const Neuron& target_neuron =
                network.m_neurons[target];
            const bool event_source =
                IsSpikingActivation(
                    source_neuron.m_activation_function_type) ||
                (source < network.NumInputs() &&
                 network.GetSpikingInputMode() != CURRENT_INPUT);

            if (!std::isfinite(
                    connection.m_synaptic_time_constant) ||
                connection.m_synaptic_time_constant <= 0.0)
            {
                throw std::domain_error(
                    "e-prop requires positive synaptic time constants");
            }
            if (event_source)
            {
                state.synaptic_trace *= std::exp(
                    -dt /
                    connection.m_synaptic_time_constant);
                state.synaptic_trace +=
                    connection.m_presynaptic_signal;
            }
            else
            {
                state.synaptic_trace =
                    source_neuron.m_activation;
            }

            if (!IsTrainable(network, index))
                continue;
            if (!std::isfinite(target_neuron.m_timeconst) ||
                target_neuron.m_timeconst <= 0.0 ||
                !std::isfinite(
                    target_neuron.m_membrane_resistance))
            {
                throw std::domain_error(
                    "e-prop requires positive membrane time constants "
                    "and finite membrane resistance");
            }

            const double membrane_fraction =
                dt / target_neuron.m_timeconst;
            const double membrane_decay =
                1.0 - membrane_fraction;
            const double old_voltage =
                state.voltage_eligibility;
            const double old_adaptation =
                state.adaptation_eligibility;
            double voltage_eligibility =
                membrane_decay * old_voltage +
                membrane_fraction *
                    target_neuron.m_membrane_resistance *
                    state.synaptic_trace;
            double adaptation_eligibility = 0.0;
            const double surrogate =
                SurrogateDerivative(target_neuron);

            if (target_neuron.m_activation_function_type ==
                SPIKING_ADAPTIVE_LIF)
            {
                if (!std::isfinite(
                        target_neuron
                            .m_adaptation_time_constant) ||
                    target_neuron
                            .m_adaptation_time_constant <= 0.0 ||
                    !std::isfinite(
                        target_neuron
                            .m_adaptation_increment))
                {
                    throw std::domain_error(
                        "e-prop requires valid adaptive-LIF "
                        "parameters");
                }
                voltage_eligibility -=
                    membrane_fraction * old_adaptation;
                adaptation_eligibility =
                    std::exp(
                        -dt /
                        target_neuron
                            .m_adaptation_time_constant) *
                        old_adaptation +
                    target_neuron.m_adaptation_increment *
                        surrogate * voltage_eligibility;
            }
            else if (
                target_neuron.m_activation_function_type ==
                SPIKING_IZHIKEVICH)
            {
                // The Izhikevich state has a stiff millisecond Jacobian.
                // A membrane-time-constant eligibility path keeps the
                // online surrogate stable while still assigning temporal
                // credit through its voltage and recovery state.
                const double recovery_decay = std::exp(
                    -dt * 1000.0 *
                    std::max(
                        0.0,
                        target_neuron.m_izhikevich_a));
                adaptation_eligibility =
                    recovery_decay * old_adaptation +
                    target_neuron.m_izhikevich_b *
                        surrogate * voltage_eligibility;
                voltage_eligibility -=
                    membrane_fraction *
                    adaptation_eligibility;
            }

            const double spike_eligibility =
                surrogate * voltage_eligibility;
            double eligibility = spike_eligibility;
            const std::size_t output_begin =
                network.NumInputs();
            const std::size_t output_end =
                output_begin + network.NumOutputs();
            if (target >= output_begin &&
                target < output_end)
            {
                switch (network.GetSpikingOutputMode())
                {
                case SPIKE_OUTPUT:
                    state.readout_eligibility =
                        spike_eligibility;
                    break;
                case FILTERED_SPIKE_OUTPUT:
                    state.readout_eligibility *= std::exp(
                        -dt /
                        target_neuron.m_rate_time_constant);
                    state.readout_eligibility +=
                        spike_eligibility /
                        target_neuron.m_rate_time_constant;
                    break;
                case FIRING_RATE_OUTPUT:
                    state.readout_eligibility +=
                        spike_eligibility;
                    break;
                case MEMBRANE_POTENTIAL_OUTPUT:
                    state.readout_eligibility =
                        voltage_eligibility;
                    break;
                default:
                    throw std::logic_error(
                        "Unsupported e-prop output decoder");
                }
                eligibility = state.readout_eligibility;
                if (network.GetSpikingOutputMode() ==
                        FIRING_RATE_OUTPUT &&
                    network.SpikingTime() > 0.0)
                {
                    eligibility /= network.SpikingTime();
                }
            }
            const double gradient =
                neuron_signals[target] * eligibility;
            if (!std::isfinite(gradient))
            {
                throw std::domain_error(
                    "e-prop eligibility gradient became non-finite");
            }
            state.gradient += gradient;

            const double reset_jump =
                target_neuron.m_spike
                    ? target_neuron.m_spike_threshold -
                          target_neuron.m_reset_potential
                    : 0.0;
            state.voltage_eligibility =
                voltage_eligibility *
                (1.0 - reset_jump * surrogate);
            state.adaptation_eligibility =
                adaptation_eligibility;
        }
        ++m_accumulated_steps;
    }

    void EPropLearner::AccumulateLearningSignals(
        NeuralNetwork& network,
        const std::vector<double>& learning_signals,
        double time_step)
    {
        ValidateTopology(network);
        std::vector<double> direct;
        if (learning_signals.size() == m_output_count)
        {
            direct = BroadcastOutputSignals(learning_signals);
        }
        else if (
            learning_signals.size() ==
            m_neuron_count - network.NumInputs())
        {
            direct.assign(m_neuron_count, 0.0);
            std::copy(
                learning_signals.begin(),
                learning_signals.end(),
                direct.begin() +
                    static_cast<std::ptrdiff_t>(
                        network.NumInputs()));
        }
        else if (learning_signals.size() == m_neuron_count)
        {
            direct = learning_signals;
        }
        else
        {
            throw std::invalid_argument(
                "e-prop learning signals must match outputs, "
                "non-input neurons, or all neurons");
        }
        AccumulateDirectSignals(
            network,
            direct,
            time_step);
    }

    EPropStepResult EPropLearner::ApplyGradients(
        NeuralNetwork& network)
    {
        ValidateConfig();
        ValidateTopology(network);
        EPropStepResult result;
        result.outputs = network.OutputDecoded();
        if (m_accumulated_steps == 0)
            return result;

        const double inverse_steps =
            1.0 /
            static_cast<double>(m_accumulated_steps);
        double squared_norm = 0.0;
        for (std::size_t index = 0;
             index < m_connection_state.size();
             ++index)
        {
            if (!IsTrainable(network, index))
                continue;
            const double gradient =
                m_connection_state[index].gradient *
                inverse_steps;
            squared_norm += gradient * gradient;
        }
        const double gradient_norm = std::sqrt(squared_norm);
        double clip_scale = 1.0;
        if (m_config.gradient_clip_norm > 0.0 &&
            gradient_norm > m_config.gradient_clip_norm)
        {
            clip_scale =
                m_config.gradient_clip_norm /
                gradient_norm;
        }

        ++m_optimizer_step;
        std::size_t updated = 0;
        const double beta1_correction =
            1.0 -
            std::pow(
                m_config.adam_beta1,
                static_cast<double>(m_optimizer_step));
        const double beta2_correction =
            1.0 -
            std::pow(
                m_config.adam_beta2,
                static_cast<double>(m_optimizer_step));
        for (std::size_t index = 0;
             index < m_connection_state.size();
             ++index)
        {
            EPropConnectionState& state =
                m_connection_state[index];
            if (!IsTrainable(network, index))
            {
                state.gradient = 0.0;
                continue;
            }
            Connection& connection =
                network.m_connections[index];
            const double gradient =
                state.gradient * inverse_steps * clip_scale;
            if (m_config.optimizer == EPROP_ADAMW)
            {
                state.first_moment =
                    m_config.adam_beta1 *
                        state.first_moment +
                    (1.0 - m_config.adam_beta1) *
                        gradient;
                state.second_moment =
                    m_config.adam_beta2 *
                        state.second_moment +
                    (1.0 - m_config.adam_beta2) *
                        gradient * gradient;
                const double first_hat =
                    state.first_moment / beta1_correction;
                const double second_hat =
                    state.second_moment / beta2_correction;
                connection.m_weight *=
                    1.0 -
                    m_config.learning_rate *
                        m_config.weight_decay;
                connection.m_weight -=
                    m_config.learning_rate *
                    first_hat /
                    (std::sqrt(second_hat) +
                     m_config.adam_epsilon);
            }
            else
            {
                connection.m_weight *=
                    1.0 -
                    m_config.learning_rate *
                        m_config.weight_decay;
                connection.m_weight -=
                    m_config.learning_rate *
                    gradient;
            }
            connection.m_weight = std::clamp(
                connection.m_weight,
                m_config.min_weight,
                m_config.max_weight);
            state.gradient = 0.0;
            ++updated;
        }
        m_accumulated_steps = 0;
        m_last_gradient_norm =
            gradient_norm * clip_scale;
        m_last_updated_connections = updated;
        result.gradient_norm = m_last_gradient_norm;
        result.updated_connections = updated;
        result.update_applied = true;
        return result;
    }

    EPropStepResult EPropLearner::TrainStep(
        NeuralNetwork& network,
        const std::vector<double>& inputs,
        const std::vector<double>& targets,
        double time_step)
    {
        ValidateConfig();
        ValidateTopology(network);
        if (targets.size() != m_output_count)
        {
            throw std::invalid_argument(
                "e-prop target count must match network outputs");
        }
        for (double target : targets)
        {
            if (!std::isfinite(target))
            {
                throw std::invalid_argument(
                    "e-prop targets must be finite");
            }
        }
        const double dt = FiniteTimeStep(network, time_step);
        EPropStepResult result;
        result.outputs =
            network.StepSpiking(inputs, dt);
        std::vector<double> output_signals(
            m_output_count,
            0.0);
        for (std::size_t output = 0;
             output < m_output_count;
             ++output)
        {
            const double error =
                result.outputs[output] - targets[output];
            if (m_config.loss ==
                EPROP_MEAN_SQUARED_ERROR)
            {
                result.loss += 0.5 * error * error;
                output_signals[output] =
                    error /
                    static_cast<double>(m_output_count);
            }
            else
            {
                const double absolute = std::abs(error);
                if (absolute <= m_config.huber_delta)
                {
                    result.loss +=
                        0.5 * error * error;
                    output_signals[output] =
                        error /
                        static_cast<double>(m_output_count);
                }
                else
                {
                    result.loss +=
                        m_config.huber_delta *
                        (absolute -
                         0.5 * m_config.huber_delta);
                    output_signals[output] =
                        std::copysign(
                            m_config.huber_delta,
                            error) /
                        static_cast<double>(m_output_count);
                }
            }
        }
        result.loss /=
            static_cast<double>(m_output_count);
        AccumulateDirectSignals(
            network,
            BroadcastOutputSignals(output_signals),
            dt);
        if (m_accumulated_steps >=
            m_config.update_interval)
        {
            EPropStepResult update =
                ApplyGradients(network);
            result.gradient_norm =
                update.gradient_norm;
            result.updated_connections =
                update.updated_connections;
            result.update_applied = true;
        }
        return result;
    }

    EPropStepResult EPropLearner::TrainStepWithSignals(
        NeuralNetwork& network,
        const std::vector<double>& inputs,
        const std::vector<double>& learning_signals,
        double time_step)
    {
        ValidateConfig();
        ValidateTopology(network);
        const double dt = FiniteTimeStep(network, time_step);
        EPropStepResult result;
        result.outputs =
            network.StepSpiking(inputs, dt);
        AccumulateLearningSignals(
            network,
            learning_signals,
            dt);
        if (m_accumulated_steps >=
            m_config.update_interval)
        {
            EPropStepResult update =
                ApplyGradients(network);
            result.gradient_norm =
                update.gradient_norm;
            result.updated_connections =
                update.updated_connections;
            result.update_applied = true;
        }
        return result;
    }

    EPropSequenceResult EPropLearner::TrainSequence(
        NeuralNetwork& network,
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets,
        double time_step,
        bool reset_network,
        bool apply_final_update)
    {
        ValidateTopology(network);
        if (inputs.size() != targets.size())
        {
            throw std::invalid_argument(
                "e-prop input and target sequences must have "
                "equal lengths");
        }
        if (reset_network)
        {
            network.Flush();
            ResetEligibility();
        }
        EPropSequenceResult result;
        result.outputs.reserve(inputs.size());
        result.losses.reserve(inputs.size());
        for (std::size_t step = 0;
             step < inputs.size();
             ++step)
        {
            EPropStepResult current =
                TrainStep(
                    network,
                    inputs[step],
                    targets[step],
                    time_step);
            result.outputs.push_back(
                std::move(current.outputs));
            result.losses.push_back(current.loss);
            result.mean_loss += current.loss;
            if (current.update_applied)
            {
                ++result.optimizer_updates;
                result.final_gradient_norm =
                    current.gradient_norm;
                result.updated_connections +=
                    current.updated_connections;
            }
        }
        if (!inputs.empty())
        {
            result.mean_loss /=
                static_cast<double>(inputs.size());
        }
        if (apply_final_update &&
            m_accumulated_steps > 0)
        {
            EPropStepResult final_update =
                ApplyGradients(network);
            ++result.optimizer_updates;
            result.final_gradient_norm =
                final_update.gradient_norm;
            result.updated_connections +=
                final_update.updated_connections;
        }
        return result;
    }

    std::string EPropLearner::Serialize() const
    {
        ValidateConfig();
        std::ostringstream output;
        output << std::setprecision(
            std::numeric_limits<double>::max_digits10);
        output << "EPropFormat 1\n";
        output << "Config "
               << m_config.learning_rate << ' '
               << static_cast<int>(m_config.optimizer) << ' '
               << static_cast<int>(m_config.feedback_mode) << ' '
               << static_cast<int>(m_config.surrogate) << ' '
               << static_cast<int>(m_config.loss) << ' '
               << m_config.surrogate_scale << ' '
               << m_config.surrogate_dampening << ' '
               << m_config.gradient_clip_norm << ' '
               << m_config.weight_decay << ' '
               << m_config.adam_beta1 << ' '
               << m_config.adam_beta2 << ' '
               << m_config.adam_epsilon << ' '
               << m_config.huber_delta << ' '
               << m_config.min_weight << ' '
               << m_config.max_weight << ' '
               << m_config.update_interval << ' '
               << m_config.random_seed << ' '
               << static_cast<int>(
                      m_config.train_input_connections) << ' '
               << static_cast<int>(
                      m_config.train_hidden_connections) << ' '
               << static_cast<int>(
                      m_config.train_output_connections) << ' '
               << static_cast<int>(
                      m_config.train_recurrent_connections) << ' '
               << static_cast<int>(m_config.allow_stdp)
               << '\n';
        output << "State "
               << m_neuron_count << ' '
               << m_input_count << ' '
               << m_output_count << ' '
               << m_connection_state.size() << ' '
               << m_accumulated_steps << ' '
               << m_optimizer_step << ' '
               << m_last_gradient_norm << ' '
               << m_last_updated_connections << '\n';
        output << "Endpoints " << m_sources.size() << '\n';
        for (std::size_t i = 0; i < m_sources.size(); ++i)
            output << m_sources[i] << ' ' << m_targets[i] << '\n';
        output << "Neurons " << m_neuron_types.size() << '\n';
        for (std::size_t i = 0; i < m_neuron_types.size(); ++i)
        {
            output << m_neuron_types[i] << ' '
                   << m_activation_types[i] << '\n';
        }
        output << "Feedback " << m_feedback.size() << '\n';
        for (double value : m_feedback)
            output << value << '\n';
        output << "Connections "
               << m_connection_state.size() << '\n';
        for (const auto& state : m_connection_state)
        {
            output << state.synaptic_trace << ' '
                   << state.voltage_eligibility << ' '
                   << state.adaptation_eligibility << ' '
                   << state.readout_eligibility << ' '
                   << state.gradient << ' '
                   << state.first_moment << ' '
                   << state.second_moment << '\n';
        }
        output << "End\n";
        return output.str();
    }

    EPropLearner EPropLearner::Deserialize(
        const std::string& data)
    {
        std::istringstream input(data);
        if (!ReadMarker(input, "EPropFormat"))
        {
            throw std::runtime_error(
                "EPropLearner::Deserialize: missing header");
        }
        int version = 0;
        if (!(input >> version) || version != 1 ||
            !ReadMarker(input, "Config"))
        {
            throw std::runtime_error(
                "EPropLearner::Deserialize: unsupported format");
        }
        EPropLearner learner;
        int optimizer = 0;
        int feedback = 0;
        int surrogate = 0;
        int loss = 0;
        int train_input = 0;
        int train_hidden = 0;
        int train_output = 0;
        int train_recurrent = 0;
        int allow_stdp = 0;
        if (!(input >>
              learner.m_config.learning_rate >>
              optimizer >>
              feedback >>
              surrogate >>
              loss >>
              learner.m_config.surrogate_scale >>
              learner.m_config.surrogate_dampening >>
              learner.m_config.gradient_clip_norm >>
              learner.m_config.weight_decay >>
              learner.m_config.adam_beta1 >>
              learner.m_config.adam_beta2 >>
              learner.m_config.adam_epsilon >>
              learner.m_config.huber_delta >>
              learner.m_config.min_weight >>
              learner.m_config.max_weight >>
              learner.m_config.update_interval >>
              learner.m_config.random_seed >>
              train_input >>
              train_hidden >>
              train_output >>
              train_recurrent >>
              allow_stdp))
        {
            throw std::runtime_error(
                "EPropLearner::Deserialize: malformed config");
        }
        learner.m_config.optimizer =
            static_cast<EPropOptimizer>(optimizer);
        learner.m_config.feedback_mode =
            static_cast<EPropFeedbackMode>(feedback);
        learner.m_config.surrogate =
            static_cast<EPropSurrogate>(surrogate);
        learner.m_config.loss =
            static_cast<EPropLoss>(loss);
        learner.m_config.train_input_connections =
            train_input != 0;
        learner.m_config.train_hidden_connections =
            train_hidden != 0;
        learner.m_config.train_output_connections =
            train_output != 0;
        learner.m_config.train_recurrent_connections =
            train_recurrent != 0;
        learner.m_config.allow_stdp =
            allow_stdp != 0;
        learner.ValidateConfig();

        std::size_t connection_count = 0;
        if (!ReadMarker(input, "State") ||
            !(input >>
              learner.m_neuron_count >>
              learner.m_input_count >>
              learner.m_output_count >>
              connection_count >>
              learner.m_accumulated_steps >>
              learner.m_optimizer_step >>
              learner.m_last_gradient_norm >>
              learner.m_last_updated_connections))
        {
            throw std::runtime_error(
                "EPropLearner::Deserialize: malformed state");
        }
        std::size_t endpoint_count = 0;
        if (!ReadMarker(input, "Endpoints") ||
            !(input >> endpoint_count) ||
            endpoint_count != connection_count)
        {
            throw std::runtime_error(
                "EPropLearner::Deserialize: malformed endpoints");
        }
        learner.m_sources.resize(endpoint_count);
        learner.m_targets.resize(endpoint_count);
        for (std::size_t i = 0; i < endpoint_count; ++i)
        {
            if (!(input >>
                  learner.m_sources[i] >>
                  learner.m_targets[i]))
            {
                throw std::runtime_error(
                    "EPropLearner::Deserialize: malformed endpoint");
            }
        }
        std::size_t neuron_signature_count = 0;
        if (!ReadMarker(input, "Neurons") ||
            !(input >> neuron_signature_count) ||
            neuron_signature_count != learner.m_neuron_count)
        {
            throw std::runtime_error(
                "EPropLearner::Deserialize: malformed neuron "
                "signature");
        }
        learner.m_neuron_types.resize(neuron_signature_count);
        learner.m_activation_types.resize(
            neuron_signature_count);
        for (std::size_t i = 0;
             i < neuron_signature_count;
             ++i)
        {
            if (!(input >>
                  learner.m_neuron_types[i] >>
                  learner.m_activation_types[i]))
            {
                throw std::runtime_error(
                    "EPropLearner::Deserialize: malformed neuron "
                    "signature value");
            }
        }
        std::size_t feedback_count = 0;
        if (!ReadMarker(input, "Feedback") ||
            !(input >> feedback_count))
        {
            throw std::runtime_error(
                "EPropLearner::Deserialize: malformed feedback");
        }
        if (feedback_count !=
            learner.m_neuron_count *
                learner.m_output_count)
        {
            throw std::runtime_error(
                "EPropLearner::Deserialize: invalid feedback size");
        }
        learner.m_feedback.resize(feedback_count);
        for (double& value : learner.m_feedback)
        {
            if (!(input >> value) || !std::isfinite(value))
            {
                throw std::runtime_error(
                    "EPropLearner::Deserialize: malformed feedback "
                    "value");
            }
        }
        std::size_t state_count = 0;
        if (!ReadMarker(input, "Connections") ||
            !(input >> state_count) ||
            state_count != connection_count)
        {
            throw std::runtime_error(
                "EPropLearner::Deserialize: malformed connection state");
        }
        learner.m_connection_state.resize(state_count);
        for (auto& state : learner.m_connection_state)
        {
            if (!(input >>
                  state.synaptic_trace >>
                  state.voltage_eligibility >>
                  state.adaptation_eligibility >>
                  state.readout_eligibility >>
                  state.gradient >>
                  state.first_moment >>
                  state.second_moment) ||
                !std::isfinite(state.synaptic_trace) ||
                !std::isfinite(state.voltage_eligibility) ||
                !std::isfinite(
                    state.adaptation_eligibility) ||
                !std::isfinite(state.readout_eligibility) ||
                !std::isfinite(state.gradient) ||
                !std::isfinite(state.first_moment) ||
                !std::isfinite(state.second_moment))
            {
                throw std::runtime_error(
                    "EPropLearner::Deserialize: malformed connection");
            }
        }
        if (!ReadMarker(input, "End"))
        {
            throw std::runtime_error(
                "EPropLearner::Deserialize: missing terminator");
        }
        std::string trailing;
        if (input >> trailing)
        {
            throw std::runtime_error(
                "EPropLearner::Deserialize: trailing data");
        }
        return learner;
    }
}
