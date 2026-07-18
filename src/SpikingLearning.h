#ifndef MULTINEAT_SPIKING_LEARNING_H
#define MULTINEAT_SPIKING_LEARNING_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "NeuralNetwork.h"

namespace NEAT
{
    enum EPropOptimizer
    {
        EPROP_ADAMW = 0,
        EPROP_SGD
    };

    enum EPropFeedbackMode
    {
        EPROP_RANDOM_FEEDBACK = 0,
        EPROP_SYMMETRIC_FEEDBACK,
        EPROP_UNIFORM_FEEDBACK
    };

    enum EPropSurrogate
    {
        EPROP_FAST_SIGMOID = 0,
        EPROP_TRIANGULAR,
        EPROP_ARCTAN
    };

    enum EPropLoss
    {
        EPROP_MEAN_SQUARED_ERROR = 0,
        EPROP_HUBER_LOSS
    };

    struct EPropConfig
    {
        double learning_rate = 0.001;
        EPropOptimizer optimizer = EPROP_ADAMW;
        EPropFeedbackMode feedback_mode = EPROP_RANDOM_FEEDBACK;
        EPropSurrogate surrogate = EPROP_FAST_SIGMOID;
        EPropLoss loss = EPROP_MEAN_SQUARED_ERROR;
        double surrogate_scale = 10.0;
        double surrogate_dampening = 0.3;
        double gradient_clip_norm = 1.0;
        double weight_decay = 0.0;
        double adam_beta1 = 0.9;
        double adam_beta2 = 0.999;
        double adam_epsilon = 1.0e-8;
        double huber_delta = 1.0;
        double min_weight = -8.0;
        double max_weight = 8.0;
        std::size_t update_interval = 1;
        std::uint64_t random_seed = UINT64_C(0x6a09e667f3bcc909);
        bool train_input_connections = true;
        bool train_hidden_connections = true;
        bool train_output_connections = true;
        bool train_recurrent_connections = true;
        bool allow_stdp = false;
    };

    struct EPropConnectionState
    {
        double synaptic_trace = 0.0;
        double voltage_eligibility = 0.0;
        double adaptation_eligibility = 0.0;
        double readout_eligibility = 0.0;
        double gradient = 0.0;
        double first_moment = 0.0;
        double second_moment = 0.0;
    };

    struct EPropStepResult
    {
        std::vector<double> outputs;
        double loss = 0.0;
        double gradient_norm = 0.0;
        std::size_t updated_connections = 0;
        bool update_applied = false;
    };

    struct EPropSequenceResult
    {
        std::vector<std::vector<double>> outputs;
        std::vector<double> losses;
        double mean_loss = 0.0;
        double final_gradient_norm = 0.0;
        std::size_t optimizer_updates = 0;
        std::size_t updated_connections = 0;
    };

    class EPropLearner
    {
        std::vector<EPropConnectionState> m_connection_state;
        std::vector<double> m_feedback;
        std::vector<int> m_sources;
        std::vector<int> m_targets;
        std::vector<int> m_neuron_types;
        std::vector<int> m_activation_types;
        std::size_t m_neuron_count = 0;
        std::size_t m_input_count = 0;
        std::size_t m_output_count = 0;
        std::size_t m_accumulated_steps = 0;
        std::uint64_t m_optimizer_step = 0;
        double m_last_gradient_norm = 0.0;
        std::size_t m_last_updated_connections = 0;

        void ValidateConfig() const;
        void ValidateTopology(const NeuralNetwork& network) const;
        bool IsTrainable(
            const NeuralNetwork& network,
            std::size_t connection_index) const;
        double SurrogateDerivative(const Neuron& neuron) const;
        std::vector<double> BroadcastOutputSignals(
            const std::vector<double>& output_signals) const;
        void AccumulateDirectSignals(
            NeuralNetwork& network,
            const std::vector<double>& neuron_signals,
            double time_step);

    public:
        EPropConfig m_config;

        EPropLearner() = default;
        explicit EPropLearner(const EPropConfig& config)
            : m_config(config)
        {
        }

        void Initialize(const NeuralNetwork& network);
        void RefreshFeedback(const NeuralNetwork& network);
        bool IsInitialized() const
        {
            return m_neuron_count > 0 ||
                   !m_connection_state.empty();
        }
        void ResetEligibility();
        void ResetOptimizer();
        void ZeroGradients();

        EPropStepResult TrainStep(
            NeuralNetwork& network,
            const std::vector<double>& inputs,
            const std::vector<double>& targets,
            double time_step = -1.0);
        EPropStepResult TrainStepWithSignals(
            NeuralNetwork& network,
            const std::vector<double>& inputs,
            const std::vector<double>& learning_signals,
            double time_step = -1.0);
        EPropSequenceResult TrainSequence(
            NeuralNetwork& network,
            const std::vector<std::vector<double>>& inputs,
            const std::vector<std::vector<double>>& targets,
            double time_step = -1.0,
            bool reset_network = true,
            bool apply_final_update = true);
        void AccumulateLearningSignals(
            NeuralNetwork& network,
            const std::vector<double>& learning_signals,
            double time_step = -1.0);
        EPropStepResult ApplyGradients(NeuralNetwork& network);

        const std::vector<EPropConnectionState>&
        ConnectionStates() const
        {
            return m_connection_state;
        }
        const std::vector<double>& FeedbackMatrix() const
        {
            return m_feedback;
        }
        std::uint64_t OptimizerStep() const
        {
            return m_optimizer_step;
        }
        std::size_t AccumulatedSteps() const
        {
            return m_accumulated_steps;
        }

        std::string Serialize() const;
        static EPropLearner Deserialize(const std::string& data);
    };
}

#endif
