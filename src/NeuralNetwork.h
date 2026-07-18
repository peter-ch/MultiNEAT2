#ifndef _PHENOTYPE_H
#define _PHENOTYPE_H

#include <cstdio>
#include <fstream>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "Genes.h"

namespace NEAT
{
    enum SpikingInputMode
    {
        CURRENT_INPUT = 0,
        BINARY_SPIKE_INPUT,
        POISSON_RATE_INPUT
    };

    enum SpikingOutputMode
    {
        SPIKE_OUTPUT = 0,
        FIRING_RATE_OUTPUT,
        FILTERED_SPIKE_OUTPUT,
        MEMBRANE_POTENTIAL_OUTPUT
    };

    struct SpikeEvent
    {
        double time = 0.0;
        int neuron_index = 0;
        double amplitude = 1.0;
        bool input = false;
    };

    struct PendingSynapticEvent
    {
        double delivery_time = 0.0;
        double amplitude = 0.0;
        double source_amplitude = 0.0;
    };

    class Connection
    {
    public:
        int m_source_neuron_idx = 0;
        int m_target_neuron_idx = 0;
        double m_weight = 0.0;
        double m_signal = 0.0;
        bool m_recur_flag = false;
        double m_hebb_rate = 0.0;
        double m_hebb_pre_rate = 0.0;
        // Appended after the historical fields so existing aggregate
        // initializers retain their original positional meaning. This source
        // activation is required for exact online RTRL gradients after
        // recurrent neuron state advances.
        double m_source_activation = 0.0;
        double m_synaptic_delay = 0.0;
        double m_synaptic_time_constant = 0.005;
        double m_synaptic_current = 0.0;
        double m_presynaptic_signal = 0.0;
        bool m_stdp_enabled = false;
        double m_stdp_plus = 0.01;
        double m_stdp_minus = 0.012;
        double m_stdp_tau_plus = 0.02;
        double m_stdp_tau_minus = 0.02;
        double m_stdp_pre_trace = 0.0;
        double m_stdp_post_trace = 0.0;
        double m_stdp_min_weight = -8.0;
        double m_stdp_max_weight = 8.0;
        std::vector<PendingSynapticEvent> m_pending_events;

        bool operator==(const Connection &other) const
        {
            return (m_source_neuron_idx == other.m_source_neuron_idx &&
                    m_target_neuron_idx == other.m_target_neuron_idx);
        }
    };

    class Neuron
    {
    public:
        double m_activesum = 0.0;
        double m_activation = 0.0;
        double m_a = 1.0;
        double m_b = 0.0;
        double m_timeconst = 1.0;
        double m_bias = 0.0;
        double m_membrane_potential = 0.0;
        ActivationFunction m_activation_function_type = UNSIGNED_SIGMOID;
        double m_x = 0.0;
        double m_y = 0.0;
        double m_z = 0.0;
        double m_sx = 0.0;
        double m_sy = 0.0;
        double m_sz = 0.0;
        std::vector<double> m_substrate_coords;
        double m_split_y = 0.0;
        NeuronType m_type = NONE;
        std::vector<std::vector<double>> m_sensitivity_matrix;
        // Pre-activation retained for exact derivatives of non-monotonic
        // activation functions during online learning.
        double m_last_input = 0.0;
        double m_spike_threshold = 1.0;
        double m_reset_potential = 0.0;
        double m_resting_potential = 0.0;
        double m_refractory_period = 0.002;
        double m_refractory_remaining = 0.0;
        double m_membrane_resistance = 1.0;
        double m_adaptation_time_constant = 0.1;
        double m_adaptation_increment = 0.1;
        double m_adaptation = 0.0;
        double m_izhikevich_a = 0.02;
        double m_izhikevich_b = 0.2;
        double m_izhikevich_c = -65.0;
        double m_izhikevich_d = 8.0;
        double m_izhikevich_recovery = -13.0;
        bool m_spike = false;
        std::uint64_t m_spike_count = 0;
        double m_last_spike_time = -1.0;
        double m_rate_trace = 0.0;
        double m_rate_time_constant = 0.05;

        bool operator==(Neuron const &other) const
        {
            return (m_type == other.m_type &&
                    m_split_y == other.m_split_y &&
                    m_activation_function_type == other.m_activation_function_type);
        }
    };

    class NeuralNetwork
    {
        double m_total_error = 0.0;
        std::vector<double> m_total_weight_change;
        std::vector<std::vector<double>> m_sparse_rtrl_sensitivities;
        double m_spiking_time = 0.0;
        double m_spiking_time_step = 0.001;
        SpikingInputMode m_spiking_input_mode = CURRENT_INPUT;
        SpikingOutputMode m_spiking_output_mode = SPIKE_OUTPUT;
        bool m_record_spikes = true;
        std::size_t m_max_recorded_spikes = 100000;
        std::uint64_t m_spiking_rng_state =
            UINT64_C(0x9e3779b97f4a7c15);
        std::vector<SpikeEvent> m_spike_history;

    public:
        unsigned int m_num_inputs = 0;
        unsigned int m_num_outputs = 0;
        std::vector<Connection> m_connections;
        std::vector<Neuron> m_neurons;

        NeuralNetwork(bool a_Minimal);
        NeuralNetwork();
        void InitRTRLMatrix();
        void InitSparseRTRLMatrix();
        void ActivateFast();
        void Activate();
        void ActivateUseInternalBias();
        void ActivateLeaky(double step);
        void RTRL_update_gradients();
        void RTRL_update_gradients_sparse();
        void RTRL_update_error(double a_target);
        void RTRL_update_error(
            const std::vector<double>& targets,
            double learning_rate = 0.0001);
        void RTRL_update_error_sparse(
            double a_target,
            double learning_rate = 0.0001);
        void RTRL_update_error_sparse(
            const std::vector<double>& targets,
            double learning_rate = 0.0001);
        void RTRL_update_weights();
        void Adapt(Parameters &a_Parameters);
        int ConnectionExists(int a_to, int a_from);
        void Flush();
        void FlushCube();
        void Input(std::vector<double> &a_Inputs);
        void InputExact(const std::vector<double>& a_Inputs);
        std::vector<double> Output();
        void ActivateSteps(unsigned int steps, bool fast = true);
        std::vector<std::vector<double>> ActivateBatch(
            const std::vector<std::vector<double>>& inputs,
            unsigned int steps = 1,
            bool use_internal_bias = false);
        std::vector<double> StepSpiking(
            const std::vector<double>& inputs,
            double time_step = -1.0);
        std::vector<std::vector<double>> SimulateSpiking(
            const std::vector<std::vector<double>>& inputs,
            double time_step = -1.0,
            bool reset = false);
        std::vector<double> OutputSpikes() const;
        std::vector<double> OutputRates() const;
        std::vector<double> OutputFilteredSpikes() const;
        std::vector<double> OutputMembranePotentials() const;
        std::vector<double> OutputDecoded() const;
        bool IsSpiking() const;
        double SpikingTime() const { return m_spiking_time; }
        double SpikingTimeStep() const { return m_spiking_time_step; }
        void SetSpikingTimeStep(double time_step);
        void SetSpikingInputMode(SpikingInputMode mode);
        SpikingInputMode GetSpikingInputMode() const
        {
            return m_spiking_input_mode;
        }
        void SetSpikingOutputMode(SpikingOutputMode mode);
        SpikingOutputMode GetSpikingOutputMode() const
        {
            return m_spiking_output_mode;
        }
        void SeedSpiking(std::uint64_t seed);
        void EnableSpikeRecording(
            bool enabled,
            std::size_t max_events = 100000);
        const std::vector<SpikeEvent>& GetSpikeHistory() const
        {
            return m_spike_history;
        }
        void ClearSpikeHistory() { m_spike_history.clear(); }
        void EnableSTDP(bool enabled);
        std::size_t SparseRTRLStateSize() const;
        void AddNeuron(const Neuron &a_n) { m_neurons.push_back(a_n); }
        void AddConnection(const Connection &a_c) { m_connections.push_back(a_c); }
        Connection GetConnectionByIndex(unsigned int a_idx) const
        {
            return m_connections.at(a_idx);
        }
        Neuron GetNeuronByIndex(unsigned int a_idx) const
        {
            return m_neurons.at(a_idx);
        }
        void SetInputOutputDimentions(const unsigned int a_i, const unsigned int a_o)
        {
            m_num_inputs = a_i;
            m_num_outputs = a_o;
        }
        void SetInputOutputDimensions(
            const unsigned int inputs,
            const unsigned int outputs)
        {
            SetInputOutputDimentions(inputs, outputs);
        }
        unsigned int NumInputs() const { return m_num_inputs; }
        unsigned int NumOutputs() const { return m_num_outputs; }
        double GetConnectionLenght(Neuron source, Neuron target);
        double GetConnectionLength(
            const Neuron& source,
            const Neuron& target)
        {
            return GetConnectionLenght(source, target);
        }
        double GetTotalConnectionLength();
        void Save(const char* a_filename);
        bool Load(const char* a_filename);
        void Save(FILE *a_file);
        bool Load(std::ifstream &a_DataFile);

        void Clear()
        {
            m_neurons.clear();
            m_connections.clear();
            m_total_weight_change.clear();
            m_sparse_rtrl_sensitivities.clear();
            m_spike_history.clear();
            m_spiking_time = 0.0;
            m_total_error = 0.0;
            SetInputOutputDimentions(0, 0);
        }

        std::string Serialize() const;
        static NeuralNetwork Deserialize(const std::string &data);
    };
}
#endif
