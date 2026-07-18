#ifndef _PHENOTYPE_H
#define _PHENOTYPE_H

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "Genes.h"

namespace NEAT
{
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
            m_total_error = 0.0;
            SetInputOutputDimentions(0, 0);
        }

        std::string Serialize() const;
        static NeuralNetwork Deserialize(const std::string &data);
    };
}
#endif
