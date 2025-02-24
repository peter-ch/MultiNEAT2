#ifndef _PHENOTYPE_H
#define _PHENOTYPE_H

#include <vector>
#include "Genes.h"

namespace NEAT
{
    class Connection
    {
    public:
        int m_source_neuron_idx;
        int m_target_neuron_idx;
        double m_weight;
        double m_signal;
        bool m_recur_flag;
        double m_hebb_rate;
        double m_hebb_pre_rate;

        bool operator==(const Connection &other) const
        {
            return (m_source_neuron_idx == other.m_source_neuron_idx &&
                    m_target_neuron_idx == other.m_target_neuron_idx);
        }
    };

    class Neuron
    {
    public:
        double m_activesum;
        double m_activation;
        double m_a, m_b, m_timeconst, m_bias;
        double m_membrane_potential;
        ActivationFunction m_activation_function_type;
        double m_x, m_y, m_z;
        double m_sx, m_sy, m_sz;
        std::vector<double> m_substrate_coords;
        double m_split_y;
        NeuronType m_type;
        std::vector<std::vector<double>> m_sensitivity_matrix;

        bool operator==(Neuron const &other) const
        {
            return (m_type == other.m_type &&
                    m_split_y == other.m_split_y &&
                    m_activation_function_type == other.m_activation_function_type);
        }
    };

    class NeuralNetwork
    {
        double m_total_error;
        std::vector<double> m_total_weight_change;

    public:
        unsigned int m_num_inputs, m_num_outputs;
        std::vector<Connection> m_connections;
        std::vector<Neuron> m_neurons;

        NeuralNetwork(bool a_Minimal);
        NeuralNetwork();
        void InitRTRLMatrix();
        void ActivateFast();
        void Activate();
        void ActivateUseInternalBias();
        void ActivateLeaky(double step);
        void RTRL_update_gradients();
        void RTRL_update_error(double a_target);
        void RTRL_update_weights();
        void Adapt(Parameters &a_Parameters);
        void Flush();
        void FlushCube();
        void Input(std::vector<double> &a_Inputs);
        std::vector<double> Output();
        void AddNeuron(const Neuron &a_n) { m_neurons.push_back(a_n); }
        void AddConnection(const Connection &a_c) { m_connections.push_back(a_c); }
        Connection GetConnectionByIndex(unsigned int a_idx) const
        {
            return m_connections[a_idx];
        }
        Neuron GetNeuronByIndex(unsigned int a_idx) const
        {
            return m_neurons[a_idx];
        }
        void SetInputOutputDimentions(const unsigned int a_i, const unsigned int a_o)
        {
            m_num_inputs = a_i;
            m_num_outputs = a_o;
        }
        unsigned int NumInputs() const { return m_num_inputs; }
        unsigned int NumOutputs() const { return m_num_outputs; }
        double GetConnectionLenght(Neuron source, Neuron target);
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
            SetInputOutputDimentions(0, 0);
        }
    };
}
#endif
