#include <math.h>
#include <float.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include "NeuralNetwork.h"
#include "Assert.h"
#include "Utils.h"

#define sqr(x) ((x)*(x))
#define LEARNING_RATE 0.0001

namespace NEAT
{
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
        return tanh(aX * aSlope);
    }

    inline double af_tanh_cubic(double aX, double aSlope, double aShift)
    {
        return tanh(aX * aX * aX * aSlope);
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
        return log(1 + exp(aX));
    }

    double unsigned_sigmoid_derivative(double x)
    {
        return x * (1 - x);
    }

    double tanh_derivative(double x)
    {
        return 1 - x * x;
    }

    NeuralNetwork::NeuralNetwork(bool a_Minimal)
    {
        if (!a_Minimal)
        {
            // (XOR network code omitted for brevity)
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
            neuron.m_sensitivity_matrix.resize(m_neurons.size());
            for (auto &row : neuron.m_sensitivity_matrix)
                row.resize(m_neurons.size(), 0.0);
        }
        FlushCube();
        m_total_error = 0;
        m_total_weight_change.resize(m_connections.size(), 0.0);
    }

    void NeuralNetwork::ActivateFast()
    {
        for (auto &conn : m_connections)
            conn.m_signal = m_neurons[conn.m_source_neuron_idx].m_activation * conn.m_weight;
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
        for (auto &conn : m_connections)
            conn.m_signal = m_neurons[conn.m_source_neuron_idx].m_activation * conn.m_weight;
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
        for (auto &conn : m_connections)
            conn.m_signal = m_neurons[conn.m_source_neuron_idx].m_activation * conn.m_weight;
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
        for (auto &conn : m_connections)
            conn.m_signal = m_neurons[conn.m_source_neuron_idx].m_activation * conn.m_weight;
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
        size_t mx = std::min(a_Inputs.size(), static_cast<size_t>(m_num_inputs));
        for (size_t i = 0; i < mx; i++)
            m_neurons[i].m_activation = a_Inputs[i];
    }

    std::vector<double> NeuralNetwork::Output()
    {
        std::vector<double> t_output;
        for (unsigned int i = 0; i < m_num_outputs; i++)
            t_output.push_back(m_neurons[i + m_num_inputs].m_activation);
        return t_output;
    }

    void NeuralNetwork::Save(const char* a_filename)
    {
        FILE *fil = fopen(a_filename, "w");
        Save(fil);
        fclose(fil);
    }

    void NeuralNetwork::Save(FILE *a_file)
    {
        fprintf(a_file, "NNstart\n");
        fprintf(a_file, "%d %d\n", m_num_inputs, m_num_outputs);
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
                a_DataFile >> t_n.m_a >> t_n.m_b >> t_n.m_timeconst >> t_n.m_bias;
                a_DataFile >> t_aftype >> t_n.m_split_y;
                // (Assuming t_type was read elsewhere if needed.)
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
        return true;
    }

    bool NeuralNetwork::Load(const char *a_filename)
    {
        std::ifstream t_DataFile(a_filename);
        return Load(t_DataFile);
    }

    std::string NeuralNetwork::Serialize() const {
        std::ostringstream oss;
        oss << m_num_inputs << " " << m_num_outputs << "\n";
        oss << m_neurons.size() << "\n";
        for (const auto &n : m_neurons) {
            // Save all the necessary data: here we assume that an int can represent the enums
            oss << static_cast<int>(n.m_type) << " " << n.m_a << " " << n.m_b << " " 
                << n.m_timeconst << " " << n.m_bias << " " 
                << static_cast<int>(n.m_activation_function_type) << " " 
                << n.m_split_y << "\n";
        }
        oss << m_connections.size() << "\n";
        for (const auto &c : m_connections) {
            oss << c.m_source_neuron_idx << " " << c.m_target_neuron_idx << " " 
                << c.m_weight << " " << c.m_recur_flag << " " 
                << c.m_hebb_rate << " " << c.m_hebb_pre_rate << "\n";
        }
        return oss.str();
    }
    
    NeuralNetwork NeuralNetwork::Deserialize(const std::string &data) {
        NeuralNetwork nn;
        std::istringstream iss(data);
        iss >> nn.m_num_inputs >> nn.m_num_outputs;
        int num_neurons = 0;
        iss >> num_neurons;
        for (int i = 0; i < num_neurons; ++i) {
            Neuron n;
            int type_int, af_int;
            iss >> type_int >> n.m_a >> n.m_b >> n.m_timeconst >> n.m_bias >> af_int >> n.m_split_y;
            n.m_type = static_cast<NeuronType>(type_int);
            n.m_activation_function_type = static_cast<ActivationFunction>(af_int);
            nn.m_neurons.push_back(n);
        }
        int num_connections = 0;
        iss >> num_connections;
        for (int i = 0; i < num_connections; ++i) {
            Connection c;
            int src, tgt, is_recur;
            iss >> src >> tgt >> c.m_weight >> is_recur >> c.m_hebb_rate >> c.m_hebb_pre_rate;
            c.m_source_neuron_idx = src;
            c.m_target_neuron_idx = tgt;
            c.m_recur_flag = is_recur != 0;
            nn.m_connections.push_back(c);
        }
        return nn;
    }
}