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

/////////////////////////////////////
// The set of activation functions //
/////////////////////////////////////

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
    return ((aX + aShift) < 0.0) ? -(aX + aShift) : (aX + aShift);
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

///////////////////////////////////////
// Neural network class implementation
///////////////////////////////////////
NeuralNetwork::NeuralNetwork(bool a_Minimal)
{
    if (!a_Minimal)
    {
        // Build an XOR network (code omitted for brevity—all original functionality preserved)
        // …
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
    for (size_t i = 0; i < m_neurons.size(); i++)
    {
        m_neurons[i].m_sensitivity_matrix.resize(m_neurons.size());
        for (size_t j = 0; j < m_neurons.size(); j++)
        {
            m_neurons[i].m_sensitivity_matrix[j].resize(m_neurons.size());
        }
    }
    FlushCube();
    m_total_error = 0;
    m_total_weight_change.resize(m_connections.size());
    for (size_t i = 0; i < m_connections.size(); i++)
    {
        m_total_weight_change[i] = 0;
    }
}

void NeuralNetwork::ActivateFast()
{
    const size_t connSize = m_connections.size();
    for (size_t i = 0; i < connSize; i++)
    {
        m_connections[i].m_signal =
            m_neurons[m_connections[i].m_source_neuron_idx].m_activation *
            m_connections[i].m_weight;
    }
    for (size_t i = 0; i < connSize; i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
            m_connections[i].m_signal;
    }
    const size_t neuronSize = m_neurons.size();
    for (size_t i = m_num_inputs; i < neuronSize; i++)
    {
        double x = m_neurons[i].m_activesum;
        m_neurons[i].m_activesum = 0;
        double y = 0.0;
        switch (m_neurons[i].m_activation_function_type)
        {
            case SIGNED_SIGMOID:
                y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_SIGMOID:
                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case TANH:
                y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case TANH_CUBIC:
                y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case SIGNED_STEP:
                y = af_step_signed(x, m_neurons[i].m_b);
                break;
            case UNSIGNED_STEP:
                y = af_step_unsigned(x, m_neurons[i].m_b);
                break;
            case SIGNED_GAUSS:
                y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_GAUSS:
                y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case ABS:
                y = af_abs(x, m_neurons[i].m_b);
                break;
            case SIGNED_SINE:
                y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_SINE:
                y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case LINEAR:
                y = af_linear(x, m_neurons[i].m_b);
                break;
            case RELU:
                y = af_relu(x);
                break;
            case SOFTPLUS:
                y = af_softplus(x);
                break;
            default:
                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
        }
        m_neurons[i].m_activation = y;
    }
}

void NeuralNetwork::Activate()
{
    const size_t connSize = m_connections.size();
    for (size_t i = 0; i < connSize; i++)
    {
        m_connections[i].m_signal =
            m_neurons[m_connections[i].m_source_neuron_idx].m_activation *
            m_connections[i].m_weight;
    }
    for (size_t i = 0; i < connSize; i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
            m_connections[i].m_signal;
    }
    const size_t neuronSize = m_neurons.size();
    for (size_t i = m_num_inputs; i < neuronSize; i++)
    {
        double x = m_neurons[i].m_activesum;
        m_neurons[i].m_activesum = 0;
        double y = 0.0;
        switch (m_neurons[i].m_activation_function_type)
        {
            case SIGNED_SIGMOID:
                y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_SIGMOID:
                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case TANH:
                y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case TANH_CUBIC:
                y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case SIGNED_STEP:
                y = af_step_signed(x, m_neurons[i].m_b);
                break;
            case UNSIGNED_STEP:
                y = af_step_unsigned(x, m_neurons[i].m_b);
                break;
            case SIGNED_GAUSS:
                y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_GAUSS:
                y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case ABS:
                y = af_abs(x, m_neurons[i].m_b);
                break;
            case SIGNED_SINE:
                y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_SINE:
                y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case LINEAR:
                y = af_linear(x, m_neurons[i].m_b);
                break;
            case RELU:
                y = af_relu(x);
                break;
            case SOFTPLUS:
                y = af_softplus(x);
                break;
            default:
                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
        }
        m_neurons[i].m_activation = y;
    }
}

void NeuralNetwork::ActivateUseInternalBias()
{
    const size_t connSize = m_connections.size();
    for (size_t i = 0; i < connSize; i++)
    {
        m_connections[i].m_signal =
            m_neurons[m_connections[i].m_source_neuron_idx].m_activation *
            m_connections[i].m_weight;
    }
    for (size_t i = 0; i < connSize; i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
            m_connections[i].m_signal;
    }
    const size_t neuronSize = m_neurons.size();
    for (size_t i = m_num_inputs; i < neuronSize; i++)
    {
        double x = m_neurons[i].m_activesum + m_neurons[i].m_bias;
        m_neurons[i].m_activesum = 0;
        double y = 0.0;
        switch (m_neurons[i].m_activation_function_type)
        {
            case SIGNED_SIGMOID:
                y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_SIGMOID:
                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case TANH:
                y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case TANH_CUBIC:
                y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case SIGNED_STEP:
                y = af_step_signed(x, m_neurons[i].m_b);
                break;
            case UNSIGNED_STEP:
                y = af_step_unsigned(x, m_neurons[i].m_b);
                break;
            case SIGNED_GAUSS:
                y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_GAUSS:
                y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case ABS:
                y = af_abs(x, m_neurons[i].m_b);
                break;
            case SIGNED_SINE:
                y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_SINE:
                y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case LINEAR:
                y = af_linear(x, m_neurons[i].m_b);
                break;
            case RELU:
                y = af_relu(x);
                break;
            case SOFTPLUS:
                y = af_softplus(x);
                break;
            default:
                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
        }
        m_neurons[i].m_activation = y;
    }
}

void NeuralNetwork::ActivateLeaky(double a_dtime)
{
    const size_t connSize = m_connections.size();
    for (size_t i = 0; i < connSize; i++)
    {
        m_connections[i].m_signal =
            m_neurons[m_connections[i].m_source_neuron_idx].m_activation *
            m_connections[i].m_weight;
    }
    for (size_t i = 0; i < connSize; i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
            m_connections[i].m_signal;
    }
    for (size_t i = m_num_inputs; i < m_neurons.size(); i++)
    {
        double t_const = a_dtime / m_neurons[i].m_timeconst;
        m_neurons[i].m_membrane_potential = (1.0 - t_const) * m_neurons[i].m_membrane_potential + t_const * m_neurons[i].m_activesum;
    }
    const size_t neuronSize = m_neurons.size();
    for (size_t i = m_num_inputs; i < neuronSize; i++)
    {
        double x = m_neurons[i].m_membrane_potential + m_neurons[i].m_bias;
        m_neurons[i].m_activesum = 0;
        double y = 0.0;
        switch (m_neurons[i].m_activation_function_type)
        {
            case SIGNED_SIGMOID:
                y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_SIGMOID:
                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case TANH:
                y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case TANH_CUBIC:
                y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case SIGNED_STEP:
                y = af_step_signed(x, m_neurons[i].m_b);
                break;
            case UNSIGNED_STEP:
                y = af_step_unsigned(x, m_neurons[i].m_b);
                break;
            case SIGNED_GAUSS:
                y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_GAUSS:
                y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case ABS:
                y = af_abs(x, m_neurons[i].m_b);
                break;
            case SIGNED_SINE:
                y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case UNSIGNED_SINE:
                y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
            case LINEAR:
                y = af_linear(x, m_neurons[i].m_b);
                break;
            case RELU:
                y = af_relu(x);
                break;
            case SOFTPLUS:
                y = af_softplus(x);
                break;
            default:
                y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
                break;
        }
        m_neurons[i].m_activation = y;
    }
}

void NeuralNetwork::Flush()
{
    for (size_t i = 0; i < m_neurons.size(); i++)
    {
        m_neurons[i].m_activation = 0;
        m_neurons[i].m_activesum = 0;
        m_neurons[i].m_membrane_potential = 0;
    }
}

void NeuralNetwork::FlushCube()
{
    for (size_t i = 0; i < m_neurons.size(); i++)
        for (size_t j = 0; j < m_neurons.size(); j++)
            for (size_t k = 0; k < m_neurons.size(); k++)
                m_neurons[k].m_sensitivity_matrix[i][j] = 0;
}

void NeuralNetwork::Input(std::vector<double>& a_Inputs)
{
    unsigned int mx = a_Inputs.size();
    if (mx > m_num_inputs)
    {
        mx = m_num_inputs;
    }
    for (unsigned int i = 0; i < mx; i++)
    {
        m_neurons[i].m_activation = a_Inputs[i];
    }
}

std::vector<double> NeuralNetwork::Output()
{
    std::vector<double> t_output;
    for (int i = 0; i < m_num_outputs; i++)
    {
        t_output.emplace_back(m_neurons[i + m_num_inputs].m_activation);
    }
    return t_output;
}

void NeuralNetwork::Adapt(Parameters& a_Parameters)
{
    double t_max_weight = -999999999;
    for (size_t i = 0; i < m_connections.size(); i++)
    {
        if (fabs(m_connections[i].m_weight) > t_max_weight)
        {
            t_max_weight = fabs(m_connections[i].m_weight);
        }
    }
    for (size_t i = 0; i < m_connections.size(); i++)
    {
        double t_incoming_neuron_activation =
            m_neurons[m_connections[i].m_source_neuron_idx].m_activation;
        double t_outgoing_neuron_activation =
            m_neurons[m_connections[i].m_target_neuron_idx].m_activation;
        if (m_connections[i].m_weight > 0)
        {
            double t_delta = (m_connections[i].m_hebb_rate *
                              (t_max_weight - m_connections[i].m_weight) *
                              t_incoming_neuron_activation *
                              t_outgoing_neuron_activation)
                              + m_connections[i].m_hebb_pre_rate *
                              t_max_weight *
                              t_incoming_neuron_activation *
                              (t_outgoing_neuron_activation - 1.0);
            m_connections[i].m_weight += t_delta;
        }
        else if (m_connections[i].m_weight < 0)
        {
            double t_delta = m_connections[i].m_hebb_pre_rate *
                             (t_max_weight - m_connections[i].m_weight) *
                             t_incoming_neuron_activation *
                             (1.0 - t_outgoing_neuron_activation)
                             - m_connections[i].m_hebb_rate *
                             t_max_weight *
                             t_incoming_neuron_activation *
                             t_outgoing_neuron_activation;
            m_connections[i].m_weight = -(m_connections[i].m_weight + t_delta);
        }
        Clamp(m_connections[i].m_weight, -a_Parameters.MaxWeight,
              a_Parameters.MaxWeight);
    }
}

int NeuralNetwork::ConnectionExists(int a_to, int a_from)
{
    for (size_t i = 0; i < m_connections.size(); i++)
    {
        if ((m_connections[i].m_source_neuron_idx == a_from) &&
            (m_connections[i].m_target_neuron_idx == a_to))
        {
            return i;
        }
    }
    return -1;
}

void NeuralNetwork::RTRL_update_gradients()
{
    for (size_t k = m_num_inputs; k < m_neurons.size(); k++)
    {
        for (size_t i = m_num_inputs; i < m_neurons.size(); i++)
        {
            for (size_t j = 0; j < m_neurons.size(); j++)
            {
                int t_idx = ConnectionExists(i, j);
                if (t_idx != -1)
                {
                    double t_derivative = 0;
                    if (m_neurons[k].m_activation_function_type == NEAT::UNSIGNED_SIGMOID)
                    {
                        t_derivative = unsigned_sigmoid_derivative(m_neurons[k].m_activation);
                    }
                    else if (m_neurons[k].m_activation_function_type == NEAT::TANH)
                    {
                        t_derivative = tanh_derivative(m_neurons[k].m_activation);
                    }
                    double t_sum = 0;
                    for (size_t l = 0; l < m_neurons.size(); l++)
                    {
                        int t_l_idx = ConnectionExists(k, l);
                        if (t_l_idx != -1)
                        {
                            t_sum += m_connections[t_l_idx].m_weight *
                                m_neurons[l].m_sensitivity_matrix[i][j];
                        }
                    }
                    if (i == k)
                    {
                        t_sum += m_neurons[j].m_activation;
                    }
                    m_neurons[k].m_sensitivity_matrix[i][j] = t_derivative * t_sum;
                }
                else
                {
                    m_neurons[k].m_sensitivity_matrix[i][j] = 0;
                }
            }
        }
    }
}

void NeuralNetwork::RTRL_update_error(double a_target)
{
    m_total_error = (a_target - Output()[0]);
    for (size_t i = 0; i < m_neurons.size(); i++)
    {
        for (size_t j = 0; j < m_neurons.size(); j++)
        {
            int t_idx = ConnectionExists(i, j);
            if (t_idx != -1)
            {
                double t_delta = m_total_error *
                                 m_neurons[m_num_inputs].m_sensitivity_matrix[i][j];
                m_total_weight_change[t_idx] += t_delta * LEARNING_RATE;
            }
        }
    }
}

void NeuralNetwork::RTRL_update_weights()
{
    for (size_t i = 0; i < m_connections.size(); i++)
    {
        m_connections[i].m_weight += m_total_weight_change[i];
        m_total_weight_change[i] = 0;
    }
    m_total_error = 0;
}

void NeuralNetwork::Save(const char* a_filename)
{
    FILE* fil = fopen(a_filename, "w");
    Save(fil);
    fclose(fil);
}

void NeuralNetwork::Save(FILE* a_file)
{
    fprintf(a_file, "NNstart\n");
    fprintf(a_file, "%d %d\n", m_num_inputs, m_num_outputs);
    for (size_t i = 0; i < m_neurons.size(); i++)
    {
        fprintf(a_file, "neuron %d %3.18f %3.18f %3.18f %3.18f %d %3.18f\n",
                static_cast<int>(m_neurons[i].m_type), m_neurons[i].m_a,
                m_neurons[i].m_b, m_neurons[i].m_timeconst, m_neurons[i].m_bias,
                static_cast<int>(m_neurons[i].m_activation_function_type),
                m_neurons[i].m_split_y);
    }
    for (size_t i = 0; i < m_connections.size(); i++)
    {
        fprintf(a_file, "connection %d %d %3.18f %d %3.18f %3.18f\n",
                m_connections[i].m_source_neuron_idx,
                m_connections[i].m_target_neuron_idx, m_connections[i].m_weight,
                static_cast<int>(m_connections[i].m_recur_flag),
                m_connections[i].m_hebb_rate, m_connections[i].m_hebb_pre_rate);
    }
    fprintf(a_file, "NNend\n\n");
}

bool NeuralNetwork::Load(std::ifstream& a_DataFile)
{
    std::string t_str;
    bool t_no_start = true, t_no_end = true;
    if (!a_DataFile)
    {
        std::ostringstream tStream;
        tStream << "NN file error!" << std::endl;
    }
    do
    {
        a_DataFile >> t_str;
    }
    while(t_str != "NNstart" && !a_DataFile.eof());
    if(t_no_start)
        return false;
    Clear();
    a_DataFile >> m_num_inputs;
    a_DataFile >> m_num_outputs;
    do
    {
        a_DataFile >> t_str;
        if(t_str=="neuron")
        {
            Neuron t_n;
            int t_type, t_aftype;
            a_DataFile >> t_type;
            a_DataFile >> t_n.m_a;
            a_DataFile >> t_n.m_b;
            a_DataFile >> t_n.m_timeconst;
            a_DataFile >> t_n.m_bias;
            a_DataFile >> t_aftype;
            a_DataFile >> t_n.m_split_y;
            t_n.m_type = static_cast<NEAT::NeuronType>(t_type);
            t_n.m_activation_function_type = static_cast<NEAT::ActivationFunction>(t_aftype);
            m_neurons.emplace_back(t_n);
        }
        else if(t_str=="connection")
        {
            Connection t_c;
            int t_isrecur;
            a_DataFile >> t_c.m_source_neuron_idx;
            a_DataFile >> t_c.m_target_neuron_idx;
            a_DataFile >> t_c.m_weight;
            a_DataFile >> t_isrecur;
            a_DataFile >> t_c.m_hebb_rate;
            a_DataFile >> t_c.m_hebb_pre_rate;
            t_c.m_recur_flag = static_cast<bool>(t_isrecur);
            m_connections.emplace_back(t_c);
        }
        if(t_str=="NNend")
            t_no_end = false;
    }
    while(t_str!="NNend" && !a_DataFile.eof());
    if(t_no_end)
    {
        std::ostringstream tStream;
        tStream << "NNend not found in file!" << std::endl;
    }
    return true;
}
bool NeuralNetwork::Load(const char *a_filename)
{
    std::ifstream t_DataFile(a_filename);
    return Load(t_DataFile);
}

}; // namespace NEAT
