#include <vector>
#include <stdexcept>
#include "NeuralNetwork.h"
#include "Utils.h"
#include "Substrate.h"

using namespace std;


namespace NEAT
{

Substrate::Substrate()
{
    m_leaky = false;
    m_with_distance = false;
    m_custom_conn_obeys_flags = true;
    m_query_weights_only = false;
    m_allow_input_hidden_links = true;
    m_allow_input_output_links = true;
    m_allow_hidden_hidden_links = false;
    m_allow_hidden_output_links = true;
    m_allow_output_hidden_links = false;
    m_allow_output_output_links = false;
    m_allow_looped_hidden_links = false;
    m_allow_looped_output_links = false;
    m_hidden_nodes_activation = UNSIGNED_SIGMOID;
    m_output_nodes_activation = UNSIGNED_SIGMOID;
    m_max_weight_and_bias = 5.0;
    m_min_time_const = 0.1;
    m_max_time_const = 1.0;
};


Substrate::Substrate(std::vector<std::vector<double> >& a_inputs,
        std::vector<std::vector<double> >& a_hidden,
        std::vector<std::vector<double> >& a_outputs)
    : m_input_coords(a_inputs),
      m_hidden_coords(a_hidden),
      m_output_coords(a_outputs),
      m_leaky(false),
      m_with_distance(false),
      m_allow_input_hidden_links(true),
      m_allow_input_output_links(false),
      m_allow_hidden_hidden_links(false),
      m_allow_hidden_output_links(true),
      m_allow_output_hidden_links(false),
      m_allow_output_output_links(false),
      m_allow_looped_hidden_links(false),
      m_allow_looped_output_links(false),
      m_custom_connectivity(),
      m_custom_conn_obeys_flags(false),
      m_query_weights_only(false),
      m_hidden_nodes_activation(NEAT::UNSIGNED_SIGMOID),
      m_output_nodes_activation(NEAT::UNSIGNED_SIGMOID),
      m_max_weight_and_bias(5.0),
      m_min_time_const(0.1),
      m_max_time_const(1.0)
{
}

void Substrate::SetCustomConnectivity(std::vector< std::vector<int> >& a_conns)
{
    auto coordinate_count = [this](NeuronType type) -> std::size_t
    {
        switch (type)
        {
        case INPUT:
        case BIAS:
            return m_input_coords.size();
        case HIDDEN:
            return m_hidden_coords.size();
        case OUTPUT:
            return m_output_coords.size();
        default:
            throw std::invalid_argument(
                "Substrate::SetCustomConnectivity: invalid neuron type.");
        }
    };

    for (const auto& connection : a_conns)
    {
        if (connection.size() != 4)
            throw std::invalid_argument(
                "Substrate::SetCustomConnectivity: every connection must "
                "contain [source_type, source_index, target_type, target_index].");

        const auto source_type = static_cast<NeuronType>(connection[0]);
        const int source_index = connection[1];
        const auto target_type = static_cast<NeuronType>(connection[2]);
        const int target_index = connection[3];
        if (source_index < 0 ||
            static_cast<std::size_t>(source_index) >= coordinate_count(source_type) ||
            target_index < 0 ||
            static_cast<std::size_t>(target_index) >= coordinate_count(target_type))
        {
            throw std::out_of_range(
                "Substrate::SetCustomConnectivity: neuron index is out of range.");
        }
    }

    m_custom_connectivity = a_conns;
}

void Substrate::ClearCustomConnectivity()
{
	m_custom_connectivity.clear();
}

int Substrate::GetMinCPPNInputs()
{
    // determine the dimensionality across the entire substrate
    int cppn_inputs = GetMaxDims() * 2; // twice, because we query 2 points at a time

    // the distance input
    if (m_with_distance)
    {
        cppn_inputs += 1;
    }

    return cppn_inputs + 1; // always count the bias
}

int Substrate::GetMinCPPNOutputs()
{
	int outs = 0;
	if (m_query_weights_only)
	{
		outs = 1;
	}
	else
	{
		outs = 2; // (link on/off, weight)
	}
    if (m_leaky)
    {
        return outs+2; // + time_const and bias
    }
    else
    {
        return outs;
    }
}

int Substrate::GetMaxDims()
{
    std::size_t max_dims = 0;
    for(std::size_t i=0; i<m_input_coords.size(); i++)
    {
        if (max_dims < m_input_coords[i].size())
        {
            max_dims = m_input_coords[i].size();
        }
    }
    for(std::size_t i=0; i<m_hidden_coords.size(); i++)
    {
        if (max_dims < m_hidden_coords[i].size())
        {
            max_dims = m_hidden_coords[i].size();
        }
    }
    for(std::size_t i=0; i<m_output_coords.size(); i++)
    {
        if (max_dims < m_output_coords[i].size())
        {
            max_dims = m_output_coords[i].size();
        }
    }
    return static_cast<int>(max_dims);
}

void Substrate::PrintInfo()
{
    std::cerr << "Inputs: " << m_input_coords.size() << "\n";
    std::cerr << "Hidden: " << m_hidden_coords.size() << "\n";
    std::cerr << "Outputs: " << m_output_coords.size() << "\n\n";
    std::cerr << "Dimensions: " << GetMinCPPNInputs() << "\n";
}
// namespace NEAT

}

