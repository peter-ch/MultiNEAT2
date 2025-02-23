#include <vector>
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
{
    m_leaky = false;
    m_with_distance = false;
    m_query_weights_only = false;
    m_hidden_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    m_output_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    m_allow_input_hidden_links = true;
    m_allow_input_output_links = false;
    m_allow_hidden_hidden_links = false;
    m_allow_hidden_output_links = true;
    m_allow_output_hidden_links = false;
    m_allow_output_output_links = false;
    m_allow_looped_hidden_links = false;
    m_allow_looped_output_links = false;

    m_max_weight_and_bias = 5.0;
    m_min_time_const = 0.1;
    m_max_time_const = 1.0;
    m_custom_conn_obeys_flags = false;

    m_input_coords = a_inputs;
    m_hidden_coords = a_hidden;
    m_output_coords = a_outputs;
}

void Substrate::SetCustomConnectivity(std::vector< std::vector<int> >& a_conns)
{
    for(unsigned int i=0; i<a_conns.size(); i++)
    {
    	NeuronType src_type = (NeuronType) a_conns[i][0];
    	int src_idx = a_conns[i][1];
    	NeuronType dst_type = (NeuronType) a_conns[i][2];
    	int dst_idx = a_conns[i][3];

    	std::vector<int> c;
    	c.emplace_back(src_type);
    	c.emplace_back(src_idx);
    	c.emplace_back(dst_type);
    	c.emplace_back(dst_idx);

    	m_custom_connectivity.emplace_back( c );
    }
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
    unsigned int max_dims = 0;
    for(unsigned int i=0; i<m_input_coords.size(); i++)
    {
        if (max_dims < m_input_coords[i].size())
        {
            max_dims = m_input_coords[i].size();
        }
    }
    for(unsigned int i=0; i<m_hidden_coords.size(); i++)
    {
        if (max_dims < m_hidden_coords[i].size())
        {
            max_dims = m_hidden_coords[i].size();
        }
    }
    for(unsigned int i=0; i<m_output_coords.size(); i++)
    {
        if (max_dims < m_output_coords[i].size())
        {
            max_dims = m_output_coords[i].size();
        }
    }
    return max_dims;
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

