#ifndef _SUBSTRATE_H
#define _SUBSTRATE_H

#include <vector>
#include "NeuralNetwork.h"

namespace NEAT
{

//-----------------------------------------------------------------------
// The substrate describes the phenotype space that is used by HyperNEAT
// It basically contains 3 lists of coordinates - for the nodes.
class Substrate
{
public:
    std::vector< std::vector<double> > m_input_coords;
    std::vector< std::vector<double> > m_hidden_coords;
    std::vector< std::vector<double> > m_output_coords;

    // the substrate is made from leaky integrator neurons?
    bool m_leaky;

    // the additional distance input is used?
    // NOTE: don't use it, not working yet
    bool m_with_distance;

    // these flags control the connectivity of the substrate
    bool m_allow_input_hidden_links;
    bool m_allow_input_output_links;
    bool m_allow_hidden_hidden_links;
    bool m_allow_hidden_output_links;
    bool m_allow_output_hidden_links;
    bool m_allow_output_output_links;
    bool m_allow_looped_hidden_links;
    bool m_allow_looped_output_links;
    
    // custom connectivity
    // if this is not empty, the phenotype builder will use this
    // to query all connections
    // it's a list of [src_code, src_idx, dst_code, dst_idx]
    // where code is NeuronType (int, the enum)
    // and idx is the index in the m_input_coords, m_hidden_coords and m_output_coords respectively
    std::vector< std::vector<int> > m_custom_connectivity;
    bool m_custom_conn_obeys_flags; // if this is true, the flags restricting the topology above will still apply

    // this enforces custom or full connectivity
    // if it is true, connections are always made and the weights will be queried only
    bool m_query_weights_only;

    // the activation functions of hidden/output neurons
    ActivationFunction m_hidden_nodes_activation;
    ActivationFunction m_output_nodes_activation;

    // additional parameters
    double m_max_weight_and_bias;
    double m_min_time_const;
    double m_max_time_const;


    Substrate();
    Substrate(std::vector< std::vector<double> >& a_inputs,
              std::vector< std::vector<double> >& a_hidden,
              std::vector< std::vector<double> >& a_outputs );

    // Sets a custom connectivity scheme
    // The neurons must be set before calling this
    void SetCustomConnectivity(std::vector< std::vector<int> >& a_conns);

    // Clears it
    void ClearCustomConnectivity();

    int GetMaxDims();

    // Return the minimum input dimensionality of the CPPN
    int GetMinCPPNInputs();
    // Return the minimum output dimensionality of the CPPN
    int GetMinCPPNOutputs();
    
    // Prints some info about itself
    void PrintInfo();
    

};

}

#endif

