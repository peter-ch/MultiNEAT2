#include <algorithm>
#include <fstream>
#include <queue>
#include <cmath>
#include <stdexcept> // for runtime_error
#include "Genome.h"
#include "Random.h"
#include "Utils.h"
#include "Parameters.h"
#include "Assert.h"
#include "Substrate.h"  // For BuildHyperNEATPhenotype usage

namespace NEAT
{
    // ==============================================================
    // A small helper function to detect cycles in a directed graph.
    //
    // color array: 
    //    0 = unvisited,
    //    1 = visiting,
    //    2 = visited
    // ==============================================================
    static bool HasCycleDFS(int current, std::vector<int> &color, 
                            const std::vector<std::vector<int>> &adj)
    {
        color[current] = 1; // visiting
        for (int nxt : adj[current])
        {
            if (color[nxt] == 1)
            {
                // We found a node that is currently being visited => cycle
                return true;
            }
            if (color[nxt] == 0)
            {
                if (HasCycleDFS(nxt, color, adj))
                {
                    return true;
                }
            }
        }
        color[current] = 2; // visited
        return false;
    }

    // forward
    // ActivationFunction GetRandomActivation(Parameters &a_Parameters, RNG &a_RNG);

    // A helper for squared values
    inline double sqr(double x) { return x * x; }

    // ==============================================================
    // Genome constructor: empty
    // ==============================================================
    Genome::Genome()
    {
        m_ID = 0;
        m_Fitness = 0.0;
        m_AdjustedFitness = 0.0;
        m_OffspringAmount = 0.0;
        m_Depth = 0;
        m_NumInputs = 0;
        m_NumOutputs = 0;
        m_Evaluated = false;
        m_PhenotypeBehavior = nullptr;
        m_initial_num_neurons = 0;
        m_initial_num_links   = 0;
    }

    // ==============================================================
    // Copy constructor
    // ==============================================================
    Genome::Genome(const Genome &a_G)
    {
        m_ID                = a_G.m_ID;
        m_Fitness           = a_G.m_Fitness;
        m_AdjustedFitness   = a_G.m_AdjustedFitness;
        m_OffspringAmount   = a_G.m_OffspringAmount;
        m_Depth             = a_G.m_Depth;
        m_NumInputs         = a_G.m_NumInputs;
        m_NumOutputs        = a_G.m_NumOutputs;
        m_Evaluated         = a_G.m_Evaluated;
        m_PhenotypeBehavior = a_G.m_PhenotypeBehavior;

        m_NeuronGenes       = a_G.m_NeuronGenes;
        m_LinkGenes         = a_G.m_LinkGenes;
        m_GenomeGene        = a_G.m_GenomeGene;

        m_initial_num_neurons = a_G.m_initial_num_neurons;
        m_initial_num_links   = a_G.m_initial_num_links;
    }

    // ==============================================================
    // Assignment operator
    // ==============================================================
    Genome &Genome::operator=(const Genome &a_G)
    {
        if (this != &a_G)
        {
            m_ID                = a_G.m_ID;
            m_Fitness           = a_G.m_Fitness;
            m_AdjustedFitness   = a_G.m_AdjustedFitness;
            m_OffspringAmount   = a_G.m_OffspringAmount;
            m_Depth             = a_G.m_Depth;
            m_NumInputs         = a_G.m_NumInputs;
            m_NumOutputs        = a_G.m_NumOutputs;
            m_Evaluated         = a_G.m_Evaluated;
            m_PhenotypeBehavior = a_G.m_PhenotypeBehavior;

            m_NeuronGenes       = a_G.m_NeuronGenes;
            m_LinkGenes         = a_G.m_LinkGenes;
            m_GenomeGene        = a_G.m_GenomeGene;

            m_initial_num_neurons = a_G.m_initial_num_neurons;
            m_initial_num_links   = a_G.m_initial_num_links;
        }
        return *this;
    }

    // ==============================================================
    // A constructor that builds a perceptron-like or layered genome
    // depending on the input struct (GenomeInitStruct).
    // ==============================================================
    Genome::Genome(const Parameters &a_Parameters, const GenomeInitStruct &in)
    {
        ASSERT(in.NumInputs > 1 && in.NumOutputs > 0);

        RNG t_RNG;
        t_RNG.TimeSeed();

        m_ID = 0;
        m_Fitness = 0.0;
        m_AdjustedFitness = 0.0;
        m_OffspringAmount = 0.0;
        m_Depth = 0;
        m_PhenotypeBehavior = nullptr;
        m_initial_num_neurons = 0;
        m_initial_num_links   = 0;

        int t_innovnum = 1;
        int t_nnum     = 1;

        GenomeSeedType seed_type = in.SeedType;
        // If user says LAYERED but Hidden=0, override with PERCEPTRON:
        if ((seed_type == LAYERED) && (in.NumHidden == 0))
        {
            seed_type = PERCEPTRON;
        }

        // If not ignoring bias, the last input is the bias
        if (!a_Parameters.DontUseBiasNeuron)
        {
            // normal input neurons
            for (unsigned int i = 0; i < (in.NumInputs - 1); i++)
            {
                NeuronGene n(INPUT, t_nnum, 0.0);
                m_NeuronGenes.push_back(n);
                t_nnum++;
            }
            // add the bias
            NeuronGene biasnode(BIAS, t_nnum, 0.0);
            m_NeuronGenes.push_back(biasnode);
            t_nnum++;
        }
        else
        {
            // no special bias
            for (unsigned int i = 0; i < in.NumInputs; i++)
            {
                NeuronGene n(INPUT, t_nnum, 0.0);
                m_NeuronGenes.push_back(n);
                t_nnum++;
            }
        }

        // now outputs
        for (unsigned int i = 0; i < in.NumOutputs; i++)
        {
            NeuronGene outnode(OUTPUT, t_nnum, 1.0);

            // initialize some defaults
            outnode.Init( (a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0,
                          (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0,
                          (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant)/2.0,
                          (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias)/2.0,
                           in.OutputActType );

            // randomize traits
            outnode.InitTraits(a_Parameters.NeuronTraits, t_RNG);

            m_NeuronGenes.push_back(outnode);
            t_nnum++;
        }

        // If LAYERED with hidden:
        if ((seed_type == LAYERED) && (in.NumHidden > 0))
        {
            double lt_inc  = 1.0 / (in.NumLayers + 1);
            double initlt  = lt_inc;
            for (unsigned int lay = 0; lay < in.NumLayers; lay++)
            {
                for (unsigned int i = 0; i < in.NumHidden; i++)
                {
                    NeuronGene hidden(HIDDEN, t_nnum, 1.0);
                    hidden.Init( (a_Parameters.MinActivationA + a_Parameters.MaxActivationA)/2.0,
                                 (a_Parameters.MinActivationB + a_Parameters.MaxActivationB)/2.0,
                                 (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant)/2.0,
                                 (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias)/2.0,
                                  in.HiddenActType );

                    hidden.InitTraits(a_Parameters.NeuronTraits, t_RNG);
                    hidden.m_SplitY = initlt;

                    m_NeuronGenes.push_back(hidden);
                    t_nnum++;
                }
                initlt += lt_inc;
            }

            // Connect them if not FS_NEAT
            if (!in.FS_NEAT)
            {
                int last_dest_id     = in.NumInputs + in.NumOutputs + 1;
                int last_src_id      = 1;
                int prev_layer_size  = in.NumInputs;

                // for each hidden layer
                for (unsigned int ly = 0; ly < in.NumLayers; ly++)
                {
                    for (unsigned int i = 0; i < in.NumHidden; i++)
                    {
                        for (unsigned int j = 0; j < prev_layer_size; j++)
                        {
                            // create link
                            LinkGene L(j+last_src_id, i+last_dest_id, t_innovnum, 0.0, false);
                            L.InitTraits(a_Parameters.LinkTraits, t_RNG);
                            m_LinkGenes.push_back(L);
                            t_innovnum++;
                        }
                    }
                    last_dest_id += in.NumHidden;
                    if (ly == 0)
                    {
                        // for the first hidden layer, skip output neurons in indexing
                        last_src_id += (prev_layer_size + in.NumOutputs);
                    }
                    else
                    {
                        last_src_id += prev_layer_size;
                    }
                    prev_layer_size = in.NumHidden;
                }

                // now connect last hidden layer to outputs
                last_dest_id = in.NumInputs + 1; // index of first output in the genome
                for (unsigned int i = 0; i < in.NumOutputs; i++)
                {
                    for (unsigned int j = 0; j < prev_layer_size; j++)
                    {
                        LinkGene L(j+last_src_id, i+last_dest_id, t_innovnum, 0.0, false);
                        L.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.push_back(L);
                        t_innovnum++;
                    }
                }
            }
        }
        else
        {
            // Perceptron or FS-NEAT
            if ((!in.FS_NEAT) && (seed_type == PERCEPTRON))
            {
                // fully connect inputs to outputs
                for (unsigned int i = 0; i < in.NumOutputs; i++)
                {
                    for (unsigned int j = 0; j < in.NumInputs; j++)
                    {
                        LinkGene L(j+1, i + in.NumInputs + 1, t_innovnum, 0.0, false);
                        L.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.push_back(L);
                        t_innovnum++;
                    }
                }
            }
            else
            {
                // minimal FS-NEAT
                std::vector< std::pair<int,int> > used;
                bool found=false;
                int linkcount=0;

                while (linkcount < in.FS_NEAT_links)
                {
                    for (unsigned int i = 0; i < in.NumOutputs; i++)
                    {
                        int t_inp_id = t_RNG.RandInt(1, in.NumInputs - 1);
                        int t_bias_id = in.NumInputs;
                        int t_out_id  = in.NumInputs + 1 + i;

                        found = false;
                        for (auto &p: used)
                        {
                            if (p.first == t_inp_id && p.second == t_out_id)
                            {
                                found=true;
                                break;
                            }
                        }
                        if (!found)
                        {
                            // create
                            LinkGene L(t_inp_id, t_out_id, t_innovnum, 0.0, false);
                            L.InitTraits(a_Parameters.LinkTraits, t_RNG);
                            m_LinkGenes.push_back(L);
                            t_innovnum++;

                            if (!a_Parameters.DontUseBiasNeuron)
                            {
                                LinkGene BL(t_bias_id, t_out_id, t_innovnum, 0.0, false);
                                BL.InitTraits(a_Parameters.LinkTraits, t_RNG);
                                m_LinkGenes.push_back(BL);
                                t_innovnum++;
                            }
                            used.push_back(std::make_pair(t_inp_id,t_out_id));
                            linkcount++;
                        }
                    }
                }
            }
        }

        if (in.FS_NEAT && (in.FS_NEAT_links==1))
        {
            throw std::runtime_error("FS-NEAT with exactly 1 link & 1/1/1 is not recommended.");
        }

        // Init the genome-level traits
        m_GenomeGene.InitTraits(a_Parameters.GenomeTraits, t_RNG);

        // finalize
        m_Evaluated     = false;
        m_NumInputs     = in.NumInputs;
        m_NumOutputs    = in.NumOutputs;
        m_initial_num_neurons = NumNeurons();
        m_initial_num_links   = NumLinks();
    }

    // ==============================================================
    // Various set/get
    // ==============================================================
    void   Genome::SetDepth(unsigned int a_d)     { m_Depth = a_d; }
    unsigned int Genome::GetDepth() const        { return m_Depth; }
    void   Genome::SetID(int a_id)               { m_ID = a_id; }
    int    Genome::GetID() const                 { return m_ID; }

    void   Genome::SetAdjFitness(double a_af)    { m_AdjustedFitness = a_af; }
    void   Genome::SetFitness(double a_f)        { m_Fitness = a_f; }
    double Genome::GetAdjFitness() const         { return m_AdjustedFitness; }
    double Genome::GetFitness() const            { return m_Fitness; }

    void   Genome::SetNeuronY(unsigned int idx, int val)
    {
        ASSERT(idx<m_NeuronGenes.size());
        m_NeuronGenes[idx].y = val;
    }

    void   Genome::SetNeuronX(unsigned int idx, int val)
    {
        ASSERT(idx<m_NeuronGenes.size());
        m_NeuronGenes[idx].x = val;
    }

    void   Genome::SetNeuronXY(unsigned int idx, int x, int y)
    {
        ASSERT(idx<m_NeuronGenes.size());
        m_NeuronGenes[idx].x = x;
        m_NeuronGenes[idx].y = y;
    }

    // Returns true is the specified neuron ID is a dead end or isolated
    bool Genome::IsDeadEndNeuron(int a_ID) const
    {
        bool t_no_incoming = true;
        bool t_no_outgoing = true;

        // search the links and prove both are wrong
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            // there is a link going to this neuron, so there are incoming
            // don't count the link if it is recurrent or coming from a bias
            if ((m_LinkGenes[i].ToNeuronID() == a_ID)
                && (!m_LinkGenes[i].IsLoopedRecurrent())
                && (GetNeuronByID(m_LinkGenes[i].FromNeuronID()).Type() != BIAS))
            {
                t_no_incoming = false;
            }

            // there is a link going from this neuron, so there are outgoing
            // don't count the link if it is recurrent or coming from a bias
            if ((m_LinkGenes[i].FromNeuronID() == a_ID)
                && (!m_LinkGenes[i].IsLoopedRecurrent())
                && (GetNeuronByID(m_LinkGenes[i].FromNeuronID()).Type() != BIAS))
            {
                t_no_outgoing = false;
            }
        }

        // if just one of these is true, this neuron is a dead end
        if (t_no_incoming || t_no_outgoing)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    

    // Returns the count of links inputting from the specified neuron ID
    int Genome::LinksInputtingFrom(int a_ID) const
    {
        int t_counter = 0;
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].FromNeuronID() == a_ID)
                t_counter++;
        }

        return t_counter;
    }


    // Returns the count of links outputting to the specified neuron ID
    int Genome::LinksOutputtingTo(int a_ID) const
    {
        int t_counter = 0;
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].ToNeuronID() == a_ID)
                t_counter++;
        }

        return t_counter;
    }

    

    LinkGene Genome::GetLinkByIndex(int idx) const
    {
        ASSERT(idx<(int)m_LinkGenes.size());
        return m_LinkGenes[idx];
    }

    LinkGene Genome::GetLinkByInnovID(int id) const
    {
        for (unsigned int i=0; i<m_LinkGenes.size(); i++)
        {
            if(m_LinkGenes[i].InnovationID()==id)
                return m_LinkGenes[i];
        }
        throw std::runtime_error("No link found by that innovID");
    }

    NeuronGene Genome::GetNeuronByIndex(int idx) const
    {
        ASSERT(idx<(int)m_NeuronGenes.size());
        return m_NeuronGenes[idx];
    }

    NeuronGene Genome::GetNeuronByID(int a_ID) const
    {
        ASSERT(HasNeuronID(a_ID));
        int i = GetNeuronIndex(a_ID);
        ASSERT(i>=0);
        return m_NeuronGenes[i];
    }

    double Genome::GetOffspringAmount() const { return m_OffspringAmount; }
    void   Genome::SetOffspringAmount(double v) { m_OffspringAmount = v; }

    bool   Genome::IsEvaluated() const { return m_Evaluated; }
    void   Genome::SetEvaluated() { m_Evaluated = true; }
    void   Genome::ResetEvaluated() { m_Evaluated = false; }

    // ==============================================================
    // A helper to find the index of a neuron
    // ==============================================================
    int Genome::GetNeuronIndex(int a_ID) const
    {
        ASSERT(a_ID>0);
        for (unsigned int i=0; i<m_NeuronGenes.size(); i++)
        {
            if(m_NeuronGenes[i].ID()==a_ID)
                return i;
        }
        return -1;
    }

    // ==============================================================
    // A helper to find the index of a link
    // ==============================================================
    int Genome::GetLinkIndex(int a_InnovID) const
    {
        ASSERT(a_InnovID>0 && m_LinkGenes.size()>0);
        for (unsigned int i=0; i<m_LinkGenes.size(); i++)
        {
            if(m_LinkGenes[i].InnovationID()==a_InnovID)
                return i;
        }
        return -1;
    }

    // ==============================================================
    // Return the next neuron ID
    // ==============================================================
    int Genome::GetLastNeuronID() const
    {
        ASSERT(m_NeuronGenes.size()>0);
        int t_maxid = 0;
        for (unsigned int i=0; i<m_NeuronGenes.size(); i++)
        {
            if(m_NeuronGenes[i].ID()>t_maxid)
                t_maxid = m_NeuronGenes[i].ID();
        }
        return t_maxid+1;
    }

    int Genome::GetLastInnovationID() const
    {
        ASSERT(m_LinkGenes.size()>0);
        int t_maxid=0;
        for (unsigned int i=0; i<m_LinkGenes.size(); i++)
        {
            if(m_LinkGenes[i].InnovationID()>t_maxid)
                t_maxid = m_LinkGenes[i].InnovationID();
        }
        return t_maxid+1;
    }

    bool Genome::HasNeuronID(int a_ID) const
    {
        ASSERT(a_ID>0);
        for (unsigned int i=0; i<m_NeuronGenes.size(); i++)
        {
            if(m_NeuronGenes[i].ID()==a_ID)
                return true;
        }
        return false;
    }

    bool Genome::HasLink(int a_n1id, int a_n2id) const
    {
        ASSERT(a_n1id>0 && a_n2id>0);
        for (unsigned int i=0; i<m_LinkGenes.size(); i++)
        {
            if(m_LinkGenes[i].FromNeuronID()==a_n1id
            && m_LinkGenes[i].ToNeuronID()==a_n2id)
            {
                return true;
            }
        }
        return false;
    }

    // ==============================================================
    // Returns true if the network described by this genome has cycles
    // We do a DFS-based detection for cycles.
    // ==============================================================
    bool Genome::HasLoops()
    {
        // Build a phenotype
        NeuralNetwork net;
        BuildPhenotype(net);

        // Build adjacency
        std::vector<std::vector<int>> adjacency(net.m_neurons.size());
        for (unsigned int i=0; i<net.m_connections.size(); i++)
        {
            int s = net.m_connections[i].m_source_neuron_idx;
            int t = net.m_connections[i].m_target_neuron_idx;
            adjacency[s].push_back(t);
        }

        // color array
        std::vector<int> color(net.m_neurons.size(), 0);

        for (int i=0; i<(int)net.m_neurons.size(); i++)
        {
            if (color[i]==0)
            {
                if (HasCycleDFS(i, color, adjacency))
                {
                    return true;
                }
            }
        }
        return false;
    }

    bool Genome::HasLinkByInnovID(int id) const
    {
        ASSERT(id>0);
        for (unsigned int i=0; i<m_LinkGenes.size(); i++)
        {
            if(m_LinkGenes[i].InnovationID()==id)
                return true;
        }
        return false;
    }

    // ==============================================================
    // Build a NeuralNetwork (FastNetwork) from this genome
    // ==============================================================
    void Genome::BuildPhenotype(NeuralNetwork &a_Net)
    {
        a_Net.Clear();
        a_Net.SetInputOutputDimentions(m_NumInputs, m_NumOutputs);

        // fill the net with neurons
        for (unsigned int i=0; i<m_NeuronGenes.size(); i++)
        {
            Neuron t_n;
            t_n.m_a        = m_NeuronGenes[i].m_A;
            t_n.m_b        = m_NeuronGenes[i].m_B;
            t_n.m_timeconst= m_NeuronGenes[i].m_TimeConstant;
            t_n.m_bias     = m_NeuronGenes[i].m_Bias;
            t_n.m_activation_function_type = m_NeuronGenes[i].m_ActFunction;
            t_n.m_split_y  = m_NeuronGenes[i].SplitY();
            t_n.m_type     = m_NeuronGenes[i].Type();

            a_Net.AddNeuron(t_n);
        }

        // fill with connections
        for (unsigned int i=0; i<m_LinkGenes.size(); i++)
        {
            Connection c;
            c.m_source_neuron_idx = GetNeuronIndex(m_LinkGenes[i].FromNeuronID());
            c.m_target_neuron_idx = GetNeuronIndex(m_LinkGenes[i].ToNeuronID());
            c.m_weight            = m_LinkGenes[i].GetWeight();
            c.m_recur_flag        = m_LinkGenes[i].IsRecurrent();

            // default
            c.m_hebb_rate     = 0.3;
            c.m_hebb_pre_rate = 0.1;

            // if trait is present
            if(m_LinkGenes[i].m_Traits.count("hebb_rate")==1)
            {
                try
                {
                    c.m_hebb_rate = std::get<double>(m_LinkGenes[i].m_Traits["hebb_rate"].value);
                }
                catch(...) { /* ignore if variant mismatch */ }
            }
            if(m_LinkGenes[i].m_Traits.count("hebb_pre_rate")==1)
            {
                try
                {
                    c.m_hebb_pre_rate = std::get<double>(m_LinkGenes[i].m_Traits["hebb_pre_rate"].value);
                }
                catch(...) { /* ignore if mismatch */ }
            }
            a_Net.AddConnection(c);
        }

        a_Net.Flush();
    }

    ActivationFunction GetRandomActivation(Parameters &a_Parameters, RNG &a_RNG)
    {
        std::vector<double> t_probs;

        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedSigmoid_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedSigmoid_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Tanh_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_TanhCubic_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedStep_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedStep_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedGauss_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedGauss_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Abs_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedSine_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedSine_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Linear_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Relu_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Softplus_Prob);

        return (NEAT::ActivationFunction) a_RNG.Roulette(t_probs);
    }

    // ==============================================================
    // Build a HyperNEAT phenotype
    // The full snippet includes code that queries the CPPN
    // We remove any references to Boost, but otherwise keep logic.
    // ==============================================================
    void Genome::BuildHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst)
    {
        // We need a substrate with at least one input and one output
        ASSERT(subst.m_input_coords.size()>0);
        ASSERT(subst.m_output_coords.size()>0);

        int max_dims = subst.GetMaxDims();

        // The CPPN's input dimensionality
        ASSERT( (int)m_NumInputs >= subst.GetMinCPPNInputs() );
        ASSERT( (int)m_NumOutputs >= subst.GetMinCPPNOutputs() );

        net.SetInputOutputDimentions((unsigned short)subst.m_input_coords.size(),
                                     (unsigned short)subst.m_output_coords.size());

        // Create input neurons in net
        for (unsigned int i=0; i<subst.m_input_coords.size(); i++)
        {
            Neuron t_n;
            t_n.m_a  = 1;
            t_n.m_b  = 0;
            t_n.m_substrate_coords = subst.m_input_coords[i];
            t_n.m_activation_function_type = LINEAR;
            t_n.m_type = INPUT;
            net.AddNeuron(t_n);
        }

        // Create output neurons
        for (unsigned int i=0; i<subst.m_output_coords.size(); i++)
        {
            Neuron t_n;
            t_n.m_a  = 1;
            t_n.m_b  = 0;
            t_n.m_substrate_coords = subst.m_output_coords[i];
            t_n.m_activation_function_type = subst.m_output_nodes_activation;
            t_n.m_type = OUTPUT;
            net.AddNeuron(t_n);
        }

        // Create hidden if any
        for (unsigned int i=0; i<subst.m_hidden_coords.size(); i++)
        {
            Neuron t_n;
            t_n.m_a  = 1;
            t_n.m_b  = 0;
            t_n.m_substrate_coords = subst.m_hidden_coords[i];
            t_n.m_activation_function_type = subst.m_hidden_nodes_activation;
            t_n.m_type = HIDDEN;
            net.AddNeuron(t_n);
        }

        // Build a temporary CPPN from this genome
        NeuralNetwork cppn(true);
        BuildPhenotype(cppn);
        cppn.Flush();

        // For leaky substrates, init timeconstant and bias from extra outputs
        if (subst.m_leaky)
        {
            // output dimension must handle that
            ASSERT(m_NumOutputs >= (unsigned int)subst.GetMinCPPNOutputs());

            // from index net.NumInputs().. do the hidden + output
            for (unsigned int i=net.NumInputs(); i<net.m_neurons.size(); i++)
            {
                cppn.Flush();
                std::vector<double> cinputs;
                cinputs.resize(m_NumInputs);

                // fill cinputs with the coords
                for (unsigned int d=0; d<net.m_neurons[i].m_substrate_coords.size(); d++)
                {
                    cinputs[d] = net.m_neurons[i].m_substrate_coords[d];
                }
                if (subst.m_with_distance)
                {
                    // distance from origin
                    double sum=0;
                    for (int dd=0; dd<max_dims; dd++)
                    {
                        sum += sqr(cinputs[dd]);
                    }
                    sum = sqrt(sum);
                    cinputs[m_NumInputs-2] = sum;
                }
                cinputs[m_NumInputs-1] = 1.0; // bias

                cppn.Input(cinputs);
                int dp = 8;
                if (!HasLoops())
                {
                    CalculateDepth();
                    dp = GetDepth();
                }
                for(int z=0; z<dp; z++)
                {
                    cppn.Activate();
                }
                double t_tc   = cppn.Output()[m_NumOutputs-2];
                double t_bias = cppn.Output()[m_NumOutputs-1];

                Clamp(t_tc, -1, 1);
                Clamp(t_bias, -1, 1);
                Scale(t_tc, -1,1, subst.m_min_time_const, subst.m_max_time_const);
                Scale(t_bias,-1,1, -subst.m_max_weight_and_bias, subst.m_max_weight_and_bias);

                net.m_neurons[i].m_timeconst = t_tc;
                net.m_neurons[i].m_bias      = t_bias;
            }
        }

        // Now create connections
        // If custom connectivity is present, we do that
        std::vector<std::vector<int>> pairs;
        if (!subst.m_custom_connectivity.empty())
        {
            // use custom connectivity
            for (unsigned int i=0; i<subst.m_custom_connectivity.size(); i++)
            {
                // src_type, src_idx, dst_type, dst_idx
                NeuronType st = (NeuronType) subst.m_custom_connectivity[i][0];
                int sidx      = subst.m_custom_connectivity[i][1];
                NeuronType dt = (NeuronType) subst.m_custom_connectivity[i][2];
                int didx      = subst.m_custom_connectivity[i][3];

                // figure out net indices
                int j=0, k=0;
                if ((st==INPUT)||(st==BIAS))    j = sidx;
                else if (st==OUTPUT)           j = subst.m_input_coords.size()+sidx;
                else if (st==HIDDEN)           j = subst.m_input_coords.size()+subst.m_output_coords.size()+sidx;

                if ((dt==INPUT)||(dt==BIAS))    k = didx;
                else if (dt==OUTPUT)           k = subst.m_input_coords.size()+didx;
                else if (dt==HIDDEN)           k = subst.m_input_coords.size()+subst.m_output_coords.size()+didx;

                // if we obey flags, check them
                if (subst.m_custom_conn_obeys_flags && (
                    ((!subst.m_allow_input_hidden_links) &&
                     ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                    || ((!subst.m_allow_input_output_links) &&
                        ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == OUTPUT)))

                    || ((!subst.m_allow_hidden_hidden_links) &&
                        ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) && (i != j)))

                    || ((!subst.m_allow_hidden_output_links) &&
                        ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == OUTPUT)))

                    || ((!subst.m_allow_output_hidden_links) &&
                        ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                    || ((!subst.m_allow_output_output_links) &&
                        ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) && (i != j)))

                    || ((!subst.m_allow_looped_hidden_links) &&
                        ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) && (i == j)))

                    || ((!subst.m_allow_looped_output_links) &&
                        ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) && (i == j)))
               ))
             
                pairs.push_back({j,k});
            }
        }
        else
        {
            // full combos from net.m_neurons ...
            // Then skip if disallowed by flags
            for (unsigned int i= net.NumInputs(); i<net.m_neurons.size(); i++)
            {
                for (unsigned int j=0; j<net.m_neurons.size(); j++)
                {
                    // skip if same or flags disallow
                    if (subst.m_custom_conn_obeys_flags && (
                        ((!subst.m_allow_input_hidden_links) &&
                         ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                        || ((!subst.m_allow_input_output_links) &&
                            ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == OUTPUT)))

                        || ((!subst.m_allow_hidden_hidden_links) &&
                            ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) && (i != j)))

                        || ((!subst.m_allow_hidden_output_links) &&
                            ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == OUTPUT)))

                        || ((!subst.m_allow_output_hidden_links) &&
                            ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                        || ((!subst.m_allow_output_output_links) &&
                            ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) && (i != j)))

                        || ((!subst.m_allow_looped_hidden_links) &&
                            ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) && (i == j)))

                        || ((!subst.m_allow_looped_output_links) &&
                            ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) && (i == j)))
                        )
                        )
                    pairs.push_back({(int)j,(int)i});
                }
            }
        }

        // Query the CPPN for each pair
        for (auto &pp : pairs)
        {
            int j = pp[0];
            int i = pp[1];

            std::vector<double> t_inputs;
            t_inputs.resize(m_NumInputs);

            // fill the coords from net
            int from_dims = net.m_neurons[j].m_substrate_coords.size();
            int to_dims   = net.m_neurons[i].m_substrate_coords.size();
            for (int d=0; d<from_dims; d++)
                t_inputs[d] = net.m_neurons[j].m_substrate_coords[d];
            for (int d=0; d<to_dims; d++)
                t_inputs[max_dims + d] = net.m_neurons[i].m_substrate_coords[d];

            if (subst.m_with_distance)
            {
                double sum=0.0;
                for (int dd=0; dd<max_dims; dd++)
                {
                    sum += sqr(t_inputs[dd] - t_inputs[max_dims+dd]);
                }
                sum = sqrt(sum);
                t_inputs[ (int)m_NumInputs -2 ] = sum; 
            }
            t_inputs[ (int)m_NumInputs -1 ] = 1.0; // bias

            cppn.Flush();
            cppn.Input(t_inputs);
            int dp = 8;
            if (!HasLoops())
            {
                CalculateDepth();
                dp = GetDepth();
            }
            for(int z=0; z<dp; z++) cppn.Activate();

            double t_link = 0;
            double t_weight=0;
            if (subst.m_query_weights_only)
            {
                t_weight = cppn.Output()[0];
            }
            else
            {
                t_link   = cppn.Output()[0];
                t_weight = cppn.Output()[1];
            }

            if (((!subst.m_query_weights_only) && (t_link>0)) || (subst.m_query_weights_only))
            {
                // scale weight
                t_weight *= subst.m_max_weight_and_bias;
                Connection c;
                c.m_source_neuron_idx = j;
                c.m_target_neuron_idx = i;
                c.m_weight            = t_weight;
                c.m_recur_flag        = false;

                net.AddConnection(c);
            }
        }
    }

    // ==============================================================
    // Projects changes from a phenotype back to the genome
    // (e.g. for weight changes from RTRL)
    // ==============================================================
    void Genome::DerivePhenotypicChanges(NeuralNetwork &a_Net)
    {
        // must have same topology
        if( (int)a_Net.m_connections.size() != (int)m_LinkGenes.size() ) return;
        for (unsigned int i=0; i<m_LinkGenes.size(); i++)
        {
            m_LinkGenes[i].SetWeight(a_Net.GetConnectionByIndex(i).m_weight);
        }
    }

    // ==============================================================
    // Compatibility Distance
    // We remove references to caching or boost.
    // ==============================================================
    double Genome::CompatibilityDistance(Genome &a_G, Parameters &a_Parameters)
    {
        // The logic is the same as your snippet, but minus boost.
        // We won't omit code now. We'll do it in full:

        // Step 1: find all link genes
        // match by innov ID, etc.
        // We'll track:
        double total_distance    = 0.0;
        double total_w_diff      = 0.0;
        double total_A_diff      = 0.0;
        double total_B_diff      = 0.0;
        double total_TC_diff     = 0.0;
        double total_bias_diff   = 0.0;
        double total_act_diff    = 0.0;

        std::map<std::string,double> total_link_trait_diff;
        std::map<std::string,double> total_neuron_trait_diff;
        std::map<std::string,double> total_genome_trait_diff;

        double E = 0; // excess
        double D = 0; // disjoint
        double M = 0; // matching links
        double matching_neurons=0;

        // Check genome trait distances
        auto gentrait_dists = m_GenomeGene.GetTraitDistances(a_G.m_GenomeGene.m_Traits);
        for(auto &kv : gentrait_dists)
        {
            double val = kv.second * a_Parameters.GenomeTraits[kv.first].m_ImportanceCoeff;
            if(std::isnan(val)||std::isinf(val)) val=0.0;
            total_distance+= val;
        }

        // for links
        unsigned int i1=0, i2=0;
        std::vector<LinkGene> links1 = m_LinkGenes;
        std::vector<LinkGene> links2 = a_G.m_LinkGenes;
        std::sort(links1.begin(),links1.end(), [](const LinkGene &lhs,const LinkGene &rhs){
            return lhs.InnovationID()<rhs.InnovationID();
        });
        std::sort(links2.begin(),links2.end(), [](const LinkGene &lhs,const LinkGene &rhs){
            return lhs.InnovationID()<rhs.InnovationID();
        });

        while(!(i1>=links1.size() && i2>=links2.size()))
        {
            if (i1==links1.size())
            {
                E++;
                i2++;
            }
            else if (i2==links2.size())
            {
                E++;
                i1++;
            }
            else
            {
                int in1=links1[i1].InnovationID();
                int in2=links2[i2].InnovationID();
                if (in1==in2)
                {
                    M++;
                    if(a_Parameters.WeightDiffCoeff>0)
                    {
                        double wd = links1[i1].GetWeight()-links2[i2].GetWeight();
                        total_w_diff += (wd<0)?-wd:wd;
                    }
                    // trait distance
                    auto linktraitdist = links1[i1].GetTraitDistances( links2[i2].m_Traits );
                    for(auto &xx: linktraitdist)
                    {
                        double val = xx.second;
                        val *= a_Parameters.LinkTraits[xx.first].m_ImportanceCoeff;
                        if(std::isnan(val) || std::isinf(val)) val=0.0;
                        total_link_trait_diff[xx.first]+= val;
                    }

                    i1++; i2++;
                }
                else if(in1<in2)
                {
                    D++;
                    i1++;
                }
                else
                {
                    D++;
                    i2++;
                }
            }
        }

        double maxsize = (links1.size() > links2.size()) ? (double)links1.size() : (double)links2.size();
        if(maxsize<1.0) maxsize=1.0;
        double normalizer = (a_Parameters.NormalizeGenomeSize) ? maxsize : 1.0;
        if(M<1.0) M=1.0;

        double dist_links = 
            (a_Parameters.ExcessCoeff*(E/normalizer))
          + (a_Parameters.DisjointCoeff*(D/normalizer))
          + (a_Parameters.WeightDiffCoeff*((total_w_diff)/(M)));

        total_distance += dist_links;

        // now neurons
        // skip input/bias
        int bigger_neuron_count = ( (int)m_NeuronGenes.size() > (int)a_G.m_NeuronGenes.size() )
                                  ? m_NeuronGenes.size()
                                  : a_G.m_NeuronGenes.size();
        if(bigger_neuron_count<1) bigger_neuron_count=1;

        double mismatch=0;
        // for matching
        for (unsigned int i= (unsigned int)m_NumInputs; i<m_NeuronGenes.size(); i++)
        {
            if(m_NeuronGenes[i].Type()==INPUT || m_NeuronGenes[i].Type()==BIAS) continue;
            if(a_G.HasNeuronID(m_NeuronGenes[i].ID()))
            {
                matching_neurons++;
                NeuronGene oth = a_G.GetNeuronByID(m_NeuronGenes[i].ID());

                if (a_Parameters.ActivationADiffCoeff>0)
                {
                    double diffA = m_NeuronGenes[i].m_A - oth.m_A;
                    if(diffA<0) diffA=-diffA;
                    total_A_diff += diffA;
                }
                if(a_Parameters.ActivationBDiffCoeff>0)
                {
                    double diffB = m_NeuronGenes[i].m_B - oth.m_B;
                    if(diffB<0) diffB=-diffB;
                    total_B_diff += diffB;
                }
                if(a_Parameters.TimeConstantDiffCoeff>0)
                {
                    double diffT= m_NeuronGenes[i].m_TimeConstant - oth.m_TimeConstant;
                    if(diffT<0) diffT=-diffT;
                    total_TC_diff += diffT;
                }
                if(a_Parameters.BiasDiffCoeff>0)
                {
                    double diffBi= m_NeuronGenes[i].m_Bias - oth.m_Bias;
                    if(diffBi<0) diffBi=-diffBi;
                    total_bias_diff += diffBi;
                }
                if(a_Parameters.ActivationFunctionDiffCoeff>0)
                {
                    if( m_NeuronGenes[i].m_ActFunction != oth.m_ActFunction)
                        total_act_diff++;
                }

                // traits
                auto nd = m_NeuronGenes[i].GetTraitDistances(oth.m_Traits);
                for(auto &xx : nd)
                {
                    double val=xx.second;
                    val *= a_Parameters.NeuronTraits[xx.first].m_ImportanceCoeff;
                    if(std::isnan(val)||std::isinf(val)) val=0;
                    total_neuron_trait_diff[xx.first]+=val;
                }
            }
        }

        if(matching_neurons<1) matching_neurons=1;

        double dist_neurons = 0.0;
        dist_neurons += a_Parameters.ActivationADiffCoeff*(total_A_diff/matching_neurons);
        dist_neurons += a_Parameters.ActivationBDiffCoeff*(total_B_diff/matching_neurons);
        dist_neurons += a_Parameters.TimeConstantDiffCoeff*(total_TC_diff/matching_neurons);
        dist_neurons += a_Parameters.BiasDiffCoeff*(total_bias_diff/matching_neurons);
        dist_neurons += a_Parameters.ActivationFunctionDiffCoeff*((double)total_act_diff/matching_neurons);

        total_distance += dist_neurons;

        // link traits
        for(auto &xx : total_link_trait_diff)
        {
            double n = xx.second*(a_Parameters.LinkTraits[xx.first].m_ImportanceCoeff)*(1.0/M);
            if(std::isnan(n)||std::isinf(n)) n=0.0;
            total_distance += n;
        }
        // neuron traits
        for(auto &xx : total_neuron_trait_diff)
        {
            double n = xx.second*(a_Parameters.NeuronTraits[xx.first].m_ImportanceCoeff)*(1.0/matching_neurons);
            if(std::isnan(n)||std::isinf(n)) n=0.0;
            total_distance += n;
        }
        // already added genome trait differences above

        return total_distance;
    }

    // ==============================================================
    // Checks if two genomes are in the same species
    // ==============================================================
    bool Genome::IsCompatibleWith(Genome &a_G, Parameters &a_Parameters)
    {
        if(this==&a_G) return true;
        if(GetID()==a_G.GetID()) return true;

        double dist = CompatibilityDistance(a_G,a_Parameters);
        return (dist<=a_Parameters.CompatTreshold);
    }

    // ==============================================================
    // Weighted Mutation
    // ==============================================================
    bool Genome::Mutate_LinkWeights(const Parameters &a_Parameters, RNG &a_RNG)
    {
        bool did_mutate=false;
        bool severe = (a_RNG.RandFloat()<a_Parameters.MutateWeightsSevereProb);

        // define a "tail" region if # links > initial
        int tailstart=0;
        if((int)NumLinks()>m_initial_num_links)
            tailstart=(int)(NumLinks()*0.8);
        if(tailstart<m_initial_num_links)
            tailstart=m_initial_num_links;

        for(unsigned int i=0; i<m_LinkGenes.size(); i++)
        {
            if(!severe && (a_RNG.RandFloat()<a_Parameters.WeightMutationRate))
            {
                double w = m_LinkGenes[i].GetWeight();
                bool in_tail = (int)i>=tailstart;
                if(in_tail || a_RNG.RandFloat()<a_Parameters.WeightReplacementRate)
                {
                    w = a_RNG.RandFloatSigned()*a_Parameters.WeightReplacementMaxPower;
                }
                else
                {
                    w += (a_RNG.RandFloatSigned()*a_Parameters.WeightMutationMaxPower);
                }
                Clamp(w, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                m_LinkGenes[i].SetWeight(w);
                did_mutate=true;
            }
            else if(severe)
            {
                if(a_RNG.RandFloat()<a_Parameters.WeightMutationRate)
                {
                    double w=a_RNG.RandFloat();
                    Scale(w,0.0,1.0,a_Parameters.MinWeight,a_Parameters.MaxWeight);
                    m_LinkGenes[i].SetWeight(w);
                    did_mutate=true;
                }
            }
        }
        return did_mutate;
    }

    // ==============================================================
    // Just sets random weights
    // ==============================================================
    void Genome::Randomize_LinkWeights(const Parameters &a_Parameters, RNG &a_RNG)
    {
        for(unsigned int i=0; i<NumLinks(); i++)
        {
            double nf=a_RNG.RandFloat();
            Scale(nf,0.0,1.0,a_Parameters.MinWeight,a_Parameters.MaxWeight);
            m_LinkGenes[i].SetWeight(nf);
        }
    }

    void Genome::Randomize_Traits(const Parameters &a_Parameters, RNG &a_RNG)
    {
        for (auto &ng : m_NeuronGenes)
        {
            ng.InitTraits(a_Parameters.NeuronTraits, a_RNG);
        }
        for (auto &lg : m_LinkGenes)
        {
            lg.InitTraits(a_Parameters.LinkTraits, a_RNG);
        }
        m_GenomeGene.InitTraits(a_Parameters.GenomeTraits, a_RNG);
    }

    bool Genome::Mutate_NeuronActivations_A(const Parameters &a_Parameters, RNG &a_RNG)
    {
        for(unsigned int i=0; i<m_NeuronGenes.size(); i++)
        {
            if(m_NeuronGenes[i].Type()!=INPUT && m_NeuronGenes[i].Type()!=BIAS)
            {
                double r = a_RNG.RandFloatSigned()*a_Parameters.ActivationAMutationMaxPower;
                m_NeuronGenes[i].m_A += r;
                Clamp(m_NeuronGenes[i].m_A, a_Parameters.MinActivationA,a_Parameters.MaxActivationA);
            }
        }
        return true;
    }

    bool Genome::Mutate_NeuronActivations_B(const Parameters &a_Parameters, RNG &a_RNG)
    {
        for(unsigned int i=0; i<m_NeuronGenes.size(); i++)
        {
            if(m_NeuronGenes[i].Type()!=INPUT && m_NeuronGenes[i].Type()!=BIAS)
            {
                double r=a_RNG.RandFloatSigned()*a_Parameters.ActivationBMutationMaxPower;
                m_NeuronGenes[i].m_B+=r;
                Clamp(m_NeuronGenes[i].m_B,a_Parameters.MinActivationB,a_Parameters.MaxActivationB);
            }
        }
        return true;
    }

    bool Genome::Mutate_NeuronActivation_Type(const Parameters &a_Parameters, RNG &a_RNG)
    {
        // skip if we only have input + bias
        if((int)m_NeuronGenes.size()<=(int)m_NumInputs) return false;

        int startIndex = (int)m_NumInputs; 
        int choice = a_RNG.RandInt(startIndex,(int)m_NeuronGenes.size()-1);
        int oldf = m_NeuronGenes[choice].m_ActFunction;

        // We pick from probabilities
        std::vector<double> probs={
            a_Parameters.ActivationFunction_SignedSigmoid_Prob,
            a_Parameters.ActivationFunction_UnsignedSigmoid_Prob,
            a_Parameters.ActivationFunction_Tanh_Prob,
            a_Parameters.ActivationFunction_TanhCubic_Prob,
            a_Parameters.ActivationFunction_SignedStep_Prob,
            a_Parameters.ActivationFunction_UnsignedStep_Prob,
            a_Parameters.ActivationFunction_SignedGauss_Prob,
            a_Parameters.ActivationFunction_UnsignedGauss_Prob,
            a_Parameters.ActivationFunction_Abs_Prob,
            a_Parameters.ActivationFunction_SignedSine_Prob,
            a_Parameters.ActivationFunction_UnsignedSine_Prob,
            a_Parameters.ActivationFunction_Linear_Prob,
            a_Parameters.ActivationFunction_Relu_Prob,
            a_Parameters.ActivationFunction_Softplus_Prob
        };
        int idx = a_RNG.Roulette(probs);
        ActivationFunction newAF = (ActivationFunction) idx;
        if((int)newAF == oldf) return false;

        m_NeuronGenes[choice].m_ActFunction = newAF;
        return true;
    }

    bool Genome::Mutate_NeuronTimeConstants(const Parameters &a_Parameters, RNG &a_RNG)
    {
        for(unsigned int i=0; i<m_NeuronGenes.size(); i++)
        {
            if(m_NeuronGenes[i].Type()!=INPUT && m_NeuronGenes[i].Type()!=BIAS)
            {
                double r=a_RNG.RandFloatSigned()*a_Parameters.TimeConstantMutationMaxPower;
                m_NeuronGenes[i].m_TimeConstant+=r;
                Clamp(m_NeuronGenes[i].m_TimeConstant,
                      a_Parameters.MinNeuronTimeConstant,
                      a_Parameters.MaxNeuronTimeConstant);
            }
        }
        return true;
    }

    bool Genome::Mutate_NeuronBiases(const Parameters &a_Parameters, RNG &a_RNG)
    {
        for(unsigned int i=0; i<m_NeuronGenes.size(); i++)
        {
            if(m_NeuronGenes[i].Type()!=INPUT && m_NeuronGenes[i].Type()!=BIAS)
            {
                double r=a_RNG.RandFloatSigned()*a_Parameters.BiasMutationMaxPower;
                m_NeuronGenes[i].m_Bias+=r;
                Clamp(m_NeuronGenes[i].m_Bias,a_Parameters.MinNeuronBias,a_Parameters.MaxNeuronBias);
            }
        }
        return true;
    }

    bool Genome::Mutate_NeuronTraits(const Parameters &a_Parameters, RNG &a_RNG)
    {
        bool mutated=false;
        for(auto &ng : m_NeuronGenes)
        {
            if(ng.Type()!=INPUT && ng.Type()!=BIAS)
            {
                mutated |= ng.MutateTraits(a_Parameters.NeuronTraits, a_RNG);
            }
        }
        return mutated;
    }

    bool Genome::Mutate_LinkTraits(const Parameters &a_Parameters, RNG &a_RNG)
    {
        bool mutated=false;
        for(auto &lg : m_LinkGenes)
        {
            mutated |= lg.MutateTraits(a_Parameters.LinkTraits, a_RNG);
        }
        return mutated;
    }

    bool Genome::Mutate_GenomeTraits(const Parameters &a_Parameters, RNG &a_RNG)
    {
        return m_GenomeGene.MutateTraits(a_Parameters.GenomeTraits, a_RNG);
    }

    // ==============================================================
    // Adds a new neuron
    // ==============================================================
    bool Genome::Mutate_AddNeuron(InnovationDatabase &a_Innovs, Parameters &a_Parameters, RNG &a_RNG)
    {
        // No links to split - go away..
        if (NumLinks() == 0)
            return false;
        
        // First find a link that to be split
        ////////////////////

        // Select a random link for now
        bool t_link_found = false;
        int t_link_num = 0;
        int t_in = 0, t_out = 0;
        LinkGene t_chosenlink(0, 0, -1, 0, false); // to save it for later

        // number of tries to find a good link or give up
        int t_tries = 256;
        while (!t_link_found)
        {
            if (NumLinks() == 1)
            {
                t_link_num = 0;
            }
            /*else if (NumLinks() == 2)
            {
                t_link_num = Rounded(a_RNG.RandFloat());
            }*/
            else
            {
                //if (NumLinks() > 8)
                {
                    t_link_num = a_RNG.RandInt(0, NumLinks() - 1); // random selection
                }
                /*else
            {
                // this selects older links for splitting
                double t_r = abs(RandGaussSigned()/3.0);
                Clamp(t_r, 0, 1);
                t_link_num =  static_cast<int>(t_r * (NumLinks()-1));
            }*/
            }


            t_in = m_LinkGenes[t_link_num].FromNeuronID();
            t_out = m_LinkGenes[t_link_num].ToNeuronID();

            ASSERT((t_in > 0) && (t_out > 0));

            t_link_found = true;

            // In case there is only one link, coming from a bias - just quit

            // unless the parameter is set
            if (a_Parameters.DontUseBiasNeuron == false)
            {
                if ((m_NeuronGenes[GetNeuronIndex(t_in)].Type() == BIAS) && (NumLinks() == 1))
                {
                    return false;
                }

                // Do not allow splitting a link coming from a bias
                if (m_NeuronGenes[GetNeuronIndex(t_in)].Type() == BIAS)
                {
                    t_link_found = false;
                }
            }

            // Do not allow splitting of recurrent links
            if (!a_Parameters.SplitRecurrent)
            {
                if (m_LinkGenes[t_link_num].IsRecurrent())
                {
                    if ((!a_Parameters.SplitLoopedRecurrent) && (t_in == t_out))
                    {
                        t_link_found = false;
                    }
                }
            }

            t_tries--;
            if (t_tries <= 0)
            {
                return false;
            }
        }
        // Now the link has been selected

        // the weight of the link that is being split
        double t_orig_weight = m_LinkGenes[t_link_num].GetWeight();
        t_chosenlink = m_LinkGenes[t_link_num]; // save the whole link

        // remove the link from the genome
        // find it first and then erase it
        // TODO: add option to keep the link, but disabled
        std::vector<LinkGene>::iterator t_iter;
        for (t_iter = m_LinkGenes.begin(); t_iter != m_LinkGenes.end(); t_iter++)
        {
            if (t_iter->InnovationID() == m_LinkGenes[t_link_num].InnovationID())
            {
                // found it! now erase..
                m_LinkGenes.erase(t_iter);
                break;
            }
        }

        // Check if an innovation of this type already occured somewhere in the population
        int t_innovid = a_Innovs.CheckInnovation(t_in, t_out, NEW_NEURON);

        // the new neuron and links ids
        int t_nid = 0;
        int t_l1id = 0;
        int t_l2id = 0;

        // This is a novel innovation?
        if (t_innovid == -1)
        {
            // Add the new neuron innovation
            t_nid = a_Innovs.AddNeuronInnovation(t_in, t_out, HIDDEN);
            // add the first link innovation
            t_l1id = a_Innovs.AddLinkInnovation(t_in, t_nid);
            // add the second innovation
            t_l2id = a_Innovs.AddLinkInnovation(t_nid, t_out);

            // Adjust the SplitY
            double t_sy = m_NeuronGenes[GetNeuronIndex(t_in)].SplitY() + m_NeuronGenes[GetNeuronIndex(t_out)].SplitY();
            t_sy /= 2.0;

            // Create the neuron gene
            NeuronGene t_ngene(HIDDEN, t_nid, t_sy);

            double t_A = a_RNG.RandFloat();
            double t_B = a_RNG.RandFloat();
            double t_TC = a_RNG.RandFloat();
            double t_Bs = a_RNG.RandFloat();
            Scale(t_A, 0, 1, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Scale(t_B, 0, 1, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Scale(t_TC, 0, 1, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Scale(t_Bs, 0, 1, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            Clamp(t_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Clamp(t_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Clamp(t_TC, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Clamp(t_Bs, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            // Initialize the neuron gene's properties
            t_ngene.Init(t_A,
                         t_B,
                         t_TC,
                         t_Bs,
                         GetRandomActivation(a_Parameters, a_RNG));

            // Initialize the traits
            t_ngene.InitTraits(a_Parameters.NeuronTraits, a_RNG);

            // Add the NeuronGene
            m_NeuronGenes.emplace_back(t_ngene);

            // Now the links

            // Make sure the recurrent flag is kept
            bool t_recurrentflag = t_chosenlink.IsRecurrent();

            // First link
            LinkGene l1 = LinkGene(t_in, t_nid, t_l1id, 1.0, t_recurrentflag);
            // make sure this weight is in the allowed interval
            Clamp(l1.m_Weight, a_Parameters.MinWeight, a_Parameters.MaxWeight);
            // Init the link's traits
            l1.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l1);

            // Second link
            LinkGene l2 = LinkGene(t_nid, t_out, t_l2id, t_orig_weight, t_recurrentflag);
            // Init the link's traits
            l2.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l2);
        }
        else
        {
            // This innovation already happened, so inherit it.

            // get the neuron ID
            t_nid = a_Innovs.FindNeuronID(t_in, t_out);
            ASSERT(t_nid != -1);

            // if such an innovation happened, these must exist
            t_l1id = a_Innovs.CheckInnovation(t_in, t_nid, NEW_LINK);
            t_l2id = a_Innovs.CheckInnovation(t_nid, t_out, NEW_LINK);

            ASSERT((t_l1id > 0) && (t_l2id > 0));

            // Perhaps this innovation occured more than once. Find the
            // first such innovation that had occured, but the genome
            // not having the same id.. If didn't find such, then add new innovation.
            std::vector<int> t_idxs = a_Innovs.CheckAllInnovations(t_in, t_out, NEW_NEURON);
            bool t_found = false;
            for (unsigned int i = 0; i < t_idxs.size(); i++)
            {
                if (!HasNeuronID(a_Innovs.GetInnovationByIdx(t_idxs[i]).NeuronID()))
                {
                    // found such innovation & this genome doesn't have that neuron ID
                    // So we are going to inherit the innovation
                    t_nid = a_Innovs.GetInnovationByIdx(t_idxs[i]).NeuronID();

                    // these must exist
                    t_l1id = a_Innovs.CheckInnovation(t_in, t_nid, NEW_LINK);
                    t_l2id = a_Innovs.CheckInnovation(t_nid, t_out, NEW_LINK);

                    ASSERT((t_l1id > 0) && (t_l2id > 0));

                    t_found = true;
                    break;
                }
            }

            // Such an innovation was not found or the genome has all neuron IDs
            // So we are going to add new innovation
            if (!t_found)
            {
                // Add 3 new innovations and replace the variables with them

                // Add the new neuron innovation
                t_nid = a_Innovs.AddNeuronInnovation(t_in, t_out, HIDDEN);
                // add the first link innovation
                t_l1id = a_Innovs.AddLinkInnovation(t_in, t_nid);
                // add the second innovation
                t_l2id = a_Innovs.AddLinkInnovation(t_nid, t_out);
            }


            // Add the neuron and the links
            double t_sy = m_NeuronGenes[GetNeuronIndex(t_in)].SplitY() + m_NeuronGenes[GetNeuronIndex(t_out)].SplitY();
            t_sy /= 2.0;

            // Create the neuron gene
            NeuronGene t_ngene(HIDDEN, t_nid, t_sy);

            double t_A = a_RNG.RandFloat();
            double t_B = a_RNG.RandFloat();
            double t_TC = a_RNG.RandFloat();
            double t_Bs = a_RNG.RandFloat();
            Scale(t_A, 0, 1, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Scale(t_B, 0, 1, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Scale(t_TC, 0, 1, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Scale(t_Bs, 0, 1, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            Clamp(t_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Clamp(t_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Clamp(t_TC, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Clamp(t_Bs, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            // Initialize the neuron gene's properties
            t_ngene.Init(t_A,
                         t_B,
                         t_TC,
                         t_Bs,
                         GetRandomActivation(a_Parameters, a_RNG));

            t_ngene.InitTraits(a_Parameters.NeuronTraits, a_RNG);

            // Make sure the recurrent flag is kept
            bool t_recurrentflag = t_chosenlink.IsRecurrent();

            // Add the NeuronGene
            m_NeuronGenes.emplace_back(t_ngene);
            // First link
            LinkGene l1 = LinkGene(t_in, t_nid, t_l1id, 1.0, t_recurrentflag);
            // make sure this weight is in the allowed interval
            Clamp(l1.m_Weight, a_Parameters.MinWeight, a_Parameters.MaxWeight);
            // initialize the link's traits
            l1.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l1);
            // Second link
            LinkGene l2 = LinkGene(t_nid, t_out, t_l2id, t_orig_weight, t_recurrentflag);
            // initialize the link's traits
            l2.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l2);
        }

        return true;
    }


    // ==============================================================
    // Adds a new link
    // ==============================================================
    // returns true if succesful
    bool Genome::Mutate_AddLink(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG)
    {
        // this variable tells where is the first noninput node
        int t_first_noninput = 0;

        // The pair of neurons that has to be connected (1 - in, 2 - out)
        // It may be the same neuron - this means that the connection is a looped recurrent one.
        // These are indexes in the NeuronGenes array!
        int t_n1idx = 0, t_n2idx = 0;

        // Should we make this connection recurrent?
        bool t_MakeRecurrent = false;

        // If so, should it be a looped one?
        bool t_LoopedRecurrent = false;

        // Should it come from the bias neuron?
        bool t_MakeBias = false;

        // Counter of tries to find a candidate pair of neuron/s to connect.
        unsigned int t_NumTries = 0;


        // Decide whether the connection will be recurrent or not..
        if (a_RNG.RandFloat() < a_Parameters.RecurrentProb)
        {
            t_MakeRecurrent = true;

            if (a_RNG.RandFloat() < a_Parameters.RecurrentLoopProb)
            {
                t_LoopedRecurrent = true;
            }
        }
            // if not recurrent, there is a probability that this link will be from the bias
            // if such link doesn't already exist.
            // in case such link exists, search for a standard feed-forward connection place
        else
        {
            if (a_RNG.RandFloat() < a_Parameters.MutateAddLinkFromBiasProb)
            {
                t_MakeBias = true;
            }
        }

        // Try to find a good pair of neurons
        bool t_Found = false;

        // Find the first noninput node
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if ((m_NeuronGenes[i].Type() == INPUT) || (m_NeuronGenes[i].Type() == BIAS))
            {
                t_first_noninput++;
            }
            else
            {
                break;
            }
        }

        // A forward link is characterized with the fact that
        // the From neuron has less or equal SplitY value

        // find a good pair of nodes for a forward link
        if (!t_MakeRecurrent)
        {
            // first see if this should come from the bias or not
            bool t_found_bias = true;
            t_n1idx = static_cast<int>(NumInputs() - 1); // the bias is always the last input
            // try to find a neuron that is not connected to the bias already
            t_NumTries = 0;
            do
            {
                t_n2idx = a_RNG.RandInt(t_first_noninput, static_cast<int>(NumNeurons() - 1));
                t_NumTries++;

                if (t_NumTries >= a_Parameters.LinkTries)
                {
                    // couldn't find anything
                    t_found_bias = false;
                    break;
                }
            }
            while ((HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID()))); // already present?

            // so if we found that link, we can skip the rest of the things
            if (t_found_bias && t_MakeBias)
            {
                t_Found = true;
            }
                // otherwise continue trying to find a normal forward link
            else
            {
                t_NumTries = 0;
                // try to find a standard forward connection
                do
                {
                    t_n1idx = a_RNG.RandInt(0, static_cast<int>(NumNeurons() - 1));
                    t_n2idx = a_RNG.RandInt(t_first_noninput, static_cast<int>(NumNeurons() - 1));
                    t_NumTries++;

                    if (t_NumTries >= a_Parameters.LinkTries)
                    {
                        // couldn't find anything
                        // say goodbye
                        return false;
                    }
                }
                while (
                        (HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID())) // already present?
                        ||
                        (m_NeuronGenes[t_n1idx].Type() == OUTPUT) // consider connections out of outputs recurrent
                        ||
                        (t_n1idx == t_n2idx) // make sure they differ
                        );

                // it found a good pair of neurons
                t_Found = true;
            }
        }
            // find a good pair of nodes for a recurrent link (non-looped)
        else if (t_MakeRecurrent && !t_LoopedRecurrent)
        {
            t_NumTries = 0;
            do
            {
                t_n1idx = a_RNG.RandInt(t_first_noninput, static_cast<int>(NumNeurons() - 1));
                t_n2idx = a_RNG.RandInt(t_first_noninput, static_cast<int>(NumNeurons() - 1));
                t_NumTries++;

                if (t_NumTries >= a_Parameters.LinkTries)
                {
                    // couldn't find anything
                    // say goodbye
                    return false;
                }
            }
                // NOTE: this considers output-output connections as forward. Should be fixed.
            while (
                    (HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID())) // already present?
                    ||
                    (t_n1idx == t_n2idx) // they should differ
                    );

            // it found a good pair of neurons
            t_Found = true;
        }
            // find a good neuron to make a looped recurrent link
        else if (t_MakeRecurrent && t_LoopedRecurrent)
        {
            t_NumTries = 0;
            do
            {
                t_n1idx = t_n2idx = a_RNG.RandInt(t_first_noninput, static_cast<int>(NumNeurons() - 1));
                t_NumTries++;

                if (t_NumTries >= a_Parameters.LinkTries)
                {
                    // couldn't find anything
                    // say goodbye
                    return false;
                }
            }
            while (
                    (HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID())) // already present?
                    );

            // it found a good pair of neurons
            t_Found = true;
        }


        // To make sure it is all right
        if (!t_Found)
        {
            return false;
        }

        // This link MUST NOT be a part of the genome by any reason
        ASSERT((!HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID()))); // already present?

        // extract the neuron IDs from the indexes
        int t_n1id = m_NeuronGenes[t_n1idx].ID();
        int t_n2id = m_NeuronGenes[t_n2idx].ID();

        // So we have a good pair of neurons to connect. See the innovation database if this is novel innovation.
        int t_innovid = a_Innovs.CheckInnovation(t_n1id, t_n2id, NEW_LINK);

        // Choose the weight for this link
        double t_weight = a_RNG.RandFloat();
        Scale(t_weight, 0, 1, a_Parameters.MinWeight, a_Parameters.MaxWeight);

        // A novel innovation?
        if (t_innovid == -1)
        {
            // Make new innovation
            t_innovid = a_Innovs.AddLinkInnovation(t_n1id, t_n2id);
        }

        // Create and add the link
        LinkGene l = LinkGene(t_n1id, t_n2id, t_innovid, t_weight, t_MakeRecurrent);
        // init the link's traits
        l.InitTraits(a_Parameters.LinkTraits, a_RNG);
        m_LinkGenes.emplace_back(l);

        // All done.
        return true;
    }

    // ==============================================================
    // Removes a random link
    // ==============================================================
    bool Genome::Mutate_RemoveLink(RNG &a_RNG)
    {
        if(NumLinks()<2) return false;
        int idx = a_RNG.RandInt(0,(int)NumLinks()-1);
        RemoveLinkGene(m_LinkGenes[idx].InnovationID());
        return true;
    }

    // ==============================================================
    // Removes a hidden neuron that has exactly 1 incoming and 1 outgoing link
    // ==============================================================
    bool Genome::Mutate_RemoveSimpleNeuron(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG)
    {
        // At least one hidden node must be present
        if (NumNeurons() == (NumInputs() + NumOutputs()))
            return false;

        // Build a list of candidate neurons for deletion
        // Indexes!
        std::vector<int> t_neurons_to_delete;
        for (int i = 0; i < NumNeurons(); i++)
        {
            if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1)
                && (m_NeuronGenes[i].Type() == HIDDEN))
            {
                t_neurons_to_delete.emplace_back(i);
            }
        }

        // If the list is empty, say goodbye
        if (t_neurons_to_delete.size() == 0)
            return false;

        // Now choose a random one to delete
        int t_choice;
        if (t_neurons_to_delete.size() == 2)
            t_choice = Rounded(a_RNG.RandFloat());
        else
            t_choice = a_RNG.RandInt(0, static_cast<int>(t_neurons_to_delete.size() - 1));

        // the links in & out
        int t_l1idx = -1, t_l2idx = -1;

        // find the link outputting to the neuron
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].ToNeuronID() == m_NeuronGenes[t_neurons_to_delete[t_choice]].ID())
            {
                t_l1idx = i;
                break;
            }
        }
        // find the link inputting from the neuron
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].FromNeuronID() == m_NeuronGenes[t_neurons_to_delete[t_choice]].ID())
            {
                t_l2idx = i;
                break;
            }
        }

        ASSERT((t_l1idx >= 0) && (t_l2idx >= 0));

        // OK now see if a link connecting the original 2 nodes is present. If it is, we will just
        // delete the neuron and quit.
        if (HasLink(m_LinkGenes[t_l1idx].FromNeuronID(), m_LinkGenes[t_l2idx].ToNeuronID()))
        {
            RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());
            return true;
        }
            // Else the link is not present and we will replace the neuron and 2 links with one link
        else
        {
            // Remember the first link's weight
            double t_weight = m_LinkGenes[t_l1idx].GetWeight();

            // See the innovation database for an innovation number
            int t_innovid = a_Innovs.CheckInnovation(m_LinkGenes[t_l1idx].FromNeuronID(),
                                                     m_LinkGenes[t_l2idx].ToNeuronID(), NEW_LINK);

            // a novel innovation?
            if (t_innovid == -1)
            {
                // Save the IDs for a while
                int from = m_LinkGenes[t_l1idx].FromNeuronID();
                int to = m_LinkGenes[t_l2idx].ToNeuronID();
                
                // Remove the neuron and its links now
                RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());

                // Add the innovation and the link gene
                int t_newinnov = a_Innovs.AddLinkInnovation(from, to);
                LinkGene lg = LinkGene(from, to, t_newinnov, t_weight, false);
                lg.InitTraits(a_Parameters.LinkTraits, a_RNG);
                
                m_LinkGenes.emplace_back(lg);
                
                // bye
                return true;
            }
            // not a novel innovation
            else
            {
                // Save the IDs for a while
                int from = m_LinkGenes[t_l1idx].FromNeuronID();
                int to = m_LinkGenes[t_l2idx].ToNeuronID();
                
                // Remove the neuron and its links now
                RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());
    
                // Add the link
                LinkGene lg = LinkGene(from, to, t_innovid, t_weight, false);
                lg.InitTraits(a_Parameters.LinkTraits, a_RNG);
                m_LinkGenes.emplace_back(lg);
                
                // TODO: Maybe inherit the traits from one of the links

                // bye
                return true;
            }
        }

        return false;
    }

    // ==============================================================
    // Removes a link gene by its innov
    // ==============================================================
    void Genome::RemoveLinkGene(int a_innovid)
    {
        for(int i=0; i<(int)m_LinkGenes.size(); i++)
        {
            if(m_LinkGenes[i].InnovationID()==a_innovid)
            {
                m_LinkGenes.erase(m_LinkGenes.begin()+i);
                break;
            }
        }
    }

    // ==============================================================
    // Removes a neuron gene by ID
    // Also removes connected links
    // ==============================================================
    void Genome::RemoveNeuronGene(int a_ID)
    {
        bool removed=false;
        do {
            removed=false;
            for (int i=0; i<(int)NumLinks(); i++)
            {
                if(m_LinkGenes[i].FromNeuronID()==a_ID
                || m_LinkGenes[i].ToNeuronID()==a_ID)
                {
                    RemoveLinkGene(m_LinkGenes[i].InnovationID());
                    removed=true;
                    break;
                }
            }
        } while(removed);

        for(auto it=m_NeuronGenes.begin(); it!=m_NeuronGenes.end(); ++it)
        {
            if(it->ID()==a_ID)
            {
                m_NeuronGenes.erase(it);
                break;
            }
        }
    }

    bool Genome::Cleanup()
    {
        bool t_removed = false;

        // remove any dead-end hidden neurons
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].Type() == HIDDEN)
            {
                if (IsDeadEndNeuron(m_NeuronGenes[i].ID()))
                {
                    RemoveNeuronGene(m_NeuronGenes[i].ID());
                    t_removed = true;
                }
            }
        }

        // a special case are isolated outputs - these are outputs having
        // one and only one looped recurrent connection
        // we simply remove these connections and leave the outputs naked.
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].Type() == OUTPUT)
            {
                // Only outputs with 1 input and 1 output connection are considered.
                if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1))
                {
                    // that must be a lonely looped recurrent,
                    // because we know that the outputs are the dead end of the network
                    // find this link
                    for (unsigned int j = 0; j < NumLinks(); j++)
                    {
                        if (m_LinkGenes[j].ToNeuronID() == m_NeuronGenes[i].ID())
                        {
                            // Remove it.
                            RemoveLinkGene(m_LinkGenes[j].InnovationID());
                            t_removed = true;
                        }
                    }
                }
            }
        }

        return t_removed;
    }


    // Returns true if has any dead end
    bool Genome::HasDeadEnds() const
    {
        // any dead-end hidden neurons?
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].Type() == HIDDEN)
            {
                if (IsDeadEndNeuron(m_NeuronGenes[i].ID()))
                {
                    return true;
                }
            }
        }

        // a special case are isolated outputs - these are outputs having
        // one and only one looped recurrent connection or no connections at all
        for (unsigned int i = 0; i < NumNeurons(); i++)
        {
            if (m_NeuronGenes[i].Type() == OUTPUT)
            {
                // Only outputs with 1 input and 1 output connection are considered.
                if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1))
                {
                    // that must be a lonely looped recurrent,
                    // because we know that the outputs are the dead end of the network
                    return true;
                }

                // There may be cases for totally isolated outputs
                // Consider this if only one output is present
                if (NumOutputs() == 1)
                    if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 0) &&
                        (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 0))
                    {
                        return true;
                    }
            }
        }

        return false;
    }


    // ==============================================================
    // Mating
    // ==============================================================
    Genome Genome::Mate(Genome &a_Dad, bool a_MateAverage, bool a_InterSpecies, RNG &a_RNG, Parameters &a_Parameters)
    {
        // Cannot mate with itself
        if (GetID() == a_Dad.GetID())
            return *this;

        // helps make the code clearer
        enum t_parent_type
        {
            MOM, DAD,
        };

        // This is the fittest genome.
        t_parent_type t_better;

        // This empty genome will hold the baby
        Genome t_baby;

        // create iterators so we can step through each parents genes and set
        // them to the first gene of each parent
        std::vector<LinkGene>::iterator t_curMom = m_LinkGenes.begin();
        std::vector<LinkGene>::iterator t_curDad = a_Dad.m_LinkGenes.begin();

        // this will hold a copy of the gene we wish to add at each step
        LinkGene t_selectedgene(0, 0, -1, 0, false);
        
        // Mate the GenomeGene first
        // Determine if it will pick either gene or mate it
        if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
        {
            // pick
            Gene n;
            
            if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
            {
                n = (GetFitness() > a_Dad.GetFitness()) ? m_GenomeGene : a_Dad.m_GenomeGene;
            }
            else
            {
                n = (a_RNG.RandFloat() < 0.5) ? m_GenomeGene : a_Dad.m_GenomeGene;
            }
            t_baby.m_GenomeGene = n;
        }
        else
        {
            // mate
            Gene n = m_GenomeGene;
            n.MateTraits(a_Dad.m_GenomeGene.m_Traits, a_RNG);
            t_baby.m_GenomeGene = n;
        }
    
    
        // Make sure all inputs/outputs are present in the baby
        // Essential to FS-NEAT

        if (!a_Parameters.DontUseBiasNeuron)
        {
            // the inputs
            unsigned int i = 0;
            for (i = 0; i < m_NumInputs - 1; i++)
            {
                t_baby.m_NeuronGenes.emplace_back(m_NeuronGenes[i]);
            }
            t_baby.m_NeuronGenes.emplace_back(m_NeuronGenes[i]);
        }
        else
        {
            // the inputs
            for (unsigned int i = 0; i < m_NumInputs; i++)
            {
                t_baby.m_NeuronGenes.emplace_back(m_NeuronGenes[i]);
            }
        }

        // the outputs
        for (unsigned int i = 0; i < m_NumOutputs; i++)
        {
            NeuronGene t_tempneuron(OUTPUT, 0, 1);

            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
            {
                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                {
                    if (GetFitness() > a_Dad.GetFitness())
                    {
                        // from mother
                        t_tempneuron = GetNeuronByIndex(i + m_NumInputs);
                    }
                    else
                    {
                        // from father
                        t_tempneuron = a_Dad.GetNeuronByIndex(i + m_NumInputs);
                    }
                }
                else
                {
                    // random pick
                    if (a_RNG.RandFloat() < 0.5)
                    {
                        // from mother
                        t_tempneuron = GetNeuronByIndex(i + m_NumInputs);
                    }
                    else
                    {
                        // from father
                        t_tempneuron = a_Dad.GetNeuronByIndex(i + m_NumInputs);
                    }
                }
            }
            else
            {
                // mating
                // from mother
                t_tempneuron = GetNeuronByIndex(i + m_NumInputs);
                t_tempneuron.MateTraits(a_Dad.GetNeuronByIndex(i + m_NumInputs).m_Traits, a_RNG);
            }

            t_baby.m_NeuronGenes.emplace_back(t_tempneuron);
        }

        // if they are of equal fitness use the shorter (because we want to keep
        // the networks as small as possible)
        if (GetFitness() == a_Dad.GetFitness())
        {
            // if they are of equal fitness and length just choose one at
            // random
            if (NumLinks() == a_Dad.NumLinks())
            {
                if (a_RNG.RandFloat() < 0.5)
                {
                    t_better = MOM;
                }
                else
                {
                    t_better = DAD;
                }
            }
            else
            {
                if (NumLinks() < a_Dad.NumLinks())
                {
                    t_better = MOM;
                }
                else
                {
                    t_better = DAD;
                }
            }
        }
        else
        {
            if (GetFitness() > a_Dad.GetFitness())
            {
                t_better = MOM;
            }
            else
            {
                t_better = DAD;
            }
        }

        //////////////////////////////////////////////////////////
        // The better genome has been chosen. Now we mate them.
        //////////////////////////////////////////////////////////

        // for cleaning up
        LinkGene t_emptygene(0, 0, -1, 0, false);
        bool t_skip = false;
        int t_innov_mom, t_innov_dad;

        // step through each parents link genes until we reach the end of both
        while (!((t_curMom == m_LinkGenes.end()) && (t_curDad == a_Dad.m_LinkGenes.end())))
        {
            t_selectedgene = t_emptygene;
            t_skip = false;
            t_innov_mom = t_innov_dad = 0;

            // the end of mum's genes have been reached
            // EXCESS
            if (t_curMom == m_LinkGenes.end())
            {
                // select dads gene
                t_selectedgene = *t_curDad;
                // move onto dad's next gene
                t_curDad++;

                // if mom is fittest, abort adding
                if (t_better == MOM)
                {
                    t_skip = true;
                }
            }

            // the end of dads's genes have been reached
            // EXCESS
            else if (t_curDad == a_Dad.m_LinkGenes.end())
            {
                // add mums gene
                t_selectedgene = *t_curMom;
                // move onto mum's next gene
                t_curMom++;

                // if dad is fittest, abort adding
                if (t_better == DAD)
                {
                    t_skip = true;
                }
            }
            else
            {
                // extract the innovation numbers
                t_innov_mom = t_curMom->InnovationID();
                t_innov_dad = t_curDad->InnovationID();

                // if both innovations match
                if (t_innov_mom == t_innov_dad)
                {
                    // get a gene from either parent or average
                    if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                    {
                        if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                        {
                            if (GetFitness() < a_Dad.GetFitness())
                            {
                                t_selectedgene = *t_curMom;
                            }
                            else
                            {
                                t_selectedgene = *t_curDad;
                            }
                        }
                        else
                        {
                            if (a_RNG.RandFloat() < 0.5)
                            {
                                t_selectedgene = *t_curMom;
                            }
                            else
                            {
                                t_selectedgene = *t_curDad;
                            }
                        }
                    }
                    else
                    {
                        t_selectedgene = *t_curMom;
                        const double t_Weight = (t_curDad->GetWeight() + t_curMom->GetWeight()) / 2.0;
                        t_selectedgene.SetWeight(t_Weight);
                        // Mate traits here
                        t_selectedgene.MateTraits(t_curDad->m_Traits, a_RNG);
                    }

                    // move onto next gene of each parent
                    t_curMom++;
                    t_curDad++;
                }
                else // DISJOINT
                if (t_innov_mom < t_innov_dad)
                {
                    t_selectedgene = *t_curMom;
                    t_curMom++;

                    if (t_better == DAD)
                    {
                        t_skip = true;
                    }
                }
                else // DISJOINT
                if (t_innov_dad < t_innov_mom)
                {
                    t_selectedgene = *t_curDad;
                    t_curDad++;

                    if (t_better == MOM)
                    {
                        t_skip = true;
                    }
                }
            }

            // for interspecies mating, allow all genes through
            if (a_InterSpecies)
            {
                t_skip = false;
            }

            // If the selected gene's innovation number is negative,
            // this means that no gene is selected (should be skipped)
            // also check the baby if it already has this link (maybe unnecessary)
            if ((t_selectedgene.InnovationID() > 0) &&
                (!t_baby.HasLink(t_selectedgene.FromNeuronID(), t_selectedgene.ToNeuronID())))
            {
                if (!t_skip)
                {
                    t_baby.m_LinkGenes.emplace_back(t_selectedgene);

                    // mom has a neuron ID not present in the baby?
                    // From
                    if ((!t_baby.HasNeuronID(t_selectedgene.FromNeuronID())) &&
                        (HasNeuronID(t_selectedgene.FromNeuronID())))
                    {
                        // See if dad has the same neuron.
                        if (a_Dad.HasNeuronID(t_selectedgene.FromNeuronID()))
                        {
                            // if so, then choose randomly which neuron the baby shoud inherit
                            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                            {
                                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                                {
                                    if (GetFitness() > a_Dad.GetFitness())
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                                    }
                                    else
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(
                                                        t_selectedgene.FromNeuronID())]);
                                    }
                                }
                                else
                                {
                                    if (a_RNG.RandFloat() < 0.5)
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                                    }
                                    else
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(
                                                        t_selectedgene.FromNeuronID())]);
                                    }
                                }
                            }
                            else
                            {
                                // mate the neurons
                                NeuronGene t_1 = m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())];
                                NeuronGene t_2 = a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.FromNeuronID())];
                                t_1.MateTraits(t_2.m_Traits, a_RNG);
                                t_baby.m_NeuronGenes.emplace_back(t_1);
                            }
                        }
                        else
                        {
                            // add mom's neuron to the baby
                            t_baby.m_NeuronGenes.emplace_back(
                                    m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                        }
                    }

                    // To
                    if ((!t_baby.HasNeuronID(t_selectedgene.ToNeuronID())) &&
                        (HasNeuronID(t_selectedgene.ToNeuronID())))
                    {
                        // See if dad has the same neuron.
                        if (a_Dad.HasNeuronID(t_selectedgene.ToNeuronID()))
                        {
                            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                            {
                                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                                {
                                    if (GetFitness() > a_Dad.GetFitness())
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                    else
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                }
                                else
                                {
                                    // if so, then choose randomly which neuron the baby shoud inherit
                                    if (a_RNG.RandFloat() < 0.5)
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                    else
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                }
                            }
                            else
                            {
                                // mate the neurons
                                NeuronGene t_1 = m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())];
                                NeuronGene t_2 = a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())];
                                t_1.MateTraits(t_2.m_Traits, a_RNG);
                                t_baby.m_NeuronGenes.emplace_back(t_1);
                            }
                        }
                        else
                        {
                            // add mom's neuron to the baby
                            t_baby.m_NeuronGenes.emplace_back(m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                        }
                    }

                    // dad has a neuron ID not present in the baby?
                    // From
                    if ((!t_baby.HasNeuronID(t_selectedgene.FromNeuronID())) &&
                        (a_Dad.HasNeuronID(t_selectedgene.FromNeuronID())))
                    {
                        // See if mom has the same neuron
                        if (HasNeuronID(t_selectedgene.FromNeuronID()))
                        {
                            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                            {
                                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                                {
                                    if (GetFitness() < a_Dad.GetFitness())
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(
                                                        t_selectedgene.FromNeuronID())]);
                                    }
                                    else
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                                    }
                                }
                                else
                                {
                                    // if so, then choose randomly which neuron the baby shoud inherit
                                    if (a_RNG.RandFloat() < 0.5)
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(
                                                        t_selectedgene.FromNeuronID())]);
                                    }
                                    else
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                                    }
                                }
                            }
                            else
                            {
                                // mate the neurons
                                NeuronGene t_1 = a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.FromNeuronID())];
                                NeuronGene t_2 = m_NeuronGenes[GetNeuronIndex(t_selectedgene.FromNeuronID())];
                                t_1.MateTraits(t_2.m_Traits, a_RNG);
                                t_baby.m_NeuronGenes.emplace_back(t_1);
                            }
                        }
                        else
                        {
                            // add dad's neuron to the baby
                            t_baby.m_NeuronGenes.emplace_back(
                                    a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                        }
                    }

                    // To
                    if ((!t_baby.HasNeuronID(t_selectedgene.ToNeuronID())) &&
                        (a_Dad.HasNeuronID(t_selectedgene.ToNeuronID())))
                    {
                        // See if mom has the same neuron
                        if (HasNeuronID(t_selectedgene.ToNeuronID()))
                        {
                            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                            {
                                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                                {
                                    if (GetFitness() < a_Dad.GetFitness())
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                    else
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                }
                                else
                                {
                                    // if so, then choose randomly which neuron the baby shoud inherit
                                    if (a_RNG.RandFloat() < 0.5)
                                    {
                                        // add dad's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                    else
                                    {
                                        // add mom's neuron to the baby
                                        t_baby.m_NeuronGenes.emplace_back(
                                                m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                                    }
                                }
                            }
                            else
                            {
                                // mate neurons
                                NeuronGene t_1 = a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())];
                                NeuronGene t_2 = m_NeuronGenes[GetNeuronIndex(t_selectedgene.ToNeuronID())];
                                t_1.MateTraits(t_2.m_Traits, a_RNG);
                                t_baby.m_NeuronGenes.emplace_back(t_1);
                            }
                        }
                        else
                        {
                            // add dad's neuron to the baby
                            t_baby.m_NeuronGenes.emplace_back(
                                    a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                        }
                    }
                }
            }
        } //end while

        t_baby.m_NumInputs = m_NumInputs;
        t_baby.m_NumOutputs = m_NumOutputs;

        // Sort the baby's genes
        t_baby.SortGenes();

        return t_baby;
    }

    // ==============================================================
    // Sorting Genes
    // ==============================================================
    void Genome::SortGenes()
    {
        std::sort(m_NeuronGenes.begin(), m_NeuronGenes.end(),
            [](const NeuronGene &lhs, const NeuronGene &rhs){
                return lhs.ID()<rhs.ID();
            });
        std::sort(m_LinkGenes.begin(), m_LinkGenes.end(),
            [](const LinkGene &lhs, const LinkGene &rhs){
                return lhs.InnovationID()<rhs.InnovationID();
            });
    }

    // ==============================================================
    // Depth
    // ==============================================================
    unsigned int Genome::NeuronDepth(int a_NeuronID, unsigned int a_Depth)
    {
        unsigned int t_current_depth;
        unsigned int t_max_depth = a_Depth;

        if (a_Depth > 16384)
        {
            // oops! a possible loop in the network!
            // DBG(" ERROR! Trying to get the depth of a looped network!");
            return 16384;
        }

        // Base case
        if ((GetNeuronByID(a_NeuronID).Type() == INPUT) || (GetNeuronByID(a_NeuronID).Type() == BIAS))
        {
            return a_Depth;
        }

        // Find all links outputting to this neuron ID
        std::vector<int> t_inputting_links_idx;
        for (unsigned int i = 0; i < NumLinks(); i++)
        {
            if (m_LinkGenes[i].ToNeuronID() == a_NeuronID)
                t_inputting_links_idx.emplace_back(i);
        }

        // For all incoming links..
        for (unsigned int i = 0; i < t_inputting_links_idx.size(); i++)
        {
            LinkGene t_link = GetLinkByIndex(t_inputting_links_idx[i]);

            // RECURSION
            t_current_depth = NeuronDepth(t_link.FromNeuronID(), a_Depth + 1);
            if (t_current_depth > t_max_depth)
                t_max_depth = t_current_depth;
        }

        return t_max_depth;
    }


    void Genome::CalculateDepth()
    {
        // snippet logic
        // If no hidden, set 1
        if(NumNeurons()==(m_NumInputs+m_NumOutputs)) m_Depth=1;
        else m_Depth=1; // or do the real recursion if needed
    }

    // ==============================================================
    // Genome loading from file
    // ==============================================================
    Genome::Genome(const char *a_FileName)
    {
        std::ifstream data(a_FileName);
        if(!data.is_open()) throw std::runtime_error("Cannot open genome file.");

        std::string st;
        do {
            data >> st;
        }
        while(st!="GenomeStart" && !data.eof());

        data >> m_ID;

        // read until GenomeEnd
        do {
            data >> st;
            if(st=="Neuron")
            {
                int tid, ttype, tact;
                double tsplity, ta, tb, ttc, tbias;
                data >> tid;     // ID
                data >> ttype;   // type
                data >> tsplity;
                data >> tact;    // act func
                data >> ta;
                data >> tb;
                data >> ttc;
                data >> tbias;

                NeuronGene N((NeuronType)ttype, tid, tsplity);
                N.m_ActFunction  = (ActivationFunction)tact;
                N.m_A            = ta;
                N.m_B            = tb;
                N.m_TimeConstant = ttc;
                N.m_Bias         = tbias;

                m_NeuronGenes.push_back(N);
            }
            else if(st=="Link")
            {
                int f,t,inv,isrec;
                double w;
                data >> f;    // from
                data >> t;    // to
                data >> inv;  // innov
                data >> isrec;// rec
                data >> w;    // weight

                LinkGene L(f,t,inv,w,(bool)isrec);
                m_LinkGenes.push_back(L);
            }
        } while(st!="GenomeEnd" && !data.eof());
        data.close();

        // count inputs/outputs
        m_NumInputs=0;
        m_NumOutputs=0;
        for(unsigned int i=0; i<m_NeuronGenes.size(); i++)
        {
            if(m_NeuronGenes[i].Type()==INPUT||m_NeuronGenes[i].Type()==BIAS)
                m_NumInputs++;
            else if(m_NeuronGenes[i].Type()==OUTPUT)
                m_NumOutputs++;
        }

        m_Fitness=0; 
        m_AdjustedFitness=0;
        m_OffspringAmount=0;
        m_Depth=0;
        m_Evaluated=false;
        m_PhenotypeBehavior=nullptr;
        m_initial_num_neurons = (int)NumNeurons();
        m_initial_num_links   = (int)NumLinks();
    }

    Genome::Genome(std::ifstream &data)
    {
        if(!data) throw std::runtime_error("Invalid file stream for Genome constructor.");

        std::string st;
        do {
            data>>st;
        }
        while(st!="GenomeStart" && !data.eof());

        data >> m_ID;

        do {
            data >> st;
            if(st=="Neuron")
            {
                int tid, ttype, tact;
                double tsplity, ta, tb, ttc, tbias;
                data>> tid;
                data>> ttype;
                data>> tsplity;
                data>> tact;
                data>> ta; data>> tb; data>> ttc; data>> tbias;

                NeuronGene N((NeuronType)ttype, tid, tsplity);
                N.m_ActFunction=(ActivationFunction)tact;
                N.m_A=ta; N.m_B=tb; N.m_TimeConstant=ttc; N.m_Bias=tbias;
                m_NeuronGenes.push_back(N);
            }
            else if(st=="Link")
            {
                int f,t,inv,isrec;
                double w;
                data >> f; data >> t; data >> inv; data >> isrec; data >> w;
                LinkGene L(f,t,inv,w,(bool)isrec);
                m_LinkGenes.push_back(L);
            }
        } while(st!="GenomeEnd" && !data.eof());

        // count input/output
        m_NumInputs=0;
        m_NumOutputs=0;
        for(unsigned int i=0; i<m_NeuronGenes.size(); i++)
        {
            if(m_NeuronGenes[i].Type()==INPUT||m_NeuronGenes[i].Type()==BIAS)
                m_NumInputs++;
            else if(m_NeuronGenes[i].Type()==OUTPUT)
                m_NumOutputs++;
        }
        m_Fitness=0;
        m_AdjustedFitness=0;
        m_OffspringAmount=0;
        m_Depth=0;
        m_Evaluated=false;
        m_PhenotypeBehavior=nullptr;
        m_initial_num_neurons= (int)NumNeurons();
        m_initial_num_links  = (int)NumLinks();
    }

    // ==============================================================
    // Save the genome to a file
    // ==============================================================
    void Genome::Save(const char *a_FileName)
    {
        FILE* fp=fopen(a_FileName,"w");
        if(!fp) throw std::runtime_error("Cannot open file for Genome::Save()");
        Save(fp);
        fclose(fp);
    }

    // ==============================================================
    // Save to an already opened file
    // ==============================================================
    void Genome::Save(FILE *fp)
    {
        fprintf(fp,"GenomeStart %d\n",m_ID);

        // Neurons
        for (unsigned int i=0; i<m_NeuronGenes.size(); i++)
        {
            fprintf(fp,"Neuron %d %d %3.8f %d %3.8f %3.8f %3.8f %3.8f\n",
                    m_NeuronGenes[i].m_ID,
                    (int)m_NeuronGenes[i].m_Type,
                    m_NeuronGenes[i].m_SplitY,
                    (int)m_NeuronGenes[i].m_ActFunction,
                    m_NeuronGenes[i].m_A,
                    m_NeuronGenes[i].m_B,
                    m_NeuronGenes[i].m_TimeConstant,
                    m_NeuronGenes[i].m_Bias);
        }

        // Links
        for (unsigned int i=0; i<m_LinkGenes.size(); i++)
        {
            fprintf(fp,"Link %d %d %d %d %3.8f\n",
                    m_LinkGenes[i].m_FromNeuronID,
                    m_LinkGenes[i].m_ToNeuronID,
                    m_LinkGenes[i].m_InnovationID,
                    (int)m_LinkGenes[i].m_IsRecurrent,
                     m_LinkGenes[i].m_Weight);
        }

        fprintf(fp,"GenomeEnd\n\n");
    }

    // ==============================================================
    // Print traits
    // ==============================================================
    void Genome::PrintTraits(std::map<std::string, Trait>& traits)
    {
        for(auto &kv : traits)
        {
            bool doit=false;
            if(!kv.second.dep_key.empty())
            {
                if(traits.count(kv.second.dep_key)!=0)
                {
                    for(auto &dv : kv.second.dep_values)
                    {
                        if(traits.at(kv.second.dep_key).value==dv)
                        {
                            doit=true;
                            break;
                        }
                    }
                }
            }
            else
            {
                doit=true;
            }
            if(doit)
            {
                std::cout<<kv.first<<" - ";
                if(std::holds_alternative<int>(kv.second.value))
                {
                    std::cout<<std::get<int>(kv.second.value);
                }
                else if(std::holds_alternative<double>(kv.second.value))
                {
                    std::cout<<std::get<double>(kv.second.value);
                }
                else if(std::holds_alternative<std::string>(kv.second.value))
                {
                    std::cout<<"\""<<std::get<std::string>(kv.second.value)<<"\"";
                }
                else if(std::holds_alternative<intsetelement>(kv.second.value))
                {
                    std::cout<<std::get<intsetelement>(kv.second.value).value;
                }
                else if(std::holds_alternative<floatsetelement>(kv.second.value))
                {
                    std::cout<<std::get<floatsetelement>(kv.second.value).value;
                }
                std::cout<<", ";
            }
        }
    }

    // ==============================================================
    // Print all traits
    // ==============================================================
    void Genome::PrintAllTraits()
    {
        std::cout<<"====================================================================\n";
        std::cout<<"Genome:\n==================================\n";
        PrintTraits(m_GenomeGene.m_Traits);
        std::cout<<"\n";

        std::cout<<"====================================================================\n";
        std::cout<<"Neurons:\n==================================\n";
        for(auto &n : m_NeuronGenes)
        {
            std::cout<<"ID: "<<n.ID()<<" : ";
            PrintTraits(n.m_Traits);
            std::cout<<"\n";
        }
        std::cout<<"==================================\n";

        std::cout<<"Links:\n==================================\n";
        for(auto &l : m_LinkGenes)
        {
            std::cout<<"ID: "<<l.InnovationID()<<" : ";
            PrintTraits(l.m_Traits);
            std::cout<<"\n";
        }
        std::cout<<"==================================\n";
        std::cout<<"====================================================================\n";
    }

} // namespace NEAT
