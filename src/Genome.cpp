#ifndef GENOME_CPP
#define GENOME_CPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <functional>
#include <map>
#include <queue>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include "Assert.h"
#include "Genome.h"
#include "Random.h"
#include "Parameters.h"
#include "Substrate.h"

namespace NEAT
{

// Helper inline
inline double sqr(double x) { return x * x; }

int Genome::GetNeuronIndex(int a_ID) const
{
    ASSERT(a_ID > 0);
    auto it = std::find_if(m_NeuronGenes.begin(), m_NeuronGenes.end(),
                           [a_ID](const NeuronGene &n) { return n.ID() == a_ID; });
    if (it != m_NeuronGenes.end())
        return static_cast<int>(std::distance(m_NeuronGenes.begin(), it));
    return -1;
}

int Genome::GetLinkIndex(int a_InnovID) const
{
    ASSERT(a_InnovID > 0 && !m_LinkGenes.empty());
    auto it = std::find_if(m_LinkGenes.begin(), m_LinkGenes.end(),
                           [a_InnovID](const LinkGene &l) { return l.InnovationID() == a_InnovID; });
    if (it != m_LinkGenes.end())
        return static_cast<int>(std::distance(m_LinkGenes.begin(), it));
    return -1;
}

void Genome::RemoveLinkGene(int a_innovid)
{
    auto it = std::find_if(m_LinkGenes.begin(), m_LinkGenes.end(),
                           [a_innovid](const LinkGene &l) { return l.InnovationID() == a_innovid; });
    if (it != m_LinkGenes.end())
        m_LinkGenes.erase(it);
}


void Genome::RemoveNeuronGene(int a_ID)
{
    m_LinkGenes.erase(std::remove_if(m_LinkGenes.begin(), m_LinkGenes.end(),
                     [a_ID](const LinkGene &l)
                     { return (l.FromNeuronID() == a_ID || l.ToNeuronID() == a_ID); }),
                     m_LinkGenes.end());
    auto it = std::find_if(m_NeuronGenes.begin(), m_NeuronGenes.end(),
                           [a_ID](const NeuronGene &ng) { return ng.ID() == a_ID; });
    if (it != m_NeuronGenes.end())
        m_NeuronGenes.erase(it);
}


bool Genome::HasNeuronID(int a_ID) const
{
    ASSERT(a_ID > 0);
    for (const auto &n : m_NeuronGenes)
    {
        if (n.ID() == a_ID)
            return true;
    }
    return false;
}


bool Genome::HasLink(int a_n1id, int a_n2id) const
{
    ASSERT(a_n1id > 0 && a_n2id > 0);
    for (const auto &l : m_LinkGenes)
    {
        if (l.FromNeuronID() == a_n1id && l.ToNeuronID() == a_n2id)
            return true;
    }
    return false;
}


Genome::Genome()
  : m_ID(0), m_Fitness(0.0), m_AdjustedFitness(0.0),
    m_OffspringAmount(0.0), m_Depth(0), m_NumInputs(0), m_NumOutputs(0),
    m_Evaluated(false), m_PhenotypeBehavior(nullptr),
    m_initial_num_neurons(0), m_initial_num_links(0)
{
}


Genome::Genome(const Genome &a_G)
  : m_ID(a_G.m_ID), m_Fitness(a_G.m_Fitness), m_AdjustedFitness(a_G.m_AdjustedFitness),
    m_OffspringAmount(a_G.m_OffspringAmount), m_Depth(a_G.m_Depth),
    m_NumInputs(a_G.m_NumInputs), m_NumOutputs(a_G.m_NumOutputs), m_Evaluated(a_G.m_Evaluated),
    m_PhenotypeBehavior(a_G.m_PhenotypeBehavior),
    m_NeuronGenes(a_G.m_NeuronGenes), m_LinkGenes(a_G.m_LinkGenes),
    m_GenomeGene(a_G.m_GenomeGene),
    m_initial_num_neurons(a_G.m_initial_num_neurons),
    m_initial_num_links(a_G.m_initial_num_links)
{
}

Genome &Genome::operator=(const Genome &a_G)
{
    if (this != &a_G)
    {
        m_ID = a_G.m_ID;
        m_Fitness = a_G.m_Fitness;
        m_AdjustedFitness = a_G.m_AdjustedFitness;
        m_OffspringAmount = a_G.m_OffspringAmount;
        m_Depth = a_G.m_Depth;
        m_NumInputs = a_G.m_NumInputs;
        m_NumOutputs = a_G.m_NumOutputs;
        m_Evaluated = a_G.m_Evaluated;
        m_PhenotypeBehavior = a_G.m_PhenotypeBehavior;

        m_NeuronGenes = a_G.m_NeuronGenes;
        m_LinkGenes = a_G.m_LinkGenes;
        m_GenomeGene = a_G.m_GenomeGene;

        m_initial_num_neurons = a_G.m_initial_num_neurons;
        m_initial_num_links = a_G.m_initial_num_links;
    }
    return *this;
}


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
    if ((seed_type == LAYERED) && (in.NumHidden == 0))
        seed_type = PERCEPTRON;

    if (!a_Parameters.DontUseBiasNeuron)
    {
        // inputs except last
        for (unsigned i = 0, end = static_cast<unsigned>(in.NumInputs - 1); i < end; ++i)
        {
            m_NeuronGenes.emplace_back(INPUT, t_nnum++, 0.0);
        }
        // bias
        m_NeuronGenes.emplace_back(BIAS, t_nnum++, 0.0);
    }
    else
    {
        for (unsigned i = 0, end = static_cast<unsigned>(in.NumInputs); i < end; ++i)
        {
            m_NeuronGenes.emplace_back(INPUT, t_nnum++, 0.0);
        }
    }

    for (unsigned i = 0; i < static_cast<unsigned>(in.NumOutputs); ++i)
    {
        NeuronGene outnode(OUTPUT, t_nnum, 1.0);
        outnode.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0,
                     (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0,
                     (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0,
                     (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0,
                     in.OutputActType);
        outnode.InitTraits(a_Parameters.NeuronTraits, t_RNG);
        m_NeuronGenes.push_back(outnode);
        ++t_nnum;
    }

    if ((seed_type == LAYERED) && (in.NumHidden > 0))
    {
        double lt_inc  = 1.0 / (in.NumLayers + 1);
        double initlt  = lt_inc;
        for (unsigned lay = 0; lay < static_cast<unsigned>(in.NumLayers); ++lay)
        {
            for (unsigned i = 0; i < static_cast<unsigned>(in.NumHidden); ++i)
            {
                NeuronGene hidden(HIDDEN, t_nnum, initlt);
                hidden.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0,
                            (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0,
                            (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0,
                            (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0,
                            in.HiddenActType);
                hidden.InitTraits(a_Parameters.NeuronTraits, t_RNG);
                m_NeuronGenes.push_back(hidden);
                ++t_nnum;
            }
            initlt += lt_inc;
        }
        if (!in.FS_NEAT)
        {
            int last_dest_id     = in.NumInputs + in.NumOutputs + 1;
            int last_src_id      = 1;
            int prev_layer_size  = in.NumInputs;
            for (unsigned ly = 0; ly < static_cast<unsigned>(in.NumLayers); ++ly)
            {
                for (unsigned i = 0; i < static_cast<unsigned>(in.NumHidden); ++i)
                {
                    for (int j = 0; j < prev_layer_size; ++j)
                    {
                        LinkGene L(j + last_src_id, i + last_dest_id, t_innovnum, 0.0, false);
                        L.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.push_back(L);
                        ++t_innovnum;
                    }
                }
                last_dest_id += in.NumHidden;
                if (ly == 0)
                    last_src_id += (prev_layer_size + in.NumOutputs);
                else
                    last_src_id += prev_layer_size;
                prev_layer_size = in.NumHidden;
            }
            last_dest_id = in.NumInputs + 1;
            for (unsigned i = 0; i < static_cast<unsigned>(in.NumOutputs); ++i)
            {
                for (int j = 0; j < prev_layer_size; ++j)
                {
                    LinkGene L(j + last_src_id, i + last_dest_id, t_innovnum, 0.0, false);
                    L.InitTraits(a_Parameters.LinkTraits, t_RNG);
                    m_LinkGenes.push_back(L);
                    ++t_innovnum;
                }
            }
        }
    }
    else
    {
        if ((!in.FS_NEAT) && (seed_type == PERCEPTRON))
        {
            for (unsigned i = 0; i < static_cast<unsigned>(in.NumOutputs); ++i)
            {
                for (unsigned j = 0; j < static_cast<unsigned>(in.NumInputs); ++j)
                {
                    LinkGene L(j + 1, i + in.NumInputs + 1, t_innovnum, 0.0, false);
                    L.InitTraits(a_Parameters.LinkTraits, t_RNG);
                    m_LinkGenes.push_back(L);
                    ++t_innovnum;
                }
            }
        }
        else
        {
            std::vector<std::pair<int, int>> used;
            bool found = false;
            int linkcount = 0;
            while (linkcount < in.FS_NEAT_links)
            {
                for (unsigned i = 0; i < static_cast<unsigned>(in.NumOutputs); ++i)
                {
                    int t_inp_id = t_RNG.RandInt(1, in.NumInputs - 1);
                    int t_bias_id = in.NumInputs;
                    int t_out_id  = in.NumInputs + 1 + i;
                    found = false;
                    for (const auto &p : used)
                    {
                        if (p.first == t_inp_id && p.second == t_out_id)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        LinkGene L(t_inp_id, t_out_id, t_innovnum, 0.0, false);
                        L.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.push_back(L);
                        ++t_innovnum;
                        if (!a_Parameters.DontUseBiasNeuron)
                        {
                            LinkGene BL(t_bias_id, t_out_id, t_innovnum, 0.0, false);
                            BL.InitTraits(a_Parameters.LinkTraits, t_RNG);
                            m_LinkGenes.push_back(BL);
                            ++t_innovnum;
                        }
                        used.push_back(std::make_pair(t_inp_id, t_out_id));
                        ++linkcount;
                    }
                }
            }
        }
    }

    if (in.FS_NEAT && (in.FS_NEAT_links == 1))
    {
        throw std::runtime_error("FS-NEAT with exactly 1 link & 1/1/1 is not recommended.");
    }

    m_GenomeGene.InitTraits(a_Parameters.GenomeTraits, t_RNG);

    m_Evaluated = false;
    m_NumInputs = in.NumInputs;
    m_NumOutputs = in.NumOutputs;
    m_initial_num_neurons = static_cast<int>(NumNeurons());
    m_initial_num_links   = static_cast<int>(NumLinks());
}


Genome::Genome(std::istream &data)
{
    if (!data)
        throw std::runtime_error("Invalid input stream provided to Genome constructor.");

    std::string token;
    // Read until we reach "GenomeStart"
    do {
        data >> token;
    } while (token != "GenomeStart" && !data.eof());

    // Read the genome ID
    data >> m_ID;

    // Read the remainder of the genome data
    do {
        data >> token;
        if (token == "Neuron")
        {
            int tid, ttype, tact;
            double tsplity, ta, tb, ttc, tbias;
            data >> tid >> ttype >> tsplity >> tact >> ta >> tb >> ttc >> tbias;
            NeuronGene N(static_cast<NeuronType>(ttype), tid, tsplity);
            N.m_ActFunction = static_cast<ActivationFunction>(tact);
            N.m_A = ta; N.m_B = tb; N.m_TimeConstant = ttc; N.m_Bias = tbias;
            m_NeuronGenes.push_back(N);
        }
        else if (token == "Link")
        {
            int from, to, innov, isrec;
            double weight;
            data >> from >> to >> innov >> isrec >> weight;
            LinkGene L(from, to, innov, weight, static_cast<bool>(isrec));
            m_LinkGenes.push_back(L);
        }
    } while (token != "GenomeEnd" && !data.eof());

    // Do not call data.close() here—since the stream might is not an fstream

    // Count inputs and outputs
    m_NumInputs = 0;
    m_NumOutputs = 0;
    for (const auto &ng : m_NeuronGenes)
    {
        if (ng.Type() == INPUT || ng.Type() == BIAS)
            ++m_NumInputs;
        else if (ng.Type() == OUTPUT)
            ++m_NumOutputs;
    }

    m_Fitness = 0;
    m_AdjustedFitness = 0;
    m_OffspringAmount = 0;
    m_Depth = 0;
    m_Evaluated = false;
    m_PhenotypeBehavior = nullptr;
    m_initial_num_neurons = static_cast<int>(m_NeuronGenes.size());
    m_initial_num_links   = static_cast<int>(m_LinkGenes.size());
}


// Use a stringstream to write out the genome (similar to your Save() method).
std::string Genome::Serialize() const {
    std::ostringstream oss;
    // For example, write out all genes (this is similar to how Save(FILE*) works):
    oss << "GenomeStart " << GetID() << "\n";
    for (const auto &ng : m_NeuronGenes) {
        oss << "Neuron " << ng.m_ID << " " << static_cast<int>(ng.m_Type) << " " 
            << ng.m_SplitY << " " << static_cast<int>(ng.m_ActFunction) << " " 
            << ng.m_A << " " << ng.m_B << " " << ng.m_TimeConstant << " " 
            << ng.m_Bias << "\n";
    }
    for (const auto &lg : m_LinkGenes) {
        oss << "Link " << lg.m_FromNeuronID << " " << lg.m_ToNeuronID << " " 
            << lg.m_InnovationID << " " << static_cast<int>(lg.m_IsRecurrent) 
            << " " << lg.m_Weight << "\n";
    }
    oss << "GenomeEnd\n";
    return oss.str();
}

Genome Genome::Deserialize(const std::string &data) {
    std::istringstream iss(data);
    return Genome(iss);  
}

void Genome::SetDepth(unsigned int a_d) { m_Depth = a_d; }
unsigned int Genome::GetDepth() const { return m_Depth; }
void Genome::SetID(int a_id) { m_ID = a_id; }
int Genome::GetID() const { return m_ID; }

void Genome::SetAdjFitness(double a_af) { m_AdjustedFitness = a_af; }
void Genome::SetFitness(double a_f) { m_Fitness = a_f; }
double Genome::GetAdjFitness() const { return m_AdjustedFitness; }
double Genome::GetFitness() const { return m_Fitness; }

void Genome::SetNeuronY(unsigned int idx, int val)
{
    ASSERT(idx < m_NeuronGenes.size());
    m_NeuronGenes[idx].y = val;
}

void Genome::SetNeuronX(unsigned int idx, int val)
{
    ASSERT(idx < m_NeuronGenes.size());
    m_NeuronGenes[idx].x = val;
}

void Genome::SetNeuronXY(unsigned int idx, int x, int y)
{
    ASSERT(idx < m_NeuronGenes.size());
    m_NeuronGenes[idx].x = x;
    m_NeuronGenes[idx].y = y;
}

bool Genome::IsDeadEndNeuron(int a_ID) const
{
    bool t_no_incoming = true;
    bool t_no_outgoing = true;

    for (size_t i = 0, end = m_LinkGenes.size(); i < end; ++i)
    {
        const LinkGene &l = m_LinkGenes[i];
        if ((l.ToNeuronID() == a_ID) && (!l.IsLoopedRecurrent()) &&
            (GetNeuronByID(l.FromNeuronID()).Type() != BIAS))
        {
            t_no_incoming = false;
        }
        if ((l.FromNeuronID() == a_ID) && (!l.IsLoopedRecurrent()) &&
            (GetNeuronByID(l.FromNeuronID()).Type() != BIAS))
        {
            t_no_outgoing = false;
        }
    }

    return (t_no_incoming || t_no_outgoing);
}

int Genome::LinksInputtingFrom(int a_ID) const
{
    int t_counter = 0;
    for (const auto &l : m_LinkGenes)
        if (l.FromNeuronID() == a_ID)
            ++t_counter;
    return t_counter;
}

int Genome::LinksOutputtingTo(int a_ID) const
{
    int t_counter = 0;
    for (const auto &l : m_LinkGenes)
        if (l.ToNeuronID() == a_ID)
            ++t_counter;
    return t_counter;
}

int Genome::GetLastNeuronID() const
{
    int last = 0;
    // Go through all neuron genes and track the maximum neuron id.
    for (const auto &ng : m_NeuronGenes)
    {
        last = std::max(last, ng.ID());
    }
    return last;
}

int Genome::GetLastInnovationID() const
{
    int last = 0;
    // Scan through the link genes to find the maximum innovation id.
    for (const auto &lg : m_LinkGenes)
    {
        last = std::max(last, lg.InnovationID());
    }
    return last;
}


LinkGene Genome::GetLinkByIndex(int idx) const
{
    ASSERT(idx < static_cast<int>(m_LinkGenes.size()));
    return m_LinkGenes[idx];
}

LinkGene Genome::GetLinkByInnovID(int id) const
{
    for (const auto &l : m_LinkGenes)
    {
        if (l.InnovationID() == id)
            return l;
    }
    throw std::runtime_error("No link found by that innovID");
}

NeuronGene Genome::GetNeuronByIndex(int idx) const
{
    ASSERT(idx < static_cast<int>(m_NeuronGenes.size()));
    return m_NeuronGenes[idx];
}

NeuronGene Genome::GetNeuronByID(int a_ID) const
{
    ASSERT(HasNeuronID(a_ID));
    int i = GetNeuronIndex(a_ID);
    ASSERT(i >= 0);
    return m_NeuronGenes[i];
}


double Genome::GetOffspringAmount() const { return m_OffspringAmount; }
void Genome::SetOffspringAmount(double v) { m_OffspringAmount = v; }

bool Genome::IsEvaluated() const { return m_Evaluated; }
void Genome::SetEvaluated() { m_Evaluated = true; }
void Genome::ResetEvaluated() { m_Evaluated = false; }

bool Genome::HasLinkByInnovID(int id) const
{
    ASSERT(id > 0);
    for (const auto &l : m_LinkGenes)
    {
        if (l.InnovationID() == id)
            return true;
    }
    return false;
}

bool Genome::HasLoops()
{
    NeuralNetwork net;
    BuildPhenotype(net);

    std::vector<std::vector<int>> adjacency(net.m_neurons.size());
    for (size_t i = 0, end = net.m_connections.size(); i < end; ++i)
    {
        const Connection &c = net.m_connections[i];
        adjacency[c.m_source_neuron_idx].push_back(c.m_target_neuron_idx);
    }

    std::vector<int> color(net.m_neurons.size(), 0);
    // DFS helper lambda (recursive)
    std::function<bool(int)> dfs = [&](int cur) -> bool
    {
        color[cur] = 1;
        for (int nxt : adjacency[cur])
        {
            if (color[nxt] == 1)
                return true;
            if (color[nxt] == 0 && dfs(nxt))
                return true;
        }
        color[cur] = 2;
        return false;
    };

    for (size_t i = 0, end = net.m_neurons.size(); i < end; ++i)
    {
        if (color[i] == 0 && dfs(static_cast<int>(i)))
            return true;
    }
    return false;
}


void Genome::BuildPhenotype(NeuralNetwork &a_Net)
{
    a_Net.Clear();
    a_Net.SetInputOutputDimentions(m_NumInputs, m_NumOutputs);

    for (const auto &ng : m_NeuronGenes)
    {
        Neuron t_n;
        t_n.m_a = ng.m_A;
        t_n.m_b = ng.m_B;
        t_n.m_timeconst = ng.m_TimeConstant;
        t_n.m_bias = ng.m_Bias;
        t_n.m_activation_function_type = ng.m_ActFunction;
        t_n.m_split_y = ng.SplitY();
        t_n.m_type = ng.Type();
        a_Net.AddNeuron(t_n);
    }

    for (const auto &lg : m_LinkGenes)
    {
        Connection c;
        c.m_source_neuron_idx = GetNeuronIndex(lg.FromNeuronID());
        c.m_target_neuron_idx = GetNeuronIndex(lg.ToNeuronID());
        c.m_weight = lg.GetWeight();
        c.m_recur_flag = lg.IsRecurrent();

        c.m_hebb_rate = 0.3;
        c.m_hebb_pre_rate = 0.1;
        if(lg.m_Traits.count("hebb_rate") == 1)
        {
            try { c.m_hebb_rate = std::get<double>(lg.m_Traits.at("hebb_rate").value); }
            catch(...) { }
        }
        if(lg.m_Traits.count("hebb_pre_rate") == 1)
        {
            try { c.m_hebb_pre_rate = std::get<double>(lg.m_Traits.at("hebb_pre_rate").value); }
            catch(...) { }
        }
        a_Net.AddConnection(c);
    }

    a_Net.Flush();
}


ActivationFunction GetRandomActivation(Parameters &a_Parameters, RNG &a_RNG)
{
    std::vector<double> t_probs = {
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

    return static_cast<ActivationFunction>(a_RNG.Roulette(t_probs));
}


void Genome::BuildHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst)
{
    ASSERT(!subst.m_input_coords.empty());
    ASSERT(!subst.m_output_coords.empty());
    int max_dims = subst.GetMaxDims();
    ASSERT(static_cast<int>(m_NumInputs) >= subst.GetMinCPPNInputs());
    ASSERT(static_cast<int>(m_NumOutputs) >= subst.GetMinCPPNOutputs());

    net.SetInputOutputDimentions(static_cast<unsigned short>(subst.m_input_coords.size()),
                                  static_cast<unsigned short>(subst.m_output_coords.size()));

    for (const auto &coord : subst.m_input_coords)
    {
        Neuron t_n;
        t_n.m_a = 1;
        t_n.m_b = 0;
        t_n.m_substrate_coords = coord;
        t_n.m_activation_function_type = LINEAR;
        t_n.m_type = INPUT;
        net.AddNeuron(t_n);
    }

    for (const auto &coord : subst.m_output_coords)
    {
        Neuron t_n;
        t_n.m_a = 1;
        t_n.m_b = 0;
        t_n.m_substrate_coords = coord;
        t_n.m_activation_function_type = subst.m_output_nodes_activation;
        t_n.m_type = OUTPUT;
        net.AddNeuron(t_n);
    }

    for (const auto &coord : subst.m_hidden_coords)
    {
        Neuron t_n;
        t_n.m_a = 1;
        t_n.m_b = 0;
        t_n.m_substrate_coords = coord;
        t_n.m_activation_function_type = subst.m_hidden_nodes_activation;
        t_n.m_type = HIDDEN;
        net.AddNeuron(t_n);
    }

    NeuralNetwork cppn(true);
    BuildPhenotype(cppn);
    cppn.Flush();

    if (subst.m_leaky)
    {
        ASSERT(static_cast<unsigned>(m_NumOutputs) >= static_cast<unsigned>(subst.GetMinCPPNOutputs()));

        for (unsigned i = net.NumInputs(); i < net.m_neurons.size(); ++i)
        {
            cppn.Flush();
            std::vector<double> cinputs(m_NumInputs, 0.0);
            unsigned from_dims = net.m_neurons[i].m_substrate_coords.size();
            for (unsigned d = 0; d < from_dims; ++d)
                cinputs[d] = net.m_neurons[i].m_substrate_coords[d];
            if (subst.m_with_distance)
            {
                double sum = 0;
                for (int dd = 0; dd < max_dims; ++dd)
                {
                    sum += sqr(cinputs[dd]);
                }
                cinputs[m_NumInputs - 2] = sqrt(sum);
            }
            cinputs[m_NumInputs - 1] = 1.0;

            cppn.Input(cinputs);
            int dp = 8;
            if (!HasLoops())
            {
                CalculateDepth();
                dp = GetDepth();
            }
            for (int z = 0; z < dp; ++z)
                cppn.Activate();

            double t_tc   = cppn.Output()[m_NumOutputs - 2];
            double t_bias = cppn.Output()[m_NumOutputs - 1];
            Clamp(t_tc, -1, 1);
            Clamp(t_bias, -1, 1);
            Scale(t_tc, -1, 1, subst.m_min_time_const, subst.m_max_time_const);
            Scale(t_bias, -1, 1, -subst.m_max_weight_and_bias, subst.m_max_weight_and_bias);
            net.m_neurons[i].m_timeconst = t_tc;
            net.m_neurons[i].m_bias      = t_bias;
        }
    }

    std::vector<std::vector<int>> pairs;
    if (!subst.m_custom_connectivity.empty())
    {
        for (const auto &conn : subst.m_custom_connectivity)
        {
            NeuronType st = static_cast<NeuronType>(conn[0]);
            int sidx       = conn[1];
            NeuronType dt = static_cast<NeuronType>(conn[2]);
            int didx       = conn[3];

            int j = 0, k = 0;
            if (st == INPUT || st == BIAS) j = sidx;
            else if (st == OUTPUT) j = static_cast<int>(subst.m_input_coords.size() + sidx);
            else if (st == HIDDEN) j = static_cast<int>(subst.m_input_coords.size() + subst.m_output_coords.size() + sidx);

            if (dt == INPUT || dt == BIAS) k = didx;
            else if (dt == OUTPUT) k = static_cast<int>(subst.m_input_coords.size() + didx);
            else if (dt == HIDDEN) k = static_cast<int>(subst.m_input_coords.size() + subst.m_output_coords.size() + didx);

            if (subst.m_custom_conn_obeys_flags && (
                ((!subst.m_allow_input_hidden_links) &&
                  ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[k].m_type == HIDDEN))) ||
                ((!subst.m_allow_input_output_links) &&
                  ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[k].m_type == OUTPUT))) ||
                ((!subst.m_allow_hidden_hidden_links) &&
                  ((net.m_neurons[j].m_type == HIDDEN) &&
                   (net.m_neurons[k].m_type == HIDDEN) && (j != k))) ||
                ((!subst.m_allow_hidden_output_links) &&
                  ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[k].m_type == OUTPUT))) ||
                ((!subst.m_allow_output_hidden_links) &&
                  ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[k].m_type == HIDDEN))) ||
                ((!subst.m_allow_output_output_links) &&
                  ((net.m_neurons[j].m_type == OUTPUT) &&
                   (net.m_neurons[k].m_type == OUTPUT) && (j != k))) ||
                ((!subst.m_allow_looped_hidden_links) &&
                  ((net.m_neurons[j].m_type == HIDDEN) && (j == k))) ||
                ((!subst.m_allow_looped_output_links) &&
                  ((net.m_neurons[j].m_type == OUTPUT) && (j == k)))
                ))
            {
                pairs.push_back({j, k});
            }
        }
    }
    else
    {
        for (unsigned i = net.NumInputs(); i < net.m_neurons.size(); ++i)
        {
            for (unsigned j = 0; j < net.m_neurons.size(); ++j)
            {
                if (subst.m_custom_conn_obeys_flags && (
                   ((!subst.m_allow_input_hidden_links) &&
                    ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == HIDDEN))) ||
                   ((!subst.m_allow_input_output_links) &&
                    ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == OUTPUT))) ||
                   ((!subst.m_allow_hidden_hidden_links) &&
                    ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) && (i != j))) ||
                   ((!subst.m_allow_hidden_output_links) &&
                    ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == OUTPUT))) ||
                   ((!subst.m_allow_output_hidden_links) &&
                    ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == HIDDEN))) ||
                   ((!subst.m_allow_output_output_links) &&
                    ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) && (i != j))) ||
                   ((!subst.m_allow_looped_hidden_links) &&
                    ((net.m_neurons[j].m_type == HIDDEN) && (i == j))) ||
                   ((!subst.m_allow_looped_output_links) &&
                    ((net.m_neurons[j].m_type == OUTPUT) && (i == j)))
                   ))
                {
                    pairs.push_back({static_cast<int>(j), static_cast<int>(i)});
                }
            }
        }
    }

    for (auto &pp : pairs)
    {
        int j = pp[0];
        int i = pp[1];
        std::vector<double> t_inputs(m_NumInputs, 0.0);
        int from_dims = static_cast<int>(net.m_neurons[j].m_substrate_coords.size());
        int to_dims = static_cast<int>(net.m_neurons[i].m_substrate_coords.size());
        for (int d = 0; d < from_dims; ++d)
            t_inputs[d] = net.m_neurons[j].m_substrate_coords[d];
        for (int d = 0; d < to_dims; ++d)
            t_inputs[max_dims + d] = net.m_neurons[i].m_substrate_coords[d];

        if (subst.m_with_distance)
        {
            double sum = 0.0;
            for (int dd = 0; dd < max_dims; ++dd)
            {
                sum += sqr(t_inputs[dd] - t_inputs[max_dims + dd]);
            }
            t_inputs[m_NumInputs - 2] = sqrt(sum);
        }
        t_inputs[m_NumInputs - 1] = 1.0;

        cppn.Flush();
        cppn.Input(t_inputs);
        int dp = 8;
        if (!HasLoops())
        {
            CalculateDepth();
            dp = GetDepth();
        }
        for (int z = 0; z < dp; ++z)
            cppn.Activate();

        double t_link = 0;
        double t_weight = 0;
        if (subst.m_query_weights_only)
            t_weight = cppn.Output()[0];
        else
        {
            t_link = cppn.Output()[0];
            t_weight = cppn.Output()[1];
        }

        if (((!subst.m_query_weights_only) && (t_link > 0)) || subst.m_query_weights_only)
        {
            t_weight *= subst.m_max_weight_and_bias;
            Connection c;
            c.m_source_neuron_idx = j;
            c.m_target_neuron_idx = i;
            c.m_weight = t_weight;
            c.m_recur_flag = false;
            net.AddConnection(c);
        }
    }
}

void Genome::DerivePhenotypicChanges(NeuralNetwork &a_Net)
{
    if (a_Net.m_connections.size() != m_LinkGenes.size())
        return;
    for (size_t i = 0, end = m_LinkGenes.size(); i < end; ++i)
    {
        m_LinkGenes[i].SetWeight(a_Net.GetConnectionByIndex(static_cast<int>(i)).m_weight);
    }
}

double Genome::CompatibilityDistance(Genome &a_G, Parameters &a_Parameters)
{
    double total_distance = 0.0, total_w_diff = 0.0, total_A_diff = 0.0,
           total_B_diff = 0.0, total_TC_diff = 0.0, total_bias_diff = 0.0, total_act_diff = 0.0;
    std::map<std::string, double> total_link_trait_diff;
    std::map<std::string, double> total_neuron_trait_diff;
    double E = 0, D = 0, M = 0, matching_neurons = 0;

    auto gentrait_dists = m_GenomeGene.GetTraitDistances(a_G.m_GenomeGene.m_Traits);
    for (const auto &kv : gentrait_dists)
    {
        double val = kv.second * a_Parameters.GenomeTraits.at(kv.first).m_ImportanceCoeff;
        if (std::isnan(val) || std::isinf(val))
            val = 0.0;
        total_distance += val;
    }

    unsigned i1 = 0, i2 = 0;
    std::vector<LinkGene> links1 = m_LinkGenes;
    std::vector<LinkGene> links2 = a_G.m_LinkGenes;
    std::sort(links1.begin(), links1.end(),
              [](const LinkGene &lhs, const LinkGene &rhs)
              { return lhs.InnovationID() < rhs.InnovationID(); });
    std::sort(links2.begin(), links2.end(),
              [](const LinkGene &lhs, const LinkGene &rhs)
              { return lhs.InnovationID() < rhs.InnovationID(); });

    while (!(i1 >= links1.size() && i2 >= links2.size()))
    {
        if (i1 == links1.size())
        {
            ++E;
            ++i2;
        }
        else if (i2 == links2.size())
        {
            ++E;
            ++i1;
        }
        else
        {
            int in1 = links1[i1].InnovationID();
            int in2 = links2[i2].InnovationID();
            if (in1 == in2)
            {
                ++M;
                if (a_Parameters.WeightDiffCoeff > 0)
                {
                    double wd = links1[i1].GetWeight() - links2[i2].GetWeight();
                    total_w_diff += (wd < 0) ? -wd : wd;
                }
                auto linktraitdist = links1[i1].GetTraitDistances(links2[i2].m_Traits);
                for (const auto &xx : linktraitdist)
                {
                    double val = xx.second * a_Parameters.LinkTraits.at(xx.first).m_ImportanceCoeff;
                    if (std::isnan(val) || std::isinf(val))
                        val = 0.0;
                    total_link_trait_diff[xx.first] += val;
                }
                ++i1;
                ++i2;
            }
            else if (in1 < in2)
            {
                ++D;
                ++i1;
            }
            else
            {
                ++D;
                ++i2;
            }
        }
    }

    double maxsize = (links1.size() > links2.size()) ? static_cast<double>(links1.size()) : static_cast<double>(links2.size());
    if (maxsize < 1.0) maxsize = 1.0;
    double normalizer = (a_Parameters.NormalizeGenomeSize) ? maxsize : 1.0;
    if(M < 1.0) M = 1.0;
    double dist_links = a_Parameters.ExcessCoeff * (E / normalizer)
                        + a_Parameters.DisjointCoeff * (D / normalizer)
                        + a_Parameters.WeightDiffCoeff * (total_w_diff / M);
    total_distance += dist_links;

    int bigger_neuron_count = (m_NeuronGenes.size() > a_G.m_NeuronGenes.size())
                                ? static_cast<int>(m_NeuronGenes.size())
                                : static_cast<int>(a_G.m_NeuronGenes.size());
    if (bigger_neuron_count < 1) bigger_neuron_count = 1;
    double mismatch = 0;
    for (size_t i = m_NumInputs; i < m_NeuronGenes.size(); ++i)
    {
        if(m_NeuronGenes[i].Type() == INPUT || m_NeuronGenes[i].Type() == BIAS)
            continue;
        if(a_G.HasNeuronID(m_NeuronGenes[i].ID()))
        {
            ++matching_neurons;
            NeuronGene oth = a_G.GetNeuronByID(m_NeuronGenes[i].ID());
            if(a_Parameters.ActivationADiffCoeff>0)
                total_A_diff += std::abs(m_NeuronGenes[i].m_A - oth.m_A);
            if(a_Parameters.ActivationBDiffCoeff>0)
                total_B_diff += std::abs(m_NeuronGenes[i].m_B - oth.m_B);
            if(a_Parameters.TimeConstantDiffCoeff>0)
                total_TC_diff += std::abs(m_NeuronGenes[i].m_TimeConstant - oth.m_TimeConstant);
            if(a_Parameters.BiasDiffCoeff>0)
                total_bias_diff += std::abs(m_NeuronGenes[i].m_Bias - oth.m_Bias);
            if(a_Parameters.ActivationFunctionDiffCoeff>0)
                if(m_NeuronGenes[i].m_ActFunction != oth.m_ActFunction)
                    total_act_diff++;
            auto nd = m_NeuronGenes[i].GetTraitDistances(oth.m_Traits);
            for (const auto &xx : nd)
            {
                double val = xx.second * a_Parameters.NeuronTraits.at(xx.first).m_ImportanceCoeff;
                if (std::isnan(val) || std::isinf(val))
                    val = 0;
                total_neuron_trait_diff[xx.first] += val;
            }
        }
    }
    if(matching_neurons < 1) matching_neurons = 1;
    double dist_neurons = a_Parameters.ActivationADiffCoeff*(total_A_diff/matching_neurons)
                          + a_Parameters.ActivationBDiffCoeff*(total_B_diff/matching_neurons)
                          + a_Parameters.TimeConstantDiffCoeff*(total_TC_diff/matching_neurons)
                          + a_Parameters.BiasDiffCoeff*(total_bias_diff/matching_neurons)
                          + a_Parameters.ActivationFunctionDiffCoeff*(total_act_diff/matching_neurons);
    total_distance += dist_neurons;
    for(const auto &xx : total_link_trait_diff)
    {
        double n = xx.second * a_Parameters.LinkTraits.at(xx.first).m_ImportanceCoeff / M;
        if(std::isnan(n) || std::isinf(n)) n=0.0;
        total_distance += n;
    }
    for(const auto &xx : total_neuron_trait_diff)
    {
        double n = xx.second * a_Parameters.NeuronTraits.at(xx.first).m_ImportanceCoeff / matching_neurons;
        if(std::isnan(n) || std::isinf(n)) n=0.0;
        total_distance += n;
    }
    return total_distance;
}

bool Genome::IsCompatibleWith(Genome &a_G, Parameters &a_Parameters)
{
    if(this == &a_G) return true;
    if(GetID() == a_G.GetID()) return true;
    double dist = CompatibilityDistance(a_G, a_Parameters);
    return (dist <= a_Parameters.CompatTreshold);
}


bool Genome::Mutate_LinkWeights(const Parameters &a_Parameters, RNG &a_RNG)
{
    bool did_mutate = false;
    bool severe = (a_RNG.RandFloat() < a_Parameters.MutateWeightsSevereProb);
    int tailstart = 0;
    if(NumLinks() > m_initial_num_links)
        tailstart = static_cast<int>(NumLinks() * 0.8);
    if(tailstart < m_initial_num_links)
        tailstart = m_initial_num_links;
    for (size_t i = 0, end = m_LinkGenes.size(); i < end; ++i)
    {
        if (!severe && (a_RNG.RandFloat() < a_Parameters.WeightMutationRate))
        {
            double w = m_LinkGenes[i].GetWeight();
            bool in_tail = (static_cast<int>(i) >= tailstart);
            if(in_tail || a_RNG.RandFloat() < a_Parameters.WeightReplacementRate)
                w = a_RNG.RandFloatSigned() * a_Parameters.WeightReplacementMaxPower;
            else
                w += a_RNG.RandFloatSigned() * a_Parameters.WeightMutationMaxPower;
            Clamp(w, a_Parameters.MinWeight, a_Parameters.MaxWeight);
            m_LinkGenes[i].SetWeight(w);
            did_mutate = true;
        }
        else if(severe)
        {
            if(a_RNG.RandFloat() < a_Parameters.WeightMutationRate)
            {
                double w = a_RNG.RandFloat();
                Scale(w, 0.0, 1.0, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                m_LinkGenes[i].SetWeight(w);
                did_mutate = true;
            }
        }
    }
    return did_mutate;
}

void Genome::Randomize_LinkWeights(const Parameters &a_Parameters, RNG &a_RNG)
{
    for (size_t i = 0, end = NumLinks(); i < end; ++i)
    {
        double nf = a_RNG.RandFloat();
        Scale(nf, 0.0, 1.0, a_Parameters.MinWeight, a_Parameters.MaxWeight);
        m_LinkGenes[i].SetWeight(nf);
    }
}

void Genome::Randomize_Traits(const Parameters &a_Parameters, RNG &a_RNG)
{
    for (auto &ng : m_NeuronGenes)
        ng.InitTraits(a_Parameters.NeuronTraits, a_RNG);
    for (auto &lg : m_LinkGenes)
        lg.InitTraits(a_Parameters.LinkTraits, a_RNG);
    m_GenomeGene.InitTraits(a_Parameters.GenomeTraits, a_RNG);
}

bool Genome::Mutate_NeuronActivations_A(const Parameters &a_Parameters, RNG &a_RNG)
{
    for (auto &ng : m_NeuronGenes)
    {
        if(ng.Type() != INPUT && ng.Type() != BIAS)
        {
            double r = a_RNG.RandFloatSigned() * a_Parameters.ActivationAMutationMaxPower;
            ng.m_A += r;
            Clamp(ng.m_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
        }
    }
    return true;
}

bool Genome::Mutate_NeuronActivations_B(const Parameters &a_Parameters, RNG &a_RNG)
{
    for (auto &ng : m_NeuronGenes)
    {
        if(ng.Type() != INPUT && ng.Type() != BIAS)
        {
            double r = a_RNG.RandFloatSigned() * a_Parameters.ActivationBMutationMaxPower;
            ng.m_B += r;
            Clamp(ng.m_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
        }
    }
    return true;
}

bool Genome::Mutate_NeuronActivation_Type(const Parameters &a_Parameters, RNG &a_RNG)
{
    if(m_NeuronGenes.size() <= m_NumInputs)
        return false;
    int startIndex = m_NumInputs; 
    int choice = a_RNG.RandInt(startIndex, static_cast<int>(m_NeuronGenes.size()) - 1);
    int oldf = m_NeuronGenes[choice].m_ActFunction;
    std::vector<double> probs = {
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
    ActivationFunction newAF = static_cast<ActivationFunction>(idx);
    if (static_cast<int>(newAF) == oldf)
        return false;
    m_NeuronGenes[choice].m_ActFunction = newAF;
    return true;
}

bool Genome::Mutate_NeuronTimeConstants(const Parameters &a_Parameters, RNG &a_RNG)
{
    for (auto &ng : m_NeuronGenes)
    {
        if(ng.Type() != INPUT && ng.Type() != BIAS)
        {
            double r = a_RNG.RandFloatSigned() * a_Parameters.TimeConstantMutationMaxPower;
            ng.m_TimeConstant += r;
            Clamp(ng.m_TimeConstant, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
        }
    }
    return true;
}

bool Genome::Mutate_NeuronBiases(const Parameters &a_Parameters, RNG &a_RNG)
{
    for (auto &ng : m_NeuronGenes)
    {
        if(ng.Type() != INPUT && ng.Type() != BIAS)
        {
            double r = a_RNG.RandFloatSigned() * a_Parameters.BiasMutationMaxPower;
            ng.m_Bias += r;
            Clamp(ng.m_Bias, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);
        }
    }
    return true;
}

bool Genome::Mutate_NeuronTraits(const Parameters &a_Parameters, RNG &a_RNG)
{
    bool mutated = false;
    for (auto &ng : m_NeuronGenes)
    {
        if (ng.Type() != INPUT && ng.Type() != BIAS)
            mutated |= ng.MutateTraits(a_Parameters.NeuronTraits, a_RNG);
    }
    return mutated;
}

bool Genome::Mutate_LinkTraits(const Parameters &a_Parameters, RNG &a_RNG)
{
    bool mutated = false;
    for (auto &lg : m_LinkGenes)
        mutated |= lg.MutateTraits(a_Parameters.LinkTraits, a_RNG);
    return mutated;
}

bool Genome::Mutate_GenomeTraits(const Parameters &a_Parameters, RNG &a_RNG)
{
    return m_GenomeGene.MutateTraits(a_Parameters.GenomeTraits, a_RNG);
}

bool Genome::Mutate_AddNeuron(InnovationDatabase &a_Innovs, Parameters &a_Parameters, RNG &a_RNG)
{
    if (NumLinks() == 0)
        return false;
    bool t_link_found = false;
    int t_link_num = 0;
    int t_in = 0, t_out = 0;
    LinkGene t_chosenlink(0, 0, -1, 0, false);
    int t_tries = 256;
    while (!t_link_found)
    {
        if (NumLinks() == 1)
        {
            t_link_num = 0;
        }
        else
        {
            t_link_num = a_RNG.RandInt(0, NumLinks() - 1);
        }
        t_in = m_LinkGenes[t_link_num].FromNeuronID();
        t_out = m_LinkGenes[t_link_num].ToNeuronID();
        ASSERT(t_in > 0 && t_out > 0);
        t_link_found = true;
        if (!a_Parameters.DontUseBiasNeuron)
        {
            if ((m_NeuronGenes[GetNeuronIndex(t_in)].Type() == BIAS) && (NumLinks() == 1))
                return false;
            if (m_NeuronGenes[GetNeuronIndex(t_in)].Type() == BIAS)
                t_link_found = false;
        }
        if (!a_Parameters.SplitRecurrent)
        {
            if (m_LinkGenes[t_link_num].IsRecurrent() &&
                (!a_Parameters.SplitLoopedRecurrent) && (t_in == t_out))
            {
                t_link_found = false;
            }
        }
        if (--t_tries <= 0)
            return false;
    }
    double t_orig_weight = m_LinkGenes[t_link_num].GetWeight();
    t_chosenlink = m_LinkGenes[t_link_num];
    RemoveLinkGene(m_LinkGenes[t_link_num].InnovationID());
    int t_innovid = a_Innovs.CheckInnovation(t_in, t_out, NEW_NEURON);
    int t_nid = 0, t_l1id = 0, t_l2id = 0;
    if (t_innovid == -1)
    {
        t_nid = a_Innovs.AddNeuronInnovation(t_in, t_out, HIDDEN);
        t_l1id = a_Innovs.AddLinkInnovation(t_in, t_nid);
        t_l2id = a_Innovs.AddLinkInnovation(t_nid, t_out);
        double t_sy = (m_NeuronGenes[GetNeuronIndex(t_in)].SplitY() + m_NeuronGenes[GetNeuronIndex(t_out)].SplitY()) / 2.0;
        NeuronGene t_ngene(HIDDEN, t_nid, t_sy);
        double t_A = a_RNG.RandFloat(), t_B = a_RNG.RandFloat();
        double t_TC = a_RNG.RandFloat(), t_Bs = a_RNG.RandFloat();
        Scale(t_A, 0, 1, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
        Scale(t_B, 0, 1, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
        Scale(t_TC, 0, 1, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
        Scale(t_Bs, 0, 1, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);
        Clamp(t_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
        Clamp(t_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
        Clamp(t_TC, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
        Clamp(t_Bs, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);
        t_ngene.Init(t_A, t_B, t_TC, t_Bs, GetRandomActivation(a_Parameters, a_RNG));
        t_ngene.InitTraits(a_Parameters.NeuronTraits, a_RNG);
        m_NeuronGenes.push_back(t_ngene);
        bool t_recurrentflag = t_chosenlink.IsRecurrent();
        LinkGene l1(t_in, t_nid, t_l1id, 1.0, t_recurrentflag);
        Clamp(l1.m_Weight, a_Parameters.MinWeight, a_Parameters.MaxWeight);
        l1.InitTraits(a_Parameters.LinkTraits, a_RNG);
        m_LinkGenes.push_back(l1);
        LinkGene l2(t_nid, t_out, t_l2id, t_orig_weight, t_recurrentflag);
        l2.InitTraits(a_Parameters.LinkTraits, a_RNG);
        m_LinkGenes.push_back(l2);
    }
    else
    {
        t_nid = a_Innovs.FindNeuronID(t_in, t_out);
        ASSERT(t_nid != -1);
        t_l1id = a_Innovs.CheckInnovation(t_in, t_nid, NEW_LINK);
        t_l2id = a_Innovs.CheckInnovation(t_nid, t_out, NEW_LINK);
        ASSERT(t_l1id > 0 && t_l2id > 0);
        std::vector<int> t_idxs = a_Innovs.CheckAllInnovations(t_in, t_out, NEW_NEURON);
        bool t_found = false;
        for (int idx : t_idxs)
        {
            if (!HasNeuronID(a_Innovs.GetInnovationByIdx(idx).NeuronID()))
            {
                t_nid = a_Innovs.GetInnovationByIdx(idx).NeuronID();
                t_l1id = a_Innovs.CheckInnovation(t_in, t_nid, NEW_LINK);
                t_l2id = a_Innovs.CheckInnovation(t_nid, t_out, NEW_LINK);
                ASSERT(t_l1id > 0 && t_l2id > 0);
                t_found = true;
                break;
            }
        }
        if (!t_found)
        {
            t_nid = a_Innovs.AddNeuronInnovation(t_in, t_out, HIDDEN);
            t_l1id = a_Innovs.AddLinkInnovation(t_in, t_nid);
            t_l2id = a_Innovs.AddLinkInnovation(t_nid, t_out);
        }
        double t_sy = (m_NeuronGenes[GetNeuronIndex(t_in)].SplitY() + m_NeuronGenes[GetNeuronIndex(t_out)].SplitY()) / 2.0;
        NeuronGene t_ngene(HIDDEN, t_nid, t_sy);
        double t_A = a_RNG.RandFloat(), t_B = a_RNG.RandFloat();
        double t_TC = a_RNG.RandFloat(), t_Bs = a_RNG.RandFloat();
        Scale(t_A, 0, 1, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
        Scale(t_B, 0, 1, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
        Scale(t_TC, 0, 1, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
        Scale(t_Bs, 0, 1, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);
        Clamp(t_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
        Clamp(t_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
        Clamp(t_TC, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
        Clamp(t_Bs, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);
        t_ngene.Init(t_A, t_B, t_TC, t_Bs, GetRandomActivation(a_Parameters, a_RNG));
        t_ngene.InitTraits(a_Parameters.NeuronTraits, a_RNG);
        bool t_recurrentflag = t_chosenlink.IsRecurrent();
        m_NeuronGenes.push_back(t_ngene);
        LinkGene l1(t_in, t_nid, t_l1id, 1.0, t_recurrentflag);
        Clamp(l1.m_Weight, a_Parameters.MinWeight, a_Parameters.MaxWeight);
        l1.InitTraits(a_Parameters.LinkTraits, a_RNG);
        m_LinkGenes.push_back(l1);
        LinkGene l2(t_nid, t_out, t_l2id, t_orig_weight, t_recurrentflag);
        l2.InitTraits(a_Parameters.LinkTraits, a_RNG);
        m_LinkGenes.push_back(l2);
    }
    return true;
}

bool Genome::Mutate_AddLink(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG)
{
    int t_first_noninput = 0;
    int t_n1idx = 0, t_n2idx = 0;
    bool t_MakeRecurrent = false;
    bool t_LoopedRecurrent = false;
    bool t_MakeBias = false;
    unsigned int t_NumTries = 0;
    if (a_RNG.RandFloat() < a_Parameters.RecurrentProb)
    {
        t_MakeRecurrent = true;
        if (a_RNG.RandFloat() < a_Parameters.RecurrentLoopProb)
            t_LoopedRecurrent = true;
    }
    else
    {
        if (a_RNG.RandFloat() < a_Parameters.MutateAddLinkFromBiasProb)
            t_MakeBias = true;
    }
    for (unsigned i = 0, n = NumNeurons(); i < n; ++i)
    {
        if (m_NeuronGenes[i].Type() == INPUT || m_NeuronGenes[i].Type() == BIAS)
            ++t_first_noninput;
        else
            break;
    }
    bool t_Found = false;
    if (!t_MakeRecurrent)
    {
        bool t_found_bias = true;
        t_n1idx = static_cast<int>(NumInputs() - 1);
        t_NumTries = 0;
        do
        {
            t_n2idx = a_RNG.RandInt(t_first_noninput, NumNeurons() - 1);
            ++t_NumTries;
            if (t_NumTries >= a_Parameters.LinkTries)
            {
                t_found_bias = false;
                break;
            }
        }
        while (HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID()));
        if (t_found_bias && t_MakeBias)
            t_Found = true;
        else
        {
            t_NumTries = 0;
            do
            {
                t_n1idx = a_RNG.RandInt(0, NumNeurons() - 1);
                t_n2idx = a_RNG.RandInt(t_first_noninput, NumNeurons() - 1);
                ++t_NumTries;
                if (t_NumTries >= a_Parameters.LinkTries)
                    return false;
            }
            while (HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID()) ||
                  (m_NeuronGenes[t_n1idx].Type() == OUTPUT) || (t_n1idx == t_n2idx));
            t_Found = true;
        }
    }
    else if (t_MakeRecurrent && !t_LoopedRecurrent)
    {
        t_NumTries = 0;
        do
        {
            t_n1idx = a_RNG.RandInt(t_first_noninput, NumNeurons() - 1);
            t_n2idx = a_RNG.RandInt(t_first_noninput, NumNeurons() - 1);
            ++t_NumTries;
            if (t_NumTries >= a_Parameters.LinkTries)
                return false;
        }
        while (HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID()) || (t_n1idx == t_n2idx));
        t_Found = true;
    }
    else if (t_MakeRecurrent && t_LoopedRecurrent)
    {
        t_NumTries = 0;
        do
        {
            t_n1idx = t_n2idx = a_RNG.RandInt(t_first_noninput, NumNeurons() - 1);
            ++t_NumTries;
            if (t_NumTries >= a_Parameters.LinkTries)
                return false;
        }
        while (HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID()));
        t_Found = true;
    }
    if (!t_Found)
        return false;
    ASSERT(!HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID()));
    int t_n1id = m_NeuronGenes[t_n1idx].ID();
    int t_n2id = m_NeuronGenes[t_n2idx].ID();
    int t_innovid = a_Innovs.CheckInnovation(t_n1id, t_n2id, NEW_LINK);
    double t_weight = a_RNG.RandFloat();
    Scale(t_weight, 0, 1, a_Parameters.MinWeight, a_Parameters.MaxWeight);
    if (t_innovid == -1)
        t_innovid = a_Innovs.AddLinkInnovation(t_n1id, t_n2id);
    LinkGene l(t_n1id, t_n2id, t_innovid, t_weight, t_MakeRecurrent);
    l.InitTraits(a_Parameters.LinkTraits, a_RNG);
    m_LinkGenes.push_back(l);
    return true;
}

bool Genome::Mutate_RemoveLink(RNG &a_RNG)
{
    if (NumLinks() < 2)
        return false;
    int idx = a_RNG.RandInt(0, NumLinks() - 1);
    RemoveLinkGene(m_LinkGenes[idx].InnovationID());
    return true;
}

bool Genome::Mutate_RemoveSimpleNeuron(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG)
{
    if (NumNeurons() == (NumInputs() + NumOutputs()))
        return false;
    std::vector<int> t_neurons_to_delete;
    for (int i = 0; i < NumNeurons(); ++i)
    {
        if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) &&
            (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1) &&
            (m_NeuronGenes[i].Type() == HIDDEN))
            t_neurons_to_delete.push_back(i);
    }
    if (t_neurons_to_delete.empty())
        return false;
    int t_choice = (t_neurons_to_delete.size() == 2) ? Rounded(a_RNG.RandFloat()) : a_RNG.RandInt(0, static_cast<int>(t_neurons_to_delete.size() - 1));
    int t_l1idx = -1, t_l2idx = -1;
    for (int i = 0; i < NumLinks(); ++i)
    {
        if (m_LinkGenes[i].ToNeuronID() == m_NeuronGenes[t_neurons_to_delete[t_choice]].ID())
        {
            t_l1idx = i;
            break;
        }
    }
    for (int i = 0; i < NumLinks(); ++i)
    {
        if (m_LinkGenes[i].FromNeuronID() == m_NeuronGenes[t_neurons_to_delete[t_choice]].ID())
        {
            t_l2idx = i;
            break;
        }
    }
    ASSERT(t_l1idx >= 0 && t_l2idx >= 0);
    if (HasLink(m_LinkGenes[t_l1idx].FromNeuronID(), m_LinkGenes[t_l2idx].ToNeuronID()))
    {
        RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());
        return true;
    }
    else
    {
        double t_weight = m_LinkGenes[t_l1idx].GetWeight();
        int t_innovid = a_Innovs.CheckInnovation(m_LinkGenes[t_l1idx].FromNeuronID(), m_LinkGenes[t_l2idx].ToNeuronID(), NEW_LINK);
        if (t_innovid == -1)
        {
            int from = m_LinkGenes[t_l1idx].FromNeuronID();
            int to = m_LinkGenes[t_l2idx].ToNeuronID();
            RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());
            int t_newinnov = a_Innovs.AddLinkInnovation(from, to);
            LinkGene lg(from, to, t_newinnov, t_weight, false);
            lg.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.push_back(lg);
            return true;
        }
        else
        {
            int from = m_LinkGenes[t_l1idx].FromNeuronID();
            int to = m_LinkGenes[t_l2idx].ToNeuronID();
            RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());
            LinkGene lg(from, to, t_innovid, t_weight, false);
            lg.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.push_back(lg);
            return true;
        }
    }
    return false;
}

bool Genome::Cleanup()
{
    bool t_removed = false;
    for (size_t i = 0, end = m_NeuronGenes.size(); i < end; ++i)
    {
        if (m_NeuronGenes[i].Type() == HIDDEN && IsDeadEndNeuron(m_NeuronGenes[i].ID()))
        {
            RemoveNeuronGene(m_NeuronGenes[i].ID());
            t_removed = true;
        }
    }
    for (size_t i = 0, end = m_NeuronGenes.size(); i < end; ++i)
    {
        if (m_NeuronGenes[i].Type() == OUTPUT)
        {
            if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1))
            {
                for (size_t j = 0, end2 = m_LinkGenes.size(); j < end2; ++j)
                {
                    if (m_LinkGenes[j].ToNeuronID() == m_NeuronGenes[i].ID())
                    {
                        RemoveLinkGene(m_LinkGenes[j].InnovationID());
                        t_removed = true;
                    }
                }
            }
            if (NumOutputs() == 1)
                if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 0) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 0))
                    return true;
        }
    }
    return t_removed;
}

bool Genome::HasDeadEnds() const
{
    for (const auto &ng : m_NeuronGenes)
    {
        if (ng.Type() == HIDDEN && IsDeadEndNeuron(ng.ID()))
            return true;
    }
    for (const auto &ng : m_NeuronGenes)
    {
        if (ng.Type() == OUTPUT)
        {
            if ((LinksInputtingFrom(ng.ID()) == 1) && (LinksOutputtingTo(ng.ID()) == 1))
                return true;
            if (NumOutputs() == 1)
                if ((LinksInputtingFrom(ng.ID()) == 0) && (LinksOutputtingTo(ng.ID()) == 0))
                    return true;
        }
    }
    return false;
}

Genome Genome::Mate(Genome &a_Dad, bool a_MateAverage, bool a_InterSpecies, RNG &a_RNG, Parameters &a_Parameters)
{
    if (GetID() == a_Dad.GetID())
        return *this;
    enum t_parent_type { MOM, DAD };
    t_parent_type t_better;
    Genome t_baby;
    auto t_curMom = m_LinkGenes.begin();
    auto t_curDad = a_Dad.m_LinkGenes.begin();
    LinkGene t_selectedgene(0,0,-1,0,false);
    if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
    {
        Gene n;
        if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
            n = (GetFitness() > a_Dad.GetFitness()) ? m_GenomeGene : a_Dad.m_GenomeGene;
        else
            n = (a_RNG.RandFloat() < 0.5) ? m_GenomeGene : a_Dad.m_GenomeGene;
        t_baby.m_GenomeGene = n;
    }
    else
    {
        Gene n = m_GenomeGene;
        n.MateTraits(a_Dad.m_GenomeGene.m_Traits, a_RNG);
        t_baby.m_GenomeGene = n;
    }
    if (!a_Parameters.DontUseBiasNeuron)
    {
        t_baby.m_NeuronGenes.reserve(m_NumInputs + m_NumOutputs);
        for (unsigned i = 0; i < static_cast<unsigned>(m_NumInputs - 1); ++i)
            t_baby.m_NeuronGenes.push_back(m_NeuronGenes[i]);
        t_baby.m_NeuronGenes.push_back(m_NeuronGenes[m_NumInputs - 1]);
    }
    else
    {
        for (unsigned i = 0; i < static_cast<unsigned>(m_NumInputs); ++i)
            t_baby.m_NeuronGenes.push_back(m_NeuronGenes[i]);
    }
    for (unsigned i = 0; i < static_cast<unsigned>(m_NumOutputs); ++i)
    {
        NeuronGene t_tempneuron(OUTPUT, 0, 1);
        if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
        {
            if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
            {
                t_tempneuron = (GetFitness() > a_Dad.GetFitness()) ? GetNeuronByIndex(i + m_NumInputs) : a_Dad.GetNeuronByIndex(i + m_NumInputs);
            }
            else
            {
                t_tempneuron = (a_RNG.RandFloat() < 0.5) ? GetNeuronByIndex(i + m_NumInputs) : a_Dad.GetNeuronByIndex(i + m_NumInputs);
            }
        }
        else
        {
            t_tempneuron = GetNeuronByIndex(i + m_NumInputs);
            t_tempneuron.MateTraits(a_Dad.GetNeuronByIndex(i + m_NumInputs).m_Traits, a_RNG);
        }
        t_baby.m_NeuronGenes.push_back(t_tempneuron);
    }
    if (GetFitness() == a_Dad.GetFitness())
    {
        if (NumLinks() == a_Dad.NumLinks())
            t_better = (a_RNG.RandFloat() < 0.5) ? MOM : DAD;
        else
            t_better = (NumLinks() < a_Dad.NumLinks()) ? MOM : DAD;
    }
    else
        t_better = (GetFitness() > a_Dad.GetFitness()) ? MOM : DAD;

    LinkGene t_emptygene(0, 0, -1, 0, false);
    bool t_skip = false;
    int t_innov_mom = 0, t_innov_dad = 0;
    while (!(t_curMom == m_LinkGenes.end() && t_curDad == a_Dad.m_LinkGenes.end()))
    {
        t_selectedgene = t_emptygene;
        t_skip = false;
        t_innov_mom = t_innov_dad = 0;
        if (t_curMom == m_LinkGenes.end())
        {
            t_selectedgene = *t_curDad;
            ++t_curDad;
            if (t_better == MOM)
                t_skip = true;
        }
        else if (t_curDad == a_Dad.m_LinkGenes.end())
        {
            t_selectedgene = *t_curMom;
            ++t_curMom;
            if (t_better == DAD)
                t_skip = true;
        }
        else
        {
            t_innov_mom = t_curMom->InnovationID();
            t_innov_dad = t_curDad->InnovationID();
            if(t_innov_mom == t_innov_dad)
            {
                if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                {
                    if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                        t_selectedgene = (GetFitness() < a_Dad.GetFitness()) ? *t_curMom : *t_curDad;
                    else
                        t_selectedgene = (a_RNG.RandFloat() < 0.5) ? *t_curMom : *t_curDad;
                }
                else
                {
                    t_selectedgene = *t_curMom;
                    double t_Weight = (t_curDad->GetWeight() + t_curMom->GetWeight()) / 2.0;
                    t_selectedgene.SetWeight(t_Weight);
                    t_selectedgene.MateTraits(t_curDad->m_Traits, a_RNG);
                }
                ++t_curMom;
                ++t_curDad;
            }
            else if(t_innov_mom < t_innov_dad)
            {
                t_selectedgene = *t_curMom;
                ++t_curMom;
                if (t_better == DAD)
                    t_skip = true;
            }
            else if(t_innov_dad < t_innov_mom)
            {
                t_selectedgene = *t_curDad;
                ++t_curDad;
                if (t_better == MOM)
                    t_skip = true;
            }
        }
        if(a_InterSpecies)
            t_skip = false;
        if(t_selectedgene.InnovationID() > 0 && !t_baby.HasLink(t_selectedgene.FromNeuronID(), t_selectedgene.ToNeuronID()))
        {
            if(!t_skip)
            {
                t_baby.m_LinkGenes.push_back(t_selectedgene);
                if(!t_baby.HasNeuronID(t_selectedgene.FromNeuronID()) && HasNeuronID(t_selectedgene.FromNeuronID()))
                {
                    if(a_Dad.HasNeuronID(t_selectedgene.FromNeuronID()))
                    {
                        if(a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                            t_baby.m_NeuronGenes.push_back((GetFitness() > a_Dad.GetFitness()) ?
                                                           GetNeuronByIndex(GetNeuronIndex(t_selectedgene.FromNeuronID())) :
                                                           a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                        else
                            t_baby.m_NeuronGenes.push_back((a_RNG.RandFloat() < 0.5) ?
                                                           GetNeuronByIndex(GetNeuronIndex(t_selectedgene.FromNeuronID())) :
                                                           a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                    }
                    else
                        t_baby.m_NeuronGenes.push_back(GetNeuronByIndex(GetNeuronIndex(t_selectedgene.FromNeuronID())));
                }
                if(!t_baby.HasNeuronID(t_selectedgene.ToNeuronID()) && HasNeuronID(t_selectedgene.ToNeuronID()))
                {
                    if(a_Dad.HasNeuronID(t_selectedgene.ToNeuronID()))
                    {
                        if(a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                            t_baby.m_NeuronGenes.push_back((GetFitness() > a_Dad.GetFitness()) ?
                                                           GetNeuronByIndex(GetNeuronIndex(t_selectedgene.ToNeuronID())) :
                                                           a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                        else
                            t_baby.m_NeuronGenes.push_back((a_RNG.RandFloat() < 0.5) ?
                                                           GetNeuronByIndex(GetNeuronIndex(t_selectedgene.ToNeuronID())) :
                                                           a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                    }
                    else
                        t_baby.m_NeuronGenes.push_back(GetNeuronByIndex(GetNeuronIndex(t_selectedgene.ToNeuronID())));
                }
                if(!t_baby.HasNeuronID(t_selectedgene.FromNeuronID()) && a_Dad.HasNeuronID(t_selectedgene.FromNeuronID()))
                {
                    if(HasNeuronID(t_selectedgene.FromNeuronID()))
                    {
                        if(a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                            t_baby.m_NeuronGenes.push_back((GetFitness() < a_Dad.GetFitness()) ?
                                                           a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.FromNeuronID())]:
                                                           GetNeuronByIndex(GetNeuronIndex(t_selectedgene.FromNeuronID())));
                        else
                            t_baby.m_NeuronGenes.push_back((a_RNG.RandFloat() < 0.5) ?
                                                           a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.FromNeuronID())]:
                                                           GetNeuronByIndex(GetNeuronIndex(t_selectedgene.FromNeuronID())));
                    }
                    else
                        t_baby.m_NeuronGenes.push_back(a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.FromNeuronID())]);
                }
                if(!t_baby.HasNeuronID(t_selectedgene.ToNeuronID()) && a_Dad.HasNeuronID(t_selectedgene.ToNeuronID()))
                {
                    if(HasNeuronID(t_selectedgene.ToNeuronID()))
                    {
                        if(a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                            t_baby.m_NeuronGenes.push_back((GetFitness() < a_Dad.GetFitness()) ?
                                                           a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())] :
                                                           GetNeuronByIndex(GetNeuronIndex(t_selectedgene.ToNeuronID())));
                        else
                            t_baby.m_NeuronGenes.push_back((a_RNG.RandFloat() < 0.5) ?
                                                           a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())] :
                                                           GetNeuronByIndex(GetNeuronIndex(t_selectedgene.ToNeuronID())));
                    }
                    else
                        t_baby.m_NeuronGenes.push_back(a_Dad.m_NeuronGenes[a_Dad.GetNeuronIndex(t_selectedgene.ToNeuronID())]);
                }
            }
        }
    }
    t_baby.m_NumInputs = m_NumInputs;
    t_baby.m_NumOutputs = m_NumOutputs;
    t_baby.SortGenes();
    return t_baby;
}

void Genome::SortGenes()
{
    std::sort(m_NeuronGenes.begin(), m_NeuronGenes.end(),
              [](const NeuronGene &lhs, const NeuronGene &rhs) { return lhs.ID() < rhs.ID(); });
    std::sort(m_LinkGenes.begin(), m_LinkGenes.end(),
              [](const LinkGene &lhs, const LinkGene &rhs) { return lhs.InnovationID() < rhs.InnovationID(); });
}

unsigned int Genome::NeuronDepth(int a_NeuronID, unsigned int a_Depth)
{
    unsigned int t_max_depth = a_Depth;
    if(a_Depth > 16384)
        return 16384;
    if(GetNeuronByID(a_NeuronID).Type() == INPUT || GetNeuronByID(a_NeuronID).Type() == BIAS)
        return a_Depth;
    std::vector<int> t_inputting_links_idx;
    for (int i = 0; i < NumLinks(); ++i)
    {
        if(m_LinkGenes[i].ToNeuronID() == a_NeuronID)
            t_inputting_links_idx.push_back(i);
    }
    for (int idx : t_inputting_links_idx)
    {
        LinkGene t_link = GetLinkByIndex(idx);
        unsigned int t_current_depth = NeuronDepth(t_link.FromNeuronID(), a_Depth + 1);
        if(t_current_depth > t_max_depth)
            t_max_depth = t_current_depth;
    }
    return t_max_depth;
}

void Genome::CalculateDepth()
{
    if(NumNeurons() == (m_NumInputs + m_NumOutputs))
        m_Depth = 1;
    else
        m_Depth = 1;
}

Genome::Genome(const char *a_FileName)
{
    std::ifstream data(a_FileName);
    if(!data.is_open())
        throw std::runtime_error("Cannot open genome file.");
    std::string st;
    do { data >> st; }
    while(st != "GenomeStart" && !data.eof());
    data >> m_ID;
    do {
        data >> st;
        if(st=="Neuron")
        {
            int tid, ttype, tact;
            double tsplity, ta, tb, ttc, tbias;
            data >> tid >> ttype >> tsplity >> tact >> ta >> tb >> ttc >> tbias;
            NeuronGene N(static_cast<NeuronType>(ttype), tid, tsplity);
            N.m_ActFunction = static_cast<ActivationFunction>(tact);
            N.m_A = ta; N.m_B = tb; N.m_TimeConstant = ttc; N.m_Bias = tbias;
            m_NeuronGenes.push_back(N);
        }
        else if(st=="Link")
        {
            int f, t, inv, isrec;
            double w;
            data >> f >> t >> inv >> isrec >> w;
            LinkGene L(f, t, inv, w, static_cast<bool>(isrec));
            m_LinkGenes.push_back(L);
        }
    }
    while(st!="GenomeEnd" && !data.eof());
    data.close();
    m_NumInputs = 0;
    m_NumOutputs = 0;
    for (const auto &ng : m_NeuronGenes)
    {
        if(ng.Type() == INPUT || ng.Type() == BIAS)
            ++m_NumInputs;
        else if(ng.Type() == OUTPUT)
            ++m_NumOutputs;
    }
    m_Fitness = 0;
    m_AdjustedFitness = 0;
    m_OffspringAmount = 0;
    m_Depth = 0;
    m_Evaluated = false;
    m_PhenotypeBehavior = nullptr;
    m_initial_num_neurons = static_cast<int>(NumNeurons());
    m_initial_num_links   = static_cast<int>(NumLinks());
}

Genome::Genome(std::ifstream &data)
{
    if(!data)
        throw std::runtime_error("Invalid file stream for Genome constructor.");
    std::string st;
    do { data >> st; }
    while(st != "GenomeStart" && !data.eof());
    data >> m_ID;
    do {
        data >> st;
        if(st=="Neuron")
        {
            int tid, ttype, tact;
            double tsplity, ta, tb, ttc, tbias;
            data >> tid >> ttype >> tsplity >> tact >> ta >> tb >> ttc >> tbias;
            NeuronGene N(static_cast<NeuronType>(ttype), tid, tsplity);
            N.m_ActFunction = static_cast<ActivationFunction>(tact);
            N.m_A = ta; N.m_B = tb; N.m_TimeConstant = ttc; N.m_Bias = tbias;
            m_NeuronGenes.push_back(N);
        }
        else if(st=="Link")
        {
            int f, t, inv, isrec;
            double w;
            data >> f >> t >> inv >> isrec >> w;
            LinkGene L(f, t, inv, w, static_cast<bool>(isrec));
            m_LinkGenes.push_back(L);
        }
    }
    while(st!="GenomeEnd" && !data.eof());
    m_NumInputs = 0;
    m_NumOutputs = 0;
    for (const auto &ng: m_NeuronGenes)
    {
        if(ng.Type()==INPUT || ng.Type()==BIAS)
            ++m_NumInputs;
        else if(ng.Type()==OUTPUT)
            ++m_NumOutputs;
    }
    m_Fitness = 0;
    m_AdjustedFitness = 0;
    m_OffspringAmount = 0;
    m_Depth = 0;
    m_Evaluated = false;
    m_PhenotypeBehavior = nullptr;
    m_initial_num_neurons = static_cast<int>(NumNeurons());
    m_initial_num_links = static_cast<int>(NumLinks());
}

void Genome::Save(const char *a_FileName)
{
    FILE* fp = fopen(a_FileName,"w");
    if(!fp)
        throw std::runtime_error("Cannot open file for Genome::Save()");
    Save(fp);
    fclose(fp);
}

void Genome::Save(FILE *fp)
{
    fprintf(fp, "GenomeStart %d\n", m_ID);
    for (const auto &ng : m_NeuronGenes)
    {
        fprintf(fp, "Neuron %d %d %3.8f %d %3.8f %3.8f %3.8f %3.8f\n",
                ng.m_ID, static_cast<int>(ng.m_Type), ng.m_SplitY,
                static_cast<int>(ng.m_ActFunction), ng.m_A, ng.m_B,
                ng.m_TimeConstant, ng.m_Bias);
    }
    for (const auto &lg : m_LinkGenes)
    {
        fprintf(fp, "Link %d %d %d %d %3.8f\n",
                lg.m_FromNeuronID, lg.m_ToNeuronID, lg.m_InnovationID,
                static_cast<int>(lg.m_IsRecurrent), lg.m_Weight);
    }
    fprintf(fp, "GenomeEnd\n\n");
}

void Genome::PrintTraits(std::map<std::string, Trait>& traits)
{
    for (auto &kv : traits)
    {
        bool doit = false;
        if(!kv.second.dep_key.empty())
        {
            if(traits.count(kv.second.dep_key) != 0)
            {
                for(auto &dv : kv.second.dep_values)
                {
                    if(traits.at(kv.second.dep_key).value == dv)
                    {
                        doit = true;
                        break;
                    }
                }
            }
        }
        else
            doit = true;
        if(doit)
        {
            std::cout << kv.first << " - ";
            if(std::holds_alternative<int>(kv.second.value))
                std::cout << std::get<int>(kv.second.value);
            else if(std::holds_alternative<double>(kv.second.value))
                std::cout << std::get<double>(kv.second.value);
            else if(std::holds_alternative<std::string>(kv.second.value))
                std::cout << "\"" << std::get<std::string>(kv.second.value) << "\"";
            else if(std::holds_alternative<intsetelement>(kv.second.value))
                std::cout << std::get<intsetelement>(kv.second.value).value;
            else if(std::holds_alternative<floatsetelement>(kv.second.value))
                std::cout << std::get<floatsetelement>(kv.second.value).value;
            std::cout << ", ";
        }
    }
}

void Genome::PrintAllTraits()
{
    std::cout << "====================================================================\n";
    std::cout << "Genome:\n==================================\n";
    PrintTraits(m_GenomeGene.m_Traits);
    std::cout << "\n";
    std::cout << "====================================================================\n";
    std::cout << "Neurons:\n==================================\n";
    for (auto &n : m_NeuronGenes)
    {
        std::cout << "ID: " << n.ID() << " : ";
        PrintTraits(n.m_Traits);
        std::cout << "\n";
    }
    std::cout << "==================================\n";
    std::cout << "Links:\n==================================\n";
    for (auto &l : m_LinkGenes)
    {
        std::cout << "ID: " << l.InnovationID() << " : ";
        PrintTraits(l.m_Traits);
        std::cout << "\n";
    }
    std::cout << "==================================\n";
    std::cout << "====================================================================\n";
}

} // namespace NEAT

#endif 