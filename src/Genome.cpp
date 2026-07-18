#ifndef GENOME_CPP
#define GENOME_CPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include "Assert.h"
#include "Genome.h"
#include "Random.h"
#include "Parameters.h"
#include "Serialization.h"
#include "FileIO.h"
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
  : m_ID(0), m_NumInputs(0), m_NumOutputs(0), m_Fitness(0.0),
    m_AdjustedFitness(0.0), m_Depth(0), m_OffspringAmount(0.0),
    m_Evaluated(false), m_initial_num_neurons(0), m_initial_num_links(0),
    m_PhenotypeBehavior(nullptr)
{
}


Genome::Genome(const Genome &a_G)
  : m_ID(a_G.m_ID), m_NumInputs(a_G.m_NumInputs),
    m_NumOutputs(a_G.m_NumOutputs), m_Fitness(a_G.m_Fitness),
    m_AdjustedFitness(a_G.m_AdjustedFitness), m_Depth(a_G.m_Depth),
    m_OffspringAmount(a_G.m_OffspringAmount),
    m_NeuronGenes(a_G.m_NeuronGenes), m_LinkGenes(a_G.m_LinkGenes),
    m_GenomeGene(a_G.m_GenomeGene), m_Evaluated(a_G.m_Evaluated),
    m_initial_num_neurons(a_G.m_initial_num_neurons),
    m_initial_num_links(a_G.m_initial_num_links),
    m_PhenotypeBehavior(a_G.m_PhenotypeBehavior)
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

bool Genome::Validate(std::string* error) const
{
    const auto fail = [error](const std::string& message)
    {
        if (error != nullptr)
        {
            *error = message;
        }
        return false;
    };

    if (m_NumInputs < 0 || m_NumOutputs < 0)
        return fail("Genome input/output counts cannot be negative");
    if (m_Depth < 0)
        return fail("Genome depth cannot be negative");
    if (!std::isfinite(m_Fitness) ||
        !std::isfinite(m_AdjustedFitness) ||
        !std::isfinite(m_OffspringAmount))
        return fail("Genome fitness and offspring state must be finite");
    if (static_cast<std::size_t>(m_NumInputs) +
            static_cast<std::size_t>(m_NumOutputs) >
        m_NeuronGenes.size())
        return fail("Genome input/output counts exceed its neuron count");
    if (m_initial_num_neurons < 0 || m_initial_num_links < 0)
        return fail("Genome initial complexity cannot be negative");

    std::map<int, bool> neuron_ids;
    int actual_inputs = 0;
    int actual_outputs = 0;
    for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i)
    {
        const auto &neuron = m_NeuronGenes[i];
        if (neuron.ID() <= 0 || !neuron_ids.emplace(neuron.ID(), true).second)
            return fail("Genome neuron IDs must be positive and unique");
        if (neuron.Type() < INPUT || neuron.Type() > OUTPUT)
            return fail("Genome contains an invalid neuron type");
        if (neuron.m_ActFunction < SIGNED_SIGMOID ||
            neuron.m_ActFunction > SOFTPLUS)
            return fail("Genome contains an invalid activation function");
        if (!std::isfinite(neuron.m_SplitY) ||
            !std::isfinite(neuron.m_A) ||
            !std::isfinite(neuron.m_B) ||
            !std::isfinite(neuron.m_TimeConstant) ||
            !std::isfinite(neuron.m_Bias))
            return fail("Genome neuron parameters must be finite");
        if (neuron.Type() == INPUT || neuron.Type() == BIAS)
            ++actual_inputs;
        else if (neuron.Type() == OUTPUT)
            ++actual_outputs;
        if (i < static_cast<std::size_t>(m_NumInputs))
        {
            if (neuron.Type() != INPUT && neuron.Type() != BIAS)
                return fail("Genome input neurons are not stored first");
        }
        else if (i < static_cast<std::size_t>(
                         m_NumInputs + m_NumOutputs) &&
                 neuron.Type() != OUTPUT)
        {
            return fail("Genome output neurons do not follow its inputs");
        }
    }
    if (actual_inputs != m_NumInputs || actual_outputs != m_NumOutputs)
        return fail("Genome input/output counts do not match its neuron types");

    std::map<int, bool> innovation_ids;
    std::set<std::pair<int, int>> link_endpoints;
    for (const auto &link : m_LinkGenes)
    {
        if (link.InnovationID() <= 0 ||
            !innovation_ids.emplace(link.InnovationID(), true).second)
            return fail("Genome innovation IDs must be positive and unique");
        if (!link_endpoints.emplace(
                link.FromNeuronID(), link.ToNeuronID()).second)
            return fail("Genome link endpoints must be unique");
        if (neuron_ids.count(link.FromNeuronID()) == 0 ||
            neuron_ids.count(link.ToNeuronID()) == 0)
            return fail("Genome link endpoint does not exist");
        if (!std::isfinite(link.GetWeight()))
            return fail("Genome link weights must be finite");
    }
    return true;
}


Genome::Genome(const Parameters &a_Parameters, const GenomeInitStruct &in)
{
    const int usable_inputs =
        a_Parameters.DontUseBiasNeuron ? in.NumInputs : in.NumInputs - 1;
    if (usable_inputs < 1 || in.NumOutputs < 1 || in.NumHidden < 0 ||
        in.NumLayers < 0)
    {
        throw std::invalid_argument(
            "Genome: input, output, hidden, and layer counts are invalid.");
    }
    if (in.FS_NEAT &&
        (in.FS_NEAT_links < 1 ||
         static_cast<long long>(in.FS_NEAT_links) >
             static_cast<long long>(usable_inputs) * in.NumOutputs))
    {
        throw std::invalid_argument(
            "Genome: FS_NEAT_links exceeds the number of unique "
            "input-to-output links.");
    }
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
    if (seed_type != PERCEPTRON && seed_type != LAYERED)
    {
        throw std::invalid_argument("Genome: unknown seed type");
    }
    if ((seed_type == LAYERED) && (in.NumHidden == 0))
    {
        seed_type = PERCEPTRON;
    }
    if (in.FS_NEAT && seed_type == LAYERED)
    {
        throw std::invalid_argument(
            "Genome: FS-NEAT initialization does not support layered seeds");
    }

    if (!a_Parameters.DontUseBiasNeuron)
    {
        // inputs except last
        for (unsigned i = 0, end = static_cast<unsigned>(in.NumInputs - 1); i < end; ++i)
        {
            NeuronGene input(INPUT, t_nnum++, 0.0);
            input.InitTraits(a_Parameters.NeuronTraits, t_RNG);
            m_NeuronGenes.push_back(input);
        }
        // bias
        NeuronGene bias(BIAS, t_nnum++, 0.0);
        bias.InitTraits(a_Parameters.NeuronTraits, t_RNG);
        m_NeuronGenes.push_back(bias);
    }
    else
    {
        for (unsigned i = 0, end = static_cast<unsigned>(in.NumInputs); i < end; ++i)
        {
            NeuronGene input(INPUT, t_nnum++, 0.0);
            input.InitTraits(a_Parameters.NeuronTraits, t_RNG);
            m_NeuronGenes.push_back(input);
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
            int linkcount = 0;
            while (linkcount < in.FS_NEAT_links)
            {
                for (unsigned i = 0; i < static_cast<unsigned>(in.NumOutputs); ++i)
                {
                    if (linkcount >= in.FS_NEAT_links)
                    {
                        break;
                    }
                    int t_inp_id = t_RNG.RandInt(1, usable_inputs);
                    int t_bias_id = in.NumInputs;
                    int t_out_id  = in.NumInputs + 1 + i;
                    bool found = false;
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

    m_GenomeGene.InitTraits(a_Parameters.GenomeTraits, t_RNG);

    m_Evaluated = false;
    m_NumInputs = in.NumInputs;
    m_NumOutputs = in.NumOutputs;
    m_initial_num_neurons = static_cast<int>(NumNeurons());
    m_initial_num_links   = static_cast<int>(NumLinks());
}


Genome::Genome(std::istream &data)
    : Genome()
{
    if (!data)
        throw std::runtime_error("Invalid input stream provided to Genome constructor.");

    std::string token;
    while (data >> token && token != "GenomeStart")
    {
    }
    if (token != "GenomeStart")
        throw std::runtime_error("Genome: missing GenomeStart marker.");

    data >> m_ID;
    if (!data)
        throw std::runtime_error("Genome: missing genome ID.");

    int format_version = 1;
    bool has_state = false;
    bool found_end = false;
    int last_neuron = -1;
    int last_link = -1;
    while (data >> token)
    {
        if (token == "GenomeEnd")
        {
            found_end = true;
            break;
        }
        if (token == "GenomeFormat")
        {
            data >> format_version;
            if (format_version < 1 || format_version > 2)
                throw std::runtime_error(
                    "Genome: unsupported serialization format.");
        }
        else if (token == "GenomeState")
        {
            int evaluated = 0;
            data >> m_Fitness >> m_AdjustedFitness >> m_OffspringAmount
                 >> m_Depth >> m_NumInputs >> m_NumOutputs >> evaluated
                 >> m_initial_num_neurons >> m_initial_num_links;
            m_Evaluated = evaluated != 0;
            has_state = true;
        }
        else if (token == "GenomeTraits")
        {
            m_GenomeGene.m_Traits = Serialization::ReadTraits(data);
        }
        else if (token == "Neuron")
        {
            int id, type, activation;
            double split_y, a, b, time_constant, bias;
            data >> id >> type >> split_y >> activation >> a >> b
                 >> time_constant >> bias;
            NeuronGene neuron(static_cast<NeuronType>(type), id, split_y);
            neuron.m_ActFunction =
                static_cast<ActivationFunction>(activation);
            neuron.m_A = a;
            neuron.m_B = b;
            neuron.m_TimeConstant = time_constant;
            neuron.m_Bias = bias;
            if (format_version >= 2)
                data >> neuron.x >> neuron.y;
            m_NeuronGenes.push_back(neuron);
            last_neuron = static_cast<int>(m_NeuronGenes.size()) - 1;
        }
        else if (token == "NeuronTraits")
        {
            if (last_neuron < 0)
                throw std::runtime_error(
                    "Genome: NeuronTraits appears before a neuron.");
            m_NeuronGenes[static_cast<std::size_t>(last_neuron)].m_Traits =
                Serialization::ReadTraits(data);
        }
        else if (token == "Link")
        {
            int from, to, innovation, recurrent;
            double weight;
            data >> from >> to >> innovation >> recurrent >> weight;
            m_LinkGenes.emplace_back(
                from, to, innovation, weight, recurrent != 0);
            last_link = static_cast<int>(m_LinkGenes.size()) - 1;
        }
        else if (token == "LinkTraits")
        {
            if (last_link < 0)
                throw std::runtime_error(
                    "Genome: LinkTraits appears before a link.");
            m_LinkGenes[static_cast<std::size_t>(last_link)].m_Traits =
                Serialization::ReadTraits(data);
        }
        else
        {
            std::string ignored;
            std::getline(data, ignored);
        }
        if (!data)
            throw std::runtime_error("Genome: malformed serialized data.");
    }
    if (!found_end)
        throw std::runtime_error("Genome: missing GenomeEnd marker.");

    if (!has_state)
    {
        m_NumInputs = 0;
        m_NumOutputs = 0;
        for (const auto &neuron : m_NeuronGenes)
        {
            if (neuron.Type() == INPUT || neuron.Type() == BIAS)
                ++m_NumInputs;
            else if (neuron.Type() == OUTPUT)
                ++m_NumOutputs;
        }
        m_initial_num_neurons = static_cast<int>(m_NeuronGenes.size());
        m_initial_num_links = static_cast<int>(m_LinkGenes.size());
    }
    m_PhenotypeBehavior = nullptr;
    std::string validation_error;
    if (!Validate(&validation_error))
    {
        throw std::runtime_error(
            "Genome: invalid serialized data: " + validation_error);
    }
}


std::string Genome::Serialize() const
{
    std::string validation_error;
    if (!Validate(&validation_error))
    {
        throw std::runtime_error(
            "Genome::Serialize: " + validation_error);
    }
    std::ostringstream output;
    Serialization::UseRoundTripPrecision(output);
    output << "GenomeStart " << GetID() << "\n";
    output << "GenomeFormat 2\n";
    output << "GenomeState " << m_Fitness << ' ' << m_AdjustedFitness << ' '
           << m_OffspringAmount << ' ' << m_Depth << ' ' << m_NumInputs << ' '
           << m_NumOutputs << ' ' << static_cast<int>(m_Evaluated) << ' '
           << m_initial_num_neurons << ' ' << m_initial_num_links << '\n';
    Serialization::WriteTraits(
        output, "GenomeTraits", m_GenomeGene.m_Traits);
    for (const auto &neuron : m_NeuronGenes)
    {
        output << "Neuron " << neuron.m_ID << ' '
               << static_cast<int>(neuron.m_Type) << ' ' << neuron.m_SplitY
               << ' ' << static_cast<int>(neuron.m_ActFunction) << ' '
               << neuron.m_A << ' ' << neuron.m_B << ' '
               << neuron.m_TimeConstant << ' ' << neuron.m_Bias << ' '
               << neuron.x << ' ' << neuron.y << '\n';
        Serialization::WriteTraits(
            output, "NeuronTraits", neuron.m_Traits);
    }
    for (const auto &link : m_LinkGenes)
    {
        output << "Link " << link.m_FromNeuronID << ' ' << link.m_ToNeuronID
               << ' ' << link.m_InnovationID << ' '
               << static_cast<int>(link.m_IsRecurrent) << ' ' << link.m_Weight
               << '\n';
        Serialization::WriteTraits(output, "LinkTraits", link.m_Traits);
    }
    output << "GenomeEnd\n";
    return output.str();
}

Genome Genome::Deserialize(const std::string &data)
{
    std::istringstream input(data);
    return Genome(input);
}

void Genome::SetDepth(unsigned int a_d)
{
    if (a_d > static_cast<unsigned int>(std::numeric_limits<int>::max()))
        throw std::out_of_range("Genome depth exceeds the supported range");
    m_Depth = static_cast<int>(a_d);
}
unsigned int Genome::GetDepth() const { return m_Depth; }
void Genome::SetID(int a_id) { m_ID = a_id; }
int Genome::GetID() const { return m_ID; }

void Genome::SetAdjFitness(double a_af) { m_AdjustedFitness = a_af; }
void Genome::SetFitness(double a_f) { m_Fitness = a_f; }
double Genome::GetAdjFitness() const { return m_AdjustedFitness; }
double Genome::GetFitness() const { return m_Fitness; }

void Genome::SetNeuronY(unsigned int idx, int val)
{
    m_NeuronGenes.at(idx).y = val;
}

void Genome::SetNeuronX(unsigned int idx, int val)
{
    m_NeuronGenes.at(idx).x = val;
}

void Genome::SetNeuronXY(unsigned int idx, int x, int y)
{
    m_NeuronGenes.at(idx).x = x;
    m_NeuronGenes.at(idx).y = y;
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
    for (const auto& l : m_LinkGenes)
    {
        if (l.FromNeuronID() == a_ID)
        {
            ++t_counter;
        }
    }
    return t_counter;
}

int Genome::LinksOutputtingTo(int a_ID) const
{
    int t_counter = 0;
    for (const auto& l : m_LinkGenes)
    {
        if (l.ToNeuronID() == a_ID)
        {
            ++t_counter;
        }
    }
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
    if (idx < 0)
    {
        throw std::out_of_range("Link index cannot be negative");
    }
    return m_LinkGenes.at(static_cast<std::size_t>(idx));
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
    if (idx < 0)
    {
        throw std::out_of_range("Neuron index cannot be negative");
    }
    return m_NeuronGenes.at(static_cast<std::size_t>(idx));
}

NeuronGene Genome::GetNeuronByID(int a_ID) const
{
    const int index = GetNeuronIndex(a_ID);
    if (index < 0)
    {
        throw std::out_of_range(
            "No neuron with ID " + std::to_string(a_ID) + " exists in the genome");
    }
    return m_NeuronGenes[static_cast<std::size_t>(index)];
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
    std::map<int, std::size_t> neuron_indices;
    for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i)
        neuron_indices.emplace(m_NeuronGenes[i].ID(), i);

    std::vector<std::vector<std::size_t>> adjacency(
        m_NeuronGenes.size());
    std::vector<std::size_t> indegree(m_NeuronGenes.size(), 0);
    for (const LinkGene& link : m_LinkGenes)
    {
        const auto source = neuron_indices.find(link.FromNeuronID());
        const auto target = neuron_indices.find(link.ToNeuronID());
        if (source == neuron_indices.end() ||
            target == neuron_indices.end())
        {
            return true;
        }
        adjacency[source->second].push_back(target->second);
        ++indegree[target->second];
    }

    std::queue<std::size_t> ready;
    for (std::size_t i = 0; i < indegree.size(); ++i)
    {
        if (indegree[i] == 0)
            ready.push(i);
    }
    std::size_t visited = 0;
    while (!ready.empty())
    {
        const std::size_t source = ready.front();
        ready.pop();
        ++visited;
        for (std::size_t target : adjacency[source])
        {
            if (--indegree[target] == 0)
                ready.push(target);
        }
    }
    return visited != m_NeuronGenes.size();
}


void Genome::BuildPhenotype(NeuralNetwork &a_Net)
{
    a_Net.Clear();
    a_Net.SetInputOutputDimentions(m_NumInputs, m_NumOutputs);

    std::map<int, int> neuron_indices;
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
        t_n.m_x = static_cast<double>(ng.x);
        t_n.m_y = static_cast<double>(ng.y);
        a_Net.AddNeuron(t_n);
        neuron_indices.emplace(
            ng.ID(),
            static_cast<int>(a_Net.m_neurons.size()) - 1);
    }

    for (const auto &lg : m_LinkGenes)
    {
        const auto source = neuron_indices.find(lg.FromNeuronID());
        const auto target = neuron_indices.find(lg.ToNeuronID());
        if (source == neuron_indices.end() ||
            target == neuron_indices.end())
        {
            throw std::runtime_error(
                "Genome contains a link whose endpoint neuron does not exist");
        }
        Connection c;
        c.m_source_neuron_idx = source->second;
        c.m_target_neuron_idx = target->second;
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

    double total = 0.0;
    for (const double probability : t_probs)
    {
        if (!std::isfinite(probability) || probability < 0.0)
        {
            throw std::invalid_argument(
                "Activation-function probabilities must be finite and "
                "non-negative");
        }
        total += probability;
    }
    if (!std::isfinite(total) || total <= 0.0)
    {
        throw std::invalid_argument(
            "At least one activation function must have positive "
            "probability");
    }
    return static_cast<ActivationFunction>(a_RNG.Roulette(t_probs));
}


void Genome::BuildHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst)
{
    if (subst.m_input_coords.empty() || subst.m_output_coords.empty())
    {
        throw std::invalid_argument(
            "A HyperNEAT substrate requires input and output coordinates");
    }
    int max_dims = subst.GetMaxDims();
    if (static_cast<int>(m_NumInputs) < subst.GetMinCPPNInputs() ||
        static_cast<int>(m_NumOutputs) < subst.GetMinCPPNOutputs())
    {
        throw std::invalid_argument(
            "The CPPN does not provide enough inputs or outputs for the substrate");
    }
    if (!std::isfinite(subst.m_max_weight_and_bias) ||
        subst.m_max_weight_and_bias < 0.0 ||
        !std::isfinite(subst.m_min_time_const) ||
        !std::isfinite(subst.m_max_time_const) ||
        subst.m_min_time_const > subst.m_max_time_const)
    {
        throw std::invalid_argument(
            "The substrate weight, bias, or time-constant range is invalid");
    }
    ASSERT(static_cast<int>(m_NumOutputs) >= subst.GetMinCPPNOutputs());

    net.Clear();
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
    int cppn_activation_steps = 8;
    if (!HasLoops())
    {
        CalculateDepth();
        cppn_activation_steps =
            std::max(1, static_cast<int>(GetDepth()));
    }

    if (subst.m_leaky)
    {
        ASSERT(static_cast<unsigned>(m_NumOutputs) >= static_cast<unsigned>(subst.GetMinCPPNOutputs()));

        for (unsigned i = net.NumInputs(); i < net.m_neurons.size(); ++i)
        {
            cppn.Flush();
            std::vector<double> cinputs(m_NumInputs, 0.0);
            const std::size_t from_dims =
                net.m_neurons[i].m_substrate_coords.size();
            for (std::size_t d = 0; d < from_dims; ++d)
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
            for (int z = 0; z < cppn_activation_steps; ++z)
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
            if (conn.size() != 4)
            {
                throw std::invalid_argument(
                    "Malformed custom substrate connection");
            }
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

            if (!subst.m_custom_conn_obeys_flags || !(
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
                if (!(
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
        for (int z = 0; z < cppn_activation_steps; ++z)
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

void Genome::BuildESHyperNEATPhenotype(
    NeuralNetwork &net,
    Substrate &subst,
    Parameters &params)
{
    if (subst.m_input_coords.empty() || subst.m_output_coords.empty())
    {
        throw std::invalid_argument(
            "An ES-HyperNEAT substrate requires input and output coordinates");
    }
    if (params.InitialDepth > params.MaxDepth)
    {
        throw std::invalid_argument(
            "ES-HyperNEAT InitialDepth cannot exceed MaxDepth");
    }
    // A depth of nine already permits 349,525 quadtree nodes per query.
    // Reject larger trees before an accidental parameter value can exhaust
    // memory or make phenotype construction effectively unbounded.
    constexpr unsigned int max_safe_depth = 9;
    if (params.MaxDepth > max_safe_depth)
    {
        throw std::invalid_argument(
            "ES-HyperNEAT MaxDepth exceeds the supported safe limit of 9");
    }
    const std::pair<const char*, double> finite_values[] = {
        {"DivisionThreshold", params.DivisionThreshold},
        {"VarianceThreshold", params.VarianceThreshold},
        {"BandThreshold", params.BandThreshold},
        {"CPPN_Bias", params.CPPN_Bias},
        {"Width", params.Width},
        {"Height", params.Height},
        {"Qtree_X", params.Qtree_X},
        {"Qtree_Y", params.Qtree_Y},
        {"LeoThreshold", params.LeoThreshold},
        {"maximum substrate weight", subst.m_max_weight_and_bias}};
    for (const auto& value : finite_values)
    {
        if (!std::isfinite(value.second))
        {
            throw std::invalid_argument(
                std::string("ES-HyperNEAT ") + value.first +
                " must be finite");
        }
    }
    if (params.DivisionThreshold < 0.0 ||
        params.VarianceThreshold < 0.0 ||
        params.BandThreshold < 0.0 ||
        params.Width <= 0.0 ||
        params.Height <= 0.0 ||
        subst.m_max_weight_and_bias < 0.0)
    {
        throw std::invalid_argument(
            "ES-HyperNEAT thresholds and weight range must be non-negative, "
            "and quadtree dimensions must be positive");
    }

    const int dimensions = std::max(2, subst.GetMaxDims());
    const int required_inputs =
        dimensions * 2 + (subst.m_with_distance ? 1 : 0) + 1;
    const int required_outputs = params.Leo ? 2 : 1;
    if (m_NumInputs < required_inputs || m_NumOutputs < required_outputs)
    {
        throw std::invalid_argument(
            "The CPPN does not provide enough inputs or outputs for "
            "ES-HyperNEAT");
    }
    const auto validate_coordinates =
        [dimensions](const std::vector<std::vector<double>>& coordinates)
    {
        for (const auto& coordinate : coordinates)
        {
            if (coordinate.empty() ||
                coordinate.size() > static_cast<std::size_t>(dimensions) ||
                !std::all_of(
                    coordinate.begin(),
                    coordinate.end(),
                    [](double value) { return std::isfinite(value); }))
            {
                return false;
            }
        }
        return true;
    };
    if (!validate_coordinates(subst.m_input_coords) ||
        !validate_coordinates(subst.m_output_coords))
    {
        throw std::invalid_argument(
            "ES-HyperNEAT substrate coordinates must be finite and "
            "dimensionally consistent");
    }
    if (subst.m_input_coords.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        subst.m_output_coords.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(
            "ES-HyperNEAT substrate exceeds the supported index range");
    }

    struct TreeNode
    {
        double x;
        double y;
        double width;
        double height;
        double weight = 0.0;
        double leo = 0.0;
        unsigned int level;
        std::array<std::unique_ptr<TreeNode>, 4> children;

        bool Divided() const
        {
            return children[0] != nullptr;
        }
    };
    struct CandidateConnection
    {
        std::vector<double> source;
        std::vector<double> target;
        double weight;
    };

    std::size_t tree_node_limit = 1;
    std::size_t nodes_at_level = 1;
    const unsigned int subdivision_levels =
        std::max(1U, params.MaxDepth);
    for (unsigned int level = 0; level < subdivision_levels; ++level)
    {
        nodes_at_level *= 4;
        tree_node_limit += nodes_at_level;
    }

    NeuralNetwork cppn(true);
    BuildPhenotype(cppn);
    int cppn_activation_steps = 8;
    if (!HasLoops())
    {
        CalculateDepth();
        cppn_activation_steps =
            std::max(1, static_cast<int>(GetDepth()));
    }

    const auto normalized_coordinate =
        [dimensions](const std::vector<double>& coordinate)
    {
        std::vector<double> result(
            static_cast<std::size_t>(dimensions), 0.0);
        std::copy(coordinate.begin(), coordinate.end(), result.begin());
        return result;
    };
    const auto query_cppn =
        [&](const std::vector<double>& source,
            const std::vector<double>& target)
    {
        std::vector<double> inputs(
            static_cast<std::size_t>(m_NumInputs), 0.0);
        for (int dimension = 0; dimension < dimensions; ++dimension)
        {
            if (static_cast<std::size_t>(dimension) < source.size())
                inputs[static_cast<std::size_t>(dimension)] =
                    source[static_cast<std::size_t>(dimension)];
            if (static_cast<std::size_t>(dimension) < target.size())
                inputs[static_cast<std::size_t>(
                    dimensions + dimension)] =
                    target[static_cast<std::size_t>(dimension)];
        }
        if (subst.m_with_distance)
        {
            double squared_distance = 0.0;
            for (int dimension = 0; dimension < dimensions; ++dimension)
            {
                const double difference =
                    inputs[static_cast<std::size_t>(dimension)] -
                    inputs[static_cast<std::size_t>(
                        dimensions + dimension)];
                squared_distance += difference * difference;
            }
            inputs[inputs.size() - 2] = std::sqrt(squared_distance);
        }
        inputs.back() = params.CPPN_Bias;

        cppn.Flush();
        cppn.Input(inputs);
        for (int step = 0; step < cppn_activation_steps; ++step)
            cppn.Activate();
        const std::vector<double> outputs = cppn.Output();
        if (outputs.size() < static_cast<std::size_t>(required_outputs) ||
            !std::isfinite(outputs.front()) ||
            (params.Leo && !std::isfinite(outputs.back())))
        {
            throw std::runtime_error(
                "ES-HyperNEAT CPPN produced an invalid output");
        }
        return std::pair<double, double>(
            outputs.front(), params.Leo ? outputs.back() : 0.0);
    };
    const auto variance = [](const TreeNode& node)
    {
        if (!node.Divided())
            return 0.0;
        double mean = 0.0;
        for (const auto& child : node.children)
            mean += child->weight;
        mean /= 4.0;
        double result = 0.0;
        for (const auto& child : node.children)
        {
            const double difference = child->weight - mean;
            result += difference * difference;
        }
        return result / 4.0;
    };
    const auto generated_coordinate =
        [dimensions](double x, double y)
    {
        std::vector<double> coordinate(
            static_cast<std::size_t>(dimensions), 0.0);
        coordinate[0] = x;
        coordinate[1] = y;
        return coordinate;
    };

    const auto sample_connections =
        [&](const std::vector<double>& fixed, bool outgoing)
    {
        auto root = std::make_unique<TreeNode>(
            TreeNode{params.Qtree_X,
                     params.Qtree_Y,
                     params.Width,
                     params.Height,
                     0.0,
                     0.0,
                     1,
                     {}});
        std::queue<TreeNode*> pending;
        pending.push(root.get());
        std::size_t tree_nodes = 1;
        while (!pending.empty())
        {
            TreeNode* parent = pending.front();
            pending.pop();
            const double child_width = parent->width / 2.0;
            const double child_height = parent->height / 2.0;
            const double xs[] = {
                parent->x - child_width,
                parent->x - child_width,
                parent->x + child_width,
                parent->x + child_width};
            const double ys[] = {
                parent->y - child_height,
                parent->y + child_height,
                parent->y + child_height,
                parent->y - child_height};
            for (std::size_t index = 0; index < 4; ++index)
            {
                auto child = std::make_unique<TreeNode>(
                    TreeNode{xs[index],
                             ys[index],
                             child_width,
                             child_height,
                             0.0,
                             0.0,
                             parent->level + 1,
                             {}});
                const std::vector<double> coordinate =
                    generated_coordinate(child->x, child->y);
                const auto outputs = outgoing
                    ? query_cppn(fixed, coordinate)
                    : query_cppn(coordinate, fixed);
                child->weight = outputs.first;
                child->leo = outputs.second;
                parent->children[index] = std::move(child);
            }
            tree_nodes += 4;
            if (tree_nodes > tree_node_limit)
            {
                throw std::runtime_error(
                    "ES-HyperNEAT quadtree exceeded its calculated limit");
            }
            if (parent->level < params.InitialDepth ||
                (parent->level < params.MaxDepth &&
                 variance(*parent) > params.DivisionThreshold))
            {
                for (auto& child : parent->children)
                    pending.push(child.get());
            }
        }

        std::vector<CandidateConnection> connections;
        std::function<void(const TreeNode&)> prune_and_express;
        prune_and_express =
            [&](const TreeNode& parent)
        {
            for (const auto& child_pointer : parent.children)
            {
                const TreeNode& child = *child_pointer;
                if (child.Divided() &&
                    variance(child) > params.VarianceThreshold)
                {
                    prune_and_express(child);
                    continue;
                }
                if (params.Leo && child.leo <= params.LeoThreshold)
                    continue;

                const std::vector<double> center =
                    generated_coordinate(child.x, child.y);
                std::vector<double> left = center;
                std::vector<double> right = center;
                std::vector<double> top = center;
                std::vector<double> bottom = center;
                left[0] -= parent.width;
                right[0] += parent.width;
                top[1] -= parent.height;
                bottom[1] += parent.height;
                const auto boundary_difference =
                    [&](const std::vector<double>& coordinate)
                {
                    const double boundary_weight = outgoing
                        ? query_cppn(fixed, coordinate).first
                        : query_cppn(coordinate, fixed).first;
                    return std::abs(child.weight - boundary_weight);
                };
                const double horizontal = std::min(
                    boundary_difference(left),
                    boundary_difference(right));
                const double vertical = std::min(
                    boundary_difference(top),
                    boundary_difference(bottom));
                if (std::max(horizontal, vertical) <= params.BandThreshold)
                    continue;

                connections.push_back(
                    outgoing
                        ? CandidateConnection{
                              fixed, center, child.weight}
                        : CandidateConnection{
                              center, fixed, child.weight});
            }
        };
        prune_and_express(*root);
        return connections;
    };

    const std::size_t input_count = subst.m_input_coords.size();
    const std::size_t output_count = subst.m_output_coords.size();
    const std::size_t hidden_offset = input_count + output_count;
    std::map<std::vector<double>, std::size_t> hidden_indices;
    std::vector<std::vector<double>> hidden_coordinates;
    const auto add_hidden =
        [&](const std::vector<double>& coordinate)
            -> std::pair<std::size_t, bool>
    {
        const auto existing = hidden_indices.find(coordinate);
        if (existing != hidden_indices.end())
            return {existing->second, false};
        if (hidden_coordinates.size() >= tree_node_limit)
        {
            throw std::length_error(
                "ES-HyperNEAT generated too many hidden nodes");
        }
        const std::size_t index = hidden_coordinates.size();
        hidden_coordinates.push_back(coordinate);
        hidden_indices.emplace(coordinate, index);
        return {index, true};
    };

    std::vector<Connection> generated_connections;
    std::set<std::pair<std::size_t, std::size_t>> connection_endpoints;
    const auto add_connection =
        [&](std::size_t source, std::size_t target, double raw_weight)
    {
        if (source == target ||
            !std::isfinite(raw_weight) ||
            !connection_endpoints.emplace(source, target).second)
        {
            return;
        }
        if (source > static_cast<std::size_t>(
                         std::numeric_limits<int>::max()) ||
            target > static_cast<std::size_t>(
                         std::numeric_limits<int>::max()))
        {
            throw std::length_error(
                "ES-HyperNEAT connection index exceeds the supported range");
        }
        Connection connection;
        connection.m_source_neuron_idx = static_cast<int>(source);
        connection.m_target_neuron_idx = static_cast<int>(target);
        connection.m_weight =
            raw_weight * subst.m_max_weight_and_bias;
        generated_connections.push_back(connection);
    };

    for (std::size_t input = 0; input < input_count; ++input)
    {
        const std::vector<double> coordinate =
            normalized_coordinate(subst.m_input_coords[input]);
        for (const auto& candidate :
             sample_connections(coordinate, true))
        {
            const auto hidden = add_hidden(candidate.target);
            add_connection(
                input, hidden_offset + hidden.first, candidate.weight);
        }
    }

    std::vector<std::vector<double>> frontier = hidden_coordinates;
    for (unsigned int iteration = 0;
         iteration < params.IterationLevel && !frontier.empty();
         ++iteration)
    {
        std::vector<std::vector<double>> next_frontier;
        for (const auto& source_coordinate : frontier)
        {
            const auto source = hidden_indices.find(source_coordinate);
            if (source == hidden_indices.end())
                throw std::logic_error(
                    "ES-HyperNEAT lost a generated hidden node");
            for (const auto& candidate :
                 sample_connections(source_coordinate, true))
            {
                const auto target = add_hidden(candidate.target);
                add_connection(
                    hidden_offset + source->second,
                    hidden_offset + target.first,
                    candidate.weight);
                if (target.second)
                    next_frontier.push_back(candidate.target);
            }
        }
        frontier = std::move(next_frontier);
    }

    for (std::size_t output = 0; output < output_count; ++output)
    {
        const std::vector<double> coordinate =
            normalized_coordinate(subst.m_output_coords[output]);
        for (const auto& candidate :
             sample_connections(coordinate, false))
        {
            const auto source = hidden_indices.find(candidate.source);
            if (source != hidden_indices.end())
            {
                add_connection(
                    hidden_offset + source->second,
                    input_count + output,
                    candidate.weight);
            }
        }
    }

    std::vector<Neuron> generated_neurons;
    generated_neurons.reserve(hidden_offset + hidden_coordinates.size());
    for (std::size_t index = 0; index < input_count; ++index)
    {
        Neuron neuron;
        neuron.m_a = 1.0;
        neuron.m_b = 0.0;
        neuron.m_substrate_coords = subst.m_input_coords[index];
        neuron.m_activation_function_type = LINEAR;
        neuron.m_type =
            index + 1 == input_count ? BIAS : INPUT;
        generated_neurons.push_back(neuron);
    }
    for (const auto& coordinate : subst.m_output_coords)
    {
        Neuron neuron;
        neuron.m_a = 1.0;
        neuron.m_b = 0.0;
        neuron.m_substrate_coords = coordinate;
        neuron.m_activation_function_type =
            subst.m_output_nodes_activation;
        neuron.m_type = OUTPUT;
        generated_neurons.push_back(neuron);
    }
    for (const auto& coordinate : hidden_coordinates)
    {
        Neuron neuron;
        neuron.m_a = 1.0;
        neuron.m_b = 0.0;
        neuron.m_substrate_coords = coordinate;
        neuron.m_activation_function_type =
            subst.m_hidden_nodes_activation;
        neuron.m_type = HIDDEN;
        generated_neurons.push_back(neuron);
    }

    // Retain only hidden nodes that are both reachable from an input and can
    // reach an output. This removes disconnected islands and entire cycles,
    // not merely nodes with an immediate missing predecessor/successor.
    const std::size_t neuron_count = generated_neurons.size();
    std::vector<bool> forward_reachable(neuron_count, false);
    std::vector<bool> backward_reachable(neuron_count, false);
    std::fill(
        forward_reachable.begin(),
        forward_reachable.begin() + static_cast<std::ptrdiff_t>(input_count),
        true);
    std::fill(
        backward_reachable.begin() +
            static_cast<std::ptrdiff_t>(input_count),
        backward_reachable.begin() +
            static_cast<std::ptrdiff_t>(hidden_offset),
        true);
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (const auto& connection : generated_connections)
        {
            const std::size_t source =
                static_cast<std::size_t>(connection.m_source_neuron_idx);
            const std::size_t target =
                static_cast<std::size_t>(connection.m_target_neuron_idx);
            if (forward_reachable[source] && !forward_reachable[target])
            {
                forward_reachable[target] = true;
                changed = true;
            }
        }
    }
    changed = true;
    while (changed)
    {
        changed = false;
        for (const auto& connection : generated_connections)
        {
            const std::size_t source =
                static_cast<std::size_t>(connection.m_source_neuron_idx);
            const std::size_t target =
                static_cast<std::size_t>(connection.m_target_neuron_idx);
            if (backward_reachable[target] && !backward_reachable[source])
            {
                backward_reachable[source] = true;
                changed = true;
            }
        }
    }

    std::vector<int> remap(neuron_count, -1);
    std::vector<Neuron> pruned_neurons;
    pruned_neurons.reserve(neuron_count);
    for (std::size_t index = 0; index < neuron_count; ++index)
    {
        const bool fixed_node = index < hidden_offset;
        if (fixed_node ||
            (forward_reachable[index] && backward_reachable[index]))
        {
            remap[index] = static_cast<int>(pruned_neurons.size());
            pruned_neurons.push_back(generated_neurons[index]);
        }
    }
    std::vector<Connection> pruned_connections;
    pruned_connections.reserve(generated_connections.size());
    for (auto connection : generated_connections)
    {
        const int source = remap[static_cast<std::size_t>(
            connection.m_source_neuron_idx)];
        const int target = remap[static_cast<std::size_t>(
            connection.m_target_neuron_idx)];
        if (source >= 0 && target >= 0)
        {
            connection.m_source_neuron_idx = source;
            connection.m_target_neuron_idx = target;
            pruned_connections.push_back(connection);
        }
    }

    net.Clear();
    net.SetInputOutputDimensions(
        static_cast<unsigned int>(input_count),
        static_cast<unsigned int>(output_count));
    net.m_neurons = std::move(pruned_neurons);
    net.m_connections = std::move(pruned_connections);
    net.Flush();
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
        const auto parameters = a_Parameters.GenomeTraits.find(kv.first);
        if (parameters == a_Parameters.GenomeTraits.end())
        {
            continue;
        }
        double val = kv.second * parameters->second.m_ImportanceCoeff;
        if (std::isnan(val) || std::isinf(val))
            val = 0.0;
        total_distance += val;
    }

    unsigned i1 = 0, i2 = 0;
    const auto by_innovation =
        [](const LinkGene &lhs, const LinkGene &rhs)
        {
            return lhs.InnovationID() < rhs.InnovationID();
        };
    std::vector<LinkGene> sorted_links1;
    std::vector<LinkGene> sorted_links2;
    const std::vector<LinkGene>* links1 = &m_LinkGenes;
    const std::vector<LinkGene>* links2 = &a_G.m_LinkGenes;
    if (!std::is_sorted(m_LinkGenes.begin(), m_LinkGenes.end(),
                        by_innovation))
    {
        sorted_links1 = m_LinkGenes;
        std::sort(
            sorted_links1.begin(), sorted_links1.end(), by_innovation);
        links1 = &sorted_links1;
    }
    if (!std::is_sorted(a_G.m_LinkGenes.begin(), a_G.m_LinkGenes.end(),
                        by_innovation))
    {
        sorted_links2 = a_G.m_LinkGenes;
        std::sort(
            sorted_links2.begin(), sorted_links2.end(), by_innovation);
        links2 = &sorted_links2;
    }

    while (!(i1 >= links1->size() && i2 >= links2->size()))
    {
        if (i1 == links1->size())
        {
            ++E;
            ++i2;
        }
        else if (i2 == links2->size())
        {
            ++E;
            ++i1;
        }
        else
        {
            int in1 = (*links1)[i1].InnovationID();
            int in2 = (*links2)[i2].InnovationID();
            if (in1 == in2)
            {
                ++M;
                if (a_Parameters.WeightDiffCoeff > 0)
                {
                    double wd =
                        (*links1)[i1].GetWeight() -
                        (*links2)[i2].GetWeight();
                    total_w_diff += (wd < 0) ? -wd : wd;
                }
                auto linktraitdist =
                    (*links1)[i1].GetTraitDistances(
                        (*links2)[i2].m_Traits);
                for (const auto &xx : linktraitdist)
                {
                    if (a_Parameters.LinkTraits.count(xx.first) == 0)
                    {
                        continue;
                    }
                    double val = xx.second;
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

    double maxsize = static_cast<double>(
        std::max(links1->size(), links2->size()));
    if (maxsize < 1.0) maxsize = 1.0;
    double normalizer = (a_Parameters.NormalizeGenomeSize) ? maxsize : 1.0;
    if(M < 1.0) M = 1.0;
    double dist_links = a_Parameters.ExcessCoeff * (E / normalizer)
                        + a_Parameters.DisjointCoeff * (D / normalizer)
                        + a_Parameters.WeightDiffCoeff * (total_w_diff / M);
    total_distance += dist_links;

    std::map<int, const NeuronGene*> other_neurons;
    for (const NeuronGene& neuron : a_G.m_NeuronGenes)
        other_neurons.emplace(neuron.ID(), &neuron);
    for (size_t i = 0; i < m_NeuronGenes.size(); ++i)
    {
        if(m_NeuronGenes[i].Type() == INPUT || m_NeuronGenes[i].Type() == BIAS)
            continue;
        const auto other_neuron =
            other_neurons.find(m_NeuronGenes[i].ID());
        if(other_neuron != other_neurons.end())
        {
            ++matching_neurons;
            const NeuronGene& oth = *other_neuron->second;
            if(a_Parameters.ActivationADiffCoeff>0)
                total_A_diff += std::abs(m_NeuronGenes[i].m_A - oth.m_A);
            if(a_Parameters.ActivationBDiffCoeff>0)
                total_B_diff += std::abs(m_NeuronGenes[i].m_B - oth.m_B);
            if(a_Parameters.TimeConstantDiffCoeff>0)
                total_TC_diff += std::abs(m_NeuronGenes[i].m_TimeConstant - oth.m_TimeConstant);
            if(a_Parameters.BiasDiffCoeff>0)
                total_bias_diff += std::abs(m_NeuronGenes[i].m_Bias - oth.m_Bias);
            if (a_Parameters.ActivationFunctionDiffCoeff > 0)
            {
                if (m_NeuronGenes[i].m_ActFunction != oth.m_ActFunction)
                {
                    total_act_diff++;
                }
            }
            auto nd = m_NeuronGenes[i].GetTraitDistances(oth.m_Traits);
            for (const auto &xx : nd)
            {
                if (a_Parameters.NeuronTraits.count(xx.first) == 0)
                {
                    continue;
                }
                double val = xx.second;
                if (std::isnan(val) || std::isinf(val))
                {
                    val = 0;
                }
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
        if (std::isnan(n) || std::isinf(n)) { n = 0.0; }
        total_distance += n;
    }
    for(const auto &xx : total_neuron_trait_diff)
    {
        double n = xx.second * a_Parameters.NeuronTraits.at(xx.first).m_ImportanceCoeff / matching_neurons;
        if (std::isnan(n) || std::isinf(n)) { n = 0.0; }
        total_distance += n;
    }
    return total_distance;
}

bool Genome::IsCompatibleWith(Genome &a_G, Parameters &a_Parameters)
{
    //if(this == &a_G) return true;
    //if(GetID() == a_G.GetID()) return true;
    double dist = CompatibilityDistance(a_G, a_Parameters);
    return (dist <= a_Parameters.CompatTreshold);
}


bool Genome::Mutate_LinkWeights(const Parameters &a_Parameters, RNG &a_RNG)
{
    bool did_mutate = false;
    bool severe = (a_RNG.RandFloat() < a_Parameters.MutateWeightsSevereProb);
    int tailstart = 0;
    if (NumLinks() > static_cast<unsigned int>(
                         std::max(0, m_initial_num_links)))
        tailstart = static_cast<int>(NumLinks() * 0.9);
    if(tailstart <= m_initial_num_links)
        tailstart = m_initial_num_links;
    for (size_t i = 0, end = m_LinkGenes.size(); i < end; ++i)
    {
        if (!severe && (a_RNG.RandFloat() < a_Parameters.WeightMutationRate))
        {
            const double original = m_LinkGenes[i].GetWeight();
            double w = original;
            bool in_tail = (static_cast<int>(i) >= tailstart);
            if(in_tail || a_RNG.RandFloat() < a_Parameters.WeightReplacementRate)
                w = a_RNG.RandFloatSigned() * a_Parameters.WeightReplacementMaxPower;
            else
                w += a_RNG.RandFloatSigned() * a_Parameters.WeightMutationMaxPower;
            Clamp(w, a_Parameters.MinWeight, a_Parameters.MaxWeight);
            if (w != original)
            {
                m_LinkGenes[i].SetWeight(w);
                did_mutate = true;
            }
        }
        else if(severe)
        {
            if(a_RNG.RandFloat() < a_Parameters.WeightMutationRate)
            {
                const double original = m_LinkGenes[i].GetWeight();
                double w = a_RNG.RandFloat();
                Scale(w, 0.0, 1.0, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                if (w != original)
                {
                    m_LinkGenes[i].SetWeight(w);
                    did_mutate = true;
                }
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
    bool did_mutate = false;
    for (auto &ng : m_NeuronGenes)
    {
        if(ng.Type() != INPUT && ng.Type() != BIAS)
        {
            const double original = ng.m_A;
            double r = a_RNG.RandFloatSigned() * a_Parameters.ActivationAMutationMaxPower;
            ng.m_A += r;
            Clamp(ng.m_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            did_mutate = did_mutate || ng.m_A != original;
        }
    }
    return did_mutate;
}

bool Genome::Mutate_NeuronActivations_B(const Parameters &a_Parameters, RNG &a_RNG)
{
    bool did_mutate = false;
    for (auto &ng : m_NeuronGenes)
    {
        if(ng.Type() != INPUT && ng.Type() != BIAS)
        {
            const double original = ng.m_B;
            double r = a_RNG.RandFloatSigned() * a_Parameters.ActivationBMutationMaxPower;
            ng.m_B += r;
            Clamp(ng.m_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            did_mutate = did_mutate || ng.m_B != original;
        }
    }
    return did_mutate;
}

bool Genome::Mutate_NeuronActivation_Type(const Parameters &a_Parameters, RNG &a_RNG)
{
    if (m_NeuronGenes.size() <= static_cast<std::size_t>(m_NumInputs))
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
    bool did_mutate = false;
    for (auto &ng : m_NeuronGenes)
    {
        if(ng.Type() != INPUT && ng.Type() != BIAS)
        {
            const double original = ng.m_TimeConstant;
            double r = a_RNG.RandFloatSigned() * a_Parameters.TimeConstantMutationMaxPower;
            ng.m_TimeConstant += r;
            Clamp(ng.m_TimeConstant, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            did_mutate = did_mutate || ng.m_TimeConstant != original;
        }
    }
    return did_mutate;
}

bool Genome::Mutate_NeuronBiases(const Parameters &a_Parameters, RNG &a_RNG)
{
    bool did_mutate = false;
    for (auto &ng : m_NeuronGenes)
    {
        if(ng.Type() != INPUT && ng.Type() != BIAS)
        {
            const double original = ng.m_Bias;
            double r = a_RNG.RandFloatSigned() * a_Parameters.BiasMutationMaxPower;
            ng.m_Bias += r;
            Clamp(ng.m_Bias, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);
            did_mutate = did_mutate || ng.m_Bias != original;
        }
    }
    return did_mutate;
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
    if (NumLinks() == 0 || a_Parameters.NeuronTries <= 0)
        return false;

    std::vector<std::size_t> eligible_links;
    eligible_links.reserve(m_LinkGenes.size());
    for (std::size_t index = 0; index < m_LinkGenes.size(); ++index)
    {
        const LinkGene& link = m_LinkGenes[index];
        const int source_index = GetNeuronIndex(link.FromNeuronID());
        if (source_index < 0)
            continue;
        if (!a_Parameters.DontUseBiasNeuron &&
            m_NeuronGenes[static_cast<std::size_t>(source_index)].Type() ==
                BIAS)
        {
            continue;
        }
        if (link.IsRecurrent())
        {
            if (link.IsLoopedRecurrent())
            {
                if (!a_Parameters.SplitLoopedRecurrent)
                    continue;
            }
            else if (!a_Parameters.SplitRecurrent)
            {
                continue;
            }
        }
        eligible_links.push_back(index);
    }
    if (eligible_links.empty())
        return false;

    const std::size_t t_link_num = eligible_links[static_cast<std::size_t>(
        a_RNG.RandInt(0, static_cast<int>(eligible_links.size()) - 1))];
    const int t_in = m_LinkGenes[t_link_num].FromNeuronID();
    const int t_out = m_LinkGenes[t_link_num].ToNeuronID();
    LinkGene t_chosenlink = m_LinkGenes[t_link_num];
    double t_orig_weight = m_LinkGenes[t_link_num].GetWeight();
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
    if (m_NeuronGenes.empty() || a_Parameters.LinkTries == 0)
        return false;

    bool t_MakeRecurrent = false;
    bool t_LoopedRecurrent = false;
    bool t_MakeBias = false;
    if (a_RNG.RandFloat() < a_Parameters.RecurrentProb)
    {
        t_MakeRecurrent = true;
        if (a_RNG.RandFloat() < a_Parameters.RecurrentLoopProb)
        {
            t_LoopedRecurrent = true;
        }
    }
    else
    {
        if (!a_Parameters.DontUseBiasNeuron &&
            a_RNG.RandFloat() < a_Parameters.MutateAddLinkFromBiasProb)
        {
            t_MakeBias = true;
        }
    }

    const auto is_non_input = [](NeuronType type)
    {
        return type != INPUT && type != BIAS;
    };

    std::map<int, std::size_t> neuron_indices;
    for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i)
        neuron_indices.emplace(m_NeuronGenes[i].ID(), i);
    std::vector<std::vector<std::size_t>> feedforward_adjacency(
        m_NeuronGenes.size());
    for (const LinkGene& link : m_LinkGenes)
    {
        if (link.IsRecurrent())
            continue;
        const auto source = neuron_indices.find(link.FromNeuronID());
        const auto target = neuron_indices.find(link.ToNeuronID());
        if (source != neuron_indices.end() &&
            target != neuron_indices.end())
        {
            feedforward_adjacency[source->second].push_back(target->second);
        }
    }

    const auto would_create_feedforward_cycle =
        [&feedforward_adjacency](std::size_t source,
                                std::size_t target)
    {
        std::vector<unsigned char> visited(
            feedforward_adjacency.size(), 0);
        std::vector<std::size_t> stack{target};
        while (!stack.empty())
        {
            const std::size_t current = stack.back();
            stack.pop_back();
            if (current == source)
                return true;
            if (visited[current] != 0)
                continue;
            visited[current] = 1;
            for (std::size_t next : feedforward_adjacency[current])
            {
                if (visited[next] == 0)
                    stack.push_back(next);
            }
        }
        return false;
    };

    std::vector<std::pair<std::size_t, std::size_t>> candidates;
    std::set<std::pair<int, int>> existing_links;
    for (const LinkGene& link : m_LinkGenes)
    {
        existing_links.emplace(
            link.FromNeuronID(), link.ToNeuronID());
    }
    for (std::size_t source = 0; source < m_NeuronGenes.size(); ++source)
    {
        const NeuronType source_type = m_NeuronGenes[source].Type();
        for (std::size_t target = 0; target < m_NeuronGenes.size(); ++target)
        {
            const NeuronType target_type = m_NeuronGenes[target].Type();
            if (!is_non_input(target_type) ||
                existing_links.count(
                    {m_NeuronGenes[source].ID(),
                     m_NeuronGenes[target].ID()}) != 0)
            {
                continue;
            }
            if (t_MakeBias)
            {
                if (source_type == BIAS)
                    candidates.emplace_back(source, target);
                continue;
            }
            if (!t_MakeRecurrent)
            {
                if (source != target && source_type != OUTPUT &&
                    !would_create_feedforward_cycle(source, target))
                {
                    candidates.emplace_back(source, target);
                }
                continue;
            }
            if (!is_non_input(source_type))
                continue;
            if (t_LoopedRecurrent)
            {
                if (source == target)
                    candidates.emplace_back(source, target);
            }
            else if (source != target)
            {
                candidates.emplace_back(source, target);
            }
        }
    }
    if (candidates.empty())
        return false;

    const auto selected = candidates[static_cast<std::size_t>(
        a_RNG.RandInt(0, static_cast<int>(candidates.size()) - 1))];
    const std::size_t t_n1idx = selected.first;
    const std::size_t t_n2idx = selected.second;
    ASSERT(!HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID()));
    int t_n1id = m_NeuronGenes[t_n1idx].ID();
    int t_n2id = m_NeuronGenes[t_n2idx].ID();
    int t_innovid = a_Innovs.CheckInnovation(t_n1id, t_n2id, NEW_LINK);
    double t_weight = a_RNG.RandFloat();
    Scale(t_weight, 0, 1, a_Parameters.MinWeight, a_Parameters.MaxWeight);
    if (t_innovid == -1)
    {
        t_innovid = a_Innovs.AddLinkInnovation(t_n1id, t_n2id);
    }
    LinkGene l(t_n1id, t_n2id, t_innovid, t_weight, t_MakeRecurrent);
    l.InitTraits(a_Parameters.LinkTraits, a_RNG);
    m_LinkGenes.push_back(l);
    return true;
}

bool Genome::Mutate_RemoveLink(RNG &a_RNG)
{
    if (NumLinks() < 2)
        return false;
    const int idx =
        a_RNG.RandInt(0, static_cast<int>(NumLinks()) - 1);
    RemoveLinkGene(m_LinkGenes[static_cast<std::size_t>(idx)].InnovationID());
    return true;
}

bool Genome::Mutate_RemoveSimpleNeuron(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG)
{
    if (NumNeurons() == (NumInputs() + NumOutputs()))
        return false;
    std::vector<int> t_neurons_to_delete;
    for (unsigned int i = 0; i < NumNeurons(); ++i)
    {
        if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) &&
            (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1) &&
            (m_NeuronGenes[i].Type() == HIDDEN))
            t_neurons_to_delete.push_back(static_cast<int>(i));
    }
    if (t_neurons_to_delete.empty())
        return false;
    const int t_choice =
        a_RNG.RandInt(0, static_cast<int>(t_neurons_to_delete.size()) - 1);
    int t_l1idx = -1, t_l2idx = -1;
    for (unsigned int i = 0; i < NumLinks(); ++i)
    {
        if (m_LinkGenes[i].ToNeuronID() == m_NeuronGenes[t_neurons_to_delete[t_choice]].ID())
        {
            t_l1idx = static_cast<int>(i);
            break;
        }
    }
    for (unsigned int i = 0; i < NumLinks(); ++i)
    {
        if (m_LinkGenes[i].FromNeuronID() == m_NeuronGenes[t_neurons_to_delete[t_choice]].ID())
        {
            t_l2idx = static_cast<int>(i);
            break;
        }
    }
    if (t_l1idx < 0 || t_l2idx < 0)
    {
        return false;
    }
    if (HasLink(m_LinkGenes[static_cast<std::size_t>(t_l1idx)].FromNeuronID(),
                m_LinkGenes[static_cast<std::size_t>(t_l2idx)].ToNeuronID()))
    {
        RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());
        return true;
    }
    else
    {
        double t_weight =
            m_LinkGenes[static_cast<std::size_t>(t_l1idx)].GetWeight();
        int t_innovid = a_Innovs.CheckInnovation(
            m_LinkGenes[static_cast<std::size_t>(t_l1idx)].FromNeuronID(),
            m_LinkGenes[static_cast<std::size_t>(t_l2idx)].ToNeuronID(),
            NEW_LINK);
        if (t_innovid == -1)
        {
            int from =
                m_LinkGenes[static_cast<std::size_t>(t_l1idx)].FromNeuronID();
            int to =
                m_LinkGenes[static_cast<std::size_t>(t_l2idx)].ToNeuronID();
            RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());
            int t_newinnov = a_Innovs.AddLinkInnovation(from, to);
            LinkGene lg(from, to, t_newinnov, t_weight, false);
            lg.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.push_back(lg);
            return true;
        }
        else
        {
            int from =
                m_LinkGenes[static_cast<std::size_t>(t_l1idx)].FromNeuronID();
            int to =
                m_LinkGenes[static_cast<std::size_t>(t_l2idx)].ToNeuronID();
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
    for (std::size_t i = 0; i < m_NeuronGenes.size();)
    {
        if (m_NeuronGenes[i].Type() == HIDDEN && IsDeadEndNeuron(m_NeuronGenes[i].ID()))
        {
            RemoveNeuronGene(m_NeuronGenes[i].ID());
            t_removed = true;
            continue;
        }
        ++i;
    }
    for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i)
    {
        if (m_NeuronGenes[i].Type() == OUTPUT)
        {
            if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1))
            {
                for (std::size_t j = 0; j < m_LinkGenes.size(); ++j)
                {
                    if (m_LinkGenes[j].ToNeuronID() == m_NeuronGenes[i].ID())
                    {
                        RemoveLinkGene(m_LinkGenes[j].InnovationID());
                        t_removed = true;
                        break;
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
        if ((ng.Type() == HIDDEN) && IsDeadEndNeuron(ng.ID()))
        {
            return true;
        }
    }
    for (const auto &ng : m_NeuronGenes)
    {
        if (ng.Type() == OUTPUT)
        {
            if ((LinksInputtingFrom(ng.ID()) == 1) && (LinksOutputtingTo(ng.ID()) == 1))
            {
                return true;
            }
            if (NumOutputs() == 1)
            {
                if ((LinksInputtingFrom(ng.ID()) == 0) && (LinksOutputtingTo(ng.ID()) == 0))
                {
                    return true;
                }
            }
        }
    }
    return false;
}

Genome Genome::Mate(Genome &a_Dad, bool a_MateAverage, bool a_InterSpecies, RNG &a_RNG, Parameters &a_Parameters)
{
    if (m_NumInputs != a_Dad.m_NumInputs ||
        m_NumOutputs != a_Dad.m_NumOutputs)
    {
        throw std::invalid_argument(
            "Cannot mate genomes with different input/output dimensions");
    }
    if (GetID() == a_Dad.GetID())
        return *this;
    enum t_parent_type { MOM, DAD };
    t_parent_type t_better;
    if (GetFitness() == a_Dad.GetFitness())
    {
        if (NumLinks() == a_Dad.NumLinks())
            t_better = (a_RNG.RandFloat() < 0.5) ? MOM : DAD;
        else
            t_better = (NumLinks() < a_Dad.NumLinks()) ? MOM : DAD;
    }
    else
    {
        t_better = (GetFitness() > a_Dad.GetFitness()) ? MOM : DAD;
    }

    Genome t_baby;
    std::vector<LinkGene> mom_links = m_LinkGenes;
    std::vector<LinkGene> dad_links = a_Dad.m_LinkGenes;
    std::sort(mom_links.begin(), mom_links.end());
    std::sort(dad_links.begin(), dad_links.end());
    auto t_curMom = mom_links.begin();
    auto t_curDad = dad_links.begin();
    if (!a_MateAverage)
    {
        Gene n;
        if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
            n = (t_better == MOM) ? m_GenomeGene : a_Dad.m_GenomeGene;
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
    t_baby.m_NeuronGenes.reserve(
        static_cast<std::size_t>(m_NumInputs + m_NumOutputs));
    for (int index = 0;
         index < m_NumInputs + m_NumOutputs;
         ++index)
    {
        const NeuronGene mom = GetNeuronByIndex(index);
        const NeuronGene dad = a_Dad.GetNeuronByIndex(index);
        if (mom.ID() != dad.ID() || mom.Type() != dad.Type())
        {
            throw std::invalid_argument(
                "Cannot mate genomes with incompatible input/output neurons");
        }
        NeuronGene child = mom;
        if (a_MateAverage)
        {
            child.MateTraits(dad.m_Traits, a_RNG);
        }
        else if (a_RNG.RandFloat() <
                 a_Parameters.PreferFitterParentRate)
        {
            child = (t_better == MOM) ? mom : dad;
        }
        else
        {
            child = (a_RNG.RandFloat() < 0.5) ? mom : dad;
        }
        t_baby.m_NeuronGenes.push_back(std::move(child));
    }
    LinkGene t_emptygene(0, 0, -1, 0, false);
    while (!(t_curMom == mom_links.end() && t_curDad == dad_links.end()))
    {
        LinkGene t_selectedgene = t_emptygene;
        bool t_skip = false;
        if (t_curMom == mom_links.end())
        {
            t_selectedgene = *t_curDad;
            ++t_curDad;
            if (t_better == MOM)
                t_skip = true;
        }
        else if (t_curDad == dad_links.end())
        {
            t_selectedgene = *t_curMom;
            ++t_curMom;
            if (t_better == DAD)
                t_skip = true;
        }
        else
        {
            const int t_innov_mom = t_curMom->InnovationID();
            const int t_innov_dad = t_curDad->InnovationID();
            if(t_innov_mom == t_innov_dad)
            {
                if (!a_MateAverage)
                {
                    if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                        t_selectedgene =
                            (t_better == MOM) ? *t_curMom : *t_curDad;
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
            else
            {
                t_selectedgene = *t_curDad;
                ++t_curDad;
                if (t_better == MOM)
                    t_skip = true;
            }
        }
        if (a_InterSpecies)
        {
            t_skip = false;
        }

        if(t_selectedgene.InnovationID() > 0 && !t_baby.HasLink(t_selectedgene.FromNeuronID(), t_selectedgene.ToNeuronID()))
        {
            if(!t_skip)
            {
                t_baby.m_LinkGenes.push_back(t_selectedgene);
                if(!t_baby.HasNeuronID(t_selectedgene.FromNeuronID()) && HasNeuronID(t_selectedgene.FromNeuronID()))
                {
                    if(a_Dad.HasNeuronID(t_selectedgene.FromNeuronID()))
                    {
                        if(!a_MateAverage)
                            t_baby.m_NeuronGenes.push_back((t_better == MOM) ?
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
                        if(!a_MateAverage)
                            t_baby.m_NeuronGenes.push_back((t_better == MOM) ?
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
                        if(!a_MateAverage)
                            t_baby.m_NeuronGenes.push_back((t_better == DAD) ?
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
                        if(!a_MateAverage)
                            t_baby.m_NeuronGenes.push_back((t_better == DAD) ?
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
    t_baby.m_initial_num_neurons = m_initial_num_neurons;
    t_baby.m_initial_num_links = m_initial_num_links;
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
    for (unsigned int i = 0; i < NumLinks(); ++i)
    {
        if(m_LinkGenes[i].ToNeuronID() == a_NeuronID)
            t_inputting_links_idx.push_back(static_cast<int>(i));
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
    if (m_NeuronGenes.empty())
    {
        m_Depth = 0;
        return;
    }

    std::map<int, std::size_t> neuron_indices;
    for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i)
    {
        neuron_indices[m_NeuronGenes[i].ID()] = i;
    }

    std::vector<std::vector<std::size_t>> outgoing(m_NeuronGenes.size());
    std::vector<std::size_t> indegree(m_NeuronGenes.size(), 0);
    for (const auto &link : m_LinkGenes)
    {
        if (link.IsRecurrent())
        {
            continue;
        }
        const auto source = neuron_indices.find(link.FromNeuronID());
        const auto target = neuron_indices.find(link.ToNeuronID());
        if (source == neuron_indices.end() || target == neuron_indices.end())
        {
            throw std::runtime_error(
                "Genome contains a link whose endpoint neuron does not exist");
        }
        outgoing[source->second].push_back(target->second);
        ++indegree[target->second];
    }

    std::vector<std::size_t> queue;
    queue.reserve(m_NeuronGenes.size());
    for (std::size_t i = 0; i < indegree.size(); ++i)
    {
        if (indegree[i] == 0)
        {
            queue.push_back(i);
        }
    }

    std::vector<unsigned int> depth(m_NeuronGenes.size(), 0);
    std::size_t head = 0;
    while (head < queue.size())
    {
        const std::size_t source = queue[head++];
        for (std::size_t target : outgoing[source])
        {
            depth[target] = std::max(depth[target], depth[source] + 1);
            if (--indegree[target] == 0)
            {
                queue.push_back(target);
            }
        }
    }
    if (queue.size() != m_NeuronGenes.size())
    {
        throw std::runtime_error(
            "Genome contains a cycle made of non-recurrent links");
    }

    unsigned int maximum = 0;
    for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i)
    {
        if (m_NeuronGenes[i].Type() == OUTPUT)
        {
            maximum = std::max(maximum, depth[i]);
        }
    }
    m_Depth = static_cast<int>(std::max(1U, maximum));
}

Genome::Genome(const char *a_FileName)
    : Genome()
{
    if (a_FileName == nullptr)
        throw std::invalid_argument("Genome filename is null");
    std::ifstream data(a_FileName);
    if (!data.is_open())
        throw std::runtime_error("Cannot open genome file.");
    *this = Genome(static_cast<std::istream&>(data));
}

Genome::Genome(std::ifstream &data)
    : Genome(static_cast<std::istream&>(data))
{
}

void Genome::Save(const char *a_FileName)
{
    FILE* fp = detail::OpenFile(a_FileName, "w");
    if (fp == nullptr)
    {
        throw std::runtime_error("Cannot open genome file for writing");
    }
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
                for (const auto &dv : kv.second.dep_values)
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

bool Genome::IsIdenticalTo(const Genome& other) const
{
    // First compare basic structural properties
    if (m_NumInputs != other.m_NumInputs || 
        m_NumOutputs != other.m_NumOutputs ||
        m_NeuronGenes.size() != other.m_NeuronGenes.size() ||
        m_LinkGenes.size() != other.m_LinkGenes.size())
    {
        return false;
    }
    if (m_GenomeGene.m_Traits != other.m_GenomeGene.m_Traits)
    {
        return false;
    }

    // Compare neuron genes (ignore IDs, focus on topology and parameters)
    for (size_t i = 0; i < m_NeuronGenes.size(); ++i)
    {
        const NeuronGene& n1 = m_NeuronGenes[i];
        const NeuronGene& n2 = other.m_NeuronGenes[i];
        
        if (n1.m_Type != n2.m_Type ||
            n1.x != n2.x ||
            n1.y != n2.y ||
            n1.m_SplitY != n2.m_SplitY ||
            n1.m_A != n2.m_A ||
            n1.m_B != n2.m_B ||
            n1.m_TimeConstant != n2.m_TimeConstant ||
            n1.m_Bias != n2.m_Bias ||
            n1.m_ActFunction != n2.m_ActFunction ||
            n1.m_Traits != n2.m_Traits)
        {
            return false;
        }
    }

    // Compare link genes (ignore innovation IDs, focus on topology and weights)
    for (size_t i = 0; i < m_LinkGenes.size(); ++i)
    {
        const LinkGene& l1 = m_LinkGenes[i];
        const LinkGene& l2 = other.m_LinkGenes[i];
        
        if (l1.m_FromNeuronID != l2.m_FromNeuronID ||
            l1.m_ToNeuronID != l2.m_ToNeuronID ||
            l1.m_Weight != l2.m_Weight ||
            l1.m_IsRecurrent != l2.m_IsRecurrent ||
            l1.m_Traits != l2.m_Traits)
        {
            return false;
        }
    }

    return true;
}

} // namespace NEAT

#endif
