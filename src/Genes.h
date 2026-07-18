#ifndef _GENES_H
#define _GENES_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <variant>
#include "Parameters.h"
#include "Traits.h"
#include "Random.h"
#include "Utils.h"

namespace NEAT
{

    //////////////////////////////////////////////
    // Enumeration for all available neuron types
    //////////////////////////////////////////////
    enum NeuronType
    {
        NONE = 0,
        INPUT,
        BIAS,
        HIDDEN,
        OUTPUT
    };

    /////////////////////////////////////////////////
    // Enumeration for all possible activation function types
    /////////////////////////////////////////////////
    enum ActivationFunction
    {
        SIGNED_SIGMOID = 0,
        UNSIGNED_SIGMOID,
        TANH,
        TANH_CUBIC,
        SIGNED_STEP,
        UNSIGNED_STEP,
        SIGNED_GAUSS,
        UNSIGNED_GAUSS,
        ABS,
        SIGNED_SINE,
        UNSIGNED_SINE,
        LINEAR,
        RELU,
        SOFTPLUS,
        // Stateful activation modes advanced by NeuralNetwork::StepSpiking.
        // They are appended to preserve the numeric values of every
        // historical activation function.
        SPIKING_LIF,
        SPIKING_ADAPTIVE_LIF,
        SPIKING_IZHIKEVICH
    };

    inline bool IsSpikingActivation(ActivationFunction function)
    {
        return function == SPIKING_LIF ||
               function == SPIKING_ADAPTIVE_LIF ||
               function == SPIKING_IZHIKEVICH;
    }

    //////////////////////////////////
    // Base Gene class
    //////////////////////////////////
    class Gene
    {
    public:
        // Arbitrary traits
        std::map<std::string, Trait> m_Traits;

        Gene& operator=(const Gene&) = default;

        // Initialize traits (randomize) based on parameters
        void InitTraits(const std::map<std::string, TraitParameters> &tp, RNG &a_RNG)
        {
            for (auto it = tp.begin(); it != tp.end(); ++it)
            {
                // Check the type and create such trait
                TraitType t;
                if (it->second.type == "int")
                {
                    IntTraitParameters itp = std::get<IntTraitParameters>(it->second.m_Details);
                    if (itp.min > itp.max)
                    {
                        throw std::invalid_argument("Integer trait minimum exceeds maximum");
                    }
                    t = a_RNG.RandInt(itp.min, itp.max);
                }
                else if (it->second.type == "float")
                {
                    FloatTraitParameters itp = std::get<FloatTraitParameters>(it->second.m_Details);
                    if (itp.min > itp.max)
                    {
                        throw std::invalid_argument("Floating-point trait minimum exceeds maximum");
                    }
                    double x = a_RNG.RandFloat();
                    Scale(x, 0, 1, itp.min, itp.max);
                    t = x;
                }
                else if (it->second.type == "str")
                {
                    StringTraitParameters itp = std::get<StringTraitParameters>(it->second.m_Details);
                    std::vector<double> probs = itp.probs;
                    if (itp.set.empty())
                    {
                        throw std::runtime_error("Empty set of string traits");
                    }
                    probs.resize(itp.set.size()); // in case it didn't match length
                    int idx = a_RNG.Roulette(probs);
                    t = itp.set[idx];
                }
                else if (it->second.type == "intset")
                {
                    IntSetTraitParameters itp = std::get<IntSetTraitParameters>(it->second.m_Details);
                    std::vector<double> probs = itp.probs;
                    if (itp.set.empty())
                    {
                        throw std::runtime_error("Empty set of int traits");
                    }
                    probs.resize(itp.set.size());
                    int idx = a_RNG.Roulette(probs);
                    t = itp.set[idx];
                }
                else if (it->second.type == "floatset")
                {
                    FloatSetTraitParameters itp = std::get<FloatSetTraitParameters>(it->second.m_Details);
                    std::vector<double> probs = itp.probs;
                    if (itp.set.empty())
                    {
                        throw std::runtime_error("Empty set of float traits");
                    }
                    probs.resize(itp.set.size());
                    int idx = a_RNG.Roulette(probs);
                    t = itp.set[idx];
                }
                else
                {
                    throw std::invalid_argument("Unknown trait type: " + it->second.type);
                }

                Trait tr;
                tr.value = t;
                tr.dep_key = it->second.dep_key;
                tr.dep_values = it->second.dep_values;
                m_Traits[it->first] = tr;
            }
        }

        // Mates traits with another parent's traits
        void MateTraits(const std::map<std::string, Trait> &t, RNG &a_RNG)
        {
            for (auto it = t.begin(); it != t.end(); ++it)
            {
                // Both must share the key
                const auto mine_it = m_Traits.find(it->first);
                if (mine_it == m_Traits.end())
                {
                    continue;
                }
                TraitType mine = mine_it->second.value;
                TraitType yours = it->second.value;

                // Type must match
                if (mine.index() != yours.index())
                {
                    throw std::runtime_error("Types of traits don't match in mating");
                }

                // 50% chance pick either-or, else attempt averaging (if numeric)
                if (a_RNG.RandFloat() < 0.5)
                {
                    // pick either one
                    m_Traits[it->first].value = (a_RNG.RandFloat() < 0.5) ? mine : yours;
                }
                else
                {
                    // try to average if numeric
                    if (std::holds_alternative<int>(mine))
                    {
                        int m1 = std::get<int>(mine);
                        int m2 = std::get<int>(yours);
                        m_Traits[it->first].value = static_cast<int>(
                            (static_cast<long long>(m1) +
                             static_cast<long long>(m2)) /
                            2LL);
                    }
                    else if (std::holds_alternative<double>(mine))
                    {
                        double m1 = std::get<double>(mine);
                        double m2 = std::get<double>(yours);
                        m_Traits[it->first].value = (m1 + m2) / 2.0;
                    }
                    else if (std::holds_alternative<std::string>(mine))
                    {
                        m_Traits[it->first].value = 
                            (a_RNG.RandFloat() < 0.5) ? mine : yours;
                    }
                    else if (std::holds_alternative<intsetelement>(mine))
                    {
                        m_Traits[it->first].value = 
                            (a_RNG.RandFloat() < 0.5) ? mine : yours;
                    }
                    else if (std::holds_alternative<floatsetelement>(mine))
                    {
                        m_Traits[it->first].value = 
                            (a_RNG.RandFloat() < 0.5) ? mine : yours;
                    }
                }
            }
        }

        // Mutates traits according to parameters
        bool MutateTraits(const std::map<std::string, TraitParameters> &tp, RNG &a_RNG)
        {
            bool did_mutate = false;
            for (auto it = tp.begin(); it != tp.end(); ++it)
            {
                auto trait_it = m_Traits.find(it->first);
                if (trait_it == m_Traits.end())
                {
                    continue;
                }

                // check if we should consider the trait given any dependency
                bool doit = false;
                if (!it->second.dep_key.empty())
                {
                    // we have a dependency
                    if (m_Traits.count(it->second.dep_key) != 0)
                    {
                        // see if the dep trait matches
                        for (const auto &dv : it->second.dep_values)
                        {
                            if (m_Traits.at(it->second.dep_key).value == dv)
                            {
                                doit = true;
                                break;
                            }
                        }
                    }
                }
                else
                {
                    // no dependencies
                    doit = true;
                }

                if (doit)
                {
                    // mutate with probability
                    if (a_RNG.RandFloat() < it->second.m_MutationProb)
                    {
                        const std::string &ty = it->second.type;
                        if (ty == "int")
                        {
                            IntTraitParameters itp = std::get<IntTraitParameters>(it->second.m_Details);
                            if (itp.min > itp.max)
                            {
                                throw std::invalid_argument("Integer trait minimum exceeds maximum");
                            }
                            int val = std::get<int>(trait_it->second.value);
                            int original = val;
                            if (a_RNG.RandFloat() < itp.mut_replace_prob)
                            {
                                if (itp.min == itp.max)
                                {
                                    continue;
                                }
                                if (original >= itp.min && original <= itp.max)
                                {
                                    val = a_RNG.RandInt(itp.min, itp.max - 1);
                                    if (val >= original)
                                    {
                                        ++val;
                                    }
                                }
                                else
                                {
                                    val = a_RNG.RandInt(itp.min, itp.max);
                                }
                                trait_it->second.value = val;
                                did_mutate = true;
                            }
                            else
                            {
                                if (itp.mut_power <= 0 || itp.min == itp.max)
                                {
                                    continue;
                                }
                                for (int attempt = 0; attempt < 32 && val == original; ++attempt)
                                {
                                    val = original + a_RNG.RandInt(-itp.mut_power, itp.mut_power);
                                    Clamp(val, itp.min, itp.max);
                                }
                                if (val == original)
                                {
                                    val = (original > itp.min) ? original - 1 : original + 1;
                                    Clamp(val, itp.min, itp.max);
                                }
                                trait_it->second.value = val;
                                did_mutate = true;
                            }
                        }
                        else if (ty == "float")
                        {
                            FloatTraitParameters itp = std::get<FloatTraitParameters>(it->second.m_Details);
                            if (itp.min > itp.max)
                            {
                                throw std::invalid_argument("Floating-point trait minimum exceeds maximum");
                            }
                            double val = std::get<double>(trait_it->second.value);
                            double original = val;
                            if (a_RNG.RandFloat() < itp.mut_replace_prob)
                            {
                                if (itp.min == itp.max)
                                {
                                    continue;
                                }
                                val = a_RNG.RandFloat();
                                Scale(val, 0.0, 1.0, itp.min, itp.max);
                                if (val == original)
                                {
                                    val = std::nextafter(original,
                                                         original < itp.max ? itp.max : itp.min);
                                }
                                trait_it->second.value = val;
                                did_mutate = true;
                            }
                            else
                            {
                                if (itp.mut_power <= 0.0 || itp.min == itp.max)
                                {
                                    continue;
                                }
                                for (int attempt = 0; attempt < 32 && val == original; ++attempt)
                                {
                                    val = original + a_RNG.RandFloatSigned() * itp.mut_power;
                                    Clamp(val, itp.min, itp.max);
                                }
                                if (val == original)
                                {
                                    val = std::nextafter(original,
                                                         original < itp.max ? itp.max : itp.min);
                                }
                                trait_it->second.value = val;
                                did_mutate = true;
                            }
                        }
                        else if (ty == "str")
                        {
                            StringTraitParameters itp = std::get<StringTraitParameters>(it->second.m_Details);
                            const std::string original = std::get<std::string>(trait_it->second.value);
                            std::vector<std::string> alternatives;
                            std::vector<double> probs;
                            for (std::size_t i = 0; i < itp.set.size(); ++i)
                            {
                                if (itp.set[i] != original)
                                {
                                    alternatives.push_back(itp.set[i]);
                                    probs.push_back(i < itp.probs.size() ? itp.probs[i] : 0.0);
                                }
                            }
                            if (alternatives.empty())
                            {
                                continue;
                            }
                            trait_it->second.value = alternatives[
                                static_cast<std::size_t>(a_RNG.Roulette(probs))];
                            did_mutate = true;
                        }
                        else if (ty == "intset")
                        {
                            IntSetTraitParameters itp = std::get<IntSetTraitParameters>(it->second.m_Details);
                            const intsetelement original =
                                std::get<intsetelement>(trait_it->second.value);
                            std::vector<intsetelement> alternatives;
                            std::vector<double> probs;
                            for (std::size_t i = 0; i < itp.set.size(); ++i)
                            {
                                if (itp.set[i].value != original.value)
                                {
                                    alternatives.push_back(itp.set[i]);
                                    probs.push_back(i < itp.probs.size() ? itp.probs[i] : 0.0);
                                }
                            }
                            if (alternatives.empty())
                            {
                                continue;
                            }
                            trait_it->second.value = alternatives[
                                static_cast<std::size_t>(a_RNG.Roulette(probs))];
                            did_mutate = true;
                        }
                        else if (ty == "floatset")
                        {
                            FloatSetTraitParameters itp = std::get<FloatSetTraitParameters>(it->second.m_Details);
                            const floatsetelement original =
                                std::get<floatsetelement>(trait_it->second.value);
                            std::vector<floatsetelement> alternatives;
                            std::vector<double> probs;
                            for (std::size_t i = 0; i < itp.set.size(); ++i)
                            {
                                if (itp.set[i].value != original.value)
                                {
                                    alternatives.push_back(itp.set[i]);
                                    probs.push_back(i < itp.probs.size() ? itp.probs[i] : 0.0);
                                }
                            }
                            if (alternatives.empty())
                            {
                                continue;
                            }
                            trait_it->second.value = alternatives[
                                static_cast<std::size_t>(a_RNG.Roulette(probs))];
                            did_mutate = true;
                        }
                    }
                }
            }
            return did_mutate;
        }

        // Compute distance of each matching trait
        // Retain the historical non-const member for source and binary
        // compatibility while also allowing distance queries on const genes.
        std::map<std::string, double> GetTraitDistances(
            const std::map<std::string, Trait> &other)
        {
            return static_cast<const Gene&>(*this)
                .GetTraitDistances(other);
        }

        std::map<std::string, double> GetTraitDistances(
            const std::map<std::string, Trait> &other) const
        {
            std::map<std::string, double> dist;
            for (auto it = other.begin(); it != other.end(); ++it)
            {
                const auto mine_it = m_Traits.find(it->first);
                if (mine_it == m_Traits.end())
                {
                    continue;
                }
                TraitType mine = mine_it->second.value;
                TraitType yours = it->second.value;

                if (mine.index() != yours.index())
                {
                    throw std::runtime_error("Types of traits don't match in distance measure");
                }

                // also check if we skip due to dependencies...
                bool doit = false;
                if (!it->second.dep_key.empty())
                {
                    // check the parent's trait
                    const auto mine_dep = m_Traits.find(it->second.dep_key);
                    const auto other_dep = other.find(it->second.dep_key);
                    if (mine_dep != m_Traits.end() && other_dep != other.end())
                    {
                        for (const auto &dv : it->second.dep_values)
                        {
                            if ((mine_dep->second.value == dv) &&
                                (other_dep->second.value == dv))
                            {
                                doit = true;
                                break;
                            }
                        }
                    }
                }
                else
                {
                    doit = true;
                }

                if (doit)
                {
                    if (std::holds_alternative<int>(mine))
                    {
                        dist[it->first] = std::abs(std::get<int>(mine) - std::get<int>(yours));
                    }
                    else if (std::holds_alternative<double>(mine))
                    {
                        dist[it->first] = std::abs(std::get<double>(mine) - std::get<double>(yours));
                    }
                    else if (std::holds_alternative<std::string>(mine))
                    {
                        dist[it->first] = 
                            (std::get<std::string>(mine) == std::get<std::string>(yours)) ? 0.0 : 1.0;
                    }
                    else if (std::holds_alternative<intsetelement>(mine))
                    {
                        dist[it->first] = std::abs(std::get<intsetelement>(mine).value 
                                                 - std::get<intsetelement>(yours).value);
                    }
                    else if (std::holds_alternative<floatsetelement>(mine))
                    {
                        dist[it->first] = std::abs(std::get<floatsetelement>(mine).value 
                                                 - std::get<floatsetelement>(yours).value);
                    }
                }
            }
            return dist;
        }
    };

    //////////////////////////////////
    // This class defines a link gene
    //////////////////////////////////
    class LinkGene : public Gene
    {
    public:
        int m_FromNeuronID, m_ToNeuronID;
        int m_InnovationID;
        double m_Weight;
        bool m_IsRecurrent;
        // Spiking synapse parameters. They are inert during the historical
        // rate-network activation paths.
        double m_SynapticDelay;
        double m_SynapticTimeConstant;
        bool m_STDPEnabled;
        double m_STDPPlus;
        double m_STDPMinus;
        double m_STDPTauPlus;
        double m_STDPTauMinus;
        double m_STDPMinWeight;
        double m_STDPMaxWeight;

        LinkGene()
        {
            m_FromNeuronID = 0;
            m_ToNeuronID = 0;
            m_InnovationID = 0;
            m_Weight = 0.0;
            m_IsRecurrent = false;
            m_SynapticDelay = 0.0;
            m_SynapticTimeConstant = 0.005;
            m_STDPEnabled = false;
            m_STDPPlus = 0.01;
            m_STDPMinus = 0.012;
            m_STDPTauPlus = 0.02;
            m_STDPTauMinus = 0.02;
            m_STDPMinWeight = -8.0;
            m_STDPMaxWeight = 8.0;
        }

        LinkGene(int a_InID, int a_OutID, int a_InnovID, double a_Wgt, bool a_Recurrent=false)
        {
            m_FromNeuronID = a_InID;
            m_ToNeuronID = a_OutID;
            m_InnovationID = a_InnovID;
            m_Weight = a_Wgt;
            m_IsRecurrent = a_Recurrent;
            m_SynapticDelay = 0.0;
            m_SynapticTimeConstant = 0.005;
            m_STDPEnabled = false;
            m_STDPPlus = 0.01;
            m_STDPMinus = 0.012;
            m_STDPTauPlus = 0.02;
            m_STDPTauMinus = 0.02;
            m_STDPMinWeight = -8.0;
            m_STDPMaxWeight = 8.0;
        }

        LinkGene& operator=(const LinkGene&) = default;

        double GetWeight() const { return m_Weight; }
        void SetWeight(double w)  { m_Weight = w; }

        int FromNeuronID() const    { return m_FromNeuronID; }
        int ToNeuronID() const      { return m_ToNeuronID; }
        int InnovationID() const    { return m_InnovationID; }
        bool IsRecurrent() const    { return m_IsRecurrent; }
        bool IsLoopedRecurrent() const { return (m_FromNeuronID == m_ToNeuronID); }

        // Compare by innovation ID
        friend bool operator<(const LinkGene &lhs, const LinkGene &rhs)
        {
            return (lhs.m_InnovationID < rhs.m_InnovationID);
        }
        friend bool operator>(const LinkGene &lhs, const LinkGene &rhs)
        {
            return (lhs.m_InnovationID > rhs.m_InnovationID);
        }
        friend bool operator!=(const LinkGene &lhs, const LinkGene &rhs)
        {
            return (lhs.m_InnovationID != rhs.m_InnovationID);
        }
        friend bool operator==(const LinkGene &lhs, const LinkGene &rhs)
        {
            return (lhs.m_FromNeuronID == rhs.m_FromNeuronID &&
                    lhs.m_ToNeuronID == rhs.m_ToNeuronID &&
                    lhs.m_Weight == rhs.m_Weight &&
                    lhs.m_IsRecurrent == rhs.m_IsRecurrent &&
                    lhs.m_SynapticDelay == rhs.m_SynapticDelay &&
                    lhs.m_SynapticTimeConstant ==
                        rhs.m_SynapticTimeConstant &&
                    lhs.m_STDPEnabled == rhs.m_STDPEnabled &&
                    lhs.m_STDPPlus == rhs.m_STDPPlus &&
                    lhs.m_STDPMinus == rhs.m_STDPMinus &&
                    lhs.m_STDPTauPlus == rhs.m_STDPTauPlus &&
                    lhs.m_STDPTauMinus == rhs.m_STDPTauMinus &&
                    lhs.m_STDPMinWeight == rhs.m_STDPMinWeight &&
                    lhs.m_STDPMaxWeight == rhs.m_STDPMaxWeight);
        }
    };

    //////////////////////////////////
    // This class defines a neuron gene
    //////////////////////////////////
    class NeuronGene : public Gene
    {
    public:
        int m_ID;
        NeuronType m_Type;
        int x, y;           // for display
        double m_SplitY;    // for structural order
        double m_A, m_B;
        double m_TimeConstant;
        double m_Bias;
        ActivationFunction m_ActFunction;
        // Parameters shared by the spiking activation modes. LIF uses the
        // threshold/reset/rest/refractory/resistance values. Adaptive LIF
        // additionally uses the adaptation values. Izhikevich uses its
        // canonical a/b/c/d parameterization.
        double m_SpikeThreshold;
        double m_ResetPotential;
        double m_RestingPotential;
        double m_RefractoryPeriod;
        double m_MembraneResistance;
        double m_AdaptationTimeConstant;
        double m_AdaptationIncrement;
        double m_RateTimeConstant;
        double m_IzhikevichA;
        double m_IzhikevichB;
        double m_IzhikevichC;
        double m_IzhikevichD;

        NeuronGene()
        {
            m_ID = 0;
            m_Type = NONE;
            x = 0;
            y = 0;
            m_SplitY = 0.0;
            m_A = 0.0;
            m_B = 0.0;
            m_TimeConstant = 0.0;
            m_Bias = 0.0;
            m_ActFunction = UNSIGNED_SIGMOID;
            InitSpikingDefaults();
        }

        NeuronGene(NeuronType a_type, int a_id, double a_splity)
        {
            m_ID = a_id;
            m_Type = a_type;
            m_SplitY = a_splity;
            x = 0;
            y = 0;
            m_A = 0.0;
            m_B = 0.0;
            m_TimeConstant = 0.0;
            m_Bias = 0.0;
            m_ActFunction = UNSIGNED_SIGMOID;
            InitSpikingDefaults();
        }

        friend bool operator==(const NeuronGene &lhs, const NeuronGene &rhs)
        {
            return (lhs.m_ID == rhs.m_ID && 
                    lhs.m_Type == rhs.m_Type &&
                    lhs.x == rhs.x &&
                    lhs.y == rhs.y &&
                    lhs.m_SplitY == rhs.m_SplitY &&
                    lhs.m_A == rhs.m_A &&
                    lhs.m_B == rhs.m_B &&
                    lhs.m_TimeConstant == rhs.m_TimeConstant &&
                    lhs.m_Bias == rhs.m_Bias &&
                    lhs.m_ActFunction == rhs.m_ActFunction &&
                    lhs.m_SpikeThreshold == rhs.m_SpikeThreshold &&
                    lhs.m_ResetPotential == rhs.m_ResetPotential &&
                    lhs.m_RestingPotential == rhs.m_RestingPotential &&
                    lhs.m_RefractoryPeriod == rhs.m_RefractoryPeriod &&
                    lhs.m_MembraneResistance ==
                        rhs.m_MembraneResistance &&
                    lhs.m_AdaptationTimeConstant ==
                        rhs.m_AdaptationTimeConstant &&
                    lhs.m_AdaptationIncrement ==
                        rhs.m_AdaptationIncrement &&
                    lhs.m_RateTimeConstant ==
                        rhs.m_RateTimeConstant &&
                    lhs.m_IzhikevichA == rhs.m_IzhikevichA &&
                    lhs.m_IzhikevichB == rhs.m_IzhikevichB &&
                    lhs.m_IzhikevichC == rhs.m_IzhikevichC &&
                    lhs.m_IzhikevichD == rhs.m_IzhikevichD);
        }

        NeuronGene& operator=(const NeuronGene&) = default;

        int ID() const        { return m_ID; }
        NeuronType Type() const { return m_Type; }
        double SplitY() const { return m_SplitY; }

        void Init(double a_A, double a_B, double a_TimeConstant, double a_Bias, ActivationFunction a_ActFunc)
        {
            m_A = a_A;
            m_B = a_B;
            m_TimeConstant = a_TimeConstant;
            m_Bias = a_Bias;
            m_ActFunction = a_ActFunc;
        }

        void InitSpikingDefaults()
        {
            m_SpikeThreshold = 1.0;
            m_ResetPotential = 0.0;
            m_RestingPotential = 0.0;
            m_RefractoryPeriod = 0.002;
            m_MembraneResistance = 1.0;
            m_AdaptationTimeConstant = 0.1;
            m_AdaptationIncrement = 0.1;
            m_RateTimeConstant = 0.05;
            m_IzhikevichA = 0.02;
            m_IzhikevichB = 0.2;
            m_IzhikevichC = -65.0;
            m_IzhikevichD = 8.0;
        }
    };

} // namespace NEAT

#endif
