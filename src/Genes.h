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
        SOFTPLUS
    };

    //////////////////////////////////
    // Base Gene class
    //////////////////////////////////
    class Gene
    {
    public:
        // Arbitrary traits
        std::map<std::string, Trait> m_Traits;

        Gene &operator=(const Gene &a_g)
        {
            if (this != &a_g)
            {
                m_Traits = a_g.m_Traits;
            }
            return *this;
        }

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
                    t = a_RNG.RandInt(itp.min, itp.max);
                }
                else if (it->second.type == "float")
                {
                    FloatTraitParameters itp = std::get<FloatTraitParameters>(it->second.m_Details);
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
                    // fallback
                    t = 0;
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
                TraitType mine = m_Traits[it->first].value;
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
                        m_Traits[it->first].value = (m1 + m2) / 2;
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
                // check if we should consider the trait given any dependency
                bool doit = false;
                if (!it->second.dep_key.empty())
                {
                    // we have a dependency
                    if (m_Traits.count(it->second.dep_key) != 0)
                    {
                        // see if the dep trait matches
                        for (auto &dv : it->second.dep_values)
                        {
                            if (m_Traits[it->second.dep_key].value == dv)
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
                            int val = std::get<int>(m_Traits[it->first].value);
                            int original = val;
                            if (a_RNG.RandFloat() < itp.mut_replace_prob)
                            {
                                // replace
                                while (val == original)
                                {
                                    val = a_RNG.RandInt(itp.min, itp.max);
                                }
                                m_Traits[it->first].value = val;
                                did_mutate = true;
                            }
                            else
                            {
                                // modify
                                while (val == original)
                                {
                                    val += a_RNG.RandInt(-itp.mut_power, itp.mut_power);
                                    Clamp(val, itp.min, itp.max);
                                }
                                m_Traits[it->first].value = val;
                                did_mutate = true;
                            }
                        }
                        else if (ty == "float")
                        {
                            FloatTraitParameters itp = std::get<FloatTraitParameters>(it->second.m_Details);
                            double val = std::get<double>(m_Traits[it->first].value);
                            double original = val;
                            if (a_RNG.RandFloat() < itp.mut_replace_prob)
                            {
                                while (val == original)
                                {
                                    val = a_RNG.RandFloat();
                                    Scale(val, 0.0, 1.0, itp.min, itp.max);
                                }
                                m_Traits[it->first].value = val;
                                did_mutate = true;
                            }
                            else
                            {
                                while (val == original)
                                {
                                    val += a_RNG.RandFloatSigned() * itp.mut_power;
                                    Clamp(val, itp.min, itp.max);
                                }
                                m_Traits[it->first].value = val;
                                did_mutate = true;
                            }
                        }
                        else if (ty == "str")
                        {
                            StringTraitParameters itp = std::get<StringTraitParameters>(it->second.m_Details);
                            std::vector<double> probs = itp.probs;
                            probs.resize(itp.set.size());
                            std::string original = std::get<std::string>(m_Traits[it->first].value);
                            int idx = a_RNG.Roulette(probs);
                            while (original == itp.set[idx])
                            {
                                idx = a_RNG.Roulette(probs);
                            }
                            m_Traits[it->first].value = itp.set[idx];
                            did_mutate = true;
                        }
                        else if (ty == "intset")
                        {
                            IntSetTraitParameters itp = std::get<IntSetTraitParameters>(it->second.m_Details);
                            std::vector<double> probs = itp.probs;
                            probs.resize(itp.set.size());
                            intsetelement original = std::get<intsetelement>(m_Traits[it->first].value);
                            int idx = a_RNG.Roulette(probs);
                            while (original.value == itp.set[idx].value)
                            {
                                idx = a_RNG.Roulette(probs);
                            }
                            m_Traits[it->first].value = itp.set[idx];
                            did_mutate = true;
                        }
                        else if (ty == "floatset")
                        {
                            FloatSetTraitParameters itp = std::get<FloatSetTraitParameters>(it->second.m_Details);
                            std::vector<double> probs = itp.probs;
                            probs.resize(itp.set.size());
                            floatsetelement original = std::get<floatsetelement>(m_Traits[it->first].value);
                            int idx = a_RNG.Roulette(probs);
                            while (original.value == itp.set[idx].value)
                            {
                                idx = a_RNG.Roulette(probs);
                            }
                            m_Traits[it->first].value = itp.set[idx];
                            did_mutate = true;
                        }
                    }
                }
            }
            return did_mutate;
        }

        // Compute distance of each matching trait
        std::map<std::string, double> GetTraitDistances(const std::map<std::string, Trait> &other)
        {
            std::map<std::string, double> dist;
            for (auto it = other.begin(); it != other.end(); ++it)
            {
                TraitType mine = m_Traits[it->first].value;
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
                    if (m_Traits.count(it->second.dep_key) != 0)
                    {
                        for (auto &dv : it->second.dep_values)
                        {
                            if ((m_Traits[it->second.dep_key].value == dv) &&
                                (other.at(it->second.dep_key).value == dv))
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

        LinkGene()
        {
            m_FromNeuronID = 0;
            m_ToNeuronID = 0;
            m_InnovationID = 0;
            m_Weight = 0.0;
            m_IsRecurrent = false;
        }

        LinkGene(int a_InID, int a_OutID, int a_InnovID, double a_Wgt, bool a_Recurrent=false)
        {
            m_FromNeuronID = a_InID;
            m_ToNeuronID = a_OutID;
            m_InnovationID = a_InnovID;
            m_Weight = a_Wgt;
            m_IsRecurrent = a_Recurrent;
        }

        LinkGene &operator=(const LinkGene &a_g)
        {
            if (this != &a_g)
            {
                m_FromNeuronID = a_g.m_FromNeuronID;
                m_ToNeuronID = a_g.m_ToNeuronID;
                m_Weight = a_g.m_Weight;
                m_IsRecurrent = a_g.m_IsRecurrent;
                m_InnovationID = a_g.m_InnovationID;
                m_Traits = a_g.m_Traits;
            }
            return *this;
        }

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
                    lhs.m_IsRecurrent == rhs.m_IsRecurrent);
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
                    lhs.m_ActFunction == rhs.m_ActFunction);
        }

        NeuronGene &operator=(const NeuronGene &a_g)
        {
            if (this != &a_g)
            {
                m_ID = a_g.m_ID;
                m_Type = a_g.m_Type;
                m_SplitY = a_g.m_SplitY;
                if ((m_Type != INPUT) && (m_Type != BIAS))
                {
                    x = a_g.x;
                    y = a_g.y;
                    m_A = a_g.m_A;
                    m_B = a_g.m_B;
                    m_TimeConstant = a_g.m_TimeConstant;
                    m_Bias = a_g.m_Bias;
                    m_ActFunction = a_g.m_ActFunction;
                    m_Traits = a_g.m_Traits;
                }
            }
            return *this;
        }

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
    };

} // namespace NEAT

#endif
