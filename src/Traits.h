#ifndef MULTINEAT_TRAITS_H
#define MULTINEAT_TRAITS_H

#include <string>
#include <vector>
#include <variant> 
#include <cmath>

namespace NEAT
{
    // Represents an element in an integer set trait
    class intsetelement
    {
    public:
        int value;

        bool operator==(const intsetelement& rhs) const
        {
            return rhs.value == value;
        }

        intsetelement &operator=(const intsetelement &a_g)
        {
            if (this != &a_g)
            {
                value = a_g.value;
            }
            return *this;
        }
    };

    // Represents an element in a float set trait
    class floatsetelement
    {
    public:
        double value;

        bool operator==(const floatsetelement& rhs) const
        {
            return rhs.value == value;
        }

        floatsetelement &operator=(const floatsetelement &a_g)
        {
            if (this != &a_g)
            {
                value = a_g.value;
            }
            return *this;
        }
    };

    // Using std::variant to store possible trait types
    typedef std::variant<int, double, std::string, intsetelement, floatsetelement> TraitType;

    class IntTraitParameters
    {
    public:
        int min, max;
        int mut_power;        // magnitude of max change up/down
        double mut_replace_prob; // probability to replace when mutating

        IntTraitParameters()
        {
            min = 0; 
            max = 0;
            mut_power = 0;
            mut_replace_prob = 0.0;
        }

        IntTraitParameters &operator=(const IntTraitParameters &a_g)
        {
            if (this != &a_g)
            {
                min = a_g.min;
                max = a_g.max;
                mut_power = a_g.mut_power;
                mut_replace_prob = a_g.mut_replace_prob;
            }
            return *this;
        }
    };

    class FloatTraitParameters
    {
    public:
        double min, max;
        double mut_power;       // magnitude of max change up/down
        double mut_replace_prob; // probability to replace when mutating

        FloatTraitParameters()
        {
            min = 0; 
            max = 0;
            mut_power = 0.0;
            mut_replace_prob = 0.0;
        }

        FloatTraitParameters &operator=(const FloatTraitParameters &a_g)
        {
            if (this != &a_g)
            {
                min = a_g.min;
                max = a_g.max;
                mut_power = a_g.mut_power;
                mut_replace_prob = a_g.mut_replace_prob;
            }
            return *this;
        }
    };

    class StringTraitParameters
    {
    public:
        std::vector<std::string> set;  // the set of possible strings
        std::vector<double> probs;     // their respective probabilities for appearance

        StringTraitParameters &operator=(const StringTraitParameters &a_g)
        {
            if (this != &a_g)
            {
                set = a_g.set;
                probs = a_g.probs;
            }
            return *this;
        }
    };

    class IntSetTraitParameters
    {
    public:
        std::vector<intsetelement> set; // the set of possible ints
        std::vector<double> probs;      // their respective probabilities for appearance

        IntSetTraitParameters &operator=(const IntSetTraitParameters &a_g)
        {
            if (this != &a_g)
            {
                set = a_g.set;
                probs = a_g.probs;
            }
            return *this;
        }
    };

    class FloatSetTraitParameters
    {
    public:
        std::vector<floatsetelement> set; // the set of possible floats
        std::vector<double> probs;        // their respective probabilities for appearance

        FloatSetTraitParameters &operator=(const FloatSetTraitParameters &a_g)
        {
            if (this != &a_g)
            {
                set = a_g.set;
                probs = a_g.probs;
            }
            return *this;
        }
    };

    // Holds parameters describing how a given trait mutates,
    // which type it is, etc.
    class TraitParameters
    {
    public:
        double m_ImportanceCoeff;
        double m_MutationProb;

        // can be "int", "float", "str", "intset", "floatset"
        std::string type;
        std::variant<
            IntTraitParameters,
            FloatTraitParameters,
            StringTraitParameters,
            IntSetTraitParameters,
            FloatSetTraitParameters
        > m_Details;

        std::string dep_key;            // a dependency trait key
        std::vector<TraitType> dep_values; // allowed values of that dependency trait for this trait to apply

        TraitParameters()
        {
            m_ImportanceCoeff = 0.0;
            m_MutationProb = 0.0;
            type = "int";
            m_Details = IntTraitParameters();
            dep_key = "";
            dep_values.emplace_back(std::string(""));
        }

        TraitParameters &operator=(const TraitParameters &a_g)
        {
            if (this != &a_g)
            {
                m_ImportanceCoeff = a_g.m_ImportanceCoeff;
                m_MutationProb = a_g.m_MutationProb;
                type = a_g.type;
                m_Details = a_g.m_Details;
                dep_key = a_g.dep_key;
                dep_values = a_g.dep_values;
            }
            return *this;
        }
    };

    // Represents an actual trait instance on a LinkGene, NeuronGene, or GenomeGene
    class Trait
    {
    public:
        TraitType value;
        std::string dep_key;  // if non-empty, we only consider this trait if dep_key is matched
        std::vector<TraitType> dep_values;

        Trait()
        {
            value = 0; // default is int=0
            dep_key = "";
            dep_values.emplace_back(0); 
        }

        Trait &operator=(const Trait &a_g)
        {
            if (this != &a_g)
            {
                value = a_g.value;
                dep_values = a_g.dep_values;
                dep_key = a_g.dep_key;
            }
            return *this;
        }
    };

} // namespace NEAT

#endif //MULTINEAT_TRAITS_H
