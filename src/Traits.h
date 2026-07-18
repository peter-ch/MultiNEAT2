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
        int value = 0;

        bool operator==(const intsetelement& rhs) const
        {
            return rhs.value == value;
        }

        intsetelement& operator=(const intsetelement&) = default;
    };

    // Represents an element in a float set trait
    class floatsetelement
    {
    public:
        double value = 0.0;

        bool operator==(const floatsetelement& rhs) const
        {
            return rhs.value == value;
        }

        floatsetelement& operator=(const floatsetelement&) = default;
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

        IntTraitParameters& operator=(const IntTraitParameters&) = default;
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

        FloatTraitParameters& operator=(const FloatTraitParameters&) = default;
    };

    class StringTraitParameters
    {
    public:
        std::vector<std::string> set;  // the set of possible strings
        std::vector<double> probs;     // their respective probabilities for appearance

        StringTraitParameters& operator=(const StringTraitParameters&) = default;
    };

    class IntSetTraitParameters
    {
    public:
        std::vector<intsetelement> set; // the set of possible ints
        std::vector<double> probs;      // their respective probabilities for appearance

        IntSetTraitParameters& operator=(const IntSetTraitParameters&) = default;
    };

    class FloatSetTraitParameters
    {
    public:
        std::vector<floatsetelement> set; // the set of possible floats
        std::vector<double> probs;        // their respective probabilities for appearance

        FloatSetTraitParameters& operator=(
            const FloatSetTraitParameters&) = default;
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
          : m_ImportanceCoeff(0.0), m_MutationProb(0.0), type("int"),
            m_Details(IntTraitParameters()), dep_key(""),
            dep_values{std::string("")}
        {
        }

        TraitParameters& operator=(const TraitParameters&) = default;
    };

    // Represents an actual trait instance on a LinkGene, NeuronGene, or GenomeGene
    class Trait
    {
    public:
        TraitType value;
        std::string dep_key;  // if non-empty, we only consider this trait if dep_key is matched
        std::vector<TraitType> dep_values;

        Trait() : value(0), dep_key(""), dep_values{0}
        {
        }

        Trait& operator=(const Trait&) = default;

        bool operator==(const Trait& rhs) const
        {
            return value == rhs.value &&
                   dep_key == rhs.dep_key &&
                   dep_values == rhs.dep_values;
        }

        bool operator!=(const Trait& rhs) const
        {
            return !(*this == rhs);
        }
    };

} // namespace NEAT

#endif //MULTINEAT_TRAITS_H
