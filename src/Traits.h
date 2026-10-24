#ifndef MULTINEAT_TRAITS_H
#define MULTINEAT_TRAITS_H

#include <cstddef>
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
    };

    class StringTraitParameters
    {
    public:
        std::vector<std::string> set;  // the set of possible strings
        std::vector<double> probs;     // their respective probabilities for appearance
    };

    class IntSetTraitParameters
    {
    public:
        std::vector<intsetelement> set; // the set of possible ints
        std::vector<double> probs;      // their respective probabilities for appearance
    };

    class FloatSetTraitParameters
    {
    public:
        std::vector<floatsetelement> set; // the set of possible floats
        std::vector<double> probs;        // their respective probabilities for appearance
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

    // Index of the TraitType alternative that a trait of the given schema
    // type string ("int", "float", "str", "intset", "floatset") stores, or
    // std::variant_npos for an unsupported type string. The same index is
    // the TraitParameters::m_Details alternative for that type.
    inline std::size_t TraitValueIndexForSchemaType(const std::string& type)
    {
        if (type == "int")
            return 0;
        if (type == "float")
            return 1;
        if (type == "str")
            return 2;
        if (type == "intset")
            return 3;
        if (type == "floatset")
            return 4;
        return std::variant_npos;
    }

    // Converts a numeric trait value to the alternative at target_index when
    // the conversion is exact. Widening conversions (int -> double, int or
    // double -> set element of the same kind) are always applied. Narrowing
    // conversions (an integral double -> int or intsetelement) are applied
    // only when allow_narrowing is true, i.e. when the caller knows the
    // schema type. Returns true when value holds target_index afterwards.
    inline bool CoerceTraitValue(
        TraitType& value,
        std::size_t target_index,
        bool allow_narrowing)
    {
        if (value.index() == target_index)
            return true;
        const auto exact_int = [](double number, int& result)
        {
            if (!std::isfinite(number) || std::floor(number) != number ||
                number < -2147483648.0 || number > 2147483647.0)
                return false;
            result = static_cast<int>(number);
            return true;
        };
        switch (target_index)
        {
        case 0: // int
            if (allow_narrowing && std::holds_alternative<double>(value))
            {
                int converted = 0;
                if (!exact_int(std::get<double>(value), converted))
                    return false;
                value = converted;
                return true;
            }
            return false;
        case 1: // double
            if (std::holds_alternative<int>(value))
            {
                value = static_cast<double>(std::get<int>(value));
                return true;
            }
            return false;
        case 3: // intsetelement
            if (std::holds_alternative<int>(value))
            {
                intsetelement element;
                element.value = std::get<int>(value);
                value = element;
                return true;
            }
            if (allow_narrowing && std::holds_alternative<double>(value))
            {
                intsetelement element;
                if (!exact_int(std::get<double>(value), element.value))
                    return false;
                value = element;
                return true;
            }
            return false;
        case 4: // floatsetelement
            if (std::holds_alternative<double>(value) ||
                std::holds_alternative<int>(value))
            {
                floatsetelement element;
                element.value = std::holds_alternative<double>(value)
                    ? std::get<double>(value)
                    : static_cast<double>(std::get<int>(value));
                value = element;
                return true;
            }
            return false;
        default:
            return false;
        }
    }

} // namespace NEAT

#endif //MULTINEAT_TRAITS_H
