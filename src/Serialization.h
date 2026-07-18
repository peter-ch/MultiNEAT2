#ifndef MULTINEAT_SERIALIZATION_H
#define MULTINEAT_SERIALIZATION_H

#include <iomanip>
#include <istream>
#include <limits>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "Traits.h"

namespace NEAT
{
namespace Serialization
{

inline void RequireStream(const std::ios& stream, const char* context)
{
    if (!stream)
        throw std::runtime_error(std::string(context) + ": malformed serialized data.");
}

inline void WriteTraitValue(std::ostream& output, const TraitType& value)
{
    output << value.index() << ' ';
    std::visit(
        [&output](const auto& item)
        {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, std::string>)
                output << std::quoted(item);
            else if constexpr (std::is_same_v<T, intsetelement> ||
                               std::is_same_v<T, floatsetelement>)
                output << item.value;
            else
                output << item;
        },
        value);
}

inline TraitType ReadTraitValue(std::istream& input)
{
    std::size_t type = 0;
    input >> type;
    switch (type)
    {
    case 0:
    {
        int value = 0;
        input >> value;
        RequireStream(input, "trait value");
        return value;
    }
    case 1:
    {
        double value = 0.0;
        input >> value;
        RequireStream(input, "trait value");
        return value;
    }
    case 2:
    {
        std::string value;
        input >> std::quoted(value);
        RequireStream(input, "trait value");
        return value;
    }
    case 3:
    {
        intsetelement value;
        input >> value.value;
        RequireStream(input, "trait value");
        return value;
    }
    case 4:
    {
        floatsetelement value;
        input >> value.value;
        RequireStream(input, "trait value");
        return value;
    }
    default:
        throw std::runtime_error("trait value: unknown variant index.");
    }
}

inline void WriteTraitValues(std::ostream& output,
                             const std::vector<TraitType>& values)
{
    output << values.size();
    for (const auto& value : values)
    {
        output << ' ';
        WriteTraitValue(output, value);
    }
}

inline std::vector<TraitType> ReadTraitValues(std::istream& input)
{
    std::size_t count = 0;
    input >> count;
    RequireStream(input, "trait value list");
    std::vector<TraitType> values;
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        values.push_back(ReadTraitValue(input));
    return values;
}

inline void WriteTraits(std::ostream& output,
                        const std::string& marker,
                        const std::map<std::string, Trait>& traits)
{
    output << marker << ' ' << traits.size() << '\n';
    for (const auto& [key, trait] : traits)
    {
        output << "Trait " << std::quoted(key) << ' ';
        WriteTraitValue(output, trait.value);
        output << ' ' << std::quoted(trait.dep_key) << ' ';
        WriteTraitValues(output, trait.dep_values);
        output << '\n';
    }
}

inline std::map<std::string, Trait> ReadTraits(std::istream& input)
{
    std::size_t count = 0;
    input >> count;
    RequireStream(input, "trait map");
    std::map<std::string, Trait> traits;
    for (std::size_t i = 0; i < count; ++i)
    {
        std::string token;
        std::string key;
        Trait trait;
        input >> token >> std::quoted(key);
        if (token != "Trait")
            throw std::runtime_error("trait map: missing Trait marker.");
        trait.value = ReadTraitValue(input);
        input >> std::quoted(trait.dep_key);
        trait.dep_values = ReadTraitValues(input);
        traits.emplace(std::move(key), std::move(trait));
    }
    return traits;
}

inline void WriteTraitParameters(
    std::ostream& output,
    const std::string& marker,
    const std::map<std::string, TraitParameters>& schemas)
{
    output << marker << ' ' << schemas.size() << '\n';
    for (const auto& [key, schema] : schemas)
    {
        output << "TraitSchema " << std::quoted(key) << ' '
               << schema.m_ImportanceCoeff << ' ' << schema.m_MutationProb << ' '
               << std::quoted(schema.type) << ' ' << std::quoted(schema.dep_key)
               << ' ';
        WriteTraitValues(output, schema.dep_values);
        output << '\n';

        output << "TraitDetails ";
        if (schema.type == "int")
        {
            const auto& details = std::get<IntTraitParameters>(schema.m_Details);
            output << details.min << ' ' << details.max << ' '
                   << details.mut_power << ' ' << details.mut_replace_prob;
        }
        else if (schema.type == "float")
        {
            const auto& details = std::get<FloatTraitParameters>(schema.m_Details);
            output << details.min << ' ' << details.max << ' '
                   << details.mut_power << ' ' << details.mut_replace_prob;
        }
        else if (schema.type == "str")
        {
            const auto& details = std::get<StringTraitParameters>(schema.m_Details);
            output << details.set.size();
            for (const auto& item : details.set)
                output << ' ' << std::quoted(item);
            output << ' ' << details.probs.size();
            for (double probability : details.probs)
                output << ' ' << probability;
        }
        else if (schema.type == "intset")
        {
            const auto& details = std::get<IntSetTraitParameters>(schema.m_Details);
            output << details.set.size();
            for (const auto& item : details.set)
                output << ' ' << item.value;
            output << ' ' << details.probs.size();
            for (double probability : details.probs)
                output << ' ' << probability;
        }
        else if (schema.type == "floatset")
        {
            const auto& details = std::get<FloatSetTraitParameters>(schema.m_Details);
            output << details.set.size();
            for (const auto& item : details.set)
                output << ' ' << item.value;
            output << ' ' << details.probs.size();
            for (double probability : details.probs)
                output << ' ' << probability;
        }
        else
        {
            throw std::runtime_error("trait schema: unsupported type '" +
                                     schema.type + "'.");
        }
        output << '\n';
    }
}

template <typename T>
inline std::vector<T> ReadNumericVector(std::istream& input)
{
    std::size_t count = 0;
    input >> count;
    RequireStream(input, "numeric vector");
    std::vector<T> values(count);
    for (auto& value : values)
        input >> value;
    RequireStream(input, "numeric vector");
    return values;
}

inline std::map<std::string, TraitParameters>
ReadTraitParameters(std::istream& input)
{
    std::size_t count = 0;
    input >> count;
    RequireStream(input, "trait schema map");
    std::map<std::string, TraitParameters> schemas;
    for (std::size_t i = 0; i < count; ++i)
    {
        std::string token;
        std::string key;
        TraitParameters schema;
        input >> token >> std::quoted(key) >> schema.m_ImportanceCoeff
              >> schema.m_MutationProb >> std::quoted(schema.type)
              >> std::quoted(schema.dep_key);
        if (token != "TraitSchema")
            throw std::runtime_error("trait schema map: missing TraitSchema marker.");
        schema.dep_values = ReadTraitValues(input);

        input >> token;
        if (token != "TraitDetails")
            throw std::runtime_error("trait schema map: missing TraitDetails marker.");
        if (schema.type == "int")
        {
            IntTraitParameters details;
            input >> details.min >> details.max >> details.mut_power
                  >> details.mut_replace_prob;
            schema.m_Details = details;
        }
        else if (schema.type == "float")
        {
            FloatTraitParameters details;
            input >> details.min >> details.max >> details.mut_power
                  >> details.mut_replace_prob;
            schema.m_Details = details;
        }
        else if (schema.type == "str")
        {
            StringTraitParameters details;
            std::size_t set_size = 0;
            input >> set_size;
            details.set.resize(set_size);
            for (auto& item : details.set)
                input >> std::quoted(item);
            details.probs = ReadNumericVector<double>(input);
            schema.m_Details = details;
        }
        else if (schema.type == "intset")
        {
            IntSetTraitParameters details;
            const auto values = ReadNumericVector<int>(input);
            details.set.reserve(values.size());
            for (int value : values)
            {
                intsetelement item;
                item.value = value;
                details.set.push_back(item);
            }
            details.probs = ReadNumericVector<double>(input);
            schema.m_Details = details;
        }
        else if (schema.type == "floatset")
        {
            FloatSetTraitParameters details;
            const auto values = ReadNumericVector<double>(input);
            details.set.reserve(values.size());
            for (double value : values)
            {
                floatsetelement item;
                item.value = value;
                details.set.push_back(item);
            }
            details.probs = ReadNumericVector<double>(input);
            schema.m_Details = details;
        }
        else
        {
            throw std::runtime_error("trait schema: unsupported type '" +
                                     schema.type + "'.");
        }
        RequireStream(input, "trait schema");
        schemas.emplace(std::move(key), std::move(schema));
    }
    return schemas;
}

inline void UseRoundTripPrecision(std::ostream& output)
{
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
}

} // namespace Serialization
} // namespace NEAT

#endif
