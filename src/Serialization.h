#ifndef MULTINEAT_SERIALIZATION_H
#define MULTINEAT_SERIALIZATION_H

#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <locale>
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
    // Counts are untrusted: grow element by element and stop on bad input
    // rather than reserving an arbitrary allocation up front.
    std::vector<TraitType> values;
    for (std::size_t i = 0; i < count; ++i)
    {
        values.push_back(ReadTraitValue(input));
        RequireStream(input, "trait value list");
    }
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
        RequireStream(input, "trait map");
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
        const auto details_for = [&key, &schema](const auto* details)
        {
            if (details == nullptr)
            {
                throw std::invalid_argument(
                    "trait schema '" + key + "': type '" + schema.type +
                    "' does not match its details.");
            }
            return details;
        };
        output << "TraitSchema " << std::quoted(key) << ' '
               << schema.m_ImportanceCoeff << ' ' << schema.m_MutationProb << ' '
               << std::quoted(schema.type) << ' ' << std::quoted(schema.dep_key)
               << ' ';
        WriteTraitValues(output, schema.dep_values);
        output << '\n';

        output << "TraitDetails ";
        if (schema.type == "int")
        {
            const auto& details = *details_for(
                std::get_if<IntTraitParameters>(&schema.m_Details));
            output << details.min << ' ' << details.max << ' '
                   << details.mut_power << ' ' << details.mut_replace_prob;
        }
        else if (schema.type == "float")
        {
            const auto& details = *details_for(
                std::get_if<FloatTraitParameters>(&schema.m_Details));
            output << details.min << ' ' << details.max << ' '
                   << details.mut_power << ' ' << details.mut_replace_prob;
        }
        else if (schema.type == "str")
        {
            const auto& details = *details_for(
                std::get_if<StringTraitParameters>(&schema.m_Details));
            output << details.set.size();
            for (const auto& item : details.set)
                output << ' ' << std::quoted(item);
            output << ' ' << details.probs.size();
            for (double probability : details.probs)
                output << ' ' << probability;
        }
        else if (schema.type == "intset")
        {
            const auto& details = *details_for(
                std::get_if<IntSetTraitParameters>(&schema.m_Details));
            output << details.set.size();
            for (const auto& item : details.set)
                output << ' ' << item.value;
            output << ' ' << details.probs.size();
            for (double probability : details.probs)
                output << ' ' << probability;
        }
        else if (schema.type == "floatset")
        {
            const auto& details = *details_for(
                std::get_if<FloatSetTraitParameters>(&schema.m_Details));
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
    std::vector<T> values;
    for (std::size_t i = 0; i < count; ++i)
    {
        T value{};
        input >> value;
        RequireStream(input, "numeric vector");
        values.push_back(value);
    }
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
        RequireStream(input, "trait schema");
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
            RequireStream(input, "string trait set");
            for (std::size_t item_index = 0; item_index < set_size; ++item_index)
            {
                std::string item;
                input >> std::quoted(item);
                RequireStream(input, "string trait set");
                details.set.push_back(std::move(item));
            }
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

// Standard streams write non-finite doubles as "inf"/"nan" but cannot read
// them back. This facet parses each whitespace-delimited floating-point token
// with strtod, which accepts those spellings, and fails on trailing garbage.
class RoundTripNumGet : public std::num_get<char>
{
protected:
    iter_type do_get(
        iter_type in,
        iter_type end,
        std::ios_base& stream,
        std::ios_base::iostate& error,
        double& value) const override
    {
        std::string token;
        while (in != end && !std::isspace(
                                static_cast<unsigned char>(*in),
                                stream.getloc()))
        {
            token.push_back(*in);
            ++in;
        }
        if (in == end)
            error |= std::ios_base::eofbit;
        if (token.empty())
        {
            error |= std::ios_base::failbit;
            return in;
        }
        char* parsed_end = nullptr;
        const double parsed = std::strtod(token.c_str(), &parsed_end);
        if (parsed_end != token.c_str() + token.size())
        {
            error |= std::ios_base::failbit;
            return in;
        }
        value = parsed;
        return in;
    }

    iter_type do_get(
        iter_type in,
        iter_type end,
        std::ios_base& stream,
        std::ios_base::iostate& error,
        float& value) const override
    {
        double parsed = 0.0;
        in = do_get(in, end, stream, error, parsed);
        if (!(error & std::ios_base::failbit))
            value = static_cast<float>(parsed);
        return in;
    }
};

// Makes `input >> double` accept the non-finite values written by an output
// stream using UseRoundTripPrecision.
inline void UseRoundTripParsing(std::istream& input)
{
    input.imbue(std::locale(input.getloc(), new RoundTripNumGet));
}

} // namespace Serialization
} // namespace NEAT

#endif
