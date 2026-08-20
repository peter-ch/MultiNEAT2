#include "Reliability.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace NEAT
{
namespace
{

std::uint64_t Mix(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t SampleSeed(const std::vector<EvaluationSample>& samples)
{
    std::uint64_t seed = 0x6a09e667f3bcc909ULL;
    for (const auto& sample : samples)
    {
        std::uint64_t score_bits = 0;
        static_assert(sizeof(score_bits) == sizeof(sample.score));
        std::memcpy(&score_bits, &sample.score, sizeof(score_bits));
        seed = Mix(seed ^ Mix(sample.case_id) ^ Mix(score_bits));
    }
    return seed;
}

} // namespace

double WilsonLowerBound(
    std::size_t successes,
    std::size_t samples,
    double z)
{
    if (samples == 0)
        return 0.0;
    if (successes > samples || !std::isfinite(z) || z <= 0.0)
        throw std::invalid_argument("Invalid Wilson interval inputs");
    const double n = static_cast<double>(samples);
    const double p = static_cast<double>(successes) / n;
    const double z_squared = z * z;
    const double denominator = 1.0 + z_squared / n;
    const double center = p + z_squared / (2.0 * n);
    const double radius = z * std::sqrt(
        (p * (1.0 - p) + z_squared / (4.0 * n)) / n);
    return std::clamp((center - radius) / denominator, 0.0, 1.0);
}

EvaluationSummary SummarizeEvaluations(
    const std::vector<EvaluationSample>& samples,
    double confidence,
    unsigned int bootstrap_samples)
{
    EvaluationSummary result;
    if (samples.empty())
        return result;
    if (!std::isfinite(confidence) || confidence <= 0.0 || confidence >= 1.0)
        throw std::invalid_argument("Evaluation confidence must be between zero and one");
    if (bootstrap_samples == 0)
        throw std::invalid_argument("Bootstrap sample count must be greater than zero");

    std::vector<double> scores;
    scores.reserve(samples.size());
    long double total = 0.0L;
    for (const auto& sample : samples)
    {
        if (!std::isfinite(sample.score))
            throw std::invalid_argument("Evaluation scores must be finite");
        for (double value : sample.behavior_descriptor)
        {
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "Behavior descriptors must contain finite values");
        }
        scores.push_back(sample.score);
        total += static_cast<long double>(sample.score);
        if (sample.has_success)
        {
            ++result.success_sample_count;
            if (sample.success)
                ++result.success_count;
        }
    }
    result.sample_count = scores.size();
    result.mean = static_cast<double>(
        total / static_cast<long double>(scores.size()));

    if (scores.size() > 1)
    {
        long double squared = 0.0L;
        for (double score : scores)
        {
            const long double delta =
                static_cast<long double>(score) - result.mean;
            squared += delta * delta;
        }
        result.standard_deviation = std::sqrt(static_cast<double>(
            squared / static_cast<long double>(scores.size() - 1U)));
        result.standard_error = result.standard_deviation /
            std::sqrt(static_cast<double>(scores.size()));
    }

    std::sort(scores.begin(), scores.end());
    const std::size_t tail_count = std::max<std::size_t>(
        1U, static_cast<std::size_t>(
                std::ceil(0.2 * static_cast<double>(scores.size()))));
    result.lower_tail_mean = std::accumulate(
        scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(tail_count),
        0.0) / static_cast<double>(tail_count);

    if (scores.size() == 1)
    {
        result.lower_confidence_bound = scores.front();
    }
    else
    {
        std::vector<double> bootstrap_means;
        bootstrap_means.reserve(bootstrap_samples);
        std::uint64_t state = SampleSeed(samples);
        for (unsigned int replicate = 0; replicate < bootstrap_samples; ++replicate)
        {
            long double bootstrap_total = 0.0L;
            for (std::size_t draw = 0; draw < scores.size(); ++draw)
            {
                state = Mix(state);
                const std::size_t index = static_cast<std::size_t>(
                    state % static_cast<std::uint64_t>(scores.size()));
                bootstrap_total += scores[index];
            }
            bootstrap_means.push_back(static_cast<double>(
                bootstrap_total / static_cast<long double>(scores.size())));
        }
        std::sort(bootstrap_means.begin(), bootstrap_means.end());
        const double lower_probability = 1.0 - confidence;
        const std::size_t lower_index = std::min(
            bootstrap_means.size() - 1U,
            static_cast<std::size_t>(std::floor(
                lower_probability *
                static_cast<double>(bootstrap_means.size() - 1U))));
        result.lower_confidence_bound = bootstrap_means[lower_index];
    }

    if (result.success_sample_count > 0)
    {
        result.success_rate =
            static_cast<double>(result.success_count) /
            static_cast<double>(result.success_sample_count);
        result.success_lower_bound = WilsonLowerBound(
            result.success_count, result.success_sample_count);
    }
    return result;
}

} // namespace NEAT
