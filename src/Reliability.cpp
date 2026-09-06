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
    // Workers may complete cases in any order. Hash a canonical ordering so
    // scheduling and checkpoint boundaries cannot change the bootstrap draws.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> evidence;
    evidence.reserve(samples.size());
    for (const auto& sample : samples)
    {
        std::uint64_t score_bits = 0;
        const double score = sample.score == 0.0 ? 0.0 : sample.score;
        static_assert(sizeof(score_bits) == sizeof(score));
        std::memcpy(&score_bits, &score, sizeof(score_bits));
        evidence.emplace_back(sample.case_id, score_bits);
    }
    std::sort(evidence.begin(), evidence.end());
    std::uint64_t seed = 0x6a09e667f3bcc909ULL;
    for (const auto& item : evidence)
    {
        seed = Mix(seed ^ Mix(item.first) ^ Mix(item.second));
    }
    return seed;
}

} // namespace

double WilsonLowerBound(
    std::size_t successes,
    std::size_t samples,
    double z)
{
    if (successes > samples || !std::isfinite(z) || z <= 0.0)
        throw std::invalid_argument("Invalid Wilson interval inputs");
    if (samples == 0 || successes == 0)
        return 0.0;
    const double n = static_cast<double>(samples);
    const double p = static_cast<double>(successes) / n;
    const double scaled_z = z / std::sqrt(n);
    // Rationalizing center - radius avoids cancellation near zero success
    // rates. An overflowing denominator at extreme z correctly yields zero.
    const double denominator = p + scaled_z * (0.5 * scaled_z +
        std::hypot(std::sqrt(p * (1.0 - p)), 0.5 * scaled_z));
    return std::clamp(p * p / denominator, 0.0, 1.0);
}

EvaluationSummary SummarizeEvaluations(
    const std::vector<EvaluationSample>& samples,
    double confidence,
    unsigned int bootstrap_samples)
{
    if (!std::isfinite(confidence) || confidence <= 0.0 || confidence >= 1.0)
        throw std::invalid_argument("Evaluation confidence must be between zero and one");
    if (bootstrap_samples == 0)
        throw std::invalid_argument("Bootstrap sample count must be greater than zero");
    EvaluationSummary result;
    if (samples.empty())
        return result;

    std::vector<double> scores;
    scores.reserve(samples.size());
    double scale = 0.0;
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
        scale = std::max(scale, std::abs(sample.score));
        if (sample.has_success)
        {
            ++result.success_sample_count;
            if (sample.success)
                ++result.success_count;
        }
    }
    result.sample_count = scores.size();
    std::sort(scores.begin(), scores.end());
    const bool constant = scores.front() == scores.back();
    const double constant_score = scores.front();
    // long double has the same range as double on MSVC. Normalize before
    // summing or squaring to avoid overflow for otherwise finite statistics.
    if (scale == 0.0)
        scale = 1.0;
    for (double& score : scores)
        score /= scale;
    const double normalized_mean = static_cast<double>(std::accumulate(
        scores.begin(), scores.end(), 0.0L) /
        static_cast<long double>(scores.size()));
    const auto rescale_mean = [scale](double value)
    { return std::clamp(value, -1.0, 1.0) * scale; };
    result.mean = constant ? constant_score : rescale_mean(normalized_mean);

    if (!constant)
    {
        long double squared = 0.0L;
        for (double score : scores)
        {
            const long double delta =
                static_cast<long double>(score) - normalized_mean;
            squared += delta * delta;
        }
        const double normalized_deviation = std::sqrt(static_cast<double>(
            squared / static_cast<long double>(scores.size() - 1U)));
        result.standard_deviation = normalized_deviation * scale;
        result.standard_error = (normalized_deviation /
            std::sqrt(static_cast<double>(scores.size()))) * scale;
    }

    const std::size_t tail_count = std::max<std::size_t>(
        1U, static_cast<std::size_t>(
                std::ceil(0.2 * static_cast<double>(scores.size()))));
    result.lower_tail_mean = constant ? constant_score : rescale_mean(
        static_cast<double>(std::accumulate(
        scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(tail_count),
        0.0L) / static_cast<long double>(tail_count)));

    if (constant)
    {
        result.lower_confidence_bound = constant_score;
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
        const double lower_probability = 1.0 - confidence;
        const std::size_t lower_index = std::min(
            bootstrap_means.size() - 1U,
            static_cast<std::size_t>(std::floor(
                lower_probability *
                static_cast<double>(bootstrap_means.size() - 1U))));
        std::nth_element(
            bootstrap_means.begin(),
            bootstrap_means.begin() + static_cast<std::ptrdiff_t>(lower_index),
            bootstrap_means.end());
        result.lower_confidence_bound = rescale_mean(bootstrap_means[lower_index]);
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
