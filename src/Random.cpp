
#include "Random.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace NEAT
{
    void RNG::Seed(long seed)
    {
        m_Engine.seed(static_cast<std::mt19937::result_type>(seed));
    }

    void RNG::TimeSeed()
    {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count();
        Seed(static_cast<long>(ms));
    }

    int RNG::RandPosNeg()
    {
        return (RandInt(0, 1) == 0) ? -1 : 1;
    }

    int RNG::RandInt(int x, int y)
    {
        if(x > y)
            throw std::invalid_argument("RNG::RandInt: invalid range (x > y).");
        std::uniform_int_distribution<int> dist(x, y);
        return dist(m_Engine);
    }

    double RNG::RandFloat()
    {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(m_Engine);
    }

    double RNG::RandFloatSigned()
    {
        return 2.0 * RandFloat() - 1.0;
    }

    double RNG::RandGaussSigned()
    {
        std::normal_distribution<double> dist(0.0, 1.0);
        double val = dist(m_Engine);
        if(val > 1.0) val = 1.0;
        if(val < -1.0) val = -1.0;
        return val;
    }

    double RNG::RandNormal(double mean, double standard_deviation)
    {
        if (!std::isfinite(mean) ||
            !std::isfinite(standard_deviation) ||
            standard_deviation <= 0.0)
        {
            throw std::invalid_argument(
                "RNG::RandNormal requires a finite mean and positive "
                "standard deviation.");
        }
        std::normal_distribution<double> distribution(
            mean, standard_deviation);
        return distribution(m_Engine);
    }

    double RNG::RandCauchy(double location, double scale)
    {
        if (!std::isfinite(location) || !std::isfinite(scale) ||
            scale <= 0.0)
        {
            throw std::invalid_argument(
                "RNG::RandCauchy requires a finite location and positive "
                "scale.");
        }
        std::cauchy_distribution<double> distribution(location, scale);
        double result = distribution(m_Engine);
        // The mathematical distribution is unbounded. Resample the extremely
        // rare non-finite floating-point result so callers always receive a
        // usable mutation.
        while (!std::isfinite(result))
            result = distribution(m_Engine);
        return result;
    }

    int RNG::Roulette(const std::vector<double>& a_probs)
    {
        if (a_probs.empty())
            throw std::invalid_argument(
                "RNG::Roulette: probability vector is empty.");

        double maximum = 0.0;
        for (double p : a_probs)
        {
            if (!std::isfinite(p))
                throw std::invalid_argument(
                    "RNG::Roulette: probabilities must be finite.");
            if (p < 0.0)
                throw std::invalid_argument(
                    "RNG::Roulette: probabilities cannot be negative.");
            maximum = std::max(maximum, p);
        }

        if (maximum <= 0.0)
        {
            int maxIndex = static_cast<int>(a_probs.size()) - 1;
            std::uniform_int_distribution<int> dist(0, maxIndex);
            return dist(m_Engine);
        }

        // Scaling every weight by the maximum leaves the categorical
        // distribution unchanged and guarantees the running total cannot
        // overflow for any realistically addressable vector.
        long double total_wide = 0.0L;
        for (double probability : a_probs)
            total_wide += probability / maximum;
        if (!std::isfinite(total_wide) || total_wide <= 0.0L ||
            total_wide >
                static_cast<long double>(
                    std::numeric_limits<double>::max()))
        {
            throw std::overflow_error(
                "RNG::Roulette: normalized probability total overflowed.");
        }
        const double total = static_cast<double>(total_wide);
        std::uniform_real_distribution<double> dist(0.0, total);
        double r = dist(m_Engine);
        double run = 0.0;
        size_t lastNonZero = 0;
        for (size_t idx = 0; idx < a_probs.size(); idx++)
        {
            const double w = a_probs[idx] / maximum;
            if (w > 0.0)  // Only consider positive probabilities for selection
            {
                lastNonZero = idx;
                if (r < run + w)
                    return static_cast<int>(idx);
                run += w;
            }
        }
        return static_cast<int>(lastNonZero);
    }

    std::string RNG::Serialize() const
    {
        std::ostringstream output;
        output << m_Engine;
        return output.str();
    }

    void RNG::Deserialize(const std::string& data)
    {
        std::istringstream input(data);
        input >> m_Engine;
        if (!input)
            throw std::invalid_argument(
                "RNG::Deserialize: invalid generator state.");
    }
}

