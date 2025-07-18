
#include "Random.h"
#include <chrono>
#include <cmath>
#include <limits>
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
            throw std::runtime_error("RNG::RandInt: invalid range (x > y).");
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

    int RNG::Roulette(const std::vector<double>& a_probs)
    {
        double total = 0.0;
        for (double p : a_probs)
            total += p;

        if (total <= 0.0 || a_probs.empty())
        {
            int maxIndex = a_probs.size() > 0 ? static_cast<int>(a_probs.size()) - 1 : 0;
            std::uniform_int_distribution<int> dist(0, maxIndex);
            return dist(m_Engine);
        }

        std::uniform_real_distribution<double> dist(0.0, total);
        double r = dist(m_Engine);
        double run = 0.0;
        size_t lastNonZero = 0;
        for (size_t idx = 0; idx < a_probs.size(); idx++)
        {
            double w = a_probs[idx];
            if (w > 0.0)
            {
                lastNonZero = idx;
                if (r < run + w)
                    return static_cast<int>(idx);
            }
            run += w;
        }
        return static_cast<int>(lastNonZero);
    }
}

