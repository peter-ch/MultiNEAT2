#include "Random.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace NEAT
{
    void RNG::Seed(long seed)
    {
        // seed the mt19937 engine
        m_Engine.seed(static_cast<std::mt19937::result_type>(seed));
    }

    void RNG::TimeSeed()
    {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count();
        Seed((long) ms);
    }

    int RNG::RandPosNeg()
    {
        // Return either 1 or -1
        // We can just pick RandInt(0,1) then transform
        int r = RandInt(0, 1);
        return (r == 0) ? -1 : 1;
    }

    int RNG::RandInt(int x, int y)
    {
        if(x > y)
        {
            throw std::runtime_error("RNG::RandInt: invalid range (x>y).");
        }
        std::uniform_int_distribution<int> dist(x, y);
        return dist(m_Engine);
    }

    double RNG::RandFloat()
    {
        // uniform distribution in [0,1)
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(m_Engine);
    }

    double RNG::RandFloatSigned()
    {
        // uniform in [-1,1)
        // or we can do 2 * RandFloat() - 1
        return 2.0 * RandFloat() - 1.0;
    }

    double RNG::RandGaussSigned()
    {
        // Use a normal distribution. We will clamp to [-1,1] like old code.
        // The original Box-Muller code effectively rarely returns > 1, but let's replicate clamp.
        std::normal_distribution<double> dist(0.0, 1.0); // mean=0, stddev=1
        double val = dist(m_Engine);
        // clamp to [-1,1]
        if(val > 1.0) val = 1.0;
        if(val < -1.0) val = -1.0;
        return val;
    }

    int RNG::Roulette(std::vector<double>& a_probs)
    {
        double total = 0.0;
        for (auto &p : a_probs) total += p;
        if(total <= 0.0)
        {
            // fallback: pick random index ignoring weights
            // or return 0
            if(a_probs.empty()) return 0;
            std::uniform_int_distribution<int> dist(0, (int)a_probs.size()-1);
            return dist(m_Engine);
        }
        std::uniform_real_distribution<double> dist(0.0, total);
        double r = dist(m_Engine);

        double run = 0.0;
        for(int idx=0; idx<(int)a_probs.size(); idx++)
        {
            run += a_probs[idx];
            if(run >= r)
            {
                return idx;
            }
        }
        // if floating error, return last
        return (int)a_probs.size()-1;
    }

} // namespace NEAT
