#ifndef _RANDOMNESS_HEADER_H
#define _RANDOMNESS_HEADER_H

#include <vector>
#include <random>
#include <string>

namespace NEAT
{
    class RNG
    {
    private:
        std::mt19937 m_Engine;

    public:
        RNG() { TimeSeed(); }
        // Accepts any 64-bit seed. Values representable as a 32-bit signed
        // or unsigned integer seed the engine exactly as earlier releases did;
        // wider values are folded (low ^ high 32 bits) deterministically.
        void Seed(long long seed);
        void TimeSeed();
        int RandPosNeg();
        int RandInt(int x, int y);
        double RandFloat();
        double RandFloatSigned();
        double RandGaussSigned();
        double RandNormal(double mean = 0.0, double standard_deviation = 1.0);
        double RandCauchy(double location = 0.0, double scale = 1.0);
        int Roulette(const std::vector<double>& a_probs);
        std::string Serialize() const;
        void Deserialize(const std::string& data);
    };
}

#endif
