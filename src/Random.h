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
        void Seed(long seed);
        void TimeSeed();
        int RandPosNeg();
        int RandInt(int x, int y);
        double RandFloat();
        double RandFloatSigned();
        double RandGaussSigned();
        int Roulette(const std::vector<double>& a_probs);
        std::string Serialize() const;
        void Deserialize(const std::string& data);
    };
}

#endif
