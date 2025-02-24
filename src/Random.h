#ifndef _RANDOMNESS_HEADER_H
#define _RANDOMNESS_HEADER_H

#include <vector>
#include <random>

namespace NEAT
{

class RNG
{
private:
    std::mt19937 m_Engine; // Our Mersenne Twister engine

public:
    // constructor
    RNG()
    {
        // By default, we seed by time
        TimeSeed();
    }

    // Seeds the random number generator with this value
    void Seed(long seed);

    // Seeds the random number generator with the current time in ms
    void TimeSeed();

    // Returns randomly either 1 or -1
    int RandPosNeg();

    // Returns a random integer between [x, y]
    int RandInt(int x, int y);

    // Returns a random number from a uniform distribution in the range of [0 .. 1]
    double RandFloat();

    // Returns a random number from a uniform distribution in the range of [-1 .. 1]
    double RandFloatSigned();

    // Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
    double RandGaussSigned();

    // Returns an index given a vector of probabilities
    int Roulette(std::vector<double>& a_probs);
};

} // namespace NEAT

#endif
