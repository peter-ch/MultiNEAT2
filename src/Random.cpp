#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "boost/date_time/posix_time/posix_time.hpp"
#include "Random.h"
#include "Utils.h"

namespace NEAT
{


// Seeds the random number generator with this value
void RNG::Seed(long a_Seed)
{
    srand(a_Seed);
}

void RNG::TimeSeed()
{
    auto now = boost::posix_time::second_clock::local_time();
    Seed(now.time_of_day().total_milliseconds());
}

// Returns randomly either 1 or -1
int RNG::RandPosNeg()
{
    int choice = rand() % 2;
    if (choice == 0)
        return -1;
    else
        return 1;
}

// Returns a random integer between X and Y
int RNG::RandInt(int aX, int aY)
{
    if (aX == aY)
    {
        return aX;
    }
    if (aX == (aY-1))
    {
        // for two consecutives, pick either with equal probability
        if (RandFloat() < 0.5)
        {
            return aX;
        }
        else
        {
            return aY;
        }
    }
    return aX + (rand() % (aY - aX + 1));
}

// Returns a random number from a uniform distribution in the range of [0 .. 1]
double RNG::RandFloat()
{
    return (double)(rand() % RAND_MAX) / RAND_MAX;
}

// Returns a random number from a uniform distribution in the range of [-1 .. 1]
double RNG::RandFloatSigned()
{
    return (RandFloat() - RandFloat());
}

// Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
double RNG::RandGaussSigned()
{
    static int t_iset=0;
    static double t_gset;
    double t_fac,t_rsq,t_v1,t_v2;
    
    if (t_iset==0)
    {
        do
        {
            t_v1=2.0*(RandFloat())-1.0;
            t_v2=2.0*(RandFloat())-1.0;
            t_rsq=t_v1*t_v1+t_v2*t_v2;
        }
        while (t_rsq>=1.0 || t_rsq==0.0);
    
        t_fac=sqrt(-2.0*log(t_rsq)/t_rsq);
        t_gset=t_v1*t_fac;
        t_iset=1;
    
        double t_tmp = t_v2*t_fac;
    
        Clamp(t_tmp, -1.0, 1.0);
        return t_tmp;
    }
    else
    {
        t_iset=0;
        double t_tmp = t_gset;
        Clamp(t_tmp, -1.0, 1.0);
        return t_tmp;
    }
}

int RNG::Roulette(std::vector<double>& a_probs)
{

    double t_marble = 0, t_spin = 0, t_total_score = 0;
    for(unsigned int i=0; i<a_probs.size(); i++)
    {
        t_total_score += a_probs[i];
    }
    t_marble = RandFloat() * t_total_score;
    
    int t_chosen = 0;
    t_spin = a_probs[t_chosen];
    while(t_spin < t_marble)
    {
        t_chosen++;
        t_spin += a_probs[t_chosen];
    }
    
    return t_chosen;
}


}
 // namespace NEAT
