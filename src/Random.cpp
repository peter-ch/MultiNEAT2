#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono> // Replaces boost/date_time
#include "Random.h"
#include "Utils.h"

namespace NEAT
{
    // Seeds the RNG with this value
    void RNG::Seed(long a_Seed)
    {
        srand((unsigned int)a_Seed);
    }

    // Seeds the RNG using the current time in ms
    void RNG::TimeSeed()
    {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count();
        Seed((long)ms);
    }

    int RNG::RandPosNeg()
    {
        return (rand() % 2) ? 1 : -1;
    }

    int RNG::RandInt(int aX, int aY)
    {
        if(aX==aY) return aX;
        return aX + (rand() % (aY - aX + 1));
    }

    double RNG::RandFloat()
    {
        return (double)rand() / (double)RAND_MAX;
    }

    double RNG::RandFloatSigned()
    {
        return RandFloat() - RandFloat();
    }

    double RNG::RandGaussSigned()
    {
        // Using Box-Muller or simple approach
        static int t_iset=0;
        static double t_gset;
        double t_fac,t_rsq,t_v1,t_v2;

        if (t_iset==0)
        {
            do {
                t_v1=2.0*RandFloat()-1.0;
                t_v2=2.0*RandFloat()-1.0;
                t_rsq = t_v1*t_v1 + t_v2*t_v2;
            } while(t_rsq>=1.0 || t_rsq==0.0);
            t_fac = sqrt(-2.0*log(t_rsq)/t_rsq);
            t_gset = t_v1*t_fac;
            t_iset = 1;
            double tmp = t_v2*t_fac;
            if(tmp>1.0) tmp=1.0;
            if(tmp<-1.0) tmp=-1.0;
            return tmp;
        }
        else
        {
            t_iset=0;
            double tmp = t_gset;
            if(tmp>1.0) tmp=1.0;
            if(tmp<-1.0) tmp=-1.0;
            return tmp;
        }
    }

    int RNG::Roulette(std::vector<double>& a_probs)
    {
        double t_total = 0.0;
        for (auto &p : a_probs) t_total += p;
        double r = RandFloat() * t_total;
        double run = 0.0;
        int idx=0;
        for (; idx<(int)a_probs.size(); idx++)
        {
            run += a_probs[idx];
            if(run>=r) break;
        }
        return idx;
    }

} // namespace NEAT
