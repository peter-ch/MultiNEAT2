#ifndef _PHENOTYPE_BEHAVIOR_H
#define _PHENOTYPE_BEHAVIOR_H


#include <vector>
#include "Assert.h"

namespace NEAT
{

class Genome;

// Always use this class as the base class when defining your own
// behavior characterization!
class PhenotypeBehavior
{
public:
    virtual ~PhenotypeBehavior(){};

    // A 2D matrix of doubles with arbitrary size
    // is enough to represent any behavior in most domains
    std::vector< std::vector<double> > m_Data;

    // This method acquires behavior data based on the genome given
    // May return true if a successful behavior was encountered during
    // evaluation
    virtual bool   Acquire(Genome* a_Genome)
    {
        return false;
    }

    // Overload this method to calcluate distance between behaviors
    virtual double Distance_To(PhenotypeBehavior* a_Other)
    {
        return 0;
    }

    // This method tells us whether the behavior is the one
    // we're looking for. Not necessary to call/overload this in open-ended evolution
    virtual bool   Successful()
    {
        return true;
    }
    
    // comparison operator 
    bool operator==(PhenotypeBehavior const& other) const { return m_Data == other.m_Data; }
};


};






#endif

