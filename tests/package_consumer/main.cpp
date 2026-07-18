#include <Genome.h>
#include <Parameters.h>
#include <SpikingLearning.h>

int main()
{
    NEAT::Parameters parameters;
    NEAT::GenomeInitStruct init;
    init.NumInputs = 2;
    init.NumOutputs = 1;
    NEAT::Genome genome(parameters, init);
    NEAT::EPropConfig eprop;
    return genome.NumInputs() == 2 &&
                   genome.NumOutputs() == 1 &&
                   eprop.learning_rate > 0.0
               ? 0
               : 1;
}
