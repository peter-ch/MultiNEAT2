#include <Genome.h>
#include <Parameters.h>

int main()
{
    NEAT::Parameters parameters;
    NEAT::GenomeInitStruct init;
    init.NumInputs = 2;
    init.NumOutputs = 1;
    NEAT::Genome genome(parameters, init);
    return genome.NumInputs() == 2 && genome.NumOutputs() == 1 ? 0 : 1;
}
