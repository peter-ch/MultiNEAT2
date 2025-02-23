#ifndef _PARAMETERS_H
#define _PARAMETERS_H


#include <map>
#include "Traits.h"

namespace NEAT
{

// forward
class Genome;

//////////////////////////////////////////////
// The NEAT Parameters class
//////////////////////////////////////////////
class Parameters
{
public:
    /////////////////////
    // Members
    /////////////////////


    ////////////////////
    // Basic parameters
    ////////////////////

    // Size of population
    unsigned int PopulationSize;
    
    // Controls the use of speciation. When off, the population will consist of only one species.
    bool Speciation;

    // If true, this enables dynamic compatibility thresholding
    // It will keep the number of species between MinSpecies and MaxSpecies
    bool DynamicCompatibility;

    // Minimum number of species
    unsigned int MinSpecies;

    // Maximum number of species
    unsigned int MaxSpecies;

    // Don't wipe the innovation database each generation?
    bool InnovationsForever;

    // Allow clones or nearly identical genomes to exist simultaneously in the population.
    // This is useful for non-deterministic environments,
    // as the same individual will get more than one chance to prove himself, also
    // there will be more chances the same individual to mutate in different ways.
    // The drawback is greatly increased time for reproduction. If you want to
    // search quickly, yet less efficient, leave this to true.
    bool AllowClones;

    // Keep an archive of genomes and don't allow any new genome to exist in the acrhive or the population
    bool ArchiveEnforcement;
    
    // Normalize genome size when calculating compatibility
    bool NormalizeGenomeSize;
    
    // Pointer to a function that specifies custom topology constraints
    // Should return true if the genome FAILS to meet the constraints
    bool (*CustomConstraints)(Genome& g);
        
    ////////////////////////////////
    // GA Parameters
    ////////////////////////////////

    // AgeGens treshold, meaning if a species is below it, it is considered young
    unsigned int YoungAgeTreshold;

    // Fitness boost multiplier for young species (1.0 means no boost)
    // Make sure it is >= 1.0 to avoid confusion
    double YoungAgeFitnessBoost;

    // Number of generations without improvement (stagnation) allowed for a species
    unsigned int SpeciesMaxStagnation;

    // Minimum jump in fitness necessary to be considered as improvement.
    // Setting this value to 0.0 makes the system to behave like regular NEAT.
    double StagnationDelta;

    // AgeGens threshold, meaning if a species if above it, it is considered old
    unsigned int OldAgeTreshold;

    // Multiplier that penalizes old species.
    // Make sure it is < 1.0 to avoid confusion.
    double OldAgePenalty;

    // Detect competetive coevolution stagnation
    // This kills the worst species of age >N (each X generations)
    bool DetectCompetetiveCoevolutionStagnation;

    // Each X generation..
    int KillWorstSpeciesEach;

    // Of age above..
    int KillWorstAge;

    // Percent of best individuals that are allowed to reproduce. 1.0 = 100%
    double SurvivalRate;

    // Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
    double CrossoverRate;

    // If a baby results from sexual reproduction, this probability determines if mutation will
    // be performed after crossover. 1.0 = 100% (always mutate after crossover)
    double OverallMutationRate;

    // Probability for a baby to result from inter-species mating.
    double InterspeciesCrossoverRate;

    // Probability for a baby gene to result from Multipoint Crossover when mating. 1.0 = 100%
    // The default if the Average mating.
    double MultipointCrossoverRate;
    
    // Probability that when doing multipoint crossover,
    // the gene of the fitter parent will be prefered, instead of choosing one at random
    double PreferFitterParentRate;
    
    // Performing roulette wheel selection or not?
    bool RouletteWheelSelection;
    
    // If true, will do tournament selection
    bool TournamentSelection;

    // For tournament selection
    unsigned int TournamentSize;

    // Fraction of individuals to be copied unchanged
    double EliteFraction;



    ///////////////////////////////////
    // Phased Search parameters   //
    ///////////////////////////////////

    // Using phased search or not
    bool PhasedSearching;

    // Using delta coding or not
    bool DeltaCoding;

    // What is the MPC + base MPC needed to begin simplifying phase
    unsigned int SimplifyingPhaseMPCTreshold;

    // How many generations of global stagnation should have passed to enter simplifying phase
    unsigned int SimplifyingPhaseStagnationTreshold;

    // How many generations of MPC stagnation are needed to turn back on complexifying
    unsigned int ComplexityFloorGenerations;


    /////////////////////////////////////
    // Novelty Search parameters       //
    /////////////////////////////////////

    // the K constant
    unsigned int NoveltySearch_K;

    // Sparseness treshold. Add to the archive if above
    double NoveltySearch_P_min;

    // Dynamic Pmin?
    bool NoveltySearch_Dynamic_Pmin;

    // How many evaluations should pass without adding to the archive
    // in order to lower Pmin
    unsigned int NoveltySearch_No_Archiving_Stagnation_Treshold;

    // How should it be multiplied (make it less than 1.0)
    double NoveltySearch_Pmin_lowering_multiplier;

    // Not lower than this value
    double NoveltySearch_Pmin_min;


    // How many one-after-another additions to the archive should
    // pass in order to raise Pmin
    unsigned int NoveltySearch_Quick_Archiving_Min_Evaluations;

    // How should it be multiplied (make it more than 1.0)
    double NoveltySearch_Pmin_raising_multiplier;

    // Per how many evaluations to recompute the sparseness
    unsigned int NoveltySearch_Recompute_Sparseness_Each;


    ///////////////////////////////////
    // Mutation parameters
    ///////////////////////////////////

    // Probability for a baby to be mutated with the Add-Neuron mutation.
    double MutateAddNeuronProb;

    // Allow splitting of any recurrent links
    bool SplitRecurrent;

    // Allow splitting of looped recurrent links
    bool SplitLoopedRecurrent;

    // Maximum number of tries to find a link to split
    int NeuronTries;

    // Probability for a baby to be mutated with the Add-Link mutation
    double MutateAddLinkProb;

    // Probability for a new incoming link to be from the bias neuron;
    double MutateAddLinkFromBiasProb;

    // Probability for a baby to be mutated with the Remove-Link mutation
    double MutateRemLinkProb;

    // Probability for a baby that a simple neuron will be replaced with a link
    double MutateRemSimpleNeuronProb;

    // Maximum number of tries to find 2 neurons to add/remove a link
    unsigned int LinkTries;
    
    // Maximum number of links in the genome (originals not counted). -1 is unlimited
    int MaxLinks;
    
    // Maximum number of neurons in the genome (originals not counted). -1 is unlimited
    int MaxNeurons;
    
    // Probability that a link mutation will be made recurrent
    double RecurrentProb;

    // Probability that a recurrent link mutation will be looped
    double RecurrentLoopProb;

    // Probability for a baby's weights to be mutated
    double MutateWeightsProb;

    // Probability for a severe (shaking) weight mutation
    double MutateWeightsSevereProb;

    // Probability for a particular gene to be mutated. 1.0 = 100%
    double WeightMutationRate;
    
    // Probability for a particular gene to be mutated via replacement of the weight. 1.0 = 100%
    double WeightReplacementRate;

    // Maximum perturbation for a weight mutation
    double WeightMutationMaxPower;

    // Maximum magnitude of a replaced weight
    double WeightReplacementMaxPower;

    // Maximum weight
    double MaxWeight;
    
    // Minimum weight
    double MinWeight;

    // Probability for a baby's A activation function parameters to be perturbed
    double MutateActivationAProb;

    // Probability for a baby's B activation function parameters to be perturbed
    double MutateActivationBProb;

    // Maximum magnitude for the A parameter perturbation
    double ActivationAMutationMaxPower;

    // Maximum magnitude for the B parameter perturbation
    double ActivationBMutationMaxPower;

    // Maximum magnitude for time costants perturbation
    double TimeConstantMutationMaxPower;

    // Maximum magnitude for biases perturbation
    double BiasMutationMaxPower;

    // Activation parameter A min/max
    double MinActivationA;
    double MaxActivationA;

    // Activation parameter B min/max
    double MinActivationB;
    double MaxActivationB;

    // Probability for a baby that an activation function type will be changed for a single neuron
    // considered a structural mutation because of the large impact on fitness
    double MutateNeuronActivationTypeProb;

    // Probabilities for a particular activation function appearance
    double ActivationFunction_SignedSigmoid_Prob;
    double ActivationFunction_UnsignedSigmoid_Prob;
    double ActivationFunction_Tanh_Prob;
    double ActivationFunction_TanhCubic_Prob;
    double ActivationFunction_SignedStep_Prob;
    double ActivationFunction_UnsignedStep_Prob;
    double ActivationFunction_SignedGauss_Prob;
    double ActivationFunction_UnsignedGauss_Prob;
    double ActivationFunction_Abs_Prob;
    double ActivationFunction_SignedSine_Prob;
    double ActivationFunction_UnsignedSine_Prob;
    double ActivationFunction_Linear_Prob;
    double ActivationFunction_Relu_Prob;
    double ActivationFunction_Softplus_Prob;

    // Probability for a baby's neuron time constant values to be mutated
    double MutateNeuronTimeConstantsProb;

    // Probability for a baby's neuron bias values to be mutated
    double MutateNeuronBiasesProb;

    // Time constant range
    double MinNeuronTimeConstant;
    double MaxNeuronTimeConstant;

    // Bias range
    double MinNeuronBias;
    double MaxNeuronBias;

    /////////////////////////////////////
    // Speciation parameters
    /////////////////////////////////////

    // Percent of disjoint genes importance
    double DisjointCoeff;

    // Percent of excess genes importance
    double ExcessCoeff;

    // Node-specific activation parameter A difference importance
    double ActivationADiffCoeff;

    // Node-specific activation parameter B difference importance
    double ActivationBDiffCoeff;

    // Average weight difference importance
    double WeightDiffCoeff;

    // Average time constant difference importance
    double TimeConstantDiffCoeff;

    // Average bias difference importance
    double BiasDiffCoeff;

    // Activation function type difference importance
    double ActivationFunctionDiffCoeff;

    // Compatibility treshold
    double CompatTreshold;

    // Minumal value of the compatibility treshold
    double MinCompatTreshold;

    // Modifier per generation for keeping the species stable
    double CompatTresholdModifier;

    // Per how many generations to change the treshold
    unsigned int CompatTreshChangeInterval_Generations;

    // Per how many evaluations to change the treshold
    unsigned int CompatTreshChangeInterval_Evaluations;
    
    // What is the minimal difference needed for not to be a clone
    double MinDeltaCompatEqualGenomes;
    
    // How many times to test a genome for constraint failure or being a clone (when AllowClones=False)
    int ConstraintTrials;
    
    /////////////////////////////
    // Genome properties params
    /////////////////////////////
    
    // When true, don't have a special bias neuron and treat all inputs equal
    bool DontUseBiasNeuron;
    bool AllowLoops;


    /////////////////////////////
    // ES HyperNEAT params
    /////////////////////////////

    double DivisionThreshold;

    double VarianceThreshold;

    // Used for Band prunning.
    double BandThreshold;

    // Max and Min Depths of the quadtree
    unsigned int InitialDepth;

    unsigned int MaxDepth;

    // How many hidden layers before connecting nodes to output. At 0 there is
    // one hidden layer. At 1, there are two and so on.
    unsigned int IterationLevel;

    // The Bias value for the CPPN queries.
    double CPPN_Bias;

    // Quadtree Dimensions
    // The range of the tree. Typically set to 2,
    double Width;
    double Height;

    // The (x, y) coordinates of the tree
    double Qtree_X;

    double Qtree_Y;

    // Use Link Expression output
    bool Leo;

    // Threshold above which a connection is expressed
    double LeoThreshold;

    // Use geometric seeding. Currently only along the X axis. 1
    bool LeoSeed;
    bool GeometrySeed;

    /////////////////////////////////////
    // Universal traits
    /////////////////////////////////////
    std::map< std::string, TraitParameters > NeuronTraits;
    std::map< std::string, TraitParameters > LinkTraits;
    std::map< std::string, TraitParameters > GenomeTraits;
    double MutateNeuronTraitsProb;
    double MutateLinkTraitsProb;
    double MutateGenomeTraitsProb;

    /////////////////////////////////////
    // Constructors
    /////////////////////////////////////

    // Load defaults
    Parameters();

    ////////////////////////////////////
    // Methods
    ////////////////////////////////////

    // Load the parameters from a file
    // returns 0 on success
    int Load(const char* filename);
    // Load the parameters from an already opened file for reading
    int Load(std::ifstream& a_DataFile);

    void Save(const char* filename);
    // Saves the parameters to an already opened file for writing
    void Save(FILE* a_fstream);

    // resets the parameters to built-in defaults
    void Reset();
    

    };


} // namespace NEAT



#endif
