#include "Parameters.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "Serialization.h"
#include "FileIO.h"

namespace NEAT
{
namespace
{

#define MULTINEAT_PARAMETER_FIELDS(X)                                      \
    X(PopulationSize)                                                       \
    X(Speciation)                                                           \
    X(DynamicCompatibility)                                                 \
    X(MinSpecies)                                                           \
    X(MaxSpecies)                                                           \
    X(InnovationsForever)                                                   \
    X(AllowClones)                                                          \
    X(ArchiveEnforcement)                                                   \
    X(NormalizeGenomeSize)                                                  \
    X(YoungAgeTreshold)                                                     \
    X(YoungAgeFitnessBoost)                                                 \
    X(SpeciesMaxStagnation)                                                 \
    X(StagnationDelta)                                                      \
    X(OldAgeTreshold)                                                       \
    X(OldAgePenalty)                                                        \
    X(DetectCompetetiveCoevolutionStagnation)                               \
    X(KillWorstSpeciesEach)                                                 \
    X(KillWorstAge)                                                         \
    X(SurvivalRate)                                                         \
    X(CrossoverRate)                                                        \
    X(OverallMutationRate)                                                  \
    X(InterspeciesCrossoverRate)                                            \
    X(MultipointCrossoverRate)                                              \
    X(PreferFitterParentRate)                                               \
    X(TruncationSelection)                                                  \
    X(RouletteWheelSelection)                                               \
    X(TournamentSelection)                                                  \
    X(TournamentSize)                                                       \
    X(EliteFraction)                                                        \
    X(PhasedSearching)                                                      \
    X(DeltaCoding)                                                          \
    X(SimplifyingPhaseMPCTreshold)                                          \
    X(SimplifyingPhaseStagnationTreshold)                                   \
    X(ComplexityFloorGenerations)                                           \
    X(NoveltySearch_K)                                                      \
    X(NoveltySearch_P_min)                                                  \
    X(NoveltySearch_Dynamic_Pmin)                                           \
    X(NoveltySearch_No_Archiving_Stagnation_Treshold)                       \
    X(NoveltySearch_Pmin_lowering_multiplier)                               \
    X(NoveltySearch_Pmin_min)                                               \
    X(NoveltySearch_Quick_Archiving_Min_Evaluations)                        \
    X(NoveltySearch_Pmin_raising_multiplier)                                \
    X(NoveltySearch_Recompute_Sparseness_Each)                              \
    X(MutateAddNeuronProb)                                                  \
    X(SplitRecurrent)                                                       \
    X(SplitLoopedRecurrent)                                                 \
    X(NeuronTries)                                                          \
    X(MutateAddLinkProb)                                                    \
    X(MutateAddLinkFromBiasProb)                                            \
    X(MutateRemLinkProb)                                                    \
    X(MutateRemSimpleNeuronProb)                                            \
    X(LinkTries)                                                            \
    X(MaxLinks)                                                             \
    X(MaxNeurons)                                                           \
    X(RecurrentProb)                                                        \
    X(RecurrentLoopProb)                                                    \
    X(MutateWeightsProb)                                                    \
    X(MutateWeightsSevereProb)                                              \
    X(WeightMutationRate)                                                   \
    X(WeightReplacementRate)                                                \
    X(WeightMutationMaxPower)                                               \
    X(WeightReplacementMaxPower)                                            \
    X(MaxWeight)                                                            \
    X(MinWeight)                                                            \
    X(MutateActivationAProb)                                                \
    X(MutateActivationBProb)                                                \
    X(ActivationAMutationMaxPower)                                          \
    X(ActivationBMutationMaxPower)                                          \
    X(TimeConstantMutationMaxPower)                                         \
    X(BiasMutationMaxPower)                                                 \
    X(MinActivationA)                                                       \
    X(MaxActivationA)                                                       \
    X(MinActivationB)                                                       \
    X(MaxActivationB)                                                       \
    X(MutateNeuronActivationTypeProb)                                       \
    X(ActivationFunction_SignedSigmoid_Prob)                                \
    X(ActivationFunction_UnsignedSigmoid_Prob)                              \
    X(ActivationFunction_Tanh_Prob)                                         \
    X(ActivationFunction_TanhCubic_Prob)                                    \
    X(ActivationFunction_SignedStep_Prob)                                   \
    X(ActivationFunction_UnsignedStep_Prob)                                 \
    X(ActivationFunction_SignedGauss_Prob)                                  \
    X(ActivationFunction_UnsignedGauss_Prob)                                \
    X(ActivationFunction_Abs_Prob)                                          \
    X(ActivationFunction_SignedSine_Prob)                                   \
    X(ActivationFunction_UnsignedSine_Prob)                                 \
    X(ActivationFunction_Linear_Prob)                                       \
    X(ActivationFunction_Relu_Prob)                                         \
    X(ActivationFunction_Softplus_Prob)                                     \
    X(ActivationFunction_SpikingLIF_Prob)                                   \
    X(ActivationFunction_SpikingAdaptiveLIF_Prob)                           \
    X(ActivationFunction_SpikingIzhikevich_Prob)                            \
    X(ActivationFunction_McCullochPitts_Prob)                               \
    X(MutateNeuronTimeConstantsProb)                                        \
    X(MutateNeuronBiasesProb)                                               \
    X(MinNeuronTimeConstant)                                                \
    X(MaxNeuronTimeConstant)                                                \
    X(MinNeuronBias)                                                        \
    X(MaxNeuronBias)                                                        \
    X(MutateNeuronSpikingParametersProb)                                    \
    X(MutateLinkSpikingParametersProb)                                      \
    X(SpikingParameterMutationRate)                                         \
    X(SpikingParameterMutationPower)                                        \
    X(InitialMCPInhibitoryVetoProb)                                         \
    X(MutateMCPInhibitoryVetoProb)                                          \
    X(MinSpikingTimeConstant)                                               \
    X(MaxSpikingTimeConstant)                                               \
    X(MinSpikeThreshold)                                                    \
    X(MaxSpikeThreshold)                                                    \
    X(MinResetPotential)                                                    \
    X(MaxResetPotential)                                                    \
    X(MinRestingPotential)                                                  \
    X(MaxRestingPotential)                                                  \
    X(MinRefractoryPeriod)                                                  \
    X(MaxRefractoryPeriod)                                                  \
    X(MinMembraneResistance)                                                \
    X(MaxMembraneResistance)                                                \
    X(MinAdaptationTimeConstant)                                            \
    X(MaxAdaptationTimeConstant)                                            \
    X(MinAdaptationIncrement)                                               \
    X(MaxAdaptationIncrement)                                               \
    X(MinSpikeRateTimeConstant)                                             \
    X(MaxSpikeRateTimeConstant)                                             \
    X(MinIzhikevichA)                                                       \
    X(MaxIzhikevichA)                                                       \
    X(MinIzhikevichThreshold)                                               \
    X(MaxIzhikevichThreshold)                                               \
    X(MinIzhikevichB)                                                       \
    X(MaxIzhikevichB)                                                       \
    X(MinIzhikevichC)                                                       \
    X(MaxIzhikevichC)                                                       \
    X(MinIzhikevichD)                                                       \
    X(MaxIzhikevichD)                                                       \
    X(MinSynapticDelay)                                                     \
    X(MaxSynapticDelay)                                                     \
    X(MinSynapticTimeConstant)                                              \
    X(MaxSynapticTimeConstant)                                              \
    X(InitialSTDPEnabledProb)                                               \
    X(MinSTDPPlus)                                                          \
    X(MaxSTDPPlus)                                                          \
    X(MinSTDPMinus)                                                         \
    X(MaxSTDPMinus)                                                         \
    X(MinSTDPTau)                                                           \
    X(MaxSTDPTau)                                                           \
    X(DisjointCoeff)                                                        \
    X(ExcessCoeff)                                                          \
    X(ActivationADiffCoeff)                                                 \
    X(ActivationBDiffCoeff)                                                 \
    X(WeightDiffCoeff)                                                      \
    X(TimeConstantDiffCoeff)                                                \
    X(BiasDiffCoeff)                                                        \
    X(ActivationFunctionDiffCoeff)                                          \
    X(SpikingNeuronDiffCoeff)                                               \
    X(SpikingLinkDiffCoeff)                                                 \
    X(CompatTreshold)                                                       \
    X(MinCompatTreshold)                                                    \
    X(CompatTresholdModifier)                                               \
    X(CompatTreshChangeInterval_Generations)                                \
    X(CompatTreshChangeInterval_Evaluations)                                \
    X(MinDeltaCompatEqualGenomes)                                           \
    X(ConstraintTrials)                                                     \
    X(DontUseBiasNeuron)                                                    \
    X(AllowLoops)                                                           \
    X(DivisionThreshold)                                                    \
    X(VarianceThreshold)                                                    \
    X(BandThreshold)                                                        \
    X(InitialDepth)                                                         \
    X(MaxDepth)                                                             \
    X(IterationLevel)                                                       \
    X(CPPN_Bias)                                                            \
    X(Width)                                                                \
    X(Height)                                                               \
    X(Depth)                                                                \
    X(Qtree_X)                                                              \
    X(Qtree_Y)                                                              \
    X(Qtree_Z)                                                              \
    X(Leo)                                                                  \
    X(LeoThreshold)                                                         \
    X(LeoSeed)                                                              \
    X(GeometrySeed)                                                         \
    X(MutateNeuronTraitsProb)                                               \
    X(MutateLinkTraitsProb)                                                 \
    X(MutateGenomeTraitsProb)                                               \
    X(ParentSelectionMode)                                                  \
    X(RankSelectionPressure)                                                \
    X(RankSelectionExponent)                                                \
    X(BoltzmannTemperature)                                                 \
    X(SinglePointCrossoverRate)                                             \
    X(BlendCrossoverRate)                                                   \
    X(SimulatedBinaryCrossoverRate)                                         \
    X(CrossoverBlendAlpha)                                                  \
    X(CrossoverSBXEta)                                                      \
    X(WeightMutationDistribution)                                           \
    X(WeightMutationSigma)                                                  \
    X(WeightMutationCauchyScale)                                            \
    X(WeightMutationPolynomialEta)                                          \
    X(SpeciesRepresentativeSelection)                                       \
    X(RepresentativeSelectionCandidates)                                    \
    X(OffspringAllocation)                                                  \
    X(MinSpeciesSize)                                                       \
    X(SpeciesElitism)                                                       \
    X(StagnationPenalty)                                                    \
    X(CompatibilityThresholdControl)                                        \
    X(TargetSpecies)                                                        \
    X(CompatibilityThresholdGain)                                           \
    X(MaxCompatTreshold)                                                    \
    X(RequireEvaluatedGenomes)                                              \
    X(RejectNonFiniteFitness)                                               \
    X(MutationOperatorsPerOffspring)                                        \
    X(AdaptiveMutationStart)                                                \
    X(AdaptiveMutationRate)                                                 \
    X(AdaptiveMutationMaxFactor)                                            \
    X(FitnessScaling)                                                       \
    X(FitnessRankPressure)                                                  \
    X(FitnessSigmaScale)                                                    \
    X(FitnessBoltzmannTemperature)

template <typename T>
void WriteScalar(std::ostream& output, const T& value)
{
    output << value;
}

void WriteScalar(std::ostream& output, bool value)
{
    output << (value ? "true" : "false");
}

template <typename T>
void ReadScalar(std::istream& input, T& value)
{
    if constexpr (std::is_enum_v<T>)
    {
        std::underlying_type_t<T> encoded{};
        input >> encoded;
        if (input)
            value = static_cast<T>(encoded);
    }
    else
    {
        input >> value;
    }
}

void ReadScalar(std::istream& input, bool& value)
{
    std::string token;
    input >> token;
    if (token == "true" || token == "1")
        value = true;
    else if (token == "false" || token == "0")
        value = false;
    else
        input.setstate(std::ios::failbit);
}

void WriteParameters(std::ostream& output, const Parameters& parameters)
{
    Serialization::UseRoundTripPrecision(output);
    output << "NEAT_ParametersStart\n";
#define MULTINEAT_WRITE_PARAMETER(name) \
    output << #name << ' ';              \
    WriteScalar(output, parameters.name); \
    output << '\n';
    MULTINEAT_PARAMETER_FIELDS(MULTINEAT_WRITE_PARAMETER)
#undef MULTINEAT_WRITE_PARAMETER

    Serialization::WriteTraitParameters(
        output, "NeuronTraitSchemas", parameters.NeuronTraits);
    Serialization::WriteTraitParameters(
        output, "LinkTraitSchemas", parameters.LinkTraits);
    Serialization::WriteTraitParameters(
        output, "GenomeTraitSchemas", parameters.GenomeTraits);
    output << "NEAT_ParametersEnd\n";
}

bool ReadParameters(std::istream& input, Parameters& parameters)
{
    std::string key;
    while (input >> key)
    {
        if (key == "NEAT_ParametersStart")
            break;
    }
    if (!input || key != "NEAT_ParametersStart")
        return false;

    parameters.Reset();
    std::unordered_map<std::string, std::function<void(std::istream&)>>
        readers;
#define MULTINEAT_MAKE_READER(name)                                      \
    readers.emplace(#name, [&parameters](std::istream& stream)           \
                    { ReadScalar(stream, parameters.name); });
    MULTINEAT_PARAMETER_FIELDS(MULTINEAT_MAKE_READER)
#undef MULTINEAT_MAKE_READER
    readers.emplace("Elitism", [&parameters](std::istream& stream)
                    { ReadScalar(stream, parameters.EliteFraction); });

    while (input >> key)
    {
        if (key == "NEAT_ParametersEnd")
            return true;
        if (key == "NeuronTraitSchemas")
        {
            parameters.NeuronTraits =
                Serialization::ReadTraitParameters(input);
            continue;
        }
        if (key == "LinkTraitSchemas")
        {
            parameters.LinkTraits =
                Serialization::ReadTraitParameters(input);
            continue;
        }
        if (key == "GenomeTraitSchemas")
        {
            parameters.GenomeTraits =
                Serialization::ReadTraitParameters(input);
            continue;
        }

        const auto reader = readers.find(key);
        if (reader != readers.end())
        {
            reader->second(input);
            if (!input)
                throw std::runtime_error(
                    "Parameters::Load: invalid value for '" + key + "'.");
        }
        else
        {
            // Forward compatibility: ignore unknown key/value lines.
            std::string ignored;
            std::getline(input, ignored);
        }
    }
    return false;
}

} // namespace

Parameters::Parameters()
{
    Reset();
}

void Parameters::Reset()
{
    PopulationSize = 300;
    Speciation = true;
    DynamicCompatibility = true;
    MinSpecies = 5;
    MaxSpecies = 10;
    InnovationsForever = true;
    AllowClones = true;
    ArchiveEnforcement = false;
    NormalizeGenomeSize = false;
    CustomConstraints = nullptr;
    m_CustomConstraintsFunction = {};

    YoungAgeTreshold = 15;
    YoungAgeFitnessBoost = 1.1;
    SpeciesMaxStagnation = 25000;
    StagnationDelta = 0.0;
    OldAgeTreshold = 80;
    OldAgePenalty = 0.75;
    DetectCompetetiveCoevolutionStagnation = false;
    KillWorstSpeciesEach = 15;
    KillWorstAge = 10;
    SurvivalRate = 0.2;
    CrossoverRate = 0.7;
    OverallMutationRate = 0.75;
    InterspeciesCrossoverRate = 0.0001;
    MultipointCrossoverRate = 0.75;
    PreferFitterParentRate = 0.5;
    TruncationSelection = true;
    RouletteWheelSelection = false;
    TournamentSelection = false;
    TournamentSize = 5;
    EliteFraction = 0.0001;

    PhasedSearching = false;
    DeltaCoding = false;
    SimplifyingPhaseMPCTreshold = 20;
    SimplifyingPhaseStagnationTreshold = 30;
    ComplexityFloorGenerations = 40;

    NoveltySearch_K = 15;
    NoveltySearch_P_min = 0.5;
    NoveltySearch_Dynamic_Pmin = true;
    NoveltySearch_No_Archiving_Stagnation_Treshold = 150;
    NoveltySearch_Pmin_lowering_multiplier = 0.9;
    NoveltySearch_Pmin_min = 0.05;
    NoveltySearch_Quick_Archiving_Min_Evaluations = 8;
    NoveltySearch_Pmin_raising_multiplier = 1.1;
    NoveltySearch_Recompute_Sparseness_Each = 25;

    MutateAddNeuronProb = 0.01;
    SplitRecurrent = false;
    SplitLoopedRecurrent = false;
    NeuronTries = 64;
    MutateAddLinkProb = 0.03;
    MutateAddLinkFromBiasProb = 0.01;
    MutateRemLinkProb = 0.0;
    MutateRemSimpleNeuronProb = 0.0;
    LinkTries = 64;
    MaxLinks = -1;
    MaxNeurons = -1;
    RecurrentProb = 0.2;
    RecurrentLoopProb = 0.5;
    MutateWeightsProb = 0.80;
    MutateWeightsSevereProb = 0.2;
    WeightMutationRate = 0.8;
    WeightReplacementRate = 0.2;
    WeightMutationMaxPower = 1.5;
    WeightReplacementMaxPower = 3.0;
    MaxWeight = 8.0;
    MinWeight = -8.0;
    MutateActivationAProb = 0.0;
    MutateActivationBProb = 0.0;
    ActivationAMutationMaxPower = 0.0;
    ActivationBMutationMaxPower = 0.0;
    TimeConstantMutationMaxPower = 0.0;
    BiasMutationMaxPower = WeightMutationMaxPower;
    MinActivationA = 4.9;
    MaxActivationA = 4.9;
    MinActivationB = 0.0;
    MaxActivationB = 0.0;
    MutateNeuronActivationTypeProb = 0.0;
    ActivationFunction_SignedSigmoid_Prob = 0.0;
    ActivationFunction_UnsignedSigmoid_Prob = 1.0;
    ActivationFunction_Tanh_Prob = 0.0;
    ActivationFunction_TanhCubic_Prob = 0.0;
    ActivationFunction_SignedStep_Prob = 0.0;
    ActivationFunction_UnsignedStep_Prob = 0.0;
    ActivationFunction_SignedGauss_Prob = 0.0;
    ActivationFunction_UnsignedGauss_Prob = 0.0;
    ActivationFunction_Abs_Prob = 0.0;
    ActivationFunction_SignedSine_Prob = 0.0;
    ActivationFunction_UnsignedSine_Prob = 0.0;
    ActivationFunction_Linear_Prob = 0.0;
    ActivationFunction_Relu_Prob = 0.0;
    ActivationFunction_Softplus_Prob = 0.0;
    ActivationFunction_SpikingLIF_Prob = 0.0;
    ActivationFunction_SpikingAdaptiveLIF_Prob = 0.0;
    ActivationFunction_SpikingIzhikevich_Prob = 0.0;
    ActivationFunction_McCullochPitts_Prob = 0.0;
    MutateNeuronTimeConstantsProb = 0.0;
    MutateNeuronBiasesProb = 0.0;
    MinNeuronTimeConstant = 0.0;
    MaxNeuronTimeConstant = 0.0;
    MinNeuronBias = 0.0;
    MaxNeuronBias = 0.0;

    MutateNeuronSpikingParametersProb = 0.0;
    MutateLinkSpikingParametersProb = 0.0;
    SpikingParameterMutationRate = 0.2;
    SpikingParameterMutationPower = 0.1;
    InitialMCPInhibitoryVetoProb = 1.0;
    MutateMCPInhibitoryVetoProb = 0.05;
    MinSpikingTimeConstant = 0.005;
    MaxSpikingTimeConstant = 0.05;
    MinSpikeThreshold = 0.5;
    MaxSpikeThreshold = 2.0;
    MinResetPotential = -0.5;
    MaxResetPotential = 0.5;
    MinRestingPotential = -0.5;
    MaxRestingPotential = 0.5;
    MinRefractoryPeriod = 0.0;
    MaxRefractoryPeriod = 0.01;
    MinMembraneResistance = 0.1;
    MaxMembraneResistance = 2.0;
    MinAdaptationTimeConstant = 0.02;
    MaxAdaptationTimeConstant = 1.0;
    MinAdaptationIncrement = 0.0;
    MaxAdaptationIncrement = 0.5;
    MinSpikeRateTimeConstant = 0.01;
    MaxSpikeRateTimeConstant = 0.2;
    MinIzhikevichA = 0.01;
    MaxIzhikevichA = 0.1;
    MinIzhikevichThreshold = 25.0;
    MaxIzhikevichThreshold = 35.0;
    MinIzhikevichB = 0.1;
    MaxIzhikevichB = 0.3;
    MinIzhikevichC = -80.0;
    MaxIzhikevichC = -50.0;
    MinIzhikevichD = 0.0;
    MaxIzhikevichD = 10.0;
    MinSynapticDelay = 0.0;
    MaxSynapticDelay = 0.02;
    MinSynapticTimeConstant = 0.001;
    MaxSynapticTimeConstant = 0.05;
    InitialSTDPEnabledProb = 0.0;
    MinSTDPPlus = 0.0;
    MaxSTDPPlus = 0.05;
    MinSTDPMinus = 0.0;
    MaxSTDPMinus = 0.05;
    MinSTDPTau = 0.005;
    MaxSTDPTau = 0.1;

    DisjointCoeff = 1.0;
    ExcessCoeff = 1.0;
    ActivationADiffCoeff = 0.0;
    ActivationBDiffCoeff = 0.0;
    WeightDiffCoeff = 0.1;
    TimeConstantDiffCoeff = 0.0;
    BiasDiffCoeff = 0.0;
    ActivationFunctionDiffCoeff = 0.0;
    SpikingNeuronDiffCoeff = 0.0;
    SpikingLinkDiffCoeff = 0.0;
    CompatTreshold = 3.0;
    MinCompatTreshold = 0.1;
    CompatTresholdModifier = 0.2;
    CompatTreshChangeInterval_Generations = 1;
    CompatTreshChangeInterval_Evaluations = 1;
    MinDeltaCompatEqualGenomes = 1e-7;
    ConstraintTrials = 2000000;

    DontUseBiasNeuron = false;
    AllowLoops = true;

    // These defaults existed in the original implementation but were
    // accidentally dropped from Reset(), leaving indeterminate values.
    DivisionThreshold = 0.03;
    VarianceThreshold = 0.03;
    BandThreshold = 0.3;
    InitialDepth = 3;
    MaxDepth = 3;
    IterationLevel = 1;
    CPPN_Bias = 1.0;
    Width = 2.0;
    Height = 2.0;
    Depth = 2.0;
    Qtree_X = 0.0;
    Qtree_Y = 0.0;
    Qtree_Z = 0.0;
    Leo = false;
    LeoThreshold = 0.1;
    LeoSeed = false;
    GeometrySeed = false;

    NeuronTraits.clear();
    LinkTraits.clear();
    GenomeTraits.clear();
    MutateNeuronTraitsProb = 0.0;
    MutateLinkTraitsProb = 0.0;
    MutateGenomeTraitsProb = 0.0;

    ParentSelectionMode = LEGACY_SELECTION;
    RankSelectionPressure = 1.7;
    RankSelectionExponent = 4.0;
    BoltzmannTemperature = 1.0;

    SinglePointCrossoverRate = 0.0;
    BlendCrossoverRate = 0.0;
    SimulatedBinaryCrossoverRate = 0.0;
    CrossoverBlendAlpha = 0.5;
    CrossoverSBXEta = 10.0;

    WeightMutationDistribution = UNIFORM_MUTATION;
    WeightMutationSigma = 1.0;
    WeightMutationCauchyScale = 1.0;
    WeightMutationPolynomialEta = 20.0;

    SpeciesRepresentativeSelection = FIRST_REPRESENTATIVE;
    RepresentativeSelectionCandidates = 0;
    OffspringAllocation = LARGEST_REMAINDER;
    MinSpeciesSize = 0;
    SpeciesElitism = 0;
    StagnationPenalty = 0.0000001;

    CompatibilityThresholdControl =
        LEGACY_COMPATIBILITY_THRESHOLD;
    TargetSpecies = 0;
    CompatibilityThresholdGain = 0.25;
    MaxCompatTreshold = 1.0e9;

    RequireEvaluatedGenomes = false;
    RejectNonFiniteFitness = false;

    MutationOperatorsPerOffspring = 1.0;
    AdaptiveMutationStart = 0;
    AdaptiveMutationRate = 0.0;
    AdaptiveMutationMaxFactor = 1.0;
    FitnessScaling = SHIFTED_FITNESS_SCALING;
    FitnessRankPressure = 1.5;
    FitnessSigmaScale = 2.0;
    FitnessBoltzmannTemperature = 1.0;
}

void Parameters::ConfigureSpiking(bool enable_stdp)
{
    ActivationFunction_SignedSigmoid_Prob = 0.0;
    ActivationFunction_UnsignedSigmoid_Prob = 0.0;
    ActivationFunction_Tanh_Prob = 0.0;
    ActivationFunction_TanhCubic_Prob = 0.0;
    ActivationFunction_SignedStep_Prob = 0.0;
    ActivationFunction_UnsignedStep_Prob = 0.0;
    ActivationFunction_SignedGauss_Prob = 0.0;
    ActivationFunction_UnsignedGauss_Prob = 0.0;
    ActivationFunction_Abs_Prob = 0.0;
    ActivationFunction_SignedSine_Prob = 0.0;
    ActivationFunction_UnsignedSine_Prob = 0.0;
    ActivationFunction_Linear_Prob = 0.0;
    ActivationFunction_Relu_Prob = 0.0;
    ActivationFunction_Softplus_Prob = 0.0;
    ActivationFunction_SpikingLIF_Prob = 0.65;
    ActivationFunction_SpikingAdaptiveLIF_Prob = 0.25;
    ActivationFunction_SpikingIzhikevich_Prob = 0.10;
    ActivationFunction_McCullochPitts_Prob = 0.0;

    MutateNeuronActivationTypeProb = 0.05;
    MutateNeuronSpikingParametersProb = 0.25;
    MutateLinkSpikingParametersProb = 0.15;
    RecurrentProb = std::max(RecurrentProb, 0.2);
    AllowLoops = true;
    InitialSTDPEnabledProb = enable_stdp ? 0.1 : 0.0;
    SpikingNeuronDiffCoeff = 0.1;
    SpikingLinkDiffCoeff = 0.1;
}

void Parameters::ConfigureMcCullochPitts(
    bool inhibitory_veto,
    bool enable_stdp)
{
    ConfigureSpiking(enable_stdp);
    ActivationFunction_SpikingLIF_Prob = 0.0;
    ActivationFunction_SpikingAdaptiveLIF_Prob = 0.0;
    ActivationFunction_SpikingIzhikevich_Prob = 0.0;
    ActivationFunction_McCullochPitts_Prob = 1.0;
    InitialMCPInhibitoryVetoProb = inhibitory_veto ? 1.0 : 0.0;
    MutateMCPInhibitoryVetoProb = 0.05;
}

int Parameters::Load(std::ifstream& input)
{
    try
    {
        Parameters loaded;
        if (!ReadParameters(input, loaded))
            return 1;
        *this = std::move(loaded);
        return 0;
    }
    catch (const std::exception&)
    {
        return 1;
    }
}

int Parameters::Load(const char* filename)
{
    if (filename == nullptr)
        return 1;
    std::ifstream input(filename);
    if (!input)
        return 1;
    return Load(input);
}

void Parameters::Save(const char* filename)
{
    if (filename == nullptr)
        throw std::invalid_argument("Parameters::Save: filename is null.");
    FILE* file = detail::OpenFile(filename, "wb");
    if (file == nullptr)
        throw std::runtime_error("Parameters::Save: cannot open output file.");
    try
    {
        Save(file);
    }
    catch (...)
    {
        std::fclose(file);
        throw;
    }
    if (std::fclose(file) != 0)
        throw std::runtime_error("Parameters::Save: failed to close output file.");
}

void Parameters::Save(FILE* file)
{
    if (file == nullptr)
        throw std::invalid_argument("Parameters::Save: file is null.");
    const std::string data = Serialize();
    if (std::fwrite(data.data(), 1, data.size(), file) != data.size())
        throw std::runtime_error("Parameters::Save: failed to write output.");
}

std::string Parameters::Serialize() const
{
    std::ostringstream output;
    WriteParameters(output, *this);
    return output.str();
}

Parameters Parameters::Deserialize(const std::string& data)
{
    Parameters parameters;
    std::istringstream input(data);
    if (!ReadParameters(input, parameters))
        throw std::runtime_error(
            "Parameters::Deserialize: missing or incomplete parameter block.");
    return parameters;
}

void Parameters::SetCustomConstraintsFunction(
    std::function<bool(Genome&)> callback)
{
    m_CustomConstraintsFunction = std::move(callback);
    if (m_CustomConstraintsFunction)
        CustomConstraints = nullptr;
}

std::function<bool(Genome&)> Parameters::GetCustomConstraintsFunction() const
{
    if (m_CustomConstraintsFunction)
        return m_CustomConstraintsFunction;
    if (CustomConstraints != nullptr)
        return CustomConstraints;
    return {};
}

bool Parameters::FailsCustomConstraints(Genome& genome) const
{
    if (m_CustomConstraintsFunction)
        return m_CustomConstraintsFunction(genome);
    return CustomConstraints != nullptr && CustomConstraints(genome);
}

bool Parameters::Validate(std::string* error) const
{
    const auto fail = [error](const std::string& message)
    {
        if (error != nullptr)
            *error = message;
        return false;
    };
    const auto finite_range =
        [&fail](const char* name, double minimum, double maximum)
    {
        if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
            minimum > maximum)
        {
            return fail(
                std::string(name) +
                " must have a finite, ordered minimum and maximum");
        }
        return true;
    };
    const auto probability =
        [&fail](const char* name, double value)
    {
        if (!std::isfinite(value) || value < 0.0 || value > 1.0)
        {
            return fail(
                std::string(name) + " must be between 0 and 1");
        }
        return true;
    };

    if (PopulationSize == 0)
        return fail("PopulationSize must be greater than zero");
    if (PopulationSize >=
        static_cast<unsigned int>(std::numeric_limits<int>::max()))
        return fail("PopulationSize exceeds the supported ID range");
    if (MinSpecies == 0 || MinSpecies > MaxSpecies)
        return fail("MinSpecies and MaxSpecies must define a non-empty range");
    if (ConstraintTrials <= 0)
        return fail("ConstraintTrials must be greater than zero");
    if (NeuronTries <= 0)
        return fail("NeuronTries must be greater than zero");
    if (LinkTries == 0)
        return fail("LinkTries must be greater than zero");
    if (MaxLinks < -1 || MaxNeurons < -1)
        return fail("MaxLinks and MaxNeurons must be -1 or non-negative");
    if (InitialDepth > MaxDepth)
        return fail("InitialDepth cannot exceed MaxDepth");
    if (MaxDepth > 9)
        return fail("MaxDepth exceeds the supported safe limit of 9");
    if (Width <= 0.0 || Height <= 0.0 || Depth <= 0.0)
        return fail("Width, Height, and Depth must be positive");
    if ((TournamentSelection || ParentSelectionMode == TOURNAMENT) &&
        TournamentSize == 0)
        return fail(
            "TournamentSize must be greater than zero when tournament "
            "selection is enabled");
    if (ParentSelectionMode < LEGACY_SELECTION ||
        ParentSelectionMode > BOLTZMANN)
    {
        return fail("ParentSelectionMode is not a supported selection mode");
    }
    if (WeightMutationDistribution < UNIFORM_MUTATION ||
        WeightMutationDistribution > POLYNOMIAL_MUTATION)
    {
        return fail(
            "WeightMutationDistribution is not a supported mutation mode");
    }
    if (SpeciesRepresentativeSelection < FIRST_REPRESENTATIVE ||
        SpeciesRepresentativeSelection > MEDOID_REPRESENTATIVE)
    {
        return fail(
            "SpeciesRepresentativeSelection is not a supported mode");
    }
    if (OffspringAllocation < LARGEST_REMAINDER ||
        OffspringAllocation > STOCHASTIC_REMAINDER)
    {
        return fail("OffspringAllocation is not a supported mode");
    }
    if (CompatibilityThresholdControl <
            LEGACY_COMPATIBILITY_THRESHOLD ||
        CompatibilityThresholdControl >
            PROPORTIONAL_COMPATIBILITY_THRESHOLD)
    {
        return fail(
            "CompatibilityThresholdControl is not a supported mode");
    }
    if (FitnessScaling < SHIFTED_FITNESS_SCALING ||
        FitnessScaling > BOLTZMANN_FITNESS_SCALING)
    {
        return fail("FitnessScaling is not a supported mode");
    }
    if (MinSpeciesSize > PopulationSize)
        return fail("MinSpeciesSize cannot exceed PopulationSize");
    if (SpeciesElitism > PopulationSize)
        return fail("SpeciesElitism cannot exceed PopulationSize");
    if (MinSpeciesSize > 0 && SpeciesElitism > 0 &&
        static_cast<std::uint64_t>(MinSpeciesSize) *
                static_cast<std::uint64_t>(SpeciesElitism) >
            PopulationSize)
    {
        return fail(
            "MinSpeciesSize times SpeciesElitism cannot exceed "
            "PopulationSize");
    }
    if (TargetSpecies > PopulationSize)
        return fail("TargetSpecies cannot exceed PopulationSize");
    if (DetectCompetetiveCoevolutionStagnation &&
        (KillWorstSpeciesEach <= 0 || KillWorstAge < 0))
    {
        return fail(
            "competitive coevolution stagnation detection requires a "
            "positive interval and non-negative age");
    }

    const std::pair<const char*, double> probabilities[] = {
        {"SurvivalRate", SurvivalRate},
        {"CrossoverRate", CrossoverRate},
        {"OverallMutationRate", OverallMutationRate},
        {"InterspeciesCrossoverRate", InterspeciesCrossoverRate},
        {"MultipointCrossoverRate", MultipointCrossoverRate},
        {"SinglePointCrossoverRate", SinglePointCrossoverRate},
        {"BlendCrossoverRate", BlendCrossoverRate},
        {"SimulatedBinaryCrossoverRate",
         SimulatedBinaryCrossoverRate},
        {"PreferFitterParentRate", PreferFitterParentRate},
        {"EliteFraction", EliteFraction},
        {"MutateAddNeuronProb", MutateAddNeuronProb},
        {"MutateAddLinkProb", MutateAddLinkProb},
        {"MutateAddLinkFromBiasProb", MutateAddLinkFromBiasProb},
        {"MutateRemLinkProb", MutateRemLinkProb},
        {"MutateRemSimpleNeuronProb", MutateRemSimpleNeuronProb},
        {"RecurrentProb", RecurrentProb},
        {"RecurrentLoopProb", RecurrentLoopProb},
        {"MutateWeightsProb", MutateWeightsProb},
        {"MutateWeightsSevereProb", MutateWeightsSevereProb},
        {"WeightMutationRate", WeightMutationRate},
        {"WeightReplacementRate", WeightReplacementRate},
        {"MutateActivationAProb", MutateActivationAProb},
        {"MutateActivationBProb", MutateActivationBProb},
        {"MutateNeuronActivationTypeProb",
         MutateNeuronActivationTypeProb},
        {"MutateNeuronTimeConstantsProb",
         MutateNeuronTimeConstantsProb},
        {"MutateNeuronBiasesProb", MutateNeuronBiasesProb},
        {"MutateNeuronTraitsProb", MutateNeuronTraitsProb},
        {"MutateLinkTraitsProb", MutateLinkTraitsProb},
        {"MutateGenomeTraitsProb", MutateGenomeTraitsProb},
        {"ActivationFunction_SignedSigmoid_Prob",
         ActivationFunction_SignedSigmoid_Prob},
        {"ActivationFunction_UnsignedSigmoid_Prob",
         ActivationFunction_UnsignedSigmoid_Prob},
        {"ActivationFunction_Tanh_Prob", ActivationFunction_Tanh_Prob},
        {"ActivationFunction_TanhCubic_Prob",
         ActivationFunction_TanhCubic_Prob},
        {"ActivationFunction_SignedStep_Prob",
         ActivationFunction_SignedStep_Prob},
        {"ActivationFunction_UnsignedStep_Prob",
         ActivationFunction_UnsignedStep_Prob},
        {"ActivationFunction_SignedGauss_Prob",
         ActivationFunction_SignedGauss_Prob},
        {"ActivationFunction_UnsignedGauss_Prob",
         ActivationFunction_UnsignedGauss_Prob},
        {"ActivationFunction_Abs_Prob", ActivationFunction_Abs_Prob},
        {"ActivationFunction_SignedSine_Prob",
         ActivationFunction_SignedSine_Prob},
        {"ActivationFunction_UnsignedSine_Prob",
         ActivationFunction_UnsignedSine_Prob},
        {"ActivationFunction_Linear_Prob",
         ActivationFunction_Linear_Prob},
        {"ActivationFunction_Relu_Prob", ActivationFunction_Relu_Prob},
        {"ActivationFunction_Softplus_Prob",
         ActivationFunction_Softplus_Prob},
        {"ActivationFunction_SpikingLIF_Prob",
         ActivationFunction_SpikingLIF_Prob},
        {"ActivationFunction_SpikingAdaptiveLIF_Prob",
         ActivationFunction_SpikingAdaptiveLIF_Prob},
        {"ActivationFunction_SpikingIzhikevich_Prob",
         ActivationFunction_SpikingIzhikevich_Prob},
        {"ActivationFunction_McCullochPitts_Prob",
         ActivationFunction_McCullochPitts_Prob},
        {"MutateNeuronSpikingParametersProb",
         MutateNeuronSpikingParametersProb},
        {"MutateLinkSpikingParametersProb",
         MutateLinkSpikingParametersProb},
        {"SpikingParameterMutationRate",
         SpikingParameterMutationRate},
        {"InitialMCPInhibitoryVetoProb",
         InitialMCPInhibitoryVetoProb},
        {"MutateMCPInhibitoryVetoProb",
         MutateMCPInhibitoryVetoProb},
        {"InitialSTDPEnabledProb", InitialSTDPEnabledProb}};
    for (const auto& item : probabilities)
    {
        if (!probability(item.first, item.second))
            return false;
    }
    const double crossover_mode_total =
        MultipointCrossoverRate + SinglePointCrossoverRate +
        BlendCrossoverRate + SimulatedBinaryCrossoverRate;
    if (!std::isfinite(crossover_mode_total) ||
        crossover_mode_total > 1.0 + 1.0e-12)
    {
        return fail(
            "crossover method probabilities must sum to at most 1");
    }

    if (!finite_range("weight range", MinWeight, MaxWeight) ||
        !finite_range(
            "activation A range", MinActivationA, MaxActivationA) ||
        !finite_range(
            "activation B range", MinActivationB, MaxActivationB) ||
        !finite_range(
            "neuron time-constant range",
            MinNeuronTimeConstant,
            MaxNeuronTimeConstant) ||
        !finite_range(
            "neuron bias range", MinNeuronBias, MaxNeuronBias))
    {
        return false;
    }
    const std::pair<const char*, std::pair<double, double>>
        spiking_ranges[] = {
            {"spiking time-constant range",
             {MinSpikingTimeConstant, MaxSpikingTimeConstant}},
            {"spike-threshold range",
             {MinSpikeThreshold, MaxSpikeThreshold}},
            {"reset-potential range",
             {MinResetPotential, MaxResetPotential}},
            {"resting-potential range",
             {MinRestingPotential, MaxRestingPotential}},
            {"refractory-period range",
             {MinRefractoryPeriod, MaxRefractoryPeriod}},
            {"membrane-resistance range",
             {MinMembraneResistance, MaxMembraneResistance}},
            {"adaptation time-constant range",
             {MinAdaptationTimeConstant,
              MaxAdaptationTimeConstant}},
            {"adaptation-increment range",
             {MinAdaptationIncrement, MaxAdaptationIncrement}},
            {"spike-rate time-constant range",
             {MinSpikeRateTimeConstant,
              MaxSpikeRateTimeConstant}},
            {"Izhikevich a range",
             {MinIzhikevichA, MaxIzhikevichA}},
            {"Izhikevich threshold range",
             {MinIzhikevichThreshold,
              MaxIzhikevichThreshold}},
            {"Izhikevich b range",
             {MinIzhikevichB, MaxIzhikevichB}},
            {"Izhikevich c range",
             {MinIzhikevichC, MaxIzhikevichC}},
            {"Izhikevich d range",
             {MinIzhikevichD, MaxIzhikevichD}},
            {"synaptic-delay range",
             {MinSynapticDelay, MaxSynapticDelay}},
            {"synaptic time-constant range",
             {MinSynapticTimeConstant,
              MaxSynapticTimeConstant}},
            {"STDP potentiation range",
             {MinSTDPPlus, MaxSTDPPlus}},
            {"STDP depression range",
             {MinSTDPMinus, MaxSTDPMinus}},
            {"STDP trace time-constant range",
             {MinSTDPTau, MaxSTDPTau}}};
    for (const auto& range : spiking_ranges)
    {
        if (!finite_range(
                range.first,
                range.second.first,
                range.second.second))
        {
            return false;
        }
    }
    if (MinSpikingTimeConstant <= 0.0 ||
        MinSynapticTimeConstant <= 0.0 ||
        MinAdaptationTimeConstant <= 0.0 ||
        MinSpikeRateTimeConstant <= 0.0 ||
        MinSTDPTau <= 0.0 ||
        MinRefractoryPeriod < 0.0 ||
        MinMembraneResistance <= 0.0 ||
        MinSynapticDelay < 0.0 ||
        MinSTDPPlus < 0.0 ||
        MinSTDPMinus < 0.0)
    {
        return fail(
            "spiking time constants and resistance must be positive; "
            "delays, refractory periods, and STDP amplitudes cannot be "
            "negative");
    }

    const std::pair<const char*, double> non_negative[] = {
        {"YoungAgeFitnessBoost", YoungAgeFitnessBoost},
        {"StagnationDelta", StagnationDelta},
        {"OldAgePenalty", OldAgePenalty},
        {"WeightMutationMaxPower", WeightMutationMaxPower},
        {"WeightReplacementMaxPower", WeightReplacementMaxPower},
        {"ActivationAMutationMaxPower", ActivationAMutationMaxPower},
        {"ActivationBMutationMaxPower", ActivationBMutationMaxPower},
        {"TimeConstantMutationMaxPower", TimeConstantMutationMaxPower},
        {"BiasMutationMaxPower", BiasMutationMaxPower},
        {"CrossoverBlendAlpha", CrossoverBlendAlpha},
        {"CrossoverSBXEta", CrossoverSBXEta},
        {"WeightMutationPolynomialEta", WeightMutationPolynomialEta},
        {"DisjointCoeff", DisjointCoeff},
        {"ExcessCoeff", ExcessCoeff},
        {"ActivationADiffCoeff", ActivationADiffCoeff},
        {"ActivationBDiffCoeff", ActivationBDiffCoeff},
        {"WeightDiffCoeff", WeightDiffCoeff},
        {"TimeConstantDiffCoeff", TimeConstantDiffCoeff},
        {"BiasDiffCoeff", BiasDiffCoeff},
        {"ActivationFunctionDiffCoeff", ActivationFunctionDiffCoeff},
        {"SpikingNeuronDiffCoeff", SpikingNeuronDiffCoeff},
        {"SpikingLinkDiffCoeff", SpikingLinkDiffCoeff},
        {"CompatTreshold", CompatTreshold},
        {"MinCompatTreshold", MinCompatTreshold},
        {"CompatTresholdModifier", CompatTresholdModifier},
        {"MinDeltaCompatEqualGenomes", MinDeltaCompatEqualGenomes},
        {"NoveltySearch_P_min", NoveltySearch_P_min},
        {"NoveltySearch_Pmin_min", NoveltySearch_Pmin_min},
        {"DivisionThreshold", DivisionThreshold},
        {"VarianceThreshold", VarianceThreshold},
        {"BandThreshold", BandThreshold},
        {"SpikingParameterMutationPower",
         SpikingParameterMutationPower}};
    for (const auto& item : non_negative)
    {
        if (!std::isfinite(item.second) || item.second < 0.0)
            return fail(std::string(item.first) +
                        " must be finite and non-negative");
    }
    const std::pair<const char*, double> new_non_negative[] = {
        {"StagnationPenalty", StagnationPenalty},
        {"CompatibilityThresholdGain",
         CompatibilityThresholdGain},
        {"AdaptiveMutationRate", AdaptiveMutationRate}};
    for (const auto& item : new_non_negative)
    {
        if (!std::isfinite(item.second) || item.second < 0.0)
            return fail(std::string(item.first) +
                        " must be finite and non-negative");
    }
    if (!std::isfinite(MaxCompatTreshold) ||
        MaxCompatTreshold < MinCompatTreshold)
    {
        return fail(
            "MaxCompatTreshold must be finite and at least "
            "MinCompatTreshold");
    }
    if (CompatTreshold > MaxCompatTreshold)
        return fail("CompatTreshold cannot exceed MaxCompatTreshold");
    if (!std::isfinite(MutationOperatorsPerOffspring) ||
        MutationOperatorsPerOffspring < 1.0 ||
        MutationOperatorsPerOffspring > 256.0)
    {
        return fail(
            "MutationOperatorsPerOffspring must be finite and in [1, 256]");
    }
    if (!std::isfinite(AdaptiveMutationMaxFactor) ||
        AdaptiveMutationMaxFactor < 1.0 ||
        AdaptiveMutationMaxFactor > 256.0 ||
        MutationOperatorsPerOffspring *
                AdaptiveMutationMaxFactor >
            1024.0)
    {
        return fail(
            "Adaptive mutation settings exceed the supported operator "
            "budget");
    }
    if (!std::isfinite(RankSelectionPressure) ||
        RankSelectionPressure < 1.0 ||
        RankSelectionPressure > 2.0)
    {
        return fail("RankSelectionPressure must be between 1 and 2");
    }
    if (!std::isfinite(FitnessRankPressure) ||
        FitnessRankPressure < 1.0 ||
        FitnessRankPressure > 2.0)
    {
        return fail("FitnessRankPressure must be between 1 and 2");
    }
    const std::pair<const char*, double> positive_values[] = {
        {"RankSelectionExponent", RankSelectionExponent},
        {"BoltzmannTemperature", BoltzmannTemperature},
        {"WeightMutationSigma", WeightMutationSigma},
        {"WeightMutationCauchyScale", WeightMutationCauchyScale},
        {"FitnessSigmaScale", FitnessSigmaScale},
        {"FitnessBoltzmannTemperature",
         FitnessBoltzmannTemperature}};
    for (const auto& item : positive_values)
    {
        if (!std::isfinite(item.second) || item.second <= 0.0)
            return fail(std::string(item.first) +
                        " must be finite and positive");
    }
    if (!std::isfinite(NoveltySearch_Pmin_lowering_multiplier) ||
        NoveltySearch_Pmin_lowering_multiplier <= 0.0 ||
        !std::isfinite(NoveltySearch_Pmin_raising_multiplier) ||
        NoveltySearch_Pmin_raising_multiplier <= 0.0)
    {
        return fail(
            "novelty threshold multipliers must be finite and positive");
    }
    const std::pair<const char*, double> finite_values[] = {
        {"CPPN_Bias", CPPN_Bias},
        {"Width", Width},
        {"Height", Height},
        {"Depth", Depth},
        {"Qtree_X", Qtree_X},
        {"Qtree_Y", Qtree_Y},
        {"Qtree_Z", Qtree_Z},
        {"LeoThreshold", LeoThreshold}};
    for (const auto& item : finite_values)
    {
        if (!std::isfinite(item.second))
            return fail(std::string(item.first) + " must be finite");
    }

    const double activation_total =
        ActivationFunction_SignedSigmoid_Prob +
        ActivationFunction_UnsignedSigmoid_Prob +
        ActivationFunction_Tanh_Prob +
        ActivationFunction_TanhCubic_Prob +
        ActivationFunction_SignedStep_Prob +
        ActivationFunction_UnsignedStep_Prob +
        ActivationFunction_SignedGauss_Prob +
        ActivationFunction_UnsignedGauss_Prob +
        ActivationFunction_Abs_Prob +
        ActivationFunction_SignedSine_Prob +
        ActivationFunction_UnsignedSine_Prob +
        ActivationFunction_Linear_Prob +
        ActivationFunction_Relu_Prob +
        ActivationFunction_Softplus_Prob +
        ActivationFunction_SpikingLIF_Prob +
        ActivationFunction_SpikingAdaptiveLIF_Prob +
        ActivationFunction_SpikingIzhikevich_Prob +
        ActivationFunction_McCullochPitts_Prob;
    if ((MutateAddNeuronProb > 0.0 ||
         MutateNeuronActivationTypeProb > 0.0) &&
        activation_total <= 0.0)
    {
        return fail(
            "at least one activation function must have positive probability");
    }

    const auto validate_traits =
        [&fail, &probability](
            const char* category,
            const std::map<std::string, TraitParameters>& schemas)
    {
        for (const auto& entry : schemas)
        {
            const TraitParameters& schema = entry.second;
            const std::string prefix =
                std::string(category) + " trait '" + entry.first + "': ";
            const auto validate_set_probabilities =
                [&fail, &prefix](
                    std::size_t set_size,
                    const std::vector<double>& probabilities)
            {
                if (!probabilities.empty() &&
                    probabilities.size() != set_size)
                {
                    return fail(
                        prefix +
                        "probability count must match the set size");
                }
                for (const double value : probabilities)
                {
                    if (!std::isfinite(value) || value < 0.0)
                    {
                        return fail(
                            prefix +
                            "set probabilities must be finite and "
                            "non-negative");
                    }
                }
                return true;
            };
            if (!std::isfinite(schema.m_ImportanceCoeff) ||
                schema.m_ImportanceCoeff < 0.0)
            {
                return fail(prefix +
                            "importance must be finite and non-negative");
            }
            if (!probability(
                    (prefix + "mutation probability").c_str(),
                    schema.m_MutationProb))
            {
                return false;
            }
            if (schema.type == "int")
            {
                if (!std::holds_alternative<IntTraitParameters>(
                        schema.m_Details))
                    return fail(prefix + "detail type does not match");
                const auto& detail =
                    std::get<IntTraitParameters>(schema.m_Details);
                if (detail.min > detail.max || detail.mut_power < 0)
                    return fail(prefix + "integer range is invalid");
                if (!probability(
                        (prefix + "replacement probability").c_str(),
                        detail.mut_replace_prob))
                    return false;
            }
            else if (schema.type == "float")
            {
                if (!std::holds_alternative<FloatTraitParameters>(
                        schema.m_Details))
                    return fail(prefix + "detail type does not match");
                const auto& detail =
                    std::get<FloatTraitParameters>(schema.m_Details);
                if (!std::isfinite(detail.min) ||
                    !std::isfinite(detail.max) ||
                    detail.min > detail.max ||
                    !std::isfinite(detail.mut_power) ||
                    detail.mut_power < 0.0)
                    return fail(prefix + "floating-point range is invalid");
                if (!probability(
                        (prefix + "replacement probability").c_str(),
                        detail.mut_replace_prob))
                    return false;
            }
            else if (schema.type == "str")
            {
                if (!std::holds_alternative<StringTraitParameters>(
                        schema.m_Details))
                    return fail(prefix + "detail type does not match");
                const auto& detail =
                    std::get<StringTraitParameters>(schema.m_Details);
                if (detail.set.empty())
                    return fail(prefix + "set cannot be empty");
                if (!validate_set_probabilities(
                        detail.set.size(), detail.probs))
                    return false;
            }
            else if (schema.type == "intset")
            {
                if (!std::holds_alternative<IntSetTraitParameters>(
                        schema.m_Details))
                    return fail(prefix + "detail type does not match");
                const auto& detail =
                    std::get<IntSetTraitParameters>(schema.m_Details);
                if (detail.set.empty())
                    return fail(prefix + "set cannot be empty");
                if (!validate_set_probabilities(
                        detail.set.size(), detail.probs))
                    return false;
            }
            else if (schema.type == "floatset")
            {
                if (!std::holds_alternative<FloatSetTraitParameters>(
                        schema.m_Details))
                    return fail(prefix + "detail type does not match");
                const auto& detail =
                    std::get<FloatSetTraitParameters>(schema.m_Details);
                if (detail.set.empty())
                    return fail(prefix + "set cannot be empty");
                if (!validate_set_probabilities(
                        detail.set.size(), detail.probs))
                    return false;
                for (const auto& value : detail.set)
                {
                    if (!std::isfinite(value.value))
                        return fail(
                            prefix +
                            "set values must be finite");
                }
            }
            else
            {
                return fail(prefix + "unsupported type");
            }
            if (!schema.dep_key.empty())
            {
                if (schemas.find(schema.dep_key) == schemas.end())
                    return fail(prefix + "dependency key does not exist");
                if (schema.dep_values.empty())
                    return fail(
                        prefix + "dependency values cannot be empty");
            }
        }
        return true;
    };
    return validate_traits("neuron", NeuronTraits) &&
           validate_traits("link", LinkTraits) &&
           validate_traits("genome", GenomeTraits);
}

} // namespace NEAT
