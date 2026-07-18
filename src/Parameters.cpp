#include "Parameters.h"

#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
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
    X(MutateNeuronTimeConstantsProb)                                        \
    X(MutateNeuronBiasesProb)                                               \
    X(MinNeuronTimeConstant)                                                \
    X(MaxNeuronTimeConstant)                                                \
    X(MinNeuronBias)                                                        \
    X(MaxNeuronBias)                                                        \
    X(DisjointCoeff)                                                        \
    X(ExcessCoeff)                                                          \
    X(ActivationADiffCoeff)                                                 \
    X(ActivationBDiffCoeff)                                                 \
    X(WeightDiffCoeff)                                                      \
    X(TimeConstantDiffCoeff)                                                \
    X(BiasDiffCoeff)                                                        \
    X(ActivationFunctionDiffCoeff)                                          \
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
    X(Qtree_X)                                                              \
    X(Qtree_Y)                                                              \
    X(Leo)                                                                  \
    X(LeoThreshold)                                                         \
    X(LeoSeed)                                                              \
    X(GeometrySeed)                                                         \
    X(MutateNeuronTraitsProb)                                               \
    X(MutateLinkTraitsProb)                                                 \
    X(MutateGenomeTraitsProb)

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
    input >> value;
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
    MutateNeuronTimeConstantsProb = 0.0;
    MutateNeuronBiasesProb = 0.0;
    MinNeuronTimeConstant = 0.0;
    MaxNeuronTimeConstant = 0.0;
    MinNeuronBias = 0.0;
    MaxNeuronBias = 0.0;

    DisjointCoeff = 1.0;
    ExcessCoeff = 1.0;
    ActivationADiffCoeff = 0.0;
    ActivationBDiffCoeff = 0.0;
    WeightDiffCoeff = 0.1;
    TimeConstantDiffCoeff = 0.0;
    BiasDiffCoeff = 0.0;
    ActivationFunctionDiffCoeff = 0.0;
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
    Qtree_X = 0.0;
    Qtree_Y = 0.0;
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

} // namespace NEAT
