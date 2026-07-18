#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "Genome.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Population.h"
#include "Random.h"
#include "Species.h"
#include "Substrate.h"
#include "Utils.h"

namespace
{
int failures = 0;

void Check(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Exception, typename Callable>
void CheckThrows(Callable&& callable, const std::string& message)
{
    try
    {
        callable();
        Check(false, message);
    }
    catch (const Exception&)
    {
    }
}

NEAT::Genome MakeGenome(NEAT::Parameters& parameters)
{
    NEAT::GenomeInitStruct init;
    init.NumInputs = 3;
    init.NumOutputs = 1;
    return NEAT::Genome(parameters, init);
}

class ScalarBehavior : public NEAT::PhenotypeBehavior
{
public:
    explicit ScalarBehavior(double value)
        : m_Value(value)
    {
        m_Data = {{value}};
    }

    double Distance_To(NEAT::PhenotypeBehavior* other) override
    {
        const auto* scalar = dynamic_cast<ScalarBehavior*>(other);
        if (scalar == nullptr)
            throw std::invalid_argument("incompatible behavior type");
        return std::abs(m_Value - scalar->m_Value);
    }

private:
    double m_Value;
};
}

int main()
{
    using namespace NEAT;

    Parameters parameters;
    Check(parameters.DivisionThreshold == 0.03, "ES-HyperNEAT defaults are initialized");
    parameters.PopulationSize = 4;
    parameters.MutateAddNeuronProb = 0.42;
    parameters.DivisionThreshold = 0.77;
    parameters.MutateGenomeTraitsProb = 0.25;
    TraitParameters label_parameters;
    label_parameters.m_ImportanceCoeff = 0.8;
    label_parameters.m_MutationProb = 0.3;
    label_parameters.type = "str";
    StringTraitParameters label_details;
    label_details.set = {"alpha beta", "gamma"};
    label_details.probs = {0.25, 0.75};
    label_parameters.m_Details = label_details;
    label_parameters.dep_key = "enabled";
    label_parameters.dep_values = {1, std::string("yes")};
    TraitParameters enabled_parameters;
    enabled_parameters.type = "int";
    enabled_parameters.m_Details = IntTraitParameters();
    parameters.GenomeTraits["enabled"] = enabled_parameters;
    parameters.GenomeTraits["label"] = label_parameters;
    parameters.Elitism = 0.125;
    Check(parameters.EliteFraction == 0.125,
          "legacy Elitism aliases EliteFraction");
    const Parameters restored_parameters =
        Parameters::Deserialize(parameters.Serialize());
    Check(restored_parameters.PopulationSize == 4, "parameter population size round-trips");
    Check(restored_parameters.MutateAddNeuronProb == 0.42,
          "mutation parameters round-trip");
    Check(restored_parameters.DivisionThreshold == 0.77,
          "ES-HyperNEAT parameters round-trip");
    Check(restored_parameters.MutateGenomeTraitsProb == 0.25,
          "trait mutation parameters round-trip");
    std::string parameter_validation_error;
    Check(parameters.Validate(&parameter_validation_error),
          "default and customized parameters pass validation");
    Parameters advanced_parameters = parameters;
    advanced_parameters.ParentSelectionMode = RANK_EXP;
    advanced_parameters.RankSelectionPressure = 1.9;
    advanced_parameters.RankSelectionExponent = 2.5;
    advanced_parameters.BoltzmannTemperature = 0.75;
    advanced_parameters.MultipointCrossoverRate = 0.2;
    advanced_parameters.SinglePointCrossoverRate = 0.2;
    advanced_parameters.BlendCrossoverRate = 0.2;
    advanced_parameters.SimulatedBinaryCrossoverRate = 0.2;
    advanced_parameters.CrossoverBlendAlpha = 0.25;
    advanced_parameters.CrossoverSBXEta = 15.0;
    advanced_parameters.WeightMutationDistribution =
        POLYNOMIAL_MUTATION;
    advanced_parameters.WeightMutationSigma = 0.75;
    advanced_parameters.WeightMutationCauchyScale = 0.5;
    advanced_parameters.WeightMutationPolynomialEta = 12.0;
    advanced_parameters.SpeciesRepresentativeSelection =
        MEDOID_REPRESENTATIVE;
    advanced_parameters.RepresentativeSelectionCandidates = 8;
    advanced_parameters.OffspringAllocation =
        STOCHASTIC_REMAINDER;
    advanced_parameters.MinSpeciesSize = 2;
    advanced_parameters.SpeciesElitism = 2;
    advanced_parameters.StagnationPenalty = 0.01;
    advanced_parameters.CompatibilityThresholdControl =
        PROPORTIONAL_COMPATIBILITY_THRESHOLD;
    advanced_parameters.TargetSpecies = 2;
    advanced_parameters.CompatibilityThresholdGain = 0.4;
    advanced_parameters.MaxCompatTreshold = 20.0;
    advanced_parameters.RequireEvaluatedGenomes = true;
    advanced_parameters.RejectNonFiniteFitness = true;
    advanced_parameters.MutationOperatorsPerOffspring = 1.5;
    advanced_parameters.AdaptiveMutationStart = 10;
    advanced_parameters.AdaptiveMutationRate = 0.05;
    advanced_parameters.AdaptiveMutationMaxFactor = 3.0;
    advanced_parameters.FitnessScaling = SIGMA_FITNESS_SCALING;
    advanced_parameters.FitnessRankPressure = 1.75;
    advanced_parameters.FitnessSigmaScale = 2.5;
    advanced_parameters.FitnessBoltzmannTemperature = 0.6;
    const Parameters restored_advanced =
        Parameters::Deserialize(advanced_parameters.Serialize());
    Check(
        restored_advanced.ParentSelectionMode == RANK_EXP &&
            restored_advanced.SinglePointCrossoverRate == 0.2 &&
            restored_advanced.WeightMutationDistribution ==
                POLYNOMIAL_MUTATION &&
            restored_advanced.WeightMutationPolynomialEta == 12.0 &&
            restored_advanced.SpeciesRepresentativeSelection ==
                MEDOID_REPRESENTATIVE &&
            restored_advanced.OffspringAllocation ==
                STOCHASTIC_REMAINDER &&
            restored_advanced.MinSpeciesSize == 2 &&
            restored_advanced.CompatibilityThresholdControl ==
                PROPORTIONAL_COMPATIBILITY_THRESHOLD &&
            restored_advanced.MutationOperatorsPerOffspring == 1.5 &&
            restored_advanced.FitnessScaling ==
                SIGMA_FITNESS_SCALING &&
            restored_advanced.FitnessSigmaScale == 2.5,
        "advanced algorithm controls round-trip");
    Check(restored_advanced.Validate(&parameter_validation_error),
          "advanced algorithm controls validate");
    std::string legacy_parameter_state = parameters.Serialize();
    for (const std::string field :
         {"ParentSelectionMode",
          "RankSelectionPressure",
          "RankSelectionExponent",
          "BoltzmannTemperature",
          "SinglePointCrossoverRate",
          "BlendCrossoverRate",
          "SimulatedBinaryCrossoverRate",
          "CrossoverBlendAlpha",
          "CrossoverSBXEta",
          "WeightMutationDistribution",
          "WeightMutationSigma",
          "WeightMutationCauchyScale",
          "WeightMutationPolynomialEta",
          "SpeciesRepresentativeSelection",
          "RepresentativeSelectionCandidates",
          "OffspringAllocation",
          "MinSpeciesSize",
          "SpeciesElitism",
          "StagnationPenalty",
          "CompatibilityThresholdControl",
          "TargetSpecies",
          "CompatibilityThresholdGain",
          "MaxCompatTreshold",
          "RequireEvaluatedGenomes",
          "RejectNonFiniteFitness",
          "MutationOperatorsPerOffspring",
          "AdaptiveMutationStart",
          "AdaptiveMutationRate",
          "AdaptiveMutationMaxFactor",
          "FitnessScaling",
          "FitnessRankPressure",
          "FitnessSigmaScale",
          "FitnessBoltzmannTemperature"})
    {
        const std::size_t start =
            legacy_parameter_state.find(field + " ");
        if (start != std::string::npos)
        {
            const std::size_t end =
                legacy_parameter_state.find('\n', start);
            legacy_parameter_state.erase(
                start,
                end == std::string::npos
                    ? std::string::npos
                    : end - start + 1);
        }
    }
    const Parameters legacy_advanced_defaults =
        Parameters::Deserialize(legacy_parameter_state);
    Check(
        legacy_advanced_defaults.ParentSelectionMode ==
                LEGACY_SELECTION &&
            legacy_advanced_defaults.SinglePointCrossoverRate == 0.0 &&
            legacy_advanced_defaults.WeightMutationDistribution ==
                UNIFORM_MUTATION &&
            legacy_advanced_defaults.SpeciesRepresentativeSelection ==
                FIRST_REPRESENTATIVE &&
            legacy_advanced_defaults.OffspringAllocation ==
                LARGEST_REMAINDER &&
            legacy_advanced_defaults.MutationOperatorsPerOffspring ==
                1.0 &&
            legacy_advanced_defaults.FitnessScaling ==
                SHIFTED_FITNESS_SCALING,
        "parameter checkpoints predating advanced operators load defaults");
    Parameters invalid_crossover_parameters = advanced_parameters;
    invalid_crossover_parameters.MultipointCrossoverRate = 0.5;
    Check(!invalid_crossover_parameters.Validate(
              &parameter_validation_error),
          "crossover method probabilities reject totals above one");
    Parameters invalid_temperature_parameters = advanced_parameters;
    invalid_temperature_parameters.BoltzmannTemperature = 0.0;
    Check(!invalid_temperature_parameters.Validate(
              &parameter_validation_error),
          "Boltzmann selection rejects a non-positive temperature");
    Parameters invalid_species_floor = advanced_parameters;
    invalid_species_floor.MinSpeciesSize = 3;
    Check(!invalid_species_floor.Validate(
              &parameter_validation_error),
          "species floors reject an impossible protected allocation");
    Parameters invalid_parameters = parameters;
    invalid_parameters.SurvivalRate = 1.5;
    Check(!invalid_parameters.Validate(&parameter_validation_error),
          "parameter validation rejects invalid probabilities");
    CheckThrows<std::invalid_argument>(
        [&] {
            Population invalid_population(
                MakeGenome(invalid_parameters),
                invalid_parameters,
                true,
                1.0,
                1);
        },
        "population construction rejects invalid parameters early");
    const auto& restored_label = restored_parameters.GenomeTraits.at("label");
    const auto& restored_label_details =
        std::get<StringTraitParameters>(restored_label.m_Details);
    Check(restored_label_details.set.front() == "alpha beta" &&
              restored_label.dep_values.size() == 2,
          "universal trait schemas and dependencies round-trip");
    const auto temporary_directory =
        std::filesystem::temp_directory_path();
    const auto parameters_path =
        temporary_directory / "multineat_parameters_legacy.txt";
    parameters.Save(parameters_path.string().c_str());
    Parameters legacy_parameters;
    Check(legacy_parameters.Load(
              parameters_path.string().c_str()) == 0 &&
              legacy_parameters.PopulationSize ==
                  parameters.PopulationSize,
          "legacy parameter files remain loadable");
    std::filesystem::remove(parameters_path);

    std::vector<double> values{2.0, 4.0, 6.0};
    Scale(values, -1.0, 1.0);
    Check(values.front() == -1.0 && values.back() == 1.0,
          "vector scaling respects its requested output range");

    RNG rng;
    rng.Seed(123);
    CheckThrows<std::invalid_argument>(
        [&rng] { rng.Roulette({}); },
        "roulette rejects empty probability vectors");
    CheckThrows<std::invalid_argument>(
        [&rng] { rng.Roulette({1.0, -0.1}); },
        "roulette rejects negative probabilities");
    Check(
        rng.Roulette(
            {std::numeric_limits<double>::max(),
             std::numeric_limits<double>::max()}) >= 0,
        "roulette normalizes extreme finite weights without overflow");
    Parameters no_activation_parameters;
    double Parameters::* const activation_probabilities[] = {
        &Parameters::ActivationFunction_SignedSigmoid_Prob,
        &Parameters::ActivationFunction_UnsignedSigmoid_Prob,
        &Parameters::ActivationFunction_Tanh_Prob,
        &Parameters::ActivationFunction_TanhCubic_Prob,
        &Parameters::ActivationFunction_SignedStep_Prob,
        &Parameters::ActivationFunction_UnsignedStep_Prob,
        &Parameters::ActivationFunction_SignedGauss_Prob,
        &Parameters::ActivationFunction_UnsignedGauss_Prob,
        &Parameters::ActivationFunction_Abs_Prob,
        &Parameters::ActivationFunction_SignedSine_Prob,
        &Parameters::ActivationFunction_UnsignedSine_Prob,
        &Parameters::ActivationFunction_Linear_Prob,
        &Parameters::ActivationFunction_Relu_Prob,
        &Parameters::ActivationFunction_Softplus_Prob};
    for (const auto probability : activation_probabilities)
        no_activation_parameters.*probability = 0.0;
    CheckThrows<std::invalid_argument>(
        [&] {
            (void)GetRandomActivation(no_activation_parameters, rng);
        },
        "activation selection rejects an empty distribution");
    rng.Seed(77);
    (void)rng.RandFloat();
    RNG restored_rng;
    restored_rng.Deserialize(rng.Serialize());
    Check(rng.RandFloat() == restored_rng.RandFloat(),
          "random-generator state round-trips exactly");
    Check(std::isfinite(rng.RandNormal()) &&
              std::isfinite(rng.RandCauchy()),
          "unbounded mutation distributions return finite samples");
    CheckThrows<std::invalid_argument>(
        [&rng] { (void)rng.RandNormal(0.0, 0.0); },
        "normal sampling validates its scale");

    TraitParameters fixed_trait_parameters;
    fixed_trait_parameters.m_MutationProb = 1.0;
    fixed_trait_parameters.type = "int";
    IntTraitParameters fixed_details;
    fixed_details.min = 4;
    fixed_details.max = 4;
    fixed_details.mut_replace_prob = 1.0;
    fixed_trait_parameters.m_Details = fixed_details;
    std::map<std::string, TraitParameters> fixed_schema{
        {"fixed", fixed_trait_parameters}};
    Gene fixed_gene;
    fixed_gene.InitTraits(fixed_schema, rng);
    Check(!fixed_gene.MutateTraits(fixed_schema, rng),
          "a singleton trait reports no mutation instead of looping");

    NeuralNetwork network;
    Neuron input;
    input.m_type = INPUT;
    Neuron output;
    output.m_type = OUTPUT;
    output.m_activation_function_type = LINEAR;
    Connection connection;
    connection.m_source_neuron_idx = 0;
    connection.m_target_neuron_idx = 1;
    connection.m_weight = 2.0;
    network.AddNeuron(input);
    network.AddNeuron(output);
    network.AddConnection(connection);
    network.SetInputOutputDimentions(1, 1);
    std::vector<double> input_values{3.0};
    network.Input(input_values);
    network.Activate();
    Check(network.Output().front() == 6.0, "neural activation produces expected output");
    network.Flush();
    network.Input(input_values);
    network.ActivateFast();
    Check(network.Output().front() == 6.0,
          "fast neural activation matches the checked path");
    network.Flush();
    network.Input(input_values);
    network.ActivateSteps(2, false);
    Check(network.Output().front() == 6.0,
          "multi-step activation validates once and advances the network");
    NeuralNetwork restored_network =
        NeuralNetwork::Deserialize(network.Serialize());
    Check(restored_network.Output().front() == 6.0,
          "neural-network runtime state round-trips");
    const auto network_path =
        temporary_directory / "multineat_network_legacy.txt";
    network.Save(network_path.string().c_str());
    NeuralNetwork legacy_network;
    Check(legacy_network.Load(network_path.string().c_str()) &&
              legacy_network.m_connections.size() == 1,
          "legacy neural-network files remain loadable");
    std::filesystem::remove(network_path);
    input.m_x = 0.0;
    output.m_x = 3.0;
    output.m_y = 4.0;
    Check(network.GetConnectionLength(input, output) == 5.0,
          "connection length API is implemented");
    CheckThrows<std::runtime_error>(
        [] {
            NeuralNetwork invalid;
            invalid.SetInputOutputDimensions(1, 1);
            invalid.Output();
        },
        "invalid neural-network dimensions are rejected");

    NeuralNetwork trainable;
    input.m_type = INPUT;
    output.m_type = OUTPUT;
    output.m_activation_function_type = UNSIGNED_SIGMOID;
    connection.m_source_neuron_idx = 0;
    connection.m_target_neuron_idx = 1;
    connection.m_weight = 0.5;
    trainable.AddNeuron(input);
    trainable.AddNeuron(output);
    trainable.AddConnection(connection);
    trainable.SetInputOutputDimensions(1, 1);
    trainable.InitRTRLMatrix();
    input_values = {1.0};
    trainable.Input(input_values);
    trainable.Activate();
    const double old_weight = trainable.GetConnectionByIndex(0).m_weight;
    trainable.RTRL_update_gradients();
    trainable.RTRL_update_error(1.0);
    trainable.RTRL_update_weights();
    Check(trainable.GetConnectionByIndex(0).m_weight != old_weight,
          "RTRL methods are defined and update weights");
    NeuralNetwork gradient_network;
    output.m_a = 2.0;
    connection.m_weight = 0.5;
    gradient_network.AddNeuron(input);
    gradient_network.AddNeuron(output);
    gradient_network.AddConnection(connection);
    gradient_network.SetInputOutputDimensions(1, 1);
    gradient_network.InitRTRLMatrix();
    input_values = {1.0};
    gradient_network.Input(input_values);
    gradient_network.Activate();
    const double gradient_output = gradient_network.Output().front();
    const double expected_gradient =
        (1.0 - gradient_output) * 2.0 * gradient_output *
        (1.0 - gradient_output);
    gradient_network.RTRL_update_gradients();
    gradient_network.RTRL_update_error({1.0}, 1.0);
    gradient_network.RTRL_update_weights();
    Check(
        std::abs(
            gradient_network.GetConnectionByIndex(0).m_weight -
            (0.5 + expected_gradient)) < 1.0e-12,
        "RTRL uses exact source state and activation slopes");
    NeuralNetwork recurrent_gradient_network;
    output.m_activation_function_type = LINEAR;
    output.m_a = 1.0;
    Connection input_to_output = connection;
    input_to_output.m_source_neuron_idx = 0;
    input_to_output.m_target_neuron_idx = 1;
    input_to_output.m_weight = 0.5;
    Connection output_loop;
    output_loop.m_source_neuron_idx = 1;
    output_loop.m_target_neuron_idx = 1;
    output_loop.m_weight = 0.25;
    recurrent_gradient_network.AddNeuron(input);
    recurrent_gradient_network.AddNeuron(output);
    recurrent_gradient_network.AddConnection(input_to_output);
    recurrent_gradient_network.AddConnection(output_loop);
    recurrent_gradient_network.SetInputOutputDimensions(1, 1);
    recurrent_gradient_network.InitRTRLMatrix();
    input_values = {1.0};
    recurrent_gradient_network.Input(input_values);
    recurrent_gradient_network.Activate();
    recurrent_gradient_network.RTRL_update_gradients();
    recurrent_gradient_network.Activate();
    recurrent_gradient_network.RTRL_update_gradients();
    const auto& recurrent_sensitivities =
        recurrent_gradient_network.m_neurons[1].m_sensitivity_matrix;
    Check(std::abs(recurrent_sensitivities[1][0] - 1.25) < 1.0e-12 &&
              std::abs(recurrent_sensitivities[1][1] - 0.5) < 1.0e-12,
          "sparse RTRL preserves exact multi-step recurrent gradients");
    recurrent_gradient_network.AddConnection(input_to_output);
    CheckThrows<std::invalid_argument>(
        [&] { recurrent_gradient_network.RTRL_update_gradients(); },
        "RTRL rejects ambiguous duplicate connection endpoints");

    const ActivationFunction differentiable_activations[] = {
        SIGNED_SIGMOID,
        UNSIGNED_SIGMOID,
        TANH,
        TANH_CUBIC,
        SIGNED_GAUSS,
        UNSIGNED_GAUSS,
        ABS,
        SIGNED_SINE,
        UNSIGNED_SINE,
        LINEAR,
        RELU,
        SOFTPLUS};
    for (const ActivationFunction activation :
         differentiable_activations)
    {
        NeuralNetwork sparse_gradient;
        Neuron sparse_input;
        sparse_input.m_type = INPUT;
        Neuron sparse_output;
        sparse_output.m_type = OUTPUT;
        sparse_output.m_activation_function_type = activation;
        sparse_output.m_a = 1.3;
        sparse_output.m_b = 0.2;
        Connection sparse_connection;
        sparse_connection.m_source_neuron_idx = 0;
        sparse_connection.m_target_neuron_idx = 1;
        sparse_connection.m_weight = 0.8;
        sparse_gradient.AddNeuron(sparse_input);
        sparse_gradient.AddNeuron(sparse_output);
        sparse_gradient.AddConnection(sparse_connection);
        sparse_gradient.SetInputOutputDimensions(1, 1);
        sparse_gradient.InitSparseRTRLMatrix();
        std::vector<double> sparse_input_value{0.37};
        sparse_gradient.InputExact(sparse_input_value);
        sparse_gradient.Activate();
        const double sparse_value =
            sparse_gradient.Output().front();

        constexpr double finite_difference_step = 1.0e-6;
        NeuralNetwork plus = sparse_gradient;
        plus.Flush();
        plus.m_connections[0].m_weight +=
            finite_difference_step;
        plus.InputExact(sparse_input_value);
        plus.Activate();
        NeuralNetwork minus = sparse_gradient;
        minus.Flush();
        minus.m_connections[0].m_weight -=
            finite_difference_step;
        minus.InputExact(sparse_input_value);
        minus.Activate();
        const double numerical_gradient =
            (plus.Output().front() - minus.Output().front()) /
            (2.0 * finite_difference_step);

        const double old_sparse_weight =
            sparse_gradient.m_connections[0].m_weight;
        sparse_gradient.RTRL_update_gradients_sparse();
        sparse_gradient.RTRL_update_error_sparse(
            {sparse_value + 1.0}, 1.0);
        sparse_gradient.RTRL_update_weights();
        const double rtrl_gradient =
            sparse_gradient.m_connections[0].m_weight -
            old_sparse_weight;
        Check(
            std::abs(rtrl_gradient - numerical_gradient) <
                2.0e-5,
            "sparse RTRL matches finite differences for activation " +
                std::to_string(static_cast<int>(activation)));
        Check(
            sparse_gradient.SparseRTRLStateSize() == 2,
            "sparse RTRL stores neuron-by-connection sensitivities");
        const NeuralNetwork restored_sparse =
            NeuralNetwork::Deserialize(
                sparse_gradient.Serialize());
        Check(
            restored_sparse.SparseRTRLStateSize() ==
                    sparse_gradient.SparseRTRLStateSize() &&
                restored_sparse.m_neurons[1].m_last_input ==
                    sparse_gradient.m_neurons[1].m_last_input,
            "sparse RTRL and pre-activation state round-trip");
    }

    NeuralNetwork batch_network;
    input.m_type = INPUT;
    output.m_type = OUTPUT;
    output.m_activation_function_type = LINEAR;
    output.m_b = 0.0;
    connection.m_source_neuron_idx = 0;
    connection.m_target_neuron_idx = 1;
    connection.m_weight = 2.0;
    batch_network.AddNeuron(input);
    batch_network.AddNeuron(output);
    batch_network.AddConnection(connection);
    batch_network.SetInputOutputDimensions(1, 1);
    const auto batch_outputs =
        batch_network.ActivateBatch({{1.0}, {2.0}, {-3.0}});
    Check(
        batch_outputs.size() == 3 &&
            batch_outputs[0][0] == 2.0 &&
            batch_outputs[1][0] == 4.0 &&
            batch_outputs[2][0] == -6.0,
        "batched activation resets state between samples");

    NeuralNetwork xor_network(false);
    Check(xor_network.m_neurons.size() == 5 &&
              xor_network.m_connections.size() == 7,
          "the legacy XOR-network constructor is restored");

    Genome genome = MakeGenome(parameters);
    genome.SetID(17);
    genome.SetFitness(-7.5);
    genome.SetAdjFitness(-3.25);
    genome.SetEvaluated();
    genome.SetNeuronXY(0, 11, 22);
    Trait trait;
    trait.value = std::string("kept");
    genome.m_GenomeGene.m_Traits["label"] = trait;
    Genome restored_genome = Genome::Deserialize(genome.Serialize());
    Check(restored_genome.GetFitness() == -7.5, "genome fitness round-trips");
    Check(restored_genome.GetAdjFitness() == -3.25,
          "genome adjusted fitness round-trips");
    Check(restored_genome.IsEvaluated(), "genome evaluation state round-trips");
    Check(restored_genome.GetNeuronByIndex(0).x == 11,
          "genome display coordinates round-trip");
    Check(std::get<std::string>(
    restored_genome.m_GenomeGene.m_Traits.at("label").value) == "kept",
          "genome traits round-trip");
    Genome trait_variant = restored_genome;
    trait_variant.m_GenomeGene.m_Traits.at("label").value =
        std::string("different");
    Check(!restored_genome.IsIdenticalTo(trait_variant),
          "clone detection includes genome traits");
    const auto genome_path =
        temporary_directory / "multineat_genome_legacy.txt";
    genome.Save(genome_path.string().c_str());
    Genome legacy_genome(genome_path.string().c_str());
    Check(legacy_genome.NumNeurons() == genome.NumNeurons() &&
              legacy_genome.NumLinks() == genome.NumLinks(),
          "legacy genome files remain loadable");
    std::filesystem::remove(genome_path);
    std::string genome_validation_error;
    Check(restored_genome.Validate(&genome_validation_error),
          "a round-tripped genome passes structural validation");

    Genome dad = genome;
    dad.SetID(18);
    dad.SetFitness(-10.0);
    for (auto& link : dad.m_LinkGenes)
        link.SetWeight(-4.0);
    for (auto& link : genome.m_LinkGenes)
        link.SetWeight(2.0);
    parameters.PreferFitterParentRate = 1.0;
    RNG legacy_mate_rng;
    RNG explicit_mate_rng;
    legacy_mate_rng.Seed(54321);
    explicit_mate_rng.Seed(54321);
    Genome legacy_multipoint = genome.Mate(
        dad, false, false, legacy_mate_rng, parameters);
    Genome explicit_multipoint = genome.MateWithMode(
        dad,
        MULTIPOINT,
        false,
        explicit_mate_rng,
        parameters);
    Check(
        legacy_multipoint.IsIdenticalTo(explicit_multipoint) &&
            legacy_mate_rng.Serialize() ==
                explicit_mate_rng.Serialize(),
        "historical multipoint mating maps exactly to its explicit mode");
    Genome multipoint = genome.Mate(dad, false, false, rng, parameters);
    Check(multipoint.GetLinkByIndex(0).GetWeight() == 2.0,
          "multipoint mating prefers the fitter parent");
    Genome averaged = genome.Mate(dad, true, false, rng, parameters);
    Check(averaged.GetLinkByIndex(0).GetWeight() == -1.0,
          "average mating honors its explicit mode argument");
    Genome single_point = genome.MateWithMode(
        dad, SINGLE_POINT, false, rng, parameters);
    bool single_point_inherits_parent_weights = true;
    for (const auto& link : single_point.m_LinkGenes)
    {
        single_point_inherits_parent_weights =
            single_point_inherits_parent_weights &&
            (link.GetWeight() == 2.0 || link.GetWeight() == -4.0);
    }
    Check(single_point_inherits_parent_weights,
          "single-point crossover switches between parental innovations");
    parameters.CrossoverBlendAlpha = 0.5;
    Genome blended = genome.MateWithMode(
        dad, BLEND, false, rng, parameters);
    bool blended_weights_in_range = true;
    for (const auto& link : blended.m_LinkGenes)
    {
        blended_weights_in_range =
            blended_weights_in_range &&
            link.GetWeight() >= -7.0 && link.GetWeight() <= 5.0;
    }
    Check(blended_weights_in_range,
          "BLX-alpha crossover samples the expanded parental interval");
    Genome simulated_binary = genome.MateWithMode(
        dad, SIMULATED_BINARY, false, rng, parameters);
    bool sbx_weights_in_range = true;
    for (const auto& link : simulated_binary.m_LinkGenes)
    {
        sbx_weights_in_range =
            sbx_weights_in_range &&
            link.GetWeight() >= parameters.MinWeight &&
            link.GetWeight() <= parameters.MaxWeight;
    }
    Check(sbx_weights_in_range,
          "simulated-binary crossover respects weight bounds");
    Check(averaged.m_initial_num_neurons == genome.m_initial_num_neurons &&
              averaged.m_initial_num_links == genome.m_initial_num_links,
          "mating preserves the seed-complexity baseline");
    for (const WeightMutationMode mode :
         {UNIFORM_MUTATION,
          GAUSSIAN_MUTATION,
          CAUCHY_MUTATION,
          POLYNOMIAL_MUTATION})
    {
        Parameters mutation_parameters = parameters;
        mutation_parameters.WeightMutationDistribution = mode;
        mutation_parameters.MutateWeightsSevereProb = 0.0;
        mutation_parameters.WeightMutationRate = 1.0;
        mutation_parameters.WeightReplacementRate = 0.0;
        mutation_parameters.WeightMutationMaxPower = 0.5;
        Genome mutated_weights = genome;
        rng.Seed(1200 + static_cast<int>(mode));
        Check(mutated_weights.Mutate_LinkWeights(
                  mutation_parameters, rng),
              "each weight mutation distribution is reachable");
        bool weights_are_bounded = true;
        for (const auto& link : mutated_weights.m_LinkGenes)
        {
            weights_are_bounded =
                weights_are_bounded &&
                std::isfinite(link.GetWeight()) &&
                link.GetWeight() >= mutation_parameters.MinWeight &&
                link.GetWeight() <= mutation_parameters.MaxWeight;
        }
        Check(weights_are_bounded,
              "weight mutation distributions retain finite bounds");
    }

    NeuralNetwork positioned;
    restored_genome.BuildPhenotype(positioned);
    Check(positioned.GetNeuronByIndex(0).m_x == 11.0 &&
              positioned.GetNeuronByIndex(0).m_y == 22.0,
          "phenotype construction preserves neuron coordinates");

    Parameters trait_distance_parameters = parameters;
    trait_distance_parameters.DisjointCoeff = 0.0;
    trait_distance_parameters.ExcessCoeff = 0.0;
    trait_distance_parameters.WeightDiffCoeff = 0.0;
    trait_distance_parameters.ActivationADiffCoeff = 0.0;
    trait_distance_parameters.ActivationBDiffCoeff = 0.0;
    trait_distance_parameters.TimeConstantDiffCoeff = 0.0;
    trait_distance_parameters.BiasDiffCoeff = 0.0;
    trait_distance_parameters.ActivationFunctionDiffCoeff = 0.0;
    TraitParameters numeric_schema;
    numeric_schema.type = "int";
    numeric_schema.m_ImportanceCoeff = 0.5;
    numeric_schema.m_Details = IntTraitParameters();
    trait_distance_parameters.GenomeTraits.clear();
    trait_distance_parameters.GenomeTraits["numeric"] = numeric_schema;
    Genome trait_left = genome;
    Genome trait_right = genome;
    Trait numeric_left;
    numeric_left.value = 0;
    Trait numeric_right;
    numeric_right.value = 10;
    trait_left.m_GenomeGene.m_Traits = {{"numeric", numeric_left}};
    trait_right.m_GenomeGene.m_Traits = {{"numeric", numeric_right}};
    Check(trait_left.CompatibilityDistance(
              trait_right, trait_distance_parameters) == 5.0,
          "trait importance is applied exactly once");

    Species species(genome, parameters, 9);
    Species restored_species = Species::Deserialize(species.Serialize());
    Check(restored_species.ID() == 9 && restored_species.NumIndividuals() == 1,
          "species serialization round-trips");
    Check(restored_species.GetLeader().GetFitness() == -7.5,
          "species handles negative-fitness leaders");
    Genome unevaluated = genome;
    unevaluated.ResetEvaluated();
    unevaluated.SetFitness(100.0);
    species.AddIndividual(unevaluated);
    species.m_Individuals.front().SetEvaluated();
    species.m_Individuals.front().SetFitness(10.0);
    species.CalculateAverageFitness();
    Check(species.m_AverageFitness == 10.0,
          "species averages include evaluated individuals only");
    Check(species.GetLeader().GetFitness() == 10.0,
          "species leaders prefer evaluated finite genomes");

    Parameters full_survival_parameters = parameters;
    full_survival_parameters.TruncationSelection = true;
    full_survival_parameters.RouletteWheelSelection = false;
    full_survival_parameters.TournamentSelection = false;
    full_survival_parameters.SurvivalRate = 1.0;
    Species full_survival(genome, full_survival_parameters, 10);
    full_survival.m_Individuals.clear();
    for (int i = 0; i < 3; ++i)
    {
        Genome candidate = genome;
        candidate.SetID(100 + i);
        candidate.SetFitness(static_cast<double>(3 - i));
        candidate.SetAdjFitness(static_cast<double>(3 - i));
        candidate.SetEvaluated();
        full_survival.m_Individuals.push_back(candidate);
    }
    full_survival.SortIndividuals();
    bool selected_worst = false;
    rng.Seed(991);
    for (int i = 0; i < 200; ++i)
    {
        selected_worst =
            selected_worst ||
            full_survival.GetIndividual(
                full_survival_parameters, rng).GetID() == 102;
    }
    Check(selected_worst,
          "a 100-percent survival rate retains every parent candidate");

    for (const SelectionMode mode :
         {ROULETTE,
          RANK_LINEAR,
          RANK_EXP,
          TOURNAMENT,
          STOCHASTIC,
          BOLTZMANN})
    {
        Parameters selection_parameters = full_survival_parameters;
        selection_parameters.ParentSelectionMode = mode;
        selection_parameters.BoltzmannTemperature = 0.5;
        selection_parameters.TournamentSize = 3;
        int best_count = 0;
        int worst_count = 0;
        rng.Seed(2000 + static_cast<int>(mode));
        for (int draw = 0; draw < 3000; ++draw)
        {
            const int selected =
                full_survival.GetIndividual(
                    selection_parameters, rng).GetID();
            best_count += selected == 100 ? 1 : 0;
            worst_count += selected == 102 ? 1 : 0;
        }
        Check(
            best_count > worst_count,
            "explicit selection modes favor higher-fitness parents");
    }
    Parameters explicit_truncation = full_survival_parameters;
    explicit_truncation.ParentSelectionMode = TRUNCATION;
    explicit_truncation.TruncationSelection = false;
    explicit_truncation.SurvivalRate = 0.34;
    bool truncation_selected_only_best = true;
    for (int draw = 0; draw < 100; ++draw)
    {
        truncation_selected_only_best =
            truncation_selected_only_best &&
            full_survival.GetIndividual(
                explicit_truncation, rng).GetID() == 100;
    }
    Check(truncation_selected_only_best,
          "explicit truncation ignores legacy selector switches");

    Population population(genome, parameters, true, 1.0, 123);
    population.m_Species.front().m_Individuals.front().SetFitness(-3.0);
    Population restored_population =
        Population::Deserialize(population.Serialize());
    Check(restored_population.NumGenomes() == parameters.PopulationSize,
          "population serialization preserves all genomes");
    Check(restored_population.GetBestGenome().GetFitness() == -3.0,
          "population best-genome selection handles negative fitness");
    Check(restored_population.m_Parameters.GenomeTraits.count("label") == 1,
          "population serialization preserves universal trait schemas");
    const auto checkpoint_path =
        std::filesystem::temp_directory_path() /
        "multineat_population_checkpoint.txt";
    population.SaveState(checkpoint_path.string().c_str());
    Population file_restored_population(checkpoint_path.string());
    std::filesystem::remove(checkpoint_path);
    Check(
        file_restored_population.NumGenomes() == population.NumGenomes() &&
            file_restored_population.m_RNG.Serialize() ==
                population.m_RNG.Serialize(),
        "complete population checkpoints resume exact state");
    const auto population_path =
        temporary_directory / "multineat_population_legacy.txt";
    population.Save(population_path.string().c_str());
    Population legacy_population(population_path.string());
    std::filesystem::remove(population_path);
    Check(
        legacy_population.NumGenomes() == population.NumGenomes(),
        "legacy population files remain loadable");
    std::string population_validation_error;
    Check(population.Validate(&population_validation_error),
          "a constructed population passes global validation");
    for (const FitnessScalingMode scaling :
         {SHIFTED_FITNESS_SCALING,
          LINEAR_RANK_FITNESS_SCALING,
          SIGMA_FITNESS_SCALING,
          BOLTZMANN_FITNESS_SCALING})
    {
        Parameters extreme_parameters;
        extreme_parameters.PopulationSize = 8;
        extreme_parameters.FitnessScaling = scaling;
        extreme_parameters.RequireEvaluatedGenomes = true;
        extreme_parameters.RejectNonFiniteFitness = true;
        Genome extreme_seed = MakeGenome(extreme_parameters);
        Population extreme_population(
            extreme_seed,
            extreme_parameters,
            true,
            1.0,
            400 + static_cast<int>(scaling));
        const double maximum =
            std::numeric_limits<double>::max();
        const double extreme_fitness_values[] = {
            -maximum,
            -maximum / 2.0,
            -1.0,
            0.0,
            1.0,
            maximum / 4.0,
            maximum / 2.0,
            maximum};
        for (int index = 0;
             index < static_cast<int>(
                         extreme_parameters.PopulationSize);
             ++index)
        {
            Genome& candidate =
                extreme_population.AccessGenomeByIndex(index);
            candidate.SetFitness(extreme_fitness_values[index]);
            candidate.SetEvaluated();
        }
        extreme_population.Epoch();
        Check(
            extreme_population.Validate(
                &population_validation_error),
            "fitness scaling mode survives the finite double range: " +
                std::to_string(static_cast<int>(scaling)));
    }
    for (int scenario = 0; scenario < 8; ++scenario)
    {
        Parameters stress_parameters;
        stress_parameters.PopulationSize = 24;
        stress_parameters.ParentSelectionMode =
            static_cast<SelectionMode>(scenario % 7);
        stress_parameters.FitnessScaling =
            static_cast<FitnessScalingMode>(scenario % 4);
        stress_parameters.MultipointCrossoverRate = 0.2;
        stress_parameters.SinglePointCrossoverRate = 0.2;
        stress_parameters.BlendCrossoverRate = 0.2;
        stress_parameters.SimulatedBinaryCrossoverRate = 0.2;
        stress_parameters.WeightMutationDistribution =
            static_cast<WeightMutationMode>(scenario % 4);
        stress_parameters.MutateAddNeuronProb = 0.12;
        stress_parameters.MutateAddLinkProb = 0.18;
        stress_parameters.MutateRemLinkProb = 0.04;
        stress_parameters.MutateRemSimpleNeuronProb = 0.03;
        stress_parameters.MutateWeightsProb = 0.6;
        stress_parameters.MaxNeurons = 24;
        stress_parameters.MaxLinks = 80;
        stress_parameters.MutationOperatorsPerOffspring = 1.5;
        stress_parameters.SpeciesRepresentativeSelection =
            static_cast<SpeciesRepresentativeMode>(scenario % 4);
        stress_parameters.RepresentativeSelectionCandidates = 8;
        stress_parameters.OffspringAllocation =
            scenario % 2 == 0
                ? LARGEST_REMAINDER
                : STOCHASTIC_REMAINDER;
        Genome stress_seed = MakeGenome(stress_parameters);
        Population stress_population(
            stress_seed,
            stress_parameters,
            true,
            1.0,
            8000 + scenario);
        for (int generation = 0; generation < 12; ++generation)
        {
            for (int index = 0;
                 index <
                 static_cast<int>(
                     stress_population.NumGenomes());
                 ++index)
            {
                Genome& candidate =
                    stress_population.AccessGenomeByIndex(index);
                candidate.SetFitness(
                    std::sin(
                        static_cast<double>(
                            candidate.GetID() * 17 +
                            generation * 13)) -
                    0.001 *
                        static_cast<double>(
                            candidate.NumLinks()));
                candidate.SetEvaluated();
            }
            stress_population.Epoch();
            Check(
                stress_population.Validate(
                    &population_validation_error),
                "randomized evolution preserves invariants in scenario " +
                    std::to_string(scenario));
        }
    }
    Population duplicate_id_population = population;
    duplicate_id_population.AccessGenomeByIndex(1).SetID(
        duplicate_id_population.AccessGenomeByIndex(0).GetID());
    Check(!duplicate_id_population.Validate(
              &population_validation_error),
          "population validation rejects duplicate genome IDs");
    for (const auto& restored_species_item : restored_population.m_Species)
    {
        for (const auto& restored_individual :
             restored_species_item.m_Individuals)
        {
            for (const auto& restored_link : restored_individual.m_LinkGenes)
            {
                Check(std::abs(restored_link.GetWeight()) <= 1.0,
                      "population initialization honors its randomization range");
            }
        }
    }

    Parameters no_speciation_parameters = parameters;
    no_speciation_parameters.Speciation = false;
    Population no_speciation(
        genome, no_speciation_parameters, true, 0.25, 321);
    Check(no_speciation.m_Species.size() == 1,
          "disabling speciation creates exactly one species");

    Parameters removal_parameters = parameters;
    removal_parameters.PopulationSize = 3;
    Population removal_population(
        genome, removal_parameters, true, 0.25, 654);
    Species crowded(genome, removal_parameters, 21);
    crowded.m_Individuals.clear();
    Species sparse(genome, removal_parameters, 22);
    sparse.m_Individuals.clear();
    for (int i = 0; i < 2; ++i)
    {
        Genome candidate = genome;
        candidate.SetID(200 + i);
        candidate.SetFitness(-10.0 + i);
        candidate.SetEvaluated();
        crowded.m_Individuals.push_back(candidate);
    }
    Genome sparse_candidate = genome;
    sparse_candidate.SetID(202);
    sparse_candidate.SetFitness(-8.0);
    sparse_candidate.SetEvaluated();
    sparse.m_Individuals.push_back(sparse_candidate);
    removal_population.m_Species = {crowded, sparse};
    Check(removal_population.RemoveWorstIndividual().GetID() == 200,
          "steady-state removal handles negative fitness before sharing");

    Population parent_choice_population(
        genome, removal_parameters, true, 0.25, 655);
    Species ineligible(genome, removal_parameters, 31);
    ineligible.m_Individuals.front().ResetEvaluated();
    ineligible.m_AverageFitness = 100.0;
    Species sole_eligible(genome, removal_parameters, 32);
    sole_eligible.m_Individuals.front().SetEvaluated();
    sole_eligible.m_AverageFitness = -5.0;
    parent_choice_population.m_Species = {ineligible, sole_eligible};
    bool chose_only_eligible = true;
    for (int i = 0; i < 100; ++i)
    {
        chose_only_eligible =
            chose_only_eligible &&
            parent_choice_population.ChooseParentSpecies() == 1;
    }
    Check(chose_only_eligible,
          "parent selection never chooses an ineligible zero-weight species");

    Parameters epoch_parameters = parameters;
    epoch_parameters.PopulationSize = 4;
    epoch_parameters.AllowClones = true;
    Population negative_epoch(genome, epoch_parameters, true, 0.5, 456);
    for (unsigned int i = 0; i < negative_epoch.NumGenomes(); ++i)
    {
        negative_epoch.AccessGenomeByIndex(static_cast<int>(i)).SetFitness(
            -10.0 + static_cast<double>(i));
    }
    negative_epoch.Epoch();
    Check(negative_epoch.NumGenomes() == epoch_parameters.PopulationSize,
          "an epoch supports negative fitness without changing population size");

    Parameters strict_parameters = epoch_parameters;
    strict_parameters.PopulationSize = 2;
    strict_parameters.RequireEvaluatedGenomes = true;
    strict_parameters.RejectNonFiniteFitness = true;
    Population strict_population(
        genome, strict_parameters, true, 0.5, 457);
    for (unsigned int index = 0;
         index < strict_population.NumGenomes();
         ++index)
    {
        strict_population.AccessGenomeByIndex(
            static_cast<int>(index)).ResetEvaluated();
    }
    CheckThrows<std::runtime_error>(
        [&] { strict_population.Epoch(); },
        "strict evolution rejects missing evaluations");
    for (unsigned int index = 0;
         index < strict_population.NumGenomes();
         ++index)
    {
        Genome& candidate =
            strict_population.AccessGenomeByIndex(
                static_cast<int>(index));
        candidate.SetFitness(static_cast<double>(index));
        candidate.SetEvaluated();
    }
    strict_population.AccessGenomeByIndex(0).SetFitness(
        std::numeric_limits<double>::quiet_NaN());
    CheckThrows<std::runtime_error>(
        [&] { strict_population.Epoch(); },
        "strict evolution rejects non-finite fitness");

    Parameters elite_parameters = epoch_parameters;
    elite_parameters.Speciation = false;
    elite_parameters.EliteFraction = 0.5;
    Population elite_population(
        genome, elite_parameters, true, 0.5, 458);
    Genome expected_best;
    Genome expected_second;
    for (unsigned int index = 0;
         index < elite_population.NumGenomes();
         ++index)
    {
        Genome& candidate =
            elite_population.AccessGenomeByIndex(
                static_cast<int>(index));
        for (auto& link : candidate.m_LinkGenes)
            link.SetWeight(static_cast<double>(index + 1));
        candidate.SetFitness(static_cast<double>(index));
        candidate.SetEvaluated();
        if (index == 3)
            expected_best = candidate;
        if (index == 2)
            expected_second = candidate;
    }
    elite_population.Epoch();
    bool retained_best = false;
    bool retained_second = false;
    for (unsigned int index = 0;
         index < elite_population.NumGenomes();
         ++index)
    {
        const Genome& candidate =
            elite_population.AccessGenomeByIndex(
                static_cast<int>(index));
        retained_best =
            retained_best || candidate.IsIdenticalTo(expected_best);
        retained_second =
            retained_second || candidate.IsIdenticalTo(expected_second);
    }
    Check(retained_best && retained_second,
          "multi-elite reproduction preserves distinct top genomes");

    Parameters niche_parameters = epoch_parameters;
    niche_parameters.PopulationSize = 6;
    niche_parameters.DynamicCompatibility = true;
    niche_parameters.MinSpecies = 1;
    niche_parameters.MaxSpecies = 3;
    niche_parameters.TargetSpecies = 2;
    niche_parameters.CompatibilityThresholdControl =
        PROPORTIONAL_COMPATIBILITY_THRESHOLD;
    niche_parameters.CompatibilityThresholdGain = 0.5;
    niche_parameters.CompatTreshold = 0.05;
    niche_parameters.MinCompatTreshold = 0.001;
    niche_parameters.MaxCompatTreshold = 2.0;
    niche_parameters.MinSpeciesSize = 2;
    niche_parameters.EliteFraction = 1.0;
    niche_parameters.SpeciesRepresentativeSelection =
        MEDOID_REPRESENTATIVE;
    niche_parameters.OffspringAllocation =
        STOCHASTIC_REMAINDER;
    Population niche_population(
        genome, niche_parameters, true, 0.5, 459);
    niche_population.m_Species.clear();
    for (int species_index = 0; species_index < 3; ++species_index)
    {
        Genome representative = genome;
        representative.SetID(species_index * 2);
        for (auto& link : representative.m_LinkGenes)
            link.SetWeight(-8.0 + 8.0 * species_index);
        representative.SetFitness(
            species_index == 0 ? 100.0 : 1.0);
        representative.SetEvaluated();
        Species niche(
            representative,
            niche_parameters,
            100 + species_index);
        Genome peer = representative;
        peer.SetID(species_index * 2 + 1);
        niche.AddIndividual(peer);
        niche_population.m_Species.push_back(niche);
    }
    niche_population.Epoch();
    std::vector<unsigned int> niche_sizes;
    for (const auto& niche : niche_population.m_Species)
        niche_sizes.push_back(niche.NumIndividuals());
    std::sort(niche_sizes.begin(), niche_sizes.end());
    Check(
        niche_sizes == std::vector<unsigned int>({2, 2, 2}),
        "minimum species size protects low-fitness viable niches");
    Check(niche_population.m_Parameters.CompatTreshold > 0.05,
          "proportional threshold control responds smoothly to excess species");

    Parameters tick_parameters;
    tick_parameters.PopulationSize = 16;
    tick_parameters.AllowClones = true;
    Genome tick_seed = MakeGenome(tick_parameters);
    Population tick_population(
        tick_seed, tick_parameters, true, 1.0, 7654);
    for (unsigned int index = 0;
         index < tick_population.NumGenomes();
         ++index)
    {
        Genome& candidate =
            tick_population.AccessGenomeByIndex(
                static_cast<int>(index));
        candidate.SetFitness(static_cast<double>(index));
        candidate.SetEvaluated();
    }
    for (int evaluation = 0; evaluation < 100; ++evaluation)
    {
        Genome deleted;
        Genome* offspring = tick_population.Tick(deleted);
        Check(offspring != nullptr && offspring->GetID() >= 0,
              "steady-state evolution returns a valid offspring");
        if (offspring != nullptr)
        {
            offspring->SetFitness(
                static_cast<double>((evaluation * 13) % 29) - 20.0);
            offspring->SetEvaluated();
        }
        std::set<int> tick_ids;
        for (const auto& tick_species : tick_population.m_Species)
        {
            for (const auto& individual : tick_species.m_Individuals)
            {
                tick_ids.insert(individual.GetID());
            }
        }
        Check(
            tick_population.NumGenomes() ==
                tick_parameters.PopulationSize &&
                tick_ids.size() == tick_parameters.PopulationSize,
            "steady-state evolution retains population size and unique IDs");
        Check(tick_population.Validate(&population_validation_error),
              "steady-state evolution retains global invariants");
    }

    Parameters stress_parameters;
    stress_parameters.PopulationSize = 32;
    stress_parameters.AllowClones = true;
    Genome stress_seed = MakeGenome(stress_parameters);
    Population stress_population(
        stress_seed, stress_parameters, true, 1.0, 9876);
    for (int generation = 0; generation < 20; ++generation)
    {
        std::set<int> genome_ids;
        for (unsigned int index = 0;
             index < stress_population.NumGenomes();
             ++index)
        {
            Genome& candidate =
                stress_population.AccessGenomeByIndex(
                    static_cast<int>(index));
            std::string validation_error;
            Check(candidate.Validate(&validation_error),
                  "evolved genomes retain structural invariants");
            genome_ids.insert(candidate.GetID());
            candidate.SetFitness(
                -100.0 + std::sin(
                    static_cast<double>(generation * 37 + index)));
            candidate.SetEvaluated();
        }
        Check(genome_ids.size() == stress_parameters.PopulationSize,
              "evolution retains unique genome IDs");
        stress_population.Epoch();
        Check(
            stress_population.NumGenomes() ==
                stress_parameters.PopulationSize,
            "repeated evolution retains the configured population size");
    }

    Parameters resume_parameters;
    resume_parameters.PopulationSize = 12;
    resume_parameters.AllowClones = true;
    Genome resume_seed = MakeGenome(resume_parameters);
    Population uninterrupted(
        resume_seed, resume_parameters, true, 1.0, 2468);
    for (unsigned int index = 0;
         index < uninterrupted.NumGenomes();
         ++index)
    {
        Genome& candidate =
            uninterrupted.AccessGenomeByIndex(
                static_cast<int>(index));
        candidate.SetFitness(static_cast<double>(index));
        candidate.SetEvaluated();
    }
    uninterrupted.Epoch();
    Population resumed =
        Population::Deserialize(uninterrupted.Serialize());
    for (int generation = 0; generation < 5; ++generation)
    {
        for (unsigned int index = 0;
             index < uninterrupted.NumGenomes();
             ++index)
        {
            Genome& first =
                uninterrupted.AccessGenomeByIndex(
                    static_cast<int>(index));
            Genome& second =
                resumed.AccessGenomeByIndex(
                    static_cast<int>(index));
            const double fitness =
                -50.0 + static_cast<double>(first.GetID() % 17);
            first.SetFitness(fitness);
            first.SetEvaluated();
            second.SetFitness(fitness);
            second.SetEvaluated();
        }
        uninterrupted.Epoch();
        resumed.Epoch();
    }
    Check(uninterrupted.Serialize() == resumed.Serialize(),
          "a resumed checkpoint evolves deterministically");

    InnovationDatabase innovations;
    innovations.Init(genome);
    const int added_innovation = innovations.AddLinkInnovation(1, 2);
    InnovationDatabase restored_innovations =
        InnovationDatabase::Deserialize(innovations.Serialize());
    Check(restored_innovations.AddLinkInnovation(2, 3) ==
              added_innovation + 1,
          "innovation counters round-trip without ID reuse");
    InnovationDatabase indexed_innovations;
    for (int index = 0; index < 20000; ++index)
    {
        indexed_innovations.AddLinkInnovation(
            1 + index % 257,
            1 + (index * 17) % 263);
    }
    const int indexed_first =
        indexed_innovations.CheckInnovation(1, 1, NEW_LINK);
    const int indexed_last =
        indexed_innovations.CheckLastInnovation(1, 1, NEW_LINK);
    Check(
        indexed_first > 0 && indexed_last >= indexed_first &&
            !indexed_innovations
                 .CheckAllInnovations(1, 1, NEW_LINK)
                 .empty(),
        "indexed innovation queries retain first/last/all semantics");
    indexed_innovations.m_Innovations[100] =
        Innovation(101, NEW_LINK, 999, 1000, NONE, -1);
    indexed_innovations.RebuildIndex();
    Check(
        indexed_innovations.CheckInnovation(
            999, 1000, NEW_LINK) == 101,
        "innovation index can be rebuilt after direct vector edits");
    std::string invalid_innovation_state = innovations.Serialize();
    const std::size_t next_counter =
        invalid_innovation_state.find("NextInnovNum: ");
    const std::size_t next_counter_end =
        invalid_innovation_state.find('\n', next_counter);
    invalid_innovation_state.replace(
        next_counter,
        next_counter_end - next_counter,
        "NextInnovNum: 1");
    CheckThrows<std::runtime_error>(
        [&] {
            InnovationDatabase::Deserialize(
                invalid_innovation_state);
        },
        "innovation persistence rejects counters that would reuse IDs");

    GenomeInitStruct impossible;
    impossible.NumInputs = 2;
    impossible.NumOutputs = 1;
    impossible.FS_NEAT = true;
    impossible.FS_NEAT_links = 2;
    CheckThrows<std::invalid_argument>(
        [&] { Genome(parameters, impossible); },
        "impossible FS-NEAT seeds fail instead of looping");

    GenomeInitStruct fs_seed;
    fs_seed.NumInputs = 3;
    fs_seed.NumOutputs = 3;
    fs_seed.FS_NEAT = true;
    fs_seed.FS_NEAT_links = 1;
    Genome sparse_seed(parameters, fs_seed);
    Check(sparse_seed.NumLinks() == 2,
          "FS-NEAT creates exactly the requested sparse link plus bias");

    Parameters structural_parameters = parameters;
    structural_parameters.RecurrentProb = 0.0;
    structural_parameters.MutateAddLinkFromBiasProb = 0.0;
    structural_parameters.LinkTries = 1;
    Genome structural = MakeGenome(structural_parameters);
    InnovationDatabase structural_innovations;
    structural_innovations.Init(structural);
    rng.Seed(4321);
    Check(structural.Mutate_AddNeuron(
              structural_innovations, structural_parameters, rng),
          "a feed-forward genome can be structurally expanded");
    while (structural.Mutate_AddLink(
        structural_innovations, structural_parameters, rng))
    {
        Check(!structural.HasLoops(),
              "feed-forward link mutation never introduces a cycle");
    }
    Check(!structural.HasLoops(),
          "exhausting feed-forward links retains an acyclic graph");

    Parameters recurrent_parameters = parameters;
    recurrent_parameters.RecurrentProb = 1.0;
    recurrent_parameters.RecurrentLoopProb = 1.0;
    recurrent_parameters.LinkTries = 1;
    Genome recurrent = MakeGenome(recurrent_parameters);
    InnovationDatabase recurrent_innovations;
    recurrent_innovations.Init(recurrent);
    Check(recurrent.Mutate_AddLink(
              recurrent_innovations, recurrent_parameters, rng) &&
              recurrent.m_LinkGenes.back().IsLoopedRecurrent(),
          "looped recurrent mutation chooses a valid self-connection");

    Parameters split_parameters = parameters;
    Genome recurrent_split = MakeGenome(split_parameters);
    const int split_output =
        recurrent_split.GetNeuronByIndex(
            static_cast<int>(recurrent_split.NumInputs())).ID();
    recurrent_split.m_LinkGenes.clear();
    recurrent_split.m_LinkGenes.emplace_back(
        1, split_output, 1, 0.5, true);
    InnovationDatabase split_innovations;
    split_innovations.Init(recurrent_split);
    split_parameters.SplitRecurrent = false;
    split_parameters.SplitLoopedRecurrent = false;
    Check(!recurrent_split.Mutate_AddNeuron(
              split_innovations, split_parameters, rng),
          "non-loop recurrent links obey SplitRecurrent");
    split_parameters.SplitRecurrent = true;
    Check(recurrent_split.Mutate_AddNeuron(
              split_innovations, split_parameters, rng),
          "SplitRecurrent enables recurrent-link splitting");

    Parameters no_op_parameters = parameters;
    no_op_parameters.ActivationAMutationMaxPower = 0.0;
    Genome no_op = MakeGenome(no_op_parameters);
    Check(!no_op.Mutate_NeuronActivations_A(no_op_parameters, rng),
          "a no-op parameter mutation reports failure");

    Parameters capped_parameters;
    capped_parameters.PopulationSize = 2;
    capped_parameters.AllowClones = true;
    capped_parameters.MaxNeurons = 0;
    capped_parameters.MaxLinks = 0;
    capped_parameters.MutateAddNeuronProb = 1.0;
    capped_parameters.MutateAddLinkProb = 1.0;
    capped_parameters.MutateWeightsProb = 1.0;
    capped_parameters.WeightMutationRate = 1.0;
    capped_parameters.MutateWeightsSevereProb = 1.0;
    Genome capped_seed = MakeGenome(capped_parameters);
    Population capped_population(
        capped_seed, capped_parameters, true, 1.0, 888);
    Genome capped_candidate =
        capped_population.AccessGenomeByIndex(0);
    const unsigned int capped_neurons = capped_candidate.NumNeurons();
    const unsigned int capped_links = capped_candidate.NumLinks();
    capped_population.m_Species.front().MutateGenome(
        false,
        capped_population,
        capped_candidate,
        capped_parameters,
        rng);
    Check(capped_candidate.NumNeurons() == capped_neurons &&
              capped_candidate.NumLinks() == capped_links,
          "declared structural complexity caps are enforced");

    Parameters repeated_mutation_parameters;
    repeated_mutation_parameters.PopulationSize = 2;
    repeated_mutation_parameters.AllowClones = true;
    repeated_mutation_parameters.MutateAddNeuronProb = 0.0;
    repeated_mutation_parameters.MutateAddLinkProb = 0.0;
    repeated_mutation_parameters.MutateRemSimpleNeuronProb = 0.0;
    repeated_mutation_parameters.MutateRemLinkProb = 0.0;
    repeated_mutation_parameters.MutateNeuronActivationTypeProb = 0.0;
    repeated_mutation_parameters.MutateWeightsProb = 1.0;
    repeated_mutation_parameters.MutateActivationAProb = 0.0;
    repeated_mutation_parameters.MutateActivationBProb = 0.0;
    repeated_mutation_parameters.MutateNeuronTimeConstantsProb = 0.0;
    repeated_mutation_parameters.MutateNeuronBiasesProb = 0.0;
    repeated_mutation_parameters.MutateNeuronTraitsProb = 0.0;
    repeated_mutation_parameters.MutateLinkTraitsProb = 0.0;
    repeated_mutation_parameters.MutateGenomeTraitsProb = 0.0;
    repeated_mutation_parameters.WeightMutationRate = 1.0;
    repeated_mutation_parameters.MutateWeightsSevereProb = 1.0;
    Genome repeated_seed = MakeGenome(repeated_mutation_parameters);
    Population repeated_population(
        repeated_seed,
        repeated_mutation_parameters,
        true,
        1.0,
        889);
    Species repeated_species(
        repeated_seed, repeated_mutation_parameters, 90);
    Genome single_mutation = repeated_seed;
    Genome triple_mutation = repeated_seed;
    RNG single_rng;
    RNG triple_rng;
    single_rng.Seed(90210);
    triple_rng.Seed(90210);
    repeated_species.MutateGenome(
        false,
        repeated_population,
        single_mutation,
        repeated_mutation_parameters,
        single_rng);
    repeated_mutation_parameters.MutationOperatorsPerOffspring = 3.0;
    repeated_species.MutateGenome(
        false,
        repeated_population,
        triple_mutation,
        repeated_mutation_parameters,
        triple_rng);
    Check(
        single_rng.Serialize() != triple_rng.Serialize() &&
            !single_mutation.IsIdenticalTo(triple_mutation),
        "multi-operator mutation applies the configured exploration budget");

    GenomeInitStruct layered;
    layered.NumInputs = 3;
    layered.NumHidden = 1;
    layered.NumOutputs = 1;
    layered.SeedType = LAYERED;
    layered.NumLayers = 2;
    Genome layered_genome(parameters, layered);
    layered_genome.CalculateDepth();
    Check(layered_genome.GetDepth() == 3,
          "network depth counts feed-forward hidden layers");

    Substrate substrate;
    substrate.m_input_coords = {{0.0}};
    substrate.m_output_coords = {{1.0}};
    std::vector<std::vector<int>> invalid_connectivity{{INPUT, 2, OUTPUT, 0}};
    CheckThrows<std::out_of_range>(
        [&] { substrate.SetCustomConnectivity(invalid_connectivity); },
        "custom substrate connectivity validates indexes");

    std::vector<std::vector<double>> substrate_inputs{{-1.0}};
    std::vector<std::vector<double>> substrate_hidden;
    std::vector<std::vector<double>> substrate_outputs{{1.0}};
    Substrate flagged_substrate(
        substrate_inputs, substrate_hidden, substrate_outputs);
    flagged_substrate.m_query_weights_only = true;
    GenomeInitStruct cppn_init;
    cppn_init.NumInputs = flagged_substrate.GetMinCPPNInputs();
    cppn_init.NumOutputs = flagged_substrate.GetMinCPPNOutputs();
    Genome cppn(parameters, cppn_init);
    NeuralNetwork substrate_network;
    cppn.BuildHyperNEATPhenotype(substrate_network, flagged_substrate);
    Check(substrate_network.m_connections.empty(),
          "full HyperNEAT connectivity obeys disabled topology flags");
    flagged_substrate.m_allow_input_output_links = true;
    cppn.BuildHyperNEATPhenotype(substrate_network, flagged_substrate);
    Check(substrate_network.m_connections.size() == 1,
          "HyperNEAT enables only the requested topology");

    Parameters es_parameters;
    es_parameters.InitialDepth = 1;
    es_parameters.MaxDepth = 1;
    es_parameters.IterationLevel = 0;
    es_parameters.DivisionThreshold = 0.0;
    es_parameters.VarianceThreshold = 10.0;
    es_parameters.BandThreshold = 0.25;
    GenomeInitStruct es_cppn_init;
    es_cppn_init.NumInputs = 5;
    es_cppn_init.NumOutputs = 1;
    es_cppn_init.OutputActType = LINEAR;
    Genome es_cppn(es_parameters, es_cppn_init);
    for (auto& link : es_cppn.m_LinkGenes)
    {
        link.SetWeight(
            (link.FromNeuronID() == es_cppn.m_NeuronGenes[0].ID() ||
             link.FromNeuronID() == es_cppn.m_NeuronGenes[2].ID())
                ? 1.0
                : 0.0);
    }
    Substrate es_substrate;
    es_substrate.m_input_coords = {{-1.0, 0.0}, {0.0, 0.0}};
    es_substrate.m_output_coords = {{1.0, 0.0}};
    es_substrate.m_query_weights_only = true;
    es_substrate.m_max_weight_and_bias = 1.0;
    NeuralNetwork es_network;
    es_cppn.BuildESHyperNEATPhenotype(
        es_network, es_substrate, es_parameters);
    Check(es_network.m_neurons.size() == 7 &&
              es_network.m_connections.size() == 12,
          "ES-HyperNEAT discovers, connects, and retains a hidden geometry");
    std::set<std::pair<int, int>> es_endpoints;
    for (const auto& es_connection : es_network.m_connections)
    {
        es_endpoints.emplace(
            es_connection.m_source_neuron_idx,
            es_connection.m_target_neuron_idx);
    }
    Check(es_endpoints.size() == es_network.m_connections.size(),
          "ES-HyperNEAT deduplicates generated links");
    es_parameters.MaxDepth = 10;
    CheckThrows<std::invalid_argument>(
        [&] {
            es_cppn.BuildESHyperNEATPhenotype(
                es_network, es_substrate, es_parameters);
        },
        "ES-HyperNEAT rejects unsafe quadtree depth");

    Parameters novelty_parameters;
    novelty_parameters.PopulationSize = 2;
    novelty_parameters.NoveltySearch_K = 1;
    Genome novelty_seed = MakeGenome(novelty_parameters);
    Population novelty_population(
        novelty_seed, novelty_parameters, true, 1.0, 24601);
    std::vector<std::shared_ptr<PhenotypeBehavior>> novelty_behaviors{
        std::make_shared<ScalarBehavior>(0.0),
        std::make_shared<ScalarBehavior>(10.0)};
    novelty_population.InitPhenotypeBehaviorData(novelty_behaviors);
    ScalarBehavior detached_behavior(4.0);
    Genome detached = novelty_seed;
    detached.m_PhenotypeBehavior = &detached_behavior;
    Check(novelty_population.ComputeSparseness(detached) == 4.0,
          "detached novelty queries retain their nearest neighbor");
    Check(
        novelty_population.ComputeSparseness(
            novelty_population.AccessGenomeByIndex(0)) == 10.0,
        "novelty sparseness excludes the queried behavior by identity");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All core tests passed\n";
    return 0;
}
