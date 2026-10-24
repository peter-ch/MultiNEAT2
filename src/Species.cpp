#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "Genome.h"
#include "Serialization.h"
#include "Species.h"
#include "Random.h"
#include "Population.h"
#include "Utils.h"
#include "Parameters.h"
#include "Assert.h"

namespace NEAT
{
    double FitnessForOrdering(double fitness)
    {
        return std::isfinite(fitness)
            ? fitness
            : std::numeric_limits<double>::lowest();
    }

    void NormalizeSelectionWeights(std::vector<double>& weights)
    {
        if (weights.empty())
        {
            throw std::invalid_argument("Selection requires at least one candidate");
        }

        double minimum = weights.front();
        for (double weight : weights)
        {
            if (!std::isfinite(weight))
            {
                throw std::invalid_argument(
                    "Selection weights must contain only finite values");
            }
            minimum = std::min(minimum, weight);
        }
        if (minimum < 0.0)
        {
            const long double shifted_maximum =
                static_cast<long double>(
                    *std::max_element(weights.begin(), weights.end())) -
                static_cast<long double>(minimum);
            for (double &weight : weights)
            {
                weight = shifted_maximum > 0.0L
                    ? static_cast<double>(
                          (static_cast<long double>(weight) -
                           static_cast<long double>(minimum)) /
                          shifted_maximum)
                    : 1.0;
            }
        }
        else
        {
            const double maximum =
                *std::max_element(weights.begin(), weights.end());
            if (maximum > 0.0)
            {
                for (double& weight : weights)
                    weight /= maximum;
            }
        }
    }

    CrossoverMode SelectCrossoverMode(
        const Parameters& parameters,
        RNG& rng)
    {
        const double draw = rng.RandFloat();
        double cumulative = parameters.MultipointCrossoverRate;
        if (draw < cumulative)
            return MULTIPOINT;
        cumulative += parameters.SinglePointCrossoverRate;
        if (draw < cumulative)
            return SINGLE_POINT;
        cumulative += parameters.BlendCrossoverRate;
        if (draw < cumulative)
            return BLEND;
        cumulative += parameters.SimulatedBinaryCrossoverRate;
        if (draw < cumulative)
            return SIMULATED_BINARY;
        return AVERAGE;
    }

    bool genome_greater(const Genome& ls, const Genome& rs)
    {
        return FitnessForOrdering(ls.GetFitness()) >
               FitnessForOrdering(rs.GetFitness());
    }

    bool idxfitnesspair_greater(const std::pair<int, double>& ls,
                                const std::pair<int, double>& rs)
    {
        return FitnessForOrdering(ls.second) >
               FitnessForOrdering(rs.second);
    }

    bool GenomesAreClones(
        Genome& lhs,
        Genome& rhs,
        Parameters& parameters)
    {
        if (lhs.IsIdenticalTo(rhs))
            return true;
        return parameters.MinDeltaCompatEqualGenomes > 0.0 &&
               lhs.CompatibilityDistance(rhs, parameters) <=
                   parameters.MinDeltaCompatEqualGenomes;
    }

    double Median(std::vector<double> values)
    {
        if (values.empty())
            return 0.0;
        std::sort(values.begin(), values.end());
        const std::size_t middle = values.size() / 2U;
        return values.size() % 2U == 0U
            ? 0.5 * (values[middle - 1U] + values[middle])
            : values[middle];
    }

    double GammaSample(double shape, RNG& rng)
    {
        if (shape <= 0.0 || !std::isfinite(shape))
            throw std::invalid_argument("Gamma shape must be finite and positive");
        if (shape < 1.0)
        {
            const double draw = std::max(
                rng.RandFloat(), std::numeric_limits<double>::min());
            return GammaSample(shape + 1.0, rng) *
                   std::pow(draw, 1.0 / shape);
        }
        const double d = shape - 1.0 / 3.0;
        const double c = 1.0 / std::sqrt(9.0 * d);
        for (;;)
        {
            const double normal = rng.RandNormal();
            const double factor = 1.0 + c * normal;
            if (factor <= 0.0)
                continue;
            const double cube = factor * factor * factor;
            const double uniform = std::max(
                rng.RandFloat(), std::numeric_limits<double>::min());
            if (uniform < 1.0 - 0.0331 * normal * normal * normal * normal ||
                std::log(uniform) <
                    0.5 * normal * normal + d * (1.0 - cube + std::log(cube)))
            {
                return d * cube;
            }
        }
    }

    double BetaSample(double alpha, double beta, RNG& rng)
    {
        const double lhs = GammaSample(alpha, rng);
        const double rhs = GammaSample(beta, rng);
        return lhs / (lhs + rhs);
    }

    double OperatorPriorProbability(
        const Parameters& parameters,
        OperatorKind operation)
    {
        switch (operation)
        {
        case WEIGHT_MUTATION_OPERATOR:
            return std::clamp(parameters.MutateWeightsProb, 0.0, 1.0);
        case STRUCTURAL_ADD_OPERATOR:
            return std::clamp(
                parameters.MutateAddNeuronProb +
                    parameters.MutateAddLinkProb,
                0.0,
                1.0);
        case STRUCTURAL_REMOVE_OPERATOR:
            return std::clamp(
                parameters.MutateRemSimpleNeuronProb +
                    parameters.MutateRemLinkProb,
                0.0,
                1.0);
        case PARAMETER_MUTATION_OPERATOR:
            return std::clamp(
                parameters.MutateNeuronActivationTypeProb +
                    parameters.MutateActivationAProb +
                    parameters.MutateActivationBProb +
                    parameters.MutateNeuronTimeConstantsProb +
                    parameters.MutateNeuronBiasesProb +
                    parameters.MutateNeuronSpikingParametersProb +
                    parameters.MutateLinkSpikingParametersProb +
                    parameters.MutateNeuronTraitsProb +
                    parameters.MutateLinkTraitsProb +
                    parameters.MutateGenomeTraitsProb,
                0.0,
                1.0);
        case CROSSOVER_OPERATOR:
            return std::clamp(parameters.CrossoverRate, 0.0, 1.0);
        case WEIGHT_EMITTER_OPERATOR:
            return parameters.EnableWeightEmitter ? 0.5 : 0.0;
        default:
            return 0.0;
        }
    }

    std::string TopologySignature(const Genome& genome)
    {
        std::ostringstream signature;
        signature << genome.m_NeuronGenes.size() << ':';
        for (const auto& neuron : genome.m_NeuronGenes)
        {
            signature << neuron.m_ID << ','
                      << static_cast<int>(neuron.m_Type) << ';';
        }
        signature << '|' << genome.m_LinkGenes.size() << ':';
        for (const auto& link : genome.m_LinkGenes)
        {
            signature << link.m_InnovationID << ',' << link.m_FromNeuronID
                      << ',' << link.m_ToNeuronID << ','
                      << link.m_IsRecurrent << ';';
        }
        return signature.str();
    }

    double NormalizeContinuous(double value, double minimum, double maximum)
    {
        if (maximum <= minimum)
            return 0.5;
        return std::clamp((value - minimum) / (maximum - minimum), 0.0, 1.0);
    }

    double DenormalizeContinuous(
        double value,
        double minimum,
        double maximum)
    {
        if (maximum <= minimum)
            return minimum;
        return minimum + std::clamp(value, 0.0, 1.0) *
            (maximum - minimum);
    }

    std::vector<double> NormalizedContinuousParameters(
        const Genome& genome,
        const Parameters& parameters)
    {
        std::vector<double> values;
        values.reserve(
            genome.m_NeuronGenes.size() * 16U +
            genome.m_LinkGenes.size() * 9U);
        const auto append = [&values](double value, double minimum, double maximum)
        {
            values.push_back(NormalizeContinuous(value, minimum, maximum));
        };
        for (const auto& neuron : genome.m_NeuronGenes)
        {
            append(neuron.m_A, parameters.MinActivationA, parameters.MaxActivationA);
            append(neuron.m_B, parameters.MinActivationB, parameters.MaxActivationB);
            append(neuron.m_TimeConstant, parameters.MinNeuronTimeConstant, parameters.MaxNeuronTimeConstant);
            append(neuron.m_Bias, parameters.MinNeuronBias, parameters.MaxNeuronBias);
            const bool izhikevich = neuron.m_ActFunction == SPIKING_IZHIKEVICH;
            append(
                neuron.m_SpikeThreshold,
                izhikevich ? parameters.MinIzhikevichThreshold : parameters.MinSpikeThreshold,
                izhikevich ? parameters.MaxIzhikevichThreshold : parameters.MaxSpikeThreshold);
            append(neuron.m_ResetPotential, parameters.MinResetPotential, parameters.MaxResetPotential);
            append(neuron.m_RestingPotential, parameters.MinRestingPotential, parameters.MaxRestingPotential);
            append(neuron.m_RefractoryPeriod, parameters.MinRefractoryPeriod, parameters.MaxRefractoryPeriod);
            append(neuron.m_MembraneResistance, parameters.MinMembraneResistance, parameters.MaxMembraneResistance);
            append(neuron.m_AdaptationTimeConstant, parameters.MinAdaptationTimeConstant, parameters.MaxAdaptationTimeConstant);
            append(neuron.m_AdaptationIncrement, parameters.MinAdaptationIncrement, parameters.MaxAdaptationIncrement);
            append(neuron.m_RateTimeConstant, parameters.MinSpikeRateTimeConstant, parameters.MaxSpikeRateTimeConstant);
            append(neuron.m_IzhikevichA, parameters.MinIzhikevichA, parameters.MaxIzhikevichA);
            append(neuron.m_IzhikevichB, parameters.MinIzhikevichB, parameters.MaxIzhikevichB);
            append(neuron.m_IzhikevichC, parameters.MinIzhikevichC, parameters.MaxIzhikevichC);
            append(neuron.m_IzhikevichD, parameters.MinIzhikevichD, parameters.MaxIzhikevichD);
        }
        for (const auto& link : genome.m_LinkGenes)
        {
            append(link.m_Weight, parameters.MinWeight, parameters.MaxWeight);
            append(link.m_SynapticDelay, parameters.MinSynapticDelay, parameters.MaxSynapticDelay);
            append(link.m_SynapticTimeConstant, parameters.MinSynapticTimeConstant, parameters.MaxSynapticTimeConstant);
            append(link.m_STDPPlus, parameters.MinSTDPPlus, parameters.MaxSTDPPlus);
            append(link.m_STDPMinus, parameters.MinSTDPMinus, parameters.MaxSTDPMinus);
            append(link.m_STDPTauPlus, parameters.MinSTDPTau, parameters.MaxSTDPTau);
            append(link.m_STDPTauMinus, parameters.MinSTDPTau, parameters.MaxSTDPTau);
            append(link.m_STDPMinWeight, parameters.MinWeight, parameters.MaxWeight);
            append(link.m_STDPMaxWeight, parameters.MinWeight, parameters.MaxWeight);
        }
        return values;
    }

    void SetNormalizedContinuousParameters(
        Genome& genome,
        const std::vector<double>& values,
        const Parameters& parameters)
    {
        const std::size_t expected =
            genome.m_NeuronGenes.size() * 16U +
            genome.m_LinkGenes.size() * 9U;
        if (values.size() != expected)
            throw std::invalid_argument("Emitter parameter count does not match topology");
        std::size_t index = 0;
        const auto take = [&values, &index](double minimum, double maximum)
        {
            return DenormalizeContinuous(values[index++], minimum, maximum);
        };
        for (auto& neuron : genome.m_NeuronGenes)
        {
            neuron.m_A = take(parameters.MinActivationA, parameters.MaxActivationA);
            neuron.m_B = take(parameters.MinActivationB, parameters.MaxActivationB);
            neuron.m_TimeConstant = take(parameters.MinNeuronTimeConstant, parameters.MaxNeuronTimeConstant);
            neuron.m_Bias = take(parameters.MinNeuronBias, parameters.MaxNeuronBias);
            const bool izhikevich = neuron.m_ActFunction == SPIKING_IZHIKEVICH;
            neuron.m_SpikeThreshold = take(
                izhikevich ? parameters.MinIzhikevichThreshold : parameters.MinSpikeThreshold,
                izhikevich ? parameters.MaxIzhikevichThreshold : parameters.MaxSpikeThreshold);
            neuron.m_ResetPotential = take(parameters.MinResetPotential, parameters.MaxResetPotential);
            neuron.m_RestingPotential = take(parameters.MinRestingPotential, parameters.MaxRestingPotential);
            neuron.m_RefractoryPeriod = take(parameters.MinRefractoryPeriod, parameters.MaxRefractoryPeriod);
            neuron.m_MembraneResistance = take(parameters.MinMembraneResistance, parameters.MaxMembraneResistance);
            neuron.m_AdaptationTimeConstant = take(parameters.MinAdaptationTimeConstant, parameters.MaxAdaptationTimeConstant);
            neuron.m_AdaptationIncrement = take(parameters.MinAdaptationIncrement, parameters.MaxAdaptationIncrement);
            neuron.m_RateTimeConstant = take(parameters.MinSpikeRateTimeConstant, parameters.MaxSpikeRateTimeConstant);
            neuron.m_IzhikevichA = take(parameters.MinIzhikevichA, parameters.MaxIzhikevichA);
            neuron.m_IzhikevichB = take(parameters.MinIzhikevichB, parameters.MaxIzhikevichB);
            neuron.m_IzhikevichC = take(parameters.MinIzhikevichC, parameters.MaxIzhikevichC);
            neuron.m_IzhikevichD = take(parameters.MinIzhikevichD, parameters.MaxIzhikevichD);
        }
        for (auto& link : genome.m_LinkGenes)
        {
            link.m_Weight = take(parameters.MinWeight, parameters.MaxWeight);
            link.m_SynapticDelay = take(parameters.MinSynapticDelay, parameters.MaxSynapticDelay);
            link.m_SynapticTimeConstant = take(parameters.MinSynapticTimeConstant, parameters.MaxSynapticTimeConstant);
            link.m_STDPPlus = take(parameters.MinSTDPPlus, parameters.MaxSTDPPlus);
            link.m_STDPMinus = take(parameters.MinSTDPMinus, parameters.MaxSTDPMinus);
            link.m_STDPTauPlus = take(parameters.MinSTDPTau, parameters.MaxSTDPTau);
            link.m_STDPTauMinus = take(parameters.MinSTDPTau, parameters.MaxSTDPTau);
            link.m_STDPMinWeight = take(parameters.MinWeight, parameters.MaxWeight);
            link.m_STDPMaxWeight = take(parameters.MinWeight, parameters.MaxWeight);
            if (link.m_STDPMinWeight > link.m_STDPMaxWeight)
                std::swap(link.m_STDPMinWeight, link.m_STDPMaxWeight);
        }
    }

    int SelectAutomaticEpsilonLexicase(
        const Species& species,
        const std::vector<std::pair<int, double>>& evaluated,
        RNG& rng)
    {
        if (evaluated.size() < 2U)
            return evaluated.empty() ? -1 : evaluated.front().first;

        std::map<std::uint64_t, bool> shared_cases;
        const auto& first_samples = species.m_Individuals[
            static_cast<std::size_t>(evaluated.front().first)]
                                        .GetEvaluationSamples();
        for (const auto& sample : first_samples)
            shared_cases.emplace(sample.case_id, true);
        for (std::size_t candidate = 1;
             candidate < evaluated.size() && !shared_cases.empty();
             ++candidate)
        {
            const auto& samples = species.m_Individuals[
                static_cast<std::size_t>(evaluated[candidate].first)]
                                      .GetEvaluationSamples();
            for (auto it = shared_cases.begin(); it != shared_cases.end();)
            {
                const bool present = std::any_of(
                    samples.begin(), samples.end(),
                    [case_id = it->first](const EvaluationSample& sample)
                    { return sample.case_id == case_id; });
                it = present ? std::next(it) : shared_cases.erase(it);
            }
        }
        if (shared_cases.size() < 2U)
            return -1;

        std::vector<std::uint64_t> cases;
        cases.reserve(shared_cases.size());
        for (const auto& item : shared_cases)
            cases.push_back(item.first);
        for (std::size_t index = 0; index + 1U < cases.size(); ++index)
        {
            const std::size_t selected = static_cast<std::size_t>(rng.RandInt(
                static_cast<int>(index), static_cast<int>(cases.size() - 1U)));
            std::swap(cases[index], cases[selected]);
        }

        std::vector<int> survivors;
        survivors.reserve(evaluated.size());
        for (const auto& candidate : evaluated)
            survivors.push_back(candidate.first);
        const auto score_for = [&species](int individual, std::uint64_t case_id)
        {
            const auto& samples = species.m_Individuals[
                static_cast<std::size_t>(individual)].GetEvaluationSamples();
            const auto found = std::find_if(
                samples.begin(), samples.end(),
                [case_id](const EvaluationSample& sample)
                { return sample.case_id == case_id; });
            if (found == samples.end())
                throw std::logic_error("Lexicase shared case disappeared");
            return found->score;
        };

        for (const std::uint64_t case_id : cases)
        {
            std::vector<double> scores;
            scores.reserve(survivors.size());
            double best = std::numeric_limits<double>::lowest();
            for (int individual : survivors)
            {
                const double score = score_for(individual, case_id);
                scores.push_back(score);
                best = std::max(best, score);
            }
            const double median = Median(scores);
            std::vector<double> deviations;
            deviations.reserve(scores.size());
            for (double score : scores)
                deviations.push_back(std::abs(score - median));
            const double epsilon = Median(std::move(deviations));
            survivors.erase(
                std::remove_if(
                    survivors.begin(), survivors.end(),
                    [&](int individual)
                    { return score_for(individual, case_id) < best - epsilon; }),
                survivors.end());
            if (survivors.size() <= 1U)
                break;
        }
        if (survivors.empty())
            return -1;
        return survivors[static_cast<std::size_t>(rng.RandInt(
            0, static_cast<int>(survivors.size() - 1U)))];
    }


    // initializes a species with a representative genome and an ID number
    Species::Species(
        const Genome& a_Genome,
        const Parameters& a_Parameters,
        int a_ID)
        : m_ID(a_ID),
          m_BestSpecies(false),
          m_WorstSpecies(false),
          m_AgeGenerations(0),
          m_AgeEvaluations(0),
          m_OffspringRqd(0.0),
          m_BestFitness(
              a_Genome.IsEvaluated()
                  ? a_Genome.GetFitness()
                  : std::numeric_limits<double>::lowest()),
          m_BestGenome(a_Genome),
          m_GensNoImprovement(0),
          m_EvalsNoImprovement(0),
          m_R(0),
          m_G(0),
          m_B(0),
          m_AverageFitness(0.0),
          m_Individuals{a_Genome}
    {
        InitializeOperatorPriors(a_Parameters);
        // Derive a stable display color from the species ID. A process-global
        // RNG made independent populations and resumed checkpoints influence
        // one another and was not thread-safe.
        const std::uint32_t color =
            static_cast<std::uint32_t>(a_ID) * 2654435761U;
        m_R = static_cast<int>((color >> 16U) & 0xffU);
        m_G = 100 + static_cast<int>((color >> 8U) & 0x9bU);
        m_B = static_cast<int>(color & 0xffU);
    }

    void Species::InitializeOperatorPriors(const Parameters& parameters)
    {
        for (std::size_t index = 0; index < OPERATOR_KIND_COUNT; ++index)
        {
            const double prior = OperatorPriorProbability(
                parameters, static_cast<OperatorKind>(index));
            m_OperatorPriorAlpha[index] = 1.0 + 4.0 * prior;
            m_OperatorPriorBeta[index] = 1.0 + 4.0 * (1.0 - prior);
            m_OperatorAlpha[index] = m_OperatorPriorAlpha[index];
            m_OperatorBeta[index] = m_OperatorPriorBeta[index];
        }
    }

    double Species::SampleOperatorPreference(
        OperatorKind operation,
        RNG& rng) const
    {
        const std::size_t index = static_cast<std::size_t>(operation);
        if (index >= OPERATOR_KIND_COUNT)
            throw std::out_of_range("Invalid adaptive operator");
        return BetaSample(
            m_OperatorAlpha[index], m_OperatorBeta[index], rng);
    }

    double Species::AdaptiveCrossoverRate(
        const Parameters& parameters,
        RNG& rng) const
    {
        if (parameters.CrossoverRate <= 0.0)
            return 0.0;
        if (!parameters.AdaptiveOperatorSelection)
            return parameters.CrossoverRate;
        return std::max(
            parameters.OperatorExplorationFloor,
            SampleOperatorPreference(CROSSOVER_OPERATOR, rng));
    }

    unsigned int Species::EmitterBatchSize() const
    {
        const std::size_t parameter_count = m_EmitterMean.size();
        if (parameter_count == 0)
            return 0U;
        return 4U + static_cast<unsigned int>(
            std::floor(3.0 * std::log(static_cast<double>(parameter_count))));
    }

    bool Species::CreateEmitterOffspring(
        Genome& baby,
        LineageRecord& lineage,
        const Parameters& parameters,
        RNG& rng)
    {
        if (m_Individuals.empty() || m_EmitterMean.empty() ||
            m_EmitterMean.size() != m_EmitterDeviation.size())
        {
            return false;
        }
        const Genome& leader = m_Individuals.front();
        if (TopologySignature(leader) != m_EmitterTopologySignature ||
            leader.m_NeuronGenes.size() * 16U +
                    leader.m_LinkGenes.size() * 9U !=
                m_EmitterMean.size())
        {
            return false;
        }
        baby = leader;
        std::vector<double> parameters_vector(m_EmitterMean.size(), 0.0);
        bool changed = false;
        for (std::size_t index = 0; index < parameters_vector.size(); ++index)
        {
            parameters_vector[index] = std::clamp(
                m_EmitterMean[index] + m_EmitterSigma *
                    m_EmitterDeviation[index] * rng.RandNormal(),
                0.0,
                1.0);
            changed = changed ||
                parameters_vector[index] != m_EmitterMean[index];
        }
        if (!changed)
            return false;
        SetNormalizedContinuousParameters(baby, parameters_vector, parameters);
        lineage.parent_ids = {leader.GetID()};
        lineage.operators = {WEIGHT_EMITTER_OPERATOR};
        lineage.best_parent_score =
            FitnessForOrdering(leader.GetFitness());
        lineage.origin_species_id = m_ID;
        lineage.emitter_parameters = std::move(parameters_vector);
        return true;
    }

    void Species::UpdateAdaptiveState(
        const Population& population,
        const Parameters& parameters,
        RNG&)
    {
        if (m_Individuals.empty())
            return;

        if (parameters.AdaptiveOperatorSelection)
        {
            const double retain = 1.0 - parameters.OperatorPosteriorDecay;
            for (std::size_t index = 0; index < OPERATOR_KIND_COUNT; ++index)
            {
                m_OperatorAlpha[index] = m_OperatorPriorAlpha[index] +
                    (m_OperatorAlpha[index] - m_OperatorPriorAlpha[index]) *
                        retain;
                m_OperatorBeta[index] = m_OperatorPriorBeta[index] +
                    (m_OperatorBeta[index] - m_OperatorPriorBeta[index]) *
                        retain;
            }
            for (const auto& candidate_species : population.m_Species)
            {
                for (const auto& genome : candidate_species.m_Individuals)
                {
                    const LineageRecord& lineage = genome.GetLineage();
                    if (lineage.origin_species_id != m_ID ||
                        lineage.operators.empty())
                    {
                        continue;
                    }
                    double reward = 0.0;
                    if (lineage.archive_replacement)
                        reward = 1.0;
                    else if (genome.GetFitness() > lineage.best_parent_score)
                        reward = 0.5;
                    const double credit = reward /
                        static_cast<double>(lineage.operators.size());
                    for (OperatorKind operation : lineage.operators)
                    {
                        const std::size_t index =
                            static_cast<std::size_t>(operation);
                        if (index >= OPERATOR_KIND_COUNT)
                            continue;
                        m_OperatorAlpha[index] += credit;
                        m_OperatorBeta[index] += 1.0 - credit;
                    }
                }
            }
        }

        const Genome& leader = m_Individuals.front();
        const std::string signature = TopologySignature(leader);
        if (signature != m_EmitterTopologySignature)
        {
            m_EmitterTopologySignature = signature;
            m_EmitterStableGenerations = 1U;
            m_EmitterMean = NormalizedContinuousParameters(leader, parameters);
            m_EmitterDeviation.assign(m_EmitterMean.size(), 0.2);
            m_EmitterSigma = 0.5;
            return;
        }
        ++m_EmitterStableGenerations;

        std::vector<std::pair<double, std::vector<double>>> candidates;
        bool emitter_improved = false;
        for (const auto& candidate_species : population.m_Species)
        {
            for (const auto& genome : candidate_species.m_Individuals)
            {
                const LineageRecord& lineage = genome.GetLineage();
                if (lineage.origin_species_id == m_ID &&
                    std::find(
                        lineage.operators.begin(),
                        lineage.operators.end(),
                        WEIGHT_EMITTER_OPERATOR) != lineage.operators.end() &&
                    lineage.emitter_parameters.size() == m_EmitterMean.size() &&
                    TopologySignature(genome) == m_EmitterTopologySignature)
                {
                    // Non-finite legacy fitness would break the strict weak
                    // ordering required by stable_sort below.
                    candidates.emplace_back(
                        FitnessForOrdering(genome.GetFitness()),
                        lineage.emitter_parameters);
                    emitter_improved = emitter_improved ||
                        genome.GetFitness() > lineage.best_parent_score;
                }
            }
        }
        if (candidates.size() < 2U)
            return;
        std::stable_sort(
            candidates.begin(), candidates.end(),
            [](const auto& lhs, const auto& rhs)
            { return lhs.first > rhs.first; });
        const std::size_t selected_count = std::max<std::size_t>(
            1U, candidates.size() / 2U);
        std::vector<double> next_mean(m_EmitterMean.size(), 0.0);
        double total_weight = 0.0;
        for (std::size_t rank = 0; rank < selected_count; ++rank)
        {
            const double weight = std::log(
                static_cast<double>(selected_count) + 0.5) -
                std::log(static_cast<double>(rank) + 1.0);
            total_weight += weight;
            for (std::size_t index = 0; index < next_mean.size(); ++index)
                next_mean[index] += weight * candidates[rank].second[index];
        }
        for (double& value : next_mean)
            value /= total_weight;
        std::vector<double> next_deviation(next_mean.size(), 0.0);
        for (std::size_t rank = 0; rank < selected_count; ++rank)
        {
            const double weight = std::log(
                static_cast<double>(selected_count) + 0.5) -
                std::log(static_cast<double>(rank) + 1.0);
            for (std::size_t index = 0; index < next_mean.size(); ++index)
            {
                const double difference =
                    candidates[rank].second[index] - next_mean[index];
                next_deviation[index] += weight * difference * difference;
            }
        }
        for (double& value : next_deviation)
            value = std::max(0.03, std::sqrt(value / total_weight));
        m_EmitterSigma = std::clamp(
            m_EmitterSigma * (emitter_improved ? 1.05 : 0.95), 0.1, 2.0);
        m_EmitterMean = std::move(next_mean);
        m_EmitterDeviation = std::move(next_deviation);
    }


    // adds a new member to the species and updates variables
    void Species::AddIndividual(Genome& a_Genome)
    {
        m_Individuals.push_back(a_Genome);
    }


    // Individual selection routine
    Genome& Species::GetIndividual(Parameters& a_Parameters, RNG& a_RNG)
    {
        if (m_Individuals.size() == 0)
        {
            throw std::runtime_error(
                "Species::GetIndividual (ID:" + std::to_string(m_ID) +
                ") - No individuals in species");
        }

        // Make a pool of only evaluated individuals!
        std::vector< std::pair<int, double> > t_Evaluated;
        t_Evaluated.reserve(m_Individuals.size());
        for (unsigned int i = 0; i < m_Individuals.size(); i++)
        {
            if (m_Individuals[i].IsEvaluated())
            {
                t_Evaluated.push_back(std::make_pair(i, m_Individuals[i].GetAdjFitness()));
            }
        }

        // None are evaluated - cannot perform selection
        if (t_Evaluated.size() == 0)
        {
            throw std::runtime_error(
                "Species::GetIndividual (ID:" + std::to_string(m_ID) +
                ") - No evaluated individuals");
        }
        if (t_Evaluated.size() == 1)
        {
            return (m_Individuals[t_Evaluated[0].first]);
        }

        const SelectionMode selection_mode =
            a_Parameters.ParentSelectionMode;

        // Explicit rank and truncation modes are robust even when called
        // outside Epoch(), where the species may not have been sorted yet.
        if (selection_mode == TRUNCATION ||
            selection_mode == RANK_LINEAR ||
            selection_mode == RANK_EXP)
        {
            if (!std::is_sorted(
                    t_Evaluated.begin(),
                    t_Evaluated.end(),
                    idxfitnesspair_greater))
            {
                std::stable_sort(
                    t_Evaluated.begin(),
                    t_Evaluated.end(),
                    idxfitnesspair_greater);
            }
        }
        else if (selection_mode == LEGACY_SELECTION &&
                 a_Parameters.TruncationSelection)
        {
            // Legacy truncation historically relied on Epoch() sorting the
            // species best-first. Real-time Tick() does not sort, so order
            // the pool by raw fitness here. The stable sort is a no-op on an
            // already sorted species and leaves generational runs unchanged.
            const auto raw_greater =
                [this](const std::pair<int, double>& lhs,
                       const std::pair<int, double>& rhs)
            {
                return FitnessForOrdering(
                           m_Individuals[static_cast<std::size_t>(lhs.first)]
                               .GetFitness()) >
                       FitnessForOrdering(
                           m_Individuals[static_cast<std::size_t>(rhs.first)]
                               .GetFitness());
            };
            if (!std::is_sorted(
                    t_Evaluated.begin(), t_Evaluated.end(), raw_greater))
            {
                std::stable_sort(
                    t_Evaluated.begin(), t_Evaluated.end(), raw_greater);
            }
        }

        int t_chosen_one = 0;

        // Truncation selection goes first if enabled
        if ((selection_mode == LEGACY_SELECTION &&
             a_Parameters.TruncationSelection) ||
            selection_mode == TRUNCATION)
        {
            int t_num_parents = static_cast<int>(
                a_Parameters.SurvivalRate *
                static_cast<double>(t_Evaluated.size()));

            if (t_num_parents >= static_cast<int>(t_Evaluated.size()))
            {
                t_num_parents = static_cast<int>(t_Evaluated.size());
            }
            if (t_num_parents < 1)
            {
                t_num_parents = 1;
            }
            // do truncation here, and other extra selections can be applied below
            t_Evaluated.resize(t_num_parents);
        }

        if (selection_mode != LEGACY_SELECTION)
        {
            const std::size_t candidate_count = t_Evaluated.size();
            if (selection_mode == AUTO_EPSILON_LEXICASE)
            {
                const int selected = SelectAutomaticEpsilonLexicase(
                    *this, t_Evaluated, a_RNG);
                if (selected >= 0)
                    return m_Individuals[static_cast<std::size_t>(selected)];

                // Scalar-only callers fall back to stable linear ranking.
                std::stable_sort(
                    t_Evaluated.begin(), t_Evaluated.end(),
                    idxfitnesspair_greater);
                std::vector<double> weights(candidate_count, 1.0);
                if (candidate_count > 1U)
                {
                    const double count = static_cast<double>(candidate_count);
                    for (std::size_t rank = 0; rank < candidate_count; ++rank)
                    {
                        weights[rank] =
                            (2.0 - a_Parameters.RankSelectionPressure) / count +
                            2.0 * static_cast<double>(candidate_count - rank - 1U) *
                                (a_Parameters.RankSelectionPressure - 1.0) /
                                (count * (count - 1.0));
                    }
                }
                t_chosen_one = t_Evaluated[static_cast<std::size_t>(
                    a_RNG.Roulette(weights))].first;
                return m_Individuals[static_cast<std::size_t>(t_chosen_one)];
            }
            switch (selection_mode)
            {
            case TRUNCATION:
                t_chosen_one = t_Evaluated[static_cast<std::size_t>(
                    a_RNG.RandInt(
                        0, static_cast<int>(candidate_count) - 1))].first;
                break;

            case ROULETTE:
            {
                std::vector<double> weights;
                weights.reserve(candidate_count);
                for (const auto& candidate : t_Evaluated)
                    weights.push_back(candidate.second);
                NormalizeSelectionWeights(weights);
                t_chosen_one = t_Evaluated[static_cast<std::size_t>(
                    a_RNG.Roulette(weights))].first;
                break;
            }

            case RANK_LINEAR:
            {
                std::vector<double> weights(candidate_count, 1.0);
                if (candidate_count > 1)
                {
                    const double count =
                        static_cast<double>(candidate_count);
                    const double pressure =
                        a_Parameters.RankSelectionPressure;
                    for (std::size_t rank = 0;
                         rank < candidate_count;
                         ++rank)
                    {
                        weights[rank] =
                            (2.0 - pressure) / count +
                            2.0 *
                                static_cast<double>(
                                    candidate_count - rank - 1) *
                                (pressure - 1.0) /
                                (count * (count - 1.0));
                    }
                }
                t_chosen_one = t_Evaluated[static_cast<std::size_t>(
                    a_RNG.Roulette(weights))].first;
                break;
            }

            case RANK_EXP:
            {
                std::vector<double> weights(candidate_count, 1.0);
                if (candidate_count > 1)
                {
                    const double denominator =
                        static_cast<double>(candidate_count - 1);
                    for (std::size_t rank = 0;
                         rank < candidate_count;
                         ++rank)
                    {
                        weights[rank] = std::exp(
                            -a_Parameters.RankSelectionExponent *
                            static_cast<double>(rank) / denominator);
                    }
                }
                t_chosen_one = t_Evaluated[static_cast<std::size_t>(
                    a_RNG.Roulette(weights))].first;
                break;
            }

            case TOURNAMENT:
            {
                std::size_t winner = static_cast<std::size_t>(
                    a_RNG.RandInt(
                        0, static_cast<int>(candidate_count) - 1));
                for (unsigned int draw = 1;
                     draw < a_Parameters.TournamentSize;
                     ++draw)
                {
                    const std::size_t challenger =
                        static_cast<std::size_t>(a_RNG.RandInt(
                            0,
                            static_cast<int>(candidate_count) - 1));
                    if (t_Evaluated[challenger].second >
                        t_Evaluated[winner].second)
                    {
                        winner = challenger;
                    }
                }
                t_chosen_one = t_Evaluated[winner].first;
                break;
            }

            case STOCHASTIC:
            {
                // Fitness-proportionate stochastic acceptance avoids a
                // cumulative scan in the common case while retaining roulette
                // probabilities exactly.
                std::vector<double> weights;
                weights.reserve(candidate_count);
                for (const auto& candidate : t_Evaluated)
                    weights.push_back(candidate.second);
                NormalizeSelectionWeights(weights);
                const double maximum =
                    *std::max_element(weights.begin(), weights.end());
                if (maximum <= 0.0)
                {
                    t_chosen_one = t_Evaluated[static_cast<std::size_t>(
                        a_RNG.RandInt(
                            0,
                            static_cast<int>(candidate_count) - 1))].first;
                    break;
                }
                bool accepted = false;
                const std::size_t maximum_attempts =
                    std::max<std::size_t>(32, candidate_count * 4);
                for (std::size_t attempt = 0;
                     attempt < maximum_attempts;
                     ++attempt)
                {
                    const std::size_t candidate =
                        static_cast<std::size_t>(a_RNG.RandInt(
                            0,
                            static_cast<int>(candidate_count) - 1));
                    if (a_RNG.RandFloat() <
                        weights[candidate] / maximum)
                    {
                        t_chosen_one = t_Evaluated[candidate].first;
                        accepted = true;
                        break;
                    }
                }
                if (!accepted)
                {
                    t_chosen_one = t_Evaluated[static_cast<std::size_t>(
                        a_RNG.Roulette(weights))].first;
                }
                break;
            }

            case BOLTZMANN:
            {
                double maximum_fitness =
                    t_Evaluated.front().second;
                for (const auto& candidate : t_Evaluated)
                {
                    maximum_fitness =
                        std::max(maximum_fitness, candidate.second);
                }
                std::vector<double> weights;
                weights.reserve(candidate_count);
                for (const auto& candidate : t_Evaluated)
                {
                    weights.push_back(std::exp(
                        (candidate.second - maximum_fitness) /
                        a_Parameters.BoltzmannTemperature));
                }
                t_chosen_one = t_Evaluated[static_cast<std::size_t>(
                    a_RNG.Roulette(weights))].first;
                break;
            }

            case LEGACY_SELECTION:
            case AUTO_EPSILON_LEXICASE:
            default:
                throw std::invalid_argument(
                    "Unsupported explicit parent selection mode");
            }

            return m_Individuals[static_cast<std::size_t>(t_chosen_one)];
        }

        if (a_Parameters.TournamentSelection && (!a_Parameters.RouletteWheelSelection)) // pure tournament without roulette
        {
            if (a_Parameters.TournamentSize == 0)
            {
                throw std::invalid_argument("TournamentSize must be greater than zero");
            }
            std::vector< std::pair<int, double> > t_picked;
            t_picked.reserve(a_Parameters.TournamentSize);
            // choose N individuals at random
            for (unsigned int i = 0; i < a_Parameters.TournamentSize; ++i)
            {
                const int c = a_RNG.RandInt(
                    0, static_cast<int>(t_Evaluated.size()) - 1);
                t_picked.push_back(t_Evaluated[static_cast<std::size_t>(c)]);
            }

            // Proper tournament selection: select the best individual in the pool
            std::sort(t_picked.begin(), t_picked.end(), idxfitnesspair_greater);
            t_chosen_one = t_picked[0].first;
        }
        else if (a_Parameters.TournamentSelection && a_Parameters.RouletteWheelSelection) // tournament with roulette applied on the picked
        {
            if (a_Parameters.TournamentSize == 0)
            {
                throw std::invalid_argument("TournamentSize must be greater than zero");
            }
            std::vector< std::pair<int, double> > t_picked;
            t_picked.reserve(a_Parameters.TournamentSize);
            // choose N individuals at random
            for (unsigned int i = 0; i < a_Parameters.TournamentSize; ++i)
            {
                const int c = a_RNG.RandInt(
                    0, static_cast<int>(t_Evaluated.size()) - 1);
                t_picked.push_back(t_Evaluated[static_cast<std::size_t>(c)]);
            }

            // do a roulette on the picked
            std::vector<double> probs;
            probs.reserve(t_picked.size());
            for (auto p : t_picked)
            {
                probs.push_back(p.second);
            }
            NormalizeSelectionWeights(probs);
            t_chosen_one = t_picked[static_cast<std::size_t>(
                a_RNG.Roulette(probs))].first;
        }
        /*else if ((!a_Parameters.RouletteWheelSelection) && (!a_Parameters.TournamentSelection)) // both off means pure truncation selection
        {
            // Truncation selection based on evaluated individuals
            int t_num_parents = (int)(a_Parameters.SurvivalRate * (double)(t_Evaluated.size()));

            if (t_num_parents >= t_Evaluated.size())
            {
                t_num_parents = t_Evaluated.size() - 1;
            }
            if (t_num_parents < 1)
            {
                t_num_parents = 1;
            }

            t_chosen_one = t_Evaluated[a_RNG.RandInt(0, t_num_parents)].first;
        }*/
        else if ((a_Parameters.RouletteWheelSelection) && (!a_Parameters.TournamentSelection)) // only roulette
        {
            // Roulette wheel selection 
            std::vector<double> t_probs;
            t_probs.reserve(t_Evaluated.size());
            for (const auto &evaluated : t_Evaluated)
            {
                t_probs.push_back(evaluated.second);
            }
            NormalizeSelectionWeights(t_probs);
            t_chosen_one = t_Evaluated[static_cast<std::size_t>(
                a_RNG.Roulette(t_probs))].first;
        }
        else
        {
            // default is pure truncation or just random search if truncation is off - the array has been resized already
            t_chosen_one = t_Evaluated[static_cast<std::size_t>(
                a_RNG.RandInt(
                    0, static_cast<int>(t_Evaluated.size()) - 1))].first;
        }

        return (m_Individuals[t_chosen_one]);
    }


    // returns a completely random individual
    Genome& Species::GetRandomIndividual(RNG& a_RNG) 
    {
        if (m_Individuals.size() == 0) // no members yet, return representative
        {
            throw std::runtime_error(
                "Attempted GetRandomIndividual() but no individuals in species ID " +
                std::to_string(m_ID));
        }
        else
            if (m_Individuals.size() == 1)
            {
                return m_Individuals[0];
            }
            else
            {
                int t_rand_choice = 0;
                t_rand_choice = a_RNG.RandInt(0, static_cast<int>(m_Individuals.size() - 1));
                return (m_Individuals[t_rand_choice]);
            }
    }

    // returns the leader (the member having the best fitness)
    Genome& Species::GetLeader() //const
    {
        // if empty, return representative
        if (m_Individuals.size() == 0)
        {
            throw std::runtime_error(
                "Attempted GetLeader() but no individuals in species ID " +
                std::to_string(m_ID));
        }

        double t_max_fitness = std::numeric_limits<double>::lowest();
        std::size_t t_leader_idx = 0;
        bool found_evaluated = false;
        for (unsigned int i = 0; i < m_Individuals.size(); i++)
        {
            if (!m_Individuals[i].IsEvaluated())
                continue;
            const double t_f =
                FitnessForOrdering(m_Individuals[i].GetFitness());
            if (!found_evaluated || t_f > t_max_fitness)
            {
                t_max_fitness = t_f;
                t_leader_idx = i;
                found_evaluated = true;
            }
        }
        if (!found_evaluated)
        {
            for (std::size_t i = 0; i < m_Individuals.size(); ++i)
            {
                const double fitness =
                    FitnessForOrdering(m_Individuals[i].GetFitness());
                if (i == 0 || fitness > t_max_fitness)
                {
                    t_max_fitness = fitness;
                    t_leader_idx = i;
                }
            }
        }

        return (m_Individuals[t_leader_idx]);
    }


    Genome& Species::GetRepresentative() //const
    {
        if (m_Individuals.size() > 0)
        {
            return m_Individuals[0];
        }
        else
        {
            throw std::runtime_error(
                "Attempted GetRepresentative() but no individuals in species ID " +
                std::to_string(m_ID));
        }
    }

    // calculates how many offspring this species should spawn
    void Species::CountOffspring()
    {
        m_OffspringRqd = 0;

        for (unsigned int i = 0; i < m_Individuals.size(); i++)
        {
            m_OffspringRqd += m_Individuals[i].GetOffspringAmount();
        }
    }


    // this method performs fitness sharing
    // it also boosts the fitness of the young and penalizes old species
    void Species::AdjustFitness(Parameters& a_Parameters)
    {
        double minimum_fitness = 0.0;
        bool found_finite = false;
        for (const auto &genome : m_Individuals)
        {
            if (std::isfinite(genome.GetFitness()))
            {
                minimum_fitness = found_finite
                    ? std::min(minimum_fitness, genome.GetFitness())
                    : genome.GetFitness();
                found_finite = true;
            }
        }
        const double offset =
            !found_finite || minimum_fitness <= 0.0
                ? -minimum_fitness + 1.0e-7
                : 0.0;
        AdjustFitness(a_Parameters, offset);
    }

    void Species::AdjustFitness(
        Parameters& a_Parameters, double a_FitnessOffset)
    {
        std::vector<double> transformed;
        transformed.reserve(m_Individuals.size());
        for (const auto& genome : m_Individuals)
        {
            const double fitness = genome.GetFitness();
            const long double shifted =
                std::isfinite(fitness)
                    ? static_cast<long double>(fitness) +
                          static_cast<long double>(a_FitnessOffset)
                    : 1.0e-7L;
            transformed.push_back(static_cast<double>(std::clamp(
                shifted,
                1.0e-7L,
                static_cast<long double>(
                    std::numeric_limits<double>::max()))));
        }
        AdjustFitness(a_Parameters, transformed);
    }

    void Species::AdjustFitness(
        Parameters& a_Parameters,
        const std::vector<double>& a_TransformedFitness)
    {
        if (m_Individuals.empty())
        {
            throw std::runtime_error(
                "Cannot adjust fitness for an empty species");
        }
        if (a_TransformedFitness.size() != m_Individuals.size())
        {
            throw std::invalid_argument(
                "Transformed fitness count must match species size");
        }

        // iterate through the members
        for (unsigned int i = 0; i < m_Individuals.size(); i++)
        {
            const double raw_fitness = m_Individuals[i].GetFitness();
            // Invalid fitness never becomes a champion and receives only the
            // smallest usable allocation weight.
            const bool valid_fitness = std::isfinite(raw_fitness);

            // update the best fitness and stagnation counter
            if (valid_fitness && raw_fitness > m_BestFitness)
            {
                if (m_BestFitness == std::numeric_limits<double>::lowest() ||
                    static_cast<long double>(raw_fitness) -
                            static_cast<long double>(m_BestFitness) >=
                        a_Parameters.StagnationDelta)
                {
                    m_GensNoImprovement = 0;
                }
                m_BestFitness = raw_fitness;
                m_BestGenome = m_Individuals[i];
            }

            long double t_fitness =
                static_cast<long double>(a_TransformedFitness[i]);
            if (!std::isfinite(a_TransformedFitness[i]) ||
                t_fitness <= 0.0L)
                t_fitness = 1.0e-12L;


            // boost the fitness up to some young age
            if (m_AgeGenerations < a_Parameters.YoungAgeTreshold)
            {
                t_fitness *= a_Parameters.YoungAgeFitnessBoost;
            }

            // penalty for old species
            if (m_AgeGenerations > a_Parameters.OldAgeTreshold)
            {
                t_fitness *= a_Parameters.OldAgePenalty;
            }

            // extreme penalty if this species is stagnating for too long time
            // one exception if this is the best species found so far
            if (m_GensNoImprovement > a_Parameters.SpeciesMaxStagnation)
            {
                // the best species is always allowed to live
                if (!m_BestSpecies)
                {
                    // when the fitness is lowered that much, the species will
                    // likely have 0 offspring and therefore will not survive
                    t_fitness *= a_Parameters.StagnationPenalty;
                }
            }

            // Compute the adjusted fitness for this member
            const long double adjusted =
                t_fitness /
                static_cast<long double>(m_Individuals.size());
            m_Individuals[i].SetAdjFitness(static_cast<double>(
                std::clamp(
                    adjusted,
                    0.0L,
                    static_cast<long double>(
                        std::numeric_limits<double>::max()))));
        }
    }


    void Species::SortIndividuals()
    {
        std::sort(m_Individuals.begin(), m_Individuals.end(), genome_greater);
    }


    // Removes an individual from the species by its index within the species
    void Species::RemoveIndividual(unsigned int a_idx)
    {
        if (a_idx >= m_Individuals.size())
        {
            throw std::out_of_range("Species individual index is out of range");
        }
        m_Individuals.erase(
            m_Individuals.begin() + static_cast<std::ptrdiff_t>(a_idx));
    }

    // Reproduce mates & mutates the individuals of the species
    // It may access the global species list in the population
    // because some babies may turn out to belong in another species
    // that have to be created.
    // Also calls Birth() for every new baby
    void Species::Reproduce(Population& a_Pop, Parameters& a_Parameters, RNG& a_RNG)
    {
        Genome t_baby; // temp genome for reproduction

        unsigned int t_offspring_count = Rounded(GetOffspringRqd());
        const unsigned int total_offspring_count = t_offspring_count;
        // ensure we have a champ when enabled
        unsigned int elite_offspring = 0;
        unsigned int elite_count = 0;
        if (a_Parameters.EliteFraction > 0)
        {
            elite_offspring = Rounded(a_Parameters.EliteFraction * m_Individuals.size());
            if (elite_offspring < 1) // can't be 0
            {
                elite_offspring = 1;
            }
            elite_offspring = std::min(
                {elite_offspring,
                 t_offspring_count,
                 static_cast<unsigned int>(m_Individuals.size())});
        }

        unsigned int emitter_remaining = 0U;
        const unsigned int emitter_batch = EmitterBatchSize();
        if (a_Parameters.EnableWeightEmitter && emitter_batch > 0U &&
            m_EmitterStableGenerations >=
                a_Parameters.WeightEmitterStableGenerations &&
            total_offspring_count >= 2U * emitter_batch)
        {
            emitter_remaining = std::min(
                emitter_batch,
                total_offspring_count - elite_offspring);
        }

        const auto select_parent = [&]()
        {
            Genome parent = GetIndividual(a_Parameters, a_RNG);
            Genome archive_parent;
            if (a_Parameters.EvolutionModeSetting == ROBUST_QD_NEAT &&
                a_RNG.RandFloat() < 0.5 &&
                a_Pop.TryGetQDParent(parent, archive_parent))
            {
                return archive_parent;
            }
            return parent;
        };

        // no offspring?! yikes.. dead species!
        if (t_offspring_count == 0)
        {
            // maybe do something else?
            return;
        }

        //////////////////////////
        // Reproduction

        // Spawn t_offspring_count babies
        bool t_baby_exists_in_pop = false;
        while (t_offspring_count--)
        {
            // clear baby just in case
            t_baby = Genome();
            LineageRecord t_lineage;
            t_lineage.origin_species_id = m_ID;

            // Select the elite first..

            if (elite_count < elite_offspring)
            {
                // SortIndividuals() runs before reproduction, so this copies
                // distinct top genomes rather than cloning one leader N
                // times when EliteFraction selects multiple elites.
                t_baby = m_Individuals[elite_count];
                t_lineage.parent_ids = {t_baby.GetID()};
                t_lineage.best_parent_score =
                    FitnessForOrdering(t_baby.GetFitness());
                elite_count++;
            }
            else
            {
                unsigned int t_constraint_trials = a_Parameters.ConstraintTrials; // to prevent infinite loops

                do // - while the baby already exists somewhere in the new population or turned invalid in some way
                {
                    t_lineage = LineageRecord{};
                    t_lineage.origin_species_id = m_ID;
                    // this tells us if the baby is a result of mating
                    bool t_mated = false;
                    const bool t_emitted = emitter_remaining > 0U &&
                        CreateEmitterOffspring(
                            t_baby,
                            t_lineage,
                            a_Parameters,
                            a_RNG);
                    if (t_emitted)
                        --emitter_remaining;

                    // There must be individuals there..
                    ASSERT(NumIndividuals() > 0);

                    if (!t_emitted)
                    {
                    // for a species of size 1 we can only mutate
                    // NOTE: but does it make sense since we know this is the champ?
                    if (NumIndividuals() == 1)
                    {
                        t_baby = select_parent();
                        t_lineage.parent_ids = {t_baby.GetID()};
                        t_lineage.best_parent_score =
                            FitnessForOrdering(t_baby.GetFitness());
                        t_mated = false;
                    }
                    // else we can mate
                    else
                    {
                        // choose whether to mate at all
                        // Do not allow crossover when in simplifying phase
                        if ((a_RNG.RandFloat() <
                             AdaptiveCrossoverRate(a_Parameters, a_RNG)) &&
                            (a_Pop.GetSearchMode() != SIMPLIFYING))
                        {
                            // get the father
                            Genome t_mom;
                            Genome t_dad;
                            bool t_interspecies = false;

                            // There is a probability that the father may come from another species
                            if ((a_RNG.RandFloat() < a_Parameters.InterspeciesCrossoverRate) &&
                                (a_Pop.m_Species.size() > 1))
                            {
                                /// Find different species via roulette over average fitness as probability
                                std::vector<double> probs;
                                std::vector<bool> eligible;
                                double minimum = 0.0;
                                bool have_eligible = false;
                                for (std::size_t i = 0;
                                     i < a_Pop.m_Species.size();
                                     ++i)
                                {
                                    if ((a_Pop.m_Species[i].m_ID == m_ID))
                                    {
                                        probs.push_back(0.0);
                                        eligible.push_back(false);
                                    }
                                    else
                                    {
                                        const double fitness =
                                            a_Pop.m_Species[i].GetLeader().GetAdjFitness();
                                        probs.push_back(fitness);
                                        eligible.push_back(true);
                                        minimum = have_eligible
                                            ? std::min(minimum, fitness)
                                            : fitness;
                                        have_eligible = true;
                                    }
                                }
                                if (have_eligible)
                                {
                                    if (minimum < 0.0)
                                    {
                                        for (std::size_t i = 0;
                                             i < probs.size(); ++i)
                                        {
                                            if (eligible[i])
                                                probs[i] -= minimum;
                                        }
                                    }
                                    bool any_positive = false;
                                    for (std::size_t i = 0;
                                         i < probs.size(); ++i)
                                    {
                                        any_positive =
                                            any_positive ||
                                            (eligible[i] && probs[i] > 0.0);
                                    }
                                    if (!any_positive)
                                    {
                                        for (std::size_t i = 0;
                                             i < probs.size(); ++i)
                                        {
                                            probs[i] = eligible[i] ? 1.0 : 0.0;
                                        }
                                    }
                                    int t_diffspec = a_RNG.Roulette(probs);
                                    t_mom = select_parent();
                                    t_dad = a_Pop.m_Species[t_diffspec].GetIndividual(a_Parameters, a_RNG);
                                    t_interspecies = true;
                                }
                                // Without an eligible partner species, fall back to
                                // within-species mating below instead of discarding
                                // this attempt (which burned a constraint trial and
                                // could leave an empty baby).
                            }
                            if (!t_interspecies)
                            {
                                // Mate within species
                                t_mom = select_parent();
                                t_dad = select_parent();

                                // The other parent should be a different one
                                // number of tries to find different parent
                                int t_tries = 32;
                                while (((t_mom.GetID() == t_dad.GetID())) && (t_tries--))
                                {
                                    t_mom = select_parent();
                                    t_dad = select_parent();
                                }

                                t_interspecies = false;
                            }

                            // OK we have both mom and dad so mate them.
                            t_baby = t_mom.MateWithMode(
                                t_dad,
                                SelectCrossoverMode(
                                    a_Parameters, a_RNG),
                                t_interspecies,
                                a_RNG,
                                a_Parameters);

                            t_mated = true;
                            t_lineage.parent_ids = {
                                t_mom.GetID(), t_dad.GetID()};
                            t_lineage.best_parent_score = std::max(
                                FitnessForOrdering(t_mom.GetFitness()),
                                FitnessForOrdering(t_dad.GetFitness()));
                            t_lineage.operators.push_back(
                                CROSSOVER_OPERATOR);
                        }
                        // don't mate - reproduce one individual asexually
                        else
                        {
                            t_baby = select_parent();
                            t_lineage.parent_ids = {t_baby.GetID()};
                            t_lineage.best_parent_score =
                                FitnessForOrdering(t_baby.GetFitness());
                            t_mated = false;
                        }
                    }

                    // Mutate the baby
                    if ((!t_mated) || (a_RNG.RandFloat() < a_Parameters.OverallMutationRate))
                    {
                        MutateGenome(
                            false,
                            a_Pop,
                            t_baby,
                            a_Parameters,
                            a_RNG,
                            &t_lineage.operators);
                    }
                    }
                    // Structural mutations can append an older innovation
                    // reused from the database. Canonical ordering keeps
                    // exact clone checks and archives independent of mutation
                    // history.
                    t_baby.SortGenes();

                    // Check if this baby is already present somewhere in the offspring
                    // we don't want that
                    t_baby_exists_in_pop = false;
                    // Unless of course, we want clones to exist
                    if (!a_Parameters.AllowClones)
                    {
                        for (unsigned int i = 0; i < a_Pop.m_TempSpecies.size(); i++)
                        {
                            for (unsigned int j = 0; j < a_Pop.m_TempSpecies[i].m_Individuals.size(); j++)
                            {
                                if (GenomesAreClones(
                                        t_baby,
                                        a_Pop.m_TempSpecies[i].m_Individuals[j],
                                        a_Parameters))
                                {
                                    t_baby_exists_in_pop = true;
                                    break;
                                }
                            }
                        }
                    }

                    // In case we want to enforce always new individuals
                    if (a_Parameters.ArchiveEnforcement)
                    {
                        for (unsigned int i = 0; i < a_Pop.m_GenomeArchive.size(); i++)
                        {
                            if (GenomesAreClones(
                                    t_baby,
                                    a_Pop.m_GenomeArchive[i],
                                    a_Parameters))
                            {
                                t_baby_exists_in_pop = true;
                                break;
                            }
                        }
                    }
                } while ((t_baby_exists_in_pop ||
                          t_baby.FailsConstraints(a_Parameters)) &&
                         (t_constraint_trials--));

                if (t_baby_exists_in_pop || t_baby.FailsConstraints(a_Parameters))
                {
                    throw std::runtime_error(
                        "Unable to reproduce a valid, non-cloning genome within ConstraintTrials");
                }
            }

            // We have a new offspring now
            // give the offspring a new ID
            t_baby.SetID(a_Pop.GetNextGenomeID());
            a_Pop.IncrementNextGenomeID();

            // sort the baby's genes
            t_baby.SortGenes();

            // clear the baby's fitness
            t_baby.SetFitness(0);
            t_baby.SetAdjFitness(0);
            t_baby.SetOffspringAmount(0);
            t_baby.SetLineage(t_lineage);

            t_baby.ResetEvaluated();

            // Archive the baby if needed
            if (a_Parameters.ArchiveEnforcement)
            {
                a_Pop.m_GenomeArchive.push_back(t_baby);
            }

            //////////////////////////////////
            // put the baby to its species  //
            //////////////////////////////////

            // before Reproduce() is invoked, it is assumed that a
            // clone of the population exists with the name of m_TempSpecies
            // we will store results there.
            // after all reproduction completes, the original species will be replaced back

            auto t_cur_species = a_Pop.m_TempSpecies.begin();

            // No species yet?
            if (t_cur_species == a_Pop.m_TempSpecies.end())
            {
                // create the first species and place the baby there
                a_Pop.m_TempSpecies.push_back(Species(t_baby, a_Parameters, a_Pop.GetNextSpeciesID()));
                a_Pop.IncrementNextSpeciesID();
            }
            else
            {
                // try to find a compatible species
                Genome t_to_compare = t_cur_species->GetRepresentative(); 

                bool t_found = false;
                while ((t_cur_species != a_Pop.m_TempSpecies.end()) && (!t_found))
                {
                    if (t_baby.IsCompatibleWith(t_to_compare, a_Parameters))
                    {
                        // found a compatible species
                        t_cur_species->AddIndividual(t_baby);
                        t_found = true; // the search is over
                    }
                    else
                    {
                        // keep searching for a matching species
                        while (1)
                        {
                            t_cur_species++;
                            if (t_cur_species == a_Pop.m_TempSpecies.end())
                            {
                                break;
                            }
                            if (t_cur_species->NumIndividuals() > 0)
                            {
                                t_to_compare = t_cur_species->GetRepresentative();
                                break;
                            }
                        }
                    }
                }

                // if couldn't find a match, make a new species
                if (!t_found)
                {
                    a_Pop.m_TempSpecies.push_back(Species(t_baby, a_Parameters, a_Pop.GetNextSpeciesID()));
                    a_Pop.IncrementNextSpeciesID();
                }
            }
        }
    }


    ////////////
    // Real-time code
    void Species::CalculateAverageFitness()
    {
        long double scale = 0.0L;
        int t_num_individuals = 0;

        // consider individuals that were evaluated only!
        for (const auto& individual : m_Individuals)
        {
            if (individual.IsEvaluated())
            {
                const double fitness = individual.GetFitness();
                if (std::isfinite(fitness))
                {
                    scale = std::max(
                        scale,
                        std::abs(
                            static_cast<long double>(fitness)));
                }
                ++t_num_individuals;
            }
        }

        if (t_num_individuals > 0)
        {
            if (scale <= 0.0L)
                scale = 1.0L;
            long double scaled_total = 0.0L;
            for (const auto& individual : m_Individuals)
            {
                if (individual.IsEvaluated() &&
                    std::isfinite(individual.GetFitness()))
                {
                    scaled_total +=
                        static_cast<long double>(
                            individual.GetFitness()) /
                        scale;
                }
            }
            const long double average =
                scale * scaled_total /
                static_cast<long double>(t_num_individuals);
            m_AverageFitness = static_cast<double>(std::clamp(
                average,
                -static_cast<long double>(
                    std::numeric_limits<double>::max()),
                static_cast<long double>(
                    std::numeric_limits<double>::max())));
        }
        else
        {
            m_AverageFitness = 0;
        }
    }


    Genome Species::ReproduceOne(Population& a_Pop, Parameters& a_Parameters, RNG& a_RNG)
    {
        //////////////////////////
        // Reproduction
        bool t_baby_exists_in_pop = false;
        int t_constraint_trials = a_Parameters.ConstraintTrials;

        // Spawn only one baby
        Genome t_baby; // for storing the result
        LineageRecord lineage;
        const auto select_parent = [&]()
        {
            Genome parent = GetIndividual(a_Parameters, a_RNG);
            Genome archive_parent;
            if (a_Parameters.EvolutionModeSetting == ROBUST_QD_NEAT &&
                a_RNG.RandFloat() < 0.5 &&
                a_Pop.TryGetQDParent(parent, archive_parent))
            {
                return archive_parent;
            }
            return parent;
        };

        do // - while the baby turned invalid in some way
        {
            t_baby = Genome(); // clear baby
            lineage = LineageRecord{};
            lineage.origin_species_id = m_ID;

            // this tells us if the baby is a result of mating
            bool t_mated = false;

            // There must be individuals there..
            ASSERT(NumIndividuals() > 0);

            // for a species of size 1 we can only mutate
            // NOTE: but does it make sense since we know this is the champ?
            if (NumIndividuals() == 1)
            {
                t_baby = select_parent();
                lineage.parent_ids = {t_baby.GetID()};
                lineage.best_parent_score =
                    FitnessForOrdering(t_baby.GetFitness());
                t_mated = false;
            }
            // else we can mate
            else
            {
                // choose whether to mate at all
                // Do not allow crossover when in simplifying phase
                if ((a_RNG.RandFloat() <
                     AdaptiveCrossoverRate(a_Parameters, a_RNG)) &&
                    (a_Pop.GetSearchMode() != SIMPLIFYING))
                {
                    // get the mother and father
                    Genome t_mom;
                    Genome t_dad;
                    bool t_interspecies = false;

                    // There is a probability that the father may come from another species
                    if ((a_RNG.RandFloat() < a_Parameters.InterspeciesCrossoverRate) &&
                        (a_Pop.m_Species.size() > 1))
                    {
                        // Find different species via roulette over average fitness as probability
                        std::vector<double> probs;
                        std::vector<bool> eligible;
                        double minimum = 0.0;
                        bool have_eligible = false;
                        for (std::size_t i = 0;
                             i < a_Pop.m_Species.size();
                             ++i)
                        {
                            if ((a_Pop.m_Species[i].m_ID == m_ID) || (a_Pop.m_Species[i].NumEvaluated() == 0))
                            {
                                probs.push_back(0.0);
                                eligible.push_back(false);
                            }
                            else
                            {
                                const double fitness =
                                    std::isfinite(
                                        a_Pop.m_Species[i].m_AverageFitness)
                                    ? a_Pop.m_Species[i].m_AverageFitness
                                    : 0.0;
                                probs.push_back(fitness);
                                eligible.push_back(true);
                                minimum = have_eligible
                                    ? std::min(minimum, fitness)
                                    : fitness;
                                have_eligible = true;
                            }
                        }
                        if (have_eligible)
                        {
                            if (minimum < 0.0)
                            {
                                for (std::size_t i = 0;
                                     i < probs.size(); ++i)
                                {
                                    if (eligible[i])
                                        probs[i] -= minimum;
                                }
                            }
                            bool any_positive = false;
                            for (std::size_t i = 0;
                                 i < probs.size(); ++i)
                            {
                                any_positive =
                                    any_positive ||
                                    (eligible[i] && probs[i] > 0.0);
                            }
                            if (!any_positive)
                            {
                                for (std::size_t i = 0;
                                     i < probs.size(); ++i)
                                {
                                    probs[i] = eligible[i] ? 1.0 : 0.0;
                                }
                            }
                            int t_diffspec = a_RNG.Roulette(probs);
                            t_mom = select_parent();
                            t_dad = a_Pop.m_Species[t_diffspec].GetIndividual(a_Parameters, a_RNG);
                            t_interspecies = true;
                        }
                        // Without an eligible partner species, fall back to
                        // within-species mating below instead of discarding
                        // this attempt (which burned a constraint trial and
                        // could leave an empty baby).
                    }
                    if (!t_interspecies)
                    {
                        // Mate within species
                        t_mom = select_parent();
                        t_dad = select_parent();

                        // The other parent should be a different one
                        // number of tries to find different parent
                        // we can mate the same mom and dad and still get different baby
                        int t_tries = 32;
                        while (((t_mom.GetID() == t_dad.GetID())) && (t_tries--))
                        {
                            t_mom = select_parent();
                            t_dad = select_parent();
                        }
                        t_interspecies = false;
                    }

                    // OK we have both mom and dad so mate them.
                    t_baby = t_mom.MateWithMode(
                        t_dad,
                        SelectCrossoverMode(a_Parameters, a_RNG),
                        t_interspecies,
                        a_RNG,
                        a_Parameters);

#ifdef VDEBUG
                    std::cout << "mated baby\n";
#endif
                    t_mated = true;
                    lineage.parent_ids = {t_mom.GetID(), t_dad.GetID()};
                    lineage.best_parent_score = std::max(
                        FitnessForOrdering(t_mom.GetFitness()),
                        FitnessForOrdering(t_dad.GetFitness()));
                    lineage.operators.push_back(CROSSOVER_OPERATOR);
                }
                // don't mate - reproduce one individual asexually
                else
                {
                    t_baby = select_parent();
                    lineage.parent_ids = {t_baby.GetID()};
                    lineage.best_parent_score =
                        FitnessForOrdering(t_baby.GetFitness());
                    t_mated = false;
                }
            }

            // Mutate the baby
            if ((!t_mated) || (a_RNG.RandFloat() < a_Parameters.OverallMutationRate))
            {
                MutateGenome(
                    false,
                    a_Pop,
                    t_baby,
                    a_Parameters,
                    a_RNG,
                    &lineage.operators);
#ifdef VDEBUG
                std::cout << "mutated baby\n";
#endif
            }
            // See the generational reproduction path above.
            t_baby.SortGenes();

            // Check if this baby is already present somewhere in the offspring
            // we don't want that
            t_baby_exists_in_pop = false;
            // Unless of course, we want clones to exist
            if (!a_Parameters.AllowClones)
            {
                for (unsigned int i = 0; i < a_Pop.m_Species.size(); i++)
                {
                    for (unsigned int j = 0; j < a_Pop.m_Species[i].m_Individuals.size(); j++)
                    {
                        if (GenomesAreClones(
                                t_baby,
                                a_Pop.m_Species[i].m_Individuals[j],
                                a_Parameters))
                        {
                            t_baby_exists_in_pop = true;
                            break;
                        }
                    }
                }
            }

            // In case we want to enforce always new individuals
            if (a_Parameters.ArchiveEnforcement && (!t_baby_exists_in_pop))
            {
                for (unsigned int i = 0; i < a_Pop.m_GenomeArchive.size(); i++)
                {
                        if (GenomesAreClones(
                                t_baby,
                                a_Pop.m_GenomeArchive[i],
                                a_Parameters))
                    {
                        t_baby_exists_in_pop = true;
                        break;
                    }
                }
            }
        } while ((t_baby_exists_in_pop ||
                  t_baby.FailsConstraints(a_Parameters)) &&
                 (t_constraint_trials--));

        if (t_baby_exists_in_pop || t_baby.FailsConstraints(a_Parameters))
        {
            throw std::runtime_error(
                "Unable to reproduce a valid, non-cloning genome within ConstraintTrials");
        }


        // We have a new offspring now
        // give the offspring a new ID
        t_baby.SetID(a_Pop.GetNextGenomeID());
        a_Pop.IncrementNextGenomeID();

        // sort the baby's genes
        t_baby.SortGenes();

        // clear the baby's fitness
        t_baby.SetFitness(0);
        t_baby.SetAdjFitness(0);
        t_baby.SetOffspringAmount(0);
        t_baby.SetLineage(lineage);

        t_baby.ResetEvaluated();

        // Compute the baby's behavior if possible, before it's added to the species
        // In case of archiving, add the new baby to the archive
        if (a_Parameters.ArchiveEnforcement)
        {
            a_Pop.m_GenomeArchive.push_back(t_baby);
        }

#ifdef VDEBUG
        std::cout << "baby success\n";
#endif

        return t_baby;
    }


    // Mutates a genome
    void
        Species::MutateGenome(
            bool t_baby_is_clone,
            Population& a_Pop,
            Genome& t_baby,
            Parameters& a_Parameters,
            RNG& a_RNG,
            std::vector<OperatorKind>* applied_operators)
    {
        // We will perform roulette wheel selection to choose the type of mutation and will mutate the baby
        // This method guarantees that the baby will be mutated at least with one mutation
        enum MutationTypes
        {
            ADD_NODE = 0, ADD_LINK, REMOVE_NODE, REMOVE_LINK, CHANGE_ACTIVATION_FUNCTION,
            MUTATE_WEIGHTS, MUTATE_ACTIVATION_A, MUTATE_ACTIVATION_B, MUTATE_TIMECONSTS, MUTATE_BIASES,
            MUTATE_NEURON_SPIKING, MUTATE_LINK_SPIKING,
            MUTATE_NEURON_TRAITS, MUTATE_LINK_TRAITS, MUTATE_GENOME_TRAITS
        };
        std::vector<double> t_mut_probs;

        // ADD_NODE;
        t_mut_probs.push_back(a_Parameters.MutateAddNeuronProb);

        // ADD_LINK;
        t_mut_probs.push_back(a_Parameters.MutateAddLinkProb);

        // REMOVE_NODE;
        t_mut_probs.push_back(a_Parameters.MutateRemSimpleNeuronProb);

        // REMOVE_LINK;
        t_mut_probs.push_back(a_Parameters.MutateRemLinkProb);

        // CHANGE_ACTIVATION_FUNCTION;
        t_mut_probs.push_back(a_Parameters.MutateNeuronActivationTypeProb);

        // MUTATE_WEIGHTS;
        t_mut_probs.push_back(a_Parameters.MutateWeightsProb);

        // MUTATE_ACTIVATION_A;
        t_mut_probs.push_back(a_Parameters.MutateActivationAProb);

        // MUTATE_ACTIVATION_B;
        t_mut_probs.push_back(a_Parameters.MutateActivationBProb);

        // MUTATE_TIMECONSTS;
        t_mut_probs.push_back(a_Parameters.MutateNeuronTimeConstantsProb);

        // MUTATE_BIASES;
        t_mut_probs.push_back(a_Parameters.MutateNeuronBiasesProb);

        // MUTATE_NEURON_SPIKING;
        t_mut_probs.push_back(
            a_Parameters.MutateNeuronSpikingParametersProb);

        // MUTATE_LINK_SPIKING;
        t_mut_probs.push_back(
            a_Parameters.MutateLinkSpikingParametersProb);

        // MUTATE_NEURON_TRAITS;
        t_mut_probs.push_back(a_Parameters.MutateNeuronTraitsProb);

        // MUTATE_LINK_TRAITS;
        t_mut_probs.push_back(a_Parameters.MutateLinkTraitsProb);

        // MUTATE_GENOME_TRAITS;
        t_mut_probs.push_back(a_Parameters.MutateGenomeTraitsProb);

        // Special consideration for phased searching - do not allow certain mutations depending on the search mode
        // also don't use additive mutations if we just want to get rid of the clones
        if ((a_Pop.GetSearchMode() == SIMPLIFYING) || t_baby_is_clone)
        {
            t_mut_probs[ADD_NODE] = 0; // add node
            t_mut_probs[ADD_LINK] = 0; // add link
        }
        if ((a_Pop.GetSearchMode() == COMPLEXIFYING) || t_baby_is_clone)
        {
            t_mut_probs[REMOVE_NODE] = 0; // rem node
            t_mut_probs[REMOVE_LINK] = 0; // rem link
        }
        if (a_Parameters.MaxNeurons >= 0)
        {
            const int added_neurons = std::max(
                0,
                static_cast<int>(t_baby.NumNeurons()) -
                    t_baby.m_initial_num_neurons);
            if (added_neurons >= a_Parameters.MaxNeurons)
                t_mut_probs[ADD_NODE] = 0.0;
        }
        if (a_Parameters.MaxLinks >= 0)
        {
            const int added_links = std::max(
                0,
                static_cast<int>(t_baby.NumLinks()) -
                    t_baby.m_initial_num_links);
            if (added_links >= a_Parameters.MaxLinks)
            {
                t_mut_probs[ADD_LINK] = 0.0;
                // Splitting a link adds one net connection.
                t_mut_probs[ADD_NODE] = 0.0;
            }
        }

        const auto mutation_category = [](int mutation)
        {
            if (mutation == ADD_NODE || mutation == ADD_LINK)
                return STRUCTURAL_ADD_OPERATOR;
            if (mutation == REMOVE_NODE || mutation == REMOVE_LINK)
                return STRUCTURAL_REMOVE_OPERATOR;
            if (mutation == MUTATE_WEIGHTS)
                return WEIGHT_MUTATION_OPERATOR;
            return PARAMETER_MUTATION_OPERATOR;
        };
        if (a_Parameters.AdaptiveOperatorSelection)
        {
            std::vector<double> category_weights(4U, 0.0);
            for (int mutation = ADD_NODE;
                 mutation <= MUTATE_GENOME_TRAITS;
                 ++mutation)
            {
                if (t_mut_probs[static_cast<std::size_t>(mutation)] > 0.0)
                {
                    const auto operation = mutation_category(mutation);
                    category_weights[static_cast<std::size_t>(operation)] =
                        std::max(
                            a_Parameters.OperatorExplorationFloor,
                            SampleOperatorPreference(operation, a_RNG));
                }
            }
            const auto selected = static_cast<OperatorKind>(
                a_RNG.Roulette(category_weights));
            for (int mutation = ADD_NODE;
                 mutation <= MUTATE_GENOME_TRAITS;
                 ++mutation)
            {
                if (mutation_category(mutation) != selected)
                    t_mut_probs[static_cast<std::size_t>(mutation)] = 0.0;
            }
        }

        bool has_possible_mutation = false;
        for (double probability : t_mut_probs)
        {
            if (!std::isfinite(probability) || probability < 0.0)
            {
                throw std::invalid_argument(
                    "Mutation probabilities must be finite and non-negative");
            }
            has_possible_mutation = has_possible_mutation || probability > 0.0;
        }
        if (!has_possible_mutation)
        {
            throw std::runtime_error(
                "No mutation type is enabled for the current search mode");
        }

        bool t_mutation_success = false;
        int chosen_mutation = -1;

        // Some structural mutations can be impossible for a particular genome.
        // Bound retries so invalid configurations fail instead of hanging forever.
        for (int mutation_attempt = 0;
             mutation_attempt < 512 && !t_mutation_success;
             ++mutation_attempt)
        {
            int ChosenMutation = a_RNG.Roulette(t_mut_probs);
            chosen_mutation = ChosenMutation;

            // Now mutate based on the choice
            switch (ChosenMutation)
            {
            case ADD_NODE:
                t_mutation_success = t_baby.Mutate_AddNeuron(a_Pop.AccessInnovationDatabase(), a_Parameters, a_RNG);
                break;

            case ADD_LINK:
                t_mutation_success = t_baby.Mutate_AddLink(a_Pop.AccessInnovationDatabase(), a_Parameters, a_RNG);
                break;

            case REMOVE_NODE:
                t_mutation_success = t_baby.Mutate_RemoveSimpleNeuron(a_Pop.AccessInnovationDatabase(), a_Parameters, a_RNG);
                break;

            case REMOVE_LINK:
            {
                // Keep doing this mutation until it is sure that the baby will not
                // end up having dead ends or no links
                Genome t_saved_baby = t_baby;
                bool t_no_links = false, t_has_dead_ends = false;

                int t_tries = 128;
                do
                {
                    t_tries--;
                    if (t_tries <= 0)
                    {
                        t_saved_baby = t_baby;
                        t_mutation_success = false;
                        break; // give up
                    }

                    t_saved_baby = t_baby;
                    t_mutation_success = t_saved_baby.Mutate_RemoveLink(a_RNG);

                    t_no_links = t_saved_baby.NumLinks() == 0;
                    t_has_dead_ends = t_saved_baby.HasDeadEnds();

                } while (t_no_links || t_has_dead_ends);

                t_baby = t_saved_baby;
            }
            break;

            case CHANGE_ACTIVATION_FUNCTION:
                t_mutation_success = t_baby.Mutate_NeuronActivation_Type(a_Parameters, a_RNG);
                break;

            case MUTATE_WEIGHTS:
                t_mutation_success = t_baby.Mutate_LinkWeights(a_Parameters, a_RNG);
                break;

            case MUTATE_ACTIVATION_A:
                t_mutation_success = t_baby.Mutate_NeuronActivations_A(a_Parameters, a_RNG);
                break;

            case MUTATE_ACTIVATION_B:
                t_mutation_success = t_baby.Mutate_NeuronActivations_B(a_Parameters, a_RNG);
                break;

            case MUTATE_TIMECONSTS:
                t_mutation_success = t_baby.Mutate_NeuronTimeConstants(a_Parameters, a_RNG);
                break;

            case MUTATE_BIASES:
                t_mutation_success = t_baby.Mutate_NeuronBiases(a_Parameters, a_RNG);
                break;

            case MUTATE_NEURON_SPIKING:
                t_mutation_success =
                    t_baby.Mutate_NeuronSpikingParameters(
                        a_Parameters, a_RNG);
                break;

            case MUTATE_LINK_SPIKING:
                t_mutation_success =
                    t_baby.Mutate_LinkSpikingParameters(
                        a_Parameters, a_RNG);
                break;

            case MUTATE_NEURON_TRAITS:
                t_mutation_success = t_baby.Mutate_NeuronTraits(a_Parameters, a_RNG);
                break;

            case MUTATE_LINK_TRAITS:
                t_mutation_success = t_baby.Mutate_LinkTraits(a_Parameters, a_RNG);
                break;

            case MUTATE_GENOME_TRAITS:
                t_mutation_success = t_baby.Mutate_GenomeTraits(a_Parameters, a_RNG);
                break;

            default:
                t_mutation_success = false;
                break;
            }
        }
        if (!t_mutation_success)
        {
            if (a_Parameters.AdaptiveOperatorSelection)
            {
                // The sampled arm can be enabled globally yet infeasible for
                // this exact topology (for example, removing from a minimal
                // graph). Retry once across all currently feasible operators
                // instead of failing the generation.
                Parameters fallback = a_Parameters;
                fallback.AdaptiveOperatorSelection = false;
                fallback.MutationOperatorsPerOffspring = 1.0;
                fallback.AdaptiveMutationRate = 0.0;
                MutateGenome(
                    t_baby_is_clone,
                    a_Pop,
                    t_baby,
                    fallback,
                    a_RNG,
                    applied_operators);
                return;
            }
            throw std::runtime_error(
                "No enabled mutation could be applied to the genome");
        }
        if (applied_operators != nullptr && chosen_mutation >= 0)
            applied_operators->push_back(mutation_category(chosen_mutation));

        double mutation_budget =
            a_Parameters.MutationOperatorsPerOffspring;
        if (a_Parameters.AdaptiveMutationRate > 0.0 &&
            a_Pop.GetStagnation() >
                a_Parameters.AdaptiveMutationStart)
        {
            const double stagnant_generations =
                static_cast<double>(
                    a_Pop.GetStagnation() -
                    a_Parameters.AdaptiveMutationStart);
            const double factor = std::min(
                a_Parameters.AdaptiveMutationMaxFactor,
                1.0 + a_Parameters.AdaptiveMutationRate *
                          stagnant_generations);
            mutation_budget *= factor;
        }
        unsigned int operator_count =
            static_cast<unsigned int>(std::floor(mutation_budget));
        const double fractional =
            mutation_budget - std::floor(mutation_budget);
        if (fractional > 0.0 && a_RNG.RandFloat() < fractional)
            ++operator_count;
        operator_count = std::max(1U, operator_count);

        if (operator_count > 1)
        {
            // Reuse the validated single-operator path. Disabling adaptation
            // in the local copy bounds recursion at exactly one level while
            // recomputing topology-dependent mutation availability after
            // every successful operation.
            Parameters single_operator = a_Parameters;
            single_operator.MutationOperatorsPerOffspring = 1.0;
            single_operator.AdaptiveMutationRate = 0.0;
            single_operator.AdaptiveMutationMaxFactor = 1.0;
            for (unsigned int operation = 1;
                 operation < operator_count;
                 ++operation)
            {
                MutateGenome(
                    t_baby_is_clone,
                    a_Pop,
                    t_baby,
                    single_operator,
                    a_RNG,
                    applied_operators);
            }
        }
    }

std::string Species::Serialize() const
{
    std::ostringstream output;
    output.precision(std::numeric_limits<double>::max_digits10);
    output << "SpeciesStart\n";
    output << "SpeciesFormat 3\n";
    output << m_ID << ' ' << m_BestSpecies << ' ' << m_WorstSpecies << ' '
           << m_AgeGenerations << ' ' << m_AgeEvaluations << ' '
           << m_OffspringRqd << ' ' << m_BestFitness << ' '
           << m_GensNoImprovement << ' ' << m_EvalsNoImprovement << ' '
           << m_R << ' ' << m_G << ' ' << m_B << ' ' << m_AverageFitness
           << '\n';
    output << "BestGenome\n" << m_BestGenome.Serialize();
    output << "Individuals " << m_Individuals.size() << '\n';
    for (const auto &genome : m_Individuals)
        output << genome.Serialize();
    output << "OperatorState";
    for (double value : m_OperatorAlpha)
        output << ' ' << value;
    for (double value : m_OperatorBeta)
        output << ' ' << value;
    for (double value : m_OperatorPriorAlpha)
        output << ' ' << value;
    for (double value : m_OperatorPriorBeta)
        output << ' ' << value;
    output << '\n';
    output << "EmitterState " << m_EmitterStableGenerations << ' '
           << m_EmitterSigma << ' '
           << std::quoted(m_EmitterTopologySignature) << ' '
           << m_EmitterMean.size();
    for (double value : m_EmitterMean)
        output << ' ' << value;
    output << ' ' << m_EmitterDeviation.size();
    for (double value : m_EmitterDeviation)
        output << ' ' << value;
    output << '\n';
    output << "SpeciesEnd\n";
    return output.str();
}

Species Species::Deserialize(const std::string &data)
{
    std::istringstream input(data);
    Serialization::UseRoundTripParsing(input);
    std::string token;
    input >> token;
    if (token != "SpeciesStart")
        throw std::runtime_error(
            "Species::Deserialize: missing SpeciesStart marker.");

    Species species;
    int format_version = 1;
    input >> token;
    if (token == "SpeciesFormat")
    {
        input >> format_version;
        if (format_version != 2 && format_version != 3)
            throw std::runtime_error(
                "Species::Deserialize: unsupported format.");
        input >> species.m_ID;
    }
    else
    {
        try
        {
            species.m_ID = std::stoi(token);
        }
        catch (const std::exception&)
        {
            throw std::runtime_error(
                "Species::Deserialize: malformed species header.");
        }
    }

    input >> species.m_BestSpecies
          >> species.m_WorstSpecies
          >> species.m_AgeGenerations
          >> species.m_AgeEvaluations
          >> species.m_OffspringRqd
          >> species.m_BestFitness
          >> species.m_GensNoImprovement
          >> species.m_EvalsNoImprovement
          >> species.m_R >> species.m_G >> species.m_B
          >> species.m_AverageFitness;
    if (!input)
        throw std::runtime_error(
            "Species::Deserialize: malformed species state.");

    if (format_version >= 2)
    {
        input >> token;
        if (token != "BestGenome")
            throw std::runtime_error(
                "Species::Deserialize: missing BestGenome marker.");
        species.m_BestGenome = Genome(input);
        input >> token;
        if (token != "Individuals")
            throw std::runtime_error(
                "Species::Deserialize: missing Individuals marker.");
    }

    std::size_t count = 0;
    input >> count;
    species.m_Individuals.clear();
    species.m_Individuals.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        species.m_Individuals.emplace_back(input);

    if (format_version >= 3)
    {
        input >> token;
        if (token != "OperatorState")
        {
            throw std::runtime_error(
                "Species::Deserialize: missing OperatorState marker.");
        }
        for (double& value : species.m_OperatorAlpha)
            input >> value;
        for (double& value : species.m_OperatorBeta)
            input >> value;
        for (double& value : species.m_OperatorPriorAlpha)
            input >> value;
        for (double& value : species.m_OperatorPriorBeta)
            input >> value;

        std::size_t mean_count = 0;
        std::size_t deviation_count = 0;
        input >> token >> species.m_EmitterStableGenerations
              >> species.m_EmitterSigma
              >> std::quoted(species.m_EmitterTopologySignature)
              >> mean_count;
        if (token != "EmitterState")
        {
            throw std::runtime_error(
                "Species::Deserialize: missing EmitterState marker.");
        }
        species.m_EmitterMean.resize(mean_count);
        for (double& value : species.m_EmitterMean)
            input >> value;
        input >> deviation_count;
        species.m_EmitterDeviation.resize(deviation_count);
        for (double& value : species.m_EmitterDeviation)
            input >> value;
        if (!input || mean_count != deviation_count ||
            !std::isfinite(species.m_EmitterSigma) ||
            species.m_EmitterSigma <= 0.0)
        {
            throw std::runtime_error(
                "Species::Deserialize: invalid adaptive state.");
        }
    }

    input >> token;
    if (token != "SpeciesEnd")
        throw std::runtime_error(
            "Species::Deserialize: missing SpeciesEnd marker.");
    if (format_version == 1 && !species.m_Individuals.empty())
        species.m_BestGenome = species.GetLeader();
    return species;
}


} // namespace NEAT
