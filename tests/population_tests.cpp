// Regression tests for Population/Species selection, real-time evolution,
// novelty search, ownership and reliability bookkeeping.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Genome.h"
#include "Parameters.h"
#include "PhenotypeBehavior.h"
#include "Population.h"
#include "Random.h"
#include "Species.h"

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

template <typename Callable>
void CheckNoThrow(Callable&& callable, const std::string& message)
{
    try
    {
        callable();
    }
    catch (const std::exception& error)
    {
        Check(false, message + " (threw: " + error.what() + ")");
    }
}

NEAT::Genome MakeGenome(NEAT::Parameters& parameters)
{
    NEAT::GenomeInitStruct init;
    init.NumInputs = 3;
    init.NumOutputs = 1;
    return NEAT::Genome(parameters, init);
}

NEAT::Parameters LegacyParameters(unsigned int population_size)
{
    NEAT::Parameters parameters;
    parameters.ConfigureLegacyV2();
    parameters.PopulationSize = population_size;
    parameters.Speciation = false;
    parameters.DynamicCompatibility = false;
    parameters.AllowClones = true;
    return parameters;
}

// Behavior characterized by the sum of link weights. Archive entries are
// stored as sliced PhenotypeBehavior copies, so distances read only m_Data.
class WeightSumBehavior : public NEAT::PhenotypeBehavior
{
public:
    static bool succeed_next;

    bool Acquire(NEAT::Genome* genome) override
    {
        double sum = 0.0;
        for (const auto& link : genome->m_LinkGenes)
            sum += link.m_Weight;
        m_Data = {{sum}};
        const bool success = succeed_next;
        succeed_next = false;
        return success;
    }

    double Distance_To(NEAT::PhenotypeBehavior* other) override
    {
        if (m_Data.empty() || other->m_Data.empty())
            return 0.0;
        return std::abs(m_Data.front().front() - other->m_Data.front().front());
    }

    bool Successful() override { return false; }
};
bool WeightSumBehavior::succeed_next = false;

std::vector<std::shared_ptr<NEAT::PhenotypeBehavior>> InitNovelty(
    NEAT::Population& population)
{
    std::vector<std::shared_ptr<NEAT::PhenotypeBehavior>> behaviors;
    for (unsigned int i = 0; i < population.NumGenomes(); ++i)
        behaviors.push_back(std::make_shared<WeightSumBehavior>());
    population.InitPhenotypeBehaviorData(behaviors);
    for (unsigned int i = 0; i < population.NumGenomes(); ++i)
    {
        NEAT::Genome& genome =
            population.AccessGenomeByIndex(static_cast<int>(i));
        genome.m_PhenotypeBehavior->Acquire(&genome);
    }
    for (unsigned int i = 0; i < population.NumGenomes(); ++i)
    {
        NEAT::Genome& genome =
            population.AccessGenomeByIndex(static_cast<int>(i));
        genome.SetFitness(population.ComputeSparseness(genome));
        genome.SetEvaluated();
    }
    return behaviors;
}

// Finding 1A: Tick must compute adjusted fitness so fitness-proportionate
// selection is not uniform in real-time mode.
void TestTickRouletteUsesAdjustedFitness()
{
    using namespace NEAT;
    Parameters parameters = LegacyParameters(10);
    parameters.TruncationSelection = false;
    parameters.RouletteWheelSelection = true;
    parameters.TournamentSelection = false;
    parameters.CrossoverRate = 0.0;
    Genome seed = MakeGenome(parameters);
    Population population(seed, parameters, true, 1.0, 1111);

    int champion_id = -1;
    for (unsigned int i = 0; i < population.NumGenomes(); ++i)
    {
        Genome& genome = population.AccessGenomeByIndex(static_cast<int>(i));
        genome.SetFitness(i == 0 ? 1000.0 : 1.0);
        genome.SetEvaluated();
        if (i == 0)
            champion_id = genome.GetID();
    }

    int champion_parented = 0;
    const int ticks = 200;
    for (int tick = 0; tick < ticks; ++tick)
    {
        Genome deleted;
        Genome* baby = population.Tick(deleted);
        Check(deleted.GetID() != champion_id,
              "real-time removal never deletes the dominant champion");
        const auto& parents = baby->GetLineage().parent_ids;
        if (parents.size() == 1 && parents.front() == champion_id)
            ++champion_parented;
        baby->SetFitness(1.0);
        baby->SetEvaluated();
    }
    std::cout << "roulette: champion parented " << champion_parented << '/'
              << ticks << " real-time offspring\n";
    Check(champion_parented > ticks * 8 / 10,
          "real-time roulette selection follows adjusted fitness "
          "(champion should parent ~99% of offspring)");
}

// Finding 1B: legacy truncation must truncate a fitness-sorted pool even
// when Tick has not sorted the species.
void TestTickLegacyTruncationSorts()
{
    using namespace NEAT;
    Parameters parameters = LegacyParameters(10);
    parameters.TruncationSelection = true;
    parameters.RouletteWheelSelection = false;
    parameters.TournamentSelection = false;
    parameters.SurvivalRate = 0.2;
    parameters.CrossoverRate = 0.0;
    Genome seed = MakeGenome(parameters);
    Population population(seed, parameters, true, 1.0, 2222);

    // Ascending fitness: species order is the worst-first order.
    int champion_id = -1;
    for (unsigned int i = 0; i < population.NumGenomes(); ++i)
    {
        Genome& genome = population.AccessGenomeByIndex(static_cast<int>(i));
        genome.SetFitness(static_cast<double>(i));
        genome.SetEvaluated();
        champion_id = genome.GetID();
    }

    int champion_parented = 0;
    const int ticks = 40;
    for (int tick = 0; tick < ticks; ++tick)
    {
        Genome deleted;
        Genome* baby = population.Tick(deleted);
        const auto& parents = baby->GetLineage().parent_ids;
        if (parents.size() == 1 && parents.front() == champion_id)
            ++champion_parented;
        // Offspring are poor and are the next ones to be replaced.
        baby->SetFitness(-100.0);
        baby->SetEvaluated();
    }
    std::cout << "truncation: champion parented " << champion_parented << '/'
              << ticks << " real-time offspring\n";
    Check(champion_parented == ticks,
          "legacy truncation in Tick selects from the best survivors");
}

// Finding 2: novelty babies must become evaluated so they can be parents.
void TestNoveltySearchMarksBabiesEvaluated()
{
    using namespace NEAT;
    Parameters parameters = LegacyParameters(6);
    parameters.NoveltySearch_K = 3;
    Genome seed = MakeGenome(parameters);
    Population population(seed, parameters, true, 1.0, 3333);
    const auto behaviors = InitNovelty(population);

    const unsigned int ticks = 4U * parameters.PopulationSize;
    unsigned int completed = 0;
    CheckNoThrow(
        [&]
        {
            for (unsigned int tick = 0; tick < ticks; ++tick)
            {
                Genome result;
                population.NoveltySearchTick(result);
                Check(result.IsEvaluated(),
                      "novelty search returns an evaluated genome");
                ++completed;
            }
        },
        "novelty search runs for more than 2*PopulationSize ticks");
    Check(completed == ticks, "every novelty tick completed");
    for (const auto& species : population.m_Species)
    {
        for (const auto& genome : species.m_Individuals)
        {
            Check(genome.IsEvaluated(),
                  "every novelty individual is evaluated after its tick");
        }
    }

    // Success path.
    WeightSumBehavior::succeed_next = true;
    Genome successful;
    bool success = false;
    CheckNoThrow([&] { success = population.NoveltySearchTick(successful); },
                 "successful novelty tick");
    Check(success, "novelty search reports an acquired success");
    Check(successful.IsEvaluated() &&
              std::isfinite(successful.GetFitness()),
          "successful novelty genome is evaluated with finite fitness");
    bool stored_evaluated = false;
    for (const auto& species : population.m_Species)
    {
        for (const auto& genome : species.m_Individuals)
        {
            if (genome.GetID() == successful.GetID())
                stored_evaluated = genome.IsEvaluated();
        }
    }
    Check(stored_evaluated,
          "successful novelty genome is evaluated inside the population");
    CheckNoThrow(
        [&]
        {
            for (unsigned int tick = 0; tick < ticks; ++tick)
            {
                Genome result;
                population.NoveltySearchTick(result);
            }
        },
        "novelty search continues after a success");
}

bool AllLineageScoresFinite(const NEAT::Population& population)
{
    for (const auto& species : population.m_Species)
    {
        for (const auto& genome : species.m_Individuals)
        {
            if (!std::isfinite(genome.GetLineage().best_parent_score))
                return false;
        }
    }
    return true;
}

// Finding 3: non-finite legacy fitness must not poison lineage records.
void TestNonFiniteFitnessIsSanitized()
{
    using namespace NEAT;
    Parameters parameters = LegacyParameters(10);
    // Uniform parent choice so non-finite parents are regularly selected.
    parameters.TruncationSelection = false;
    parameters.RouletteWheelSelection = false;
    parameters.TournamentSelection = false;
    parameters.AdaptiveOperatorSelection = true;
    Genome seed = MakeGenome(parameters);
    Population population(seed, parameters, true, 1.0, 4444);

    const double poison[] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    for (int generation = 0; generation < 4; ++generation)
    {
        for (unsigned int i = 0; i < population.NumGenomes(); ++i)
        {
            Genome& genome =
                population.AccessGenomeByIndex(static_cast<int>(i));
            // The best genome stays finite; half the others are invalid.
            genome.SetFitness(i % 2 == 1 ? poison[i % 3]
                                         : 10.0 + static_cast<double>(i));
            genome.SetEvaluated();
        }
        CheckNoThrow([&] { population.Epoch(); },
                     "legacy Epoch tolerates non-finite fitness");
        Check(AllLineageScoresFinite(population),
              "Epoch offspring lineage parent scores stay finite");
        CheckNoThrow([&] { (void)population.Serialize(); },
                     "population serializes after non-finite fitness");
    }

    // Checkpoints taken while genomes still hold non-finite fitness must be
    // readable: streams write "nan"/"inf" but could not parse them back.
    for (unsigned int i = 0; i < population.NumGenomes(); ++i)
    {
        Genome& genome = population.AccessGenomeByIndex(static_cast<int>(i));
        genome.SetFitness(i % 2 == 1 ? poison[i % 3] : static_cast<double>(i));
    }
    CheckNoThrow(
        [&]
        {
            const std::string state = population.Serialize();
            const Population restored = Population::Deserialize(state);
            Check(restored.Serialize() == state,
                  "non-finite fitness checkpoints round-trip exactly");
        },
        "non-finite fitness checkpoints can be restored");

    // Real-time path.
    for (unsigned int i = 0; i < population.NumGenomes(); ++i)
    {
        Genome& genome = population.AccessGenomeByIndex(static_cast<int>(i));
        genome.SetFitness(i % 2 == 1 ? poison[i % 3] : static_cast<double>(i));
        genome.SetEvaluated();
    }
    CheckNoThrow(
        [&]
        {
            for (int tick = 0; tick < 30; ++tick)
            {
                Genome deleted;
                Genome* baby = population.Tick(deleted);
                baby->SetFitness(tick % 3 == 0
                                     ? std::numeric_limits<double>::quiet_NaN()
                                     : static_cast<double>(tick));
                baby->SetEvaluated();
            }
        },
        "Tick tolerates non-finite legacy fitness");
    Check(std::isfinite(population.GetBestFitnessEver()),
          "best fitness ever stays finite in real-time mode");
    // Leave every individual finite so the checkpoint itself is valid.
    for (unsigned int i = 0; i < population.NumGenomes(); ++i)
    {
        Genome& genome = population.AccessGenomeByIndex(static_cast<int>(i));
        if (!std::isfinite(genome.GetFitness()))
            genome.SetFitness(0.0);
    }
    Check(AllLineageScoresFinite(population),
          "real-time offspring lineage parent scores stay finite");
    CheckNoThrow([&] { (void)population.Serialize(); },
                 "population serializes after real-time non-finite fitness");
}

// Finding 4: copying/moving a population must not alias the source's
// owned novelty archive.
void TestBehaviorArchiveCopyMove()
{
    using namespace NEAT;
    Parameters parameters = LegacyParameters(6);
    parameters.NoveltySearch_K = 3;
    parameters.NoveltySearch_P_min = 0.0;
    parameters.NoveltySearch_Dynamic_Pmin = false;
    Genome seed = MakeGenome(parameters);
    Population source(seed, parameters, true, 1.0, 5555);
    const auto behaviors = InitNovelty(source);
    for (int tick = 0; tick < 3; ++tick)
    {
        Genome result;
        source.NoveltySearchTick(result);
    }
    const std::size_t archived = source.GetBehaviorArchive().size();
    Check(archived > 0, "novelty archive receives behaviors");

    Population copy = source;
    Check(&copy.GetBehaviorArchive() != &source.GetBehaviorArchive(),
          "copied population owns its own novelty archive");
    Check(copy.GetBehaviorArchive().size() == archived,
          "copied population carries the archive contents");

    Population assigned;
    assigned = source;
    Check(&assigned.GetBehaviorArchive() != &source.GetBehaviorArchive(),
          "copy-assigned population owns its own novelty archive");

    for (int tick = 0; tick < 3; ++tick)
    {
        Genome result;
        source.NoveltySearchTick(result);
    }
    Check(copy.GetBehaviorArchive().size() == archived,
          "source archive growth does not leak into the copy");

    Population moved = std::move(copy);
    Check(&moved.GetBehaviorArchive() != &copy.GetBehaviorArchive(),
          "moved population does not point into the moved-from object");
    Check(moved.GetBehaviorArchive().size() == archived,
          "moved population keeps the archive contents");

    Population move_assigned;
    move_assigned = std::move(moved);
    Check(&move_assigned.GetBehaviorArchive() != &moved.GetBehaviorArchive(),
          "move-assigned population does not point into the source");

    {
        Population scoped = source;
        source = std::move(scoped);
    }
    CheckNoThrow(
        [&]
        {
            for (int tick = 0; tick < 3; ++tick)
            {
                Genome result;
                source.NoveltySearchTick(result);
            }
        },
        "novelty search continues after the original object is destroyed");
}

NEAT::Parameters ReliableParameters()
{
    NEAT::Parameters parameters;
    parameters.PopulationSize = 4;
    parameters.Speciation = false;
    parameters.EvaluationMinSamples = 2;
    parameters.EvaluationMaxSamples = 8;
    parameters.EvaluationBootstrapSamples = 128;
    parameters.ValidationSamples = 4;
    parameters.CertificationSamples = 8;
    parameters.QDArchiveReevaluationFraction = 0.0;
    return parameters;
}

// Finding 5: Lamarckian improvement is applied once per genome per
// generation, including across checkpoints.
void TestLamarckianAppliedOncePerGeneration()
{
    using namespace NEAT;
    Parameters parameters = ReliableParameters();
    Population population(MakeGenome(parameters), parameters, true, 1.0, 6666);
    int applications = 0;
    const auto hook = [&applications](Genome& genome)
    {
        ++applications;
        genome.m_LinkGenes.front().m_Weight += 0.25;
    };
    population.SetLocalImprovementHook(hook, true);

    auto requests = population.GetEvaluationRequests();
    std::map<int, double> original_weight;
    for (const auto& request : requests)
    {
        original_weight.emplace(
            request.genome_id,
            population.AccessGenomeByID(request.genome_id)
                .m_LinkGenes.front().m_Weight);
    }
    // Every case is requested (and retried) several times.
    for (int repeat = 0; repeat < 3; ++repeat)
    {
        for (const auto& request : requests)
            (void)population.GetEvaluationGenome(request);
    }
    Check(applications == static_cast<int>(original_weight.size()),
          "Lamarckian improvement runs once per genome per generation");
    for (const auto& item : original_weight)
    {
        Check(std::abs(population.AccessGenomeByID(item.first)
                           .m_LinkGenes.front().m_Weight -
                       (item.second + 0.25)) < 1.0e-12,
              "stored genotype carries exactly one improvement");
    }

    // Resuming from a checkpoint must not re-apply the improvement.
    Population resumed = Population::Deserialize(population.Serialize());
    resumed.SetLocalImprovementHook(hook, true);
    const int before_resume = applications;
    for (const auto& request : resumed.GetEvaluationRequests())
        (void)resumed.GetEvaluationGenome(request);
    Check(applications == before_resume,
          "checkpoint resume remembers already-improved genomes");

    // Finish the generation and verify the next one is improved again.
    while (!(requests = resumed.GetEvaluationRequests()).empty())
    {
        for (const auto& request : requests)
        {
            const Genome evaluated = resumed.GetEvaluationGenome(request);
            resumed.SubmitEvaluation(
                request.genome_id, request.case_id,
                evaluated.m_LinkGenes.front().m_Weight +
                    0.001 * static_cast<double>(request.case_id & 0xffU));
        }
    }
    const int before_epoch = applications;
    CheckNoThrow([&] { resumed.Epoch(); }, "reliable Epoch after racing");
    requests = resumed.GetEvaluationRequests();
    for (int repeat = 0; repeat < 2; ++repeat)
    {
        for (const auto& request : requests)
            (void)resumed.GetEvaluationGenome(request);
    }
    std::set<int> next_ids;
    for (const auto& request : requests)
        next_ids.insert(request.genome_id);
    Check(applications - before_epoch == static_cast<int>(next_ids.size()),
          "the next generation is improved once per genome");
}

// Finding 6: interspecies mating without an eligible partner falls back to
// within-species mating instead of burning constraint trials.
void TestInterspeciesFallback()
{
    using namespace NEAT;
    Parameters parameters = LegacyParameters(6);
    parameters.CrossoverRate = 1.0;
    parameters.InterspeciesCrossoverRate = 1.0;
    parameters.ConstraintTrials = 3;
    Genome seed = MakeGenome(parameters);
    Population population(seed, parameters, true, 1.0, 7777);
    Check(population.m_Species.size() == 1, "fallback setup has one species");

    Genome moved_a = population.m_Species.front().m_Individuals[4];
    Genome moved_b = population.m_Species.front().m_Individuals[5];
    population.m_Species.front().RemoveIndividual(5);
    population.m_Species.front().RemoveIndividual(4);
    for (auto& genome : population.m_Species.front().m_Individuals)
    {
        genome.SetFitness(static_cast<double>(genome.GetID()) + 1.0);
        genome.SetEvaluated();
    }
    moved_a.ResetEvaluated();
    moved_b.ResetEvaluated();
    Species unevaluated(
        moved_a, population.m_Parameters,
        static_cast<int>(population.GetNextSpeciesID()));
    population.IncrementNextSpeciesID();
    unevaluated.AddIndividual(moved_b);
    population.m_Species.push_back(unevaluated);
    std::string error;
    Check(population.Validate(&error), "fallback setup is valid: " + error);

    for (int tick = 0; tick < 5; ++tick)
    {
        Genome* baby = nullptr;
        CheckNoThrow(
            [&]
            {
                Genome deleted;
                baby = population.Tick(deleted);
            },
            "interspecies mating without partners falls back");
        if (baby == nullptr)
            return;
        Check(baby->NumInputs() > 0 && baby->NumLinks() > 0,
              "fallback offspring is a real genome");
        Check(baby->GetLineage().parent_ids.size() == 2,
              "fallback offspring came from within-species mating");
        baby->SetFitness(0.5);
        baby->SetEvaluated();
    }
}

// Finding 7: re-binning while archive re-evaluations are pending must not
// silently invalidate pending archive indices.
void TestRebinRejectedWhileArchiveRequestsPending()
{
    using namespace NEAT;
    Parameters parameters = ReliableParameters();
    parameters.EvolutionModeSetting = ROBUST_QD_NEAT;
    parameters.QDDescriptorMinimums = {0.0};
    parameters.QDDescriptorMaximums = {1.0};
    parameters.QDDescriptorBins = {2U};
    parameters.QDArchiveReevaluationFraction = 1.0;
    Population population(MakeGenome(parameters), parameters, true, 1.0, 8888);

    bool archive_pending = false;
    auto requests = population.GetEvaluationRequests();
    int guard = 0;
    while (!requests.empty() && guard++ < 100)
    {
        archive_pending = std::any_of(
            requests.begin(), requests.end(),
            [](const EvaluationRequest& request)
            { return request.archive_evaluation; });
        if (archive_pending)
            break;
        for (const auto& request : requests)
        {
            population.SubmitEvaluation(
                request.genome_id, request.case_id,
                10.0 - request.genome_id, false, false,
                {request.genome_id % 2 == 0 ? 0.25 : 0.75});
        }
        requests = population.GetEvaluationRequests();
    }
    Check(archive_pending, "QD archive re-evaluation requests are scheduled");
    if (!archive_pending)
        return;

    population.m_Parameters.QDDescriptorBins = {4U};
    bool rejected = false;
    try
    {
        population.RebinQDArchive();
    }
    catch (const std::logic_error&)
    {
        rejected = true;
    }
    Check(rejected,
          "RebinQDArchive is rejected while archive requests are pending");
    population.m_Parameters.QDDescriptorBins = {2U};

    CheckNoThrow(
        [&]
        {
            while (!(requests = population.GetEvaluationRequests()).empty())
            {
                for (const auto& request : requests)
                {
                    population.SubmitEvaluation(
                        request.genome_id, request.case_id, 1.0, false,
                        false, {0.25});
                }
            }
        },
        "pending archive requests still complete after a rejected rebin");
    population.m_Parameters.QDDescriptorBins = {4U};
    CheckNoThrow([&] { population.RebinQDArchive(); },
                 "RebinQDArchive works once no archive request is pending");
}

// Population::Save / Population(file) must read back what it writes.
void TestLegacyFileRoundTrip()
{
    using namespace NEAT;
    Parameters parameters = LegacyParameters(8);
    parameters.Speciation = true;
    Genome seed = MakeGenome(parameters);
    Population population(seed, parameters, true, 1.0, 9999);
    for (unsigned int i = 0; i < population.NumGenomes(); ++i)
    {
        Genome& genome = population.AccessGenomeByIndex(static_cast<int>(i));
        genome.SetFitness(static_cast<double>(i));
        genome.SetEvaluated();
    }
    population.Epoch();

    const auto path = std::filesystem::temp_directory_path() /
        "multineat_population_roundtrip.txt";
    try
    {
        population.Save(path.string().c_str());
        Population loaded(path.string());
        std::filesystem::remove(path);
        std::map<int, const Genome*> original;
        for (const auto& species : population.m_Species)
        {
            for (const auto& genome : species.m_Individuals)
                original[genome.GetID()] = &genome;
        }
        Check(loaded.NumGenomes() == population.NumGenomes(),
              "file round trip preserves the genome count");
        std::string error;
        Check(loaded.Validate(&error),
              "file round trip yields a valid population: " + error);
        bool genomes_match = true;
        for (const auto& species : loaded.m_Species)
        {
            for (const auto& genome : species.m_Individuals)
            {
                const auto found = original.find(genome.GetID());
                if (found == original.end() ||
                    found->second->NumNeurons() != genome.NumNeurons() ||
                    found->second->NumLinks() != genome.NumLinks())
                {
                    genomes_match = false;
                    continue;
                }
                for (std::size_t link = 0; link < genome.m_LinkGenes.size();
                     ++link)
                {
                    if (std::abs(found->second->m_LinkGenes[link].m_Weight -
                                 genome.m_LinkGenes[link].m_Weight) > 1.0e-6)
                        genomes_match = false;
                }
            }
        }
        Check(genomes_match,
              "file round trip preserves genome IDs, structure and weights");
        Check(loaded.GetNextGenomeID() >= population.GetNextGenomeID() ||
                  loaded.GetNextGenomeID() > 0,
              "file round trip leaves a usable next genome ID");
    }
    catch (const std::exception& error)
    {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        Check(false, std::string("population file round trip threw: ") +
                         error.what());
    }
}
} // namespace

int main()
{
    TestTickRouletteUsesAdjustedFitness();
    TestTickLegacyTruncationSorts();
    TestNoveltySearchMarksBabiesEvaluated();
    TestNonFiniteFitnessIsSanitized();
    TestBehaviorArchiveCopyMove();
    TestLamarckianAppliedOncePerGeneration();
    TestInterspeciesFallback();
    TestRebinRejectedWhileArchiveRequestsPending();
    TestLegacyFileRoundTrip();

    if (failures != 0)
    {
        std::cerr << failures << " population test(s) failed\n";
        return 1;
    }
    std::cout << "All population tests passed\n";
    return 0;
}
