#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "Genome.h"
#include "Species.h"
#include "Random.h"
#include "Population.h"
#include "Utils.h"
#include "Parameters.h"
#include "Assert.h"

namespace NEAT
{
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
            for (double &weight : weights)
            {
                weight -= minimum;
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
        return (ls.GetFitness() > rs.GetFitness());
    }

    bool idxfitnesspair_greater(const std::pair<int, double>& ls,
                                const std::pair<int, double>& rs)
    {
        return (ls.second > rs.second);
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


    // initializes a species with a representative genome and an ID number
    Species::Species(const Genome& a_Genome, const Parameters&, int a_ID)
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
        // Derive a stable display color from the species ID. A process-global
        // RNG made independent populations and resumed checkpoints influence
        // one another and was not thread-safe.
        const std::uint32_t color =
            static_cast<std::uint32_t>(a_ID) * 2654435761U;
        m_R = static_cast<int>((color >> 16U) & 0xffU);
        m_G = 100 + static_cast<int>((color >> 8U) & 0x9bU);
        m_B = static_cast<int>(color & 0xffU);
    }

    Species& Species::operator=(const Species& a_S)
    {
        // self assignment guard
        if (this != &a_S)
        {
            m_ID = a_S.m_ID;
            m_BestGenome = a_S.m_BestGenome;
            m_BestSpecies = a_S.m_BestSpecies;
            m_WorstSpecies = a_S.m_WorstSpecies;
            m_BestFitness = a_S.m_BestFitness;
            m_GensNoImprovement = a_S.m_GensNoImprovement;
            m_EvalsNoImprovement = a_S.m_EvalsNoImprovement;
            m_AverageFitness = a_S.m_AverageFitness;
            m_AgeGenerations = a_S.m_AgeGenerations;
            m_AgeEvaluations = a_S.m_AgeEvaluations;
            m_OffspringRqd = a_S.m_OffspringRqd;
            m_R = a_S.m_R;
            m_G = a_S.m_G;
            m_B = a_S.m_B;
            m_Individuals = a_S.m_Individuals;
        }

        return *this;
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
            std::stable_sort(
                t_Evaluated.begin(),
                t_Evaluated.end(),
                idxfitnesspair_greater);
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
            // choose N individuals at random
            for (unsigned int i = 0; i < a_Parameters.TournamentSize; ++i)
            {
                const int c = a_RNG.RandInt(
                    0, static_cast<int>(t_Evaluated.size()) - 1);
                t_picked.push_back(t_Evaluated[static_cast<std::size_t>(c)]);
            }

            // do a roulette on the picked
            std::vector<double> probs;
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
        int t_leader_idx = 0;
        for (unsigned int i = 0; i < m_Individuals.size(); i++)
        {
            double t_f = m_Individuals[i].GetFitness();
            if (t_f > t_max_fitness)
            {
                t_max_fitness = t_f;
                t_leader_idx = i;
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
        if (m_Individuals.empty())
        {
            throw std::runtime_error(
                "Cannot adjust fitness for an empty species");
        }

        // iterate through the members
        for (unsigned int i = 0; i < m_Individuals.size(); i++)
        {
            double t_fitness = m_Individuals[i].GetFitness();
            // this prevents nan or infinity to be fitness
            if (std::isnan(t_fitness)) t_fitness = 0.0000001;
            if (std::isinf(t_fitness)) t_fitness = 0.0000001;

            // update the best fitness and stagnation counter
            if (t_fitness > m_BestFitness)
            {
                if (m_BestFitness == std::numeric_limits<double>::lowest() ||
                    t_fitness - m_BestFitness >=
                        a_Parameters.StagnationDelta)
                {
                    m_GensNoImprovement = 0;
                }
                m_BestFitness = t_fitness;
                m_BestGenome = m_Individuals[i];
            }

            t_fitness += a_FitnessOffset;
            if (t_fitness <= 0.0)
            {
                t_fitness = 0.0000001;
            }


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
                    t_fitness *= 0.0000001;
                }
            }

            // Compute the adjusted fitness for this member
            m_Individuals[i].SetAdjFitness(
                t_fitness / static_cast<double>(m_Individuals.size()));
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
        }

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

            // Select the elite first..

            if (elite_count < elite_offspring)
            {
                t_baby = GetLeader();
                elite_count++;
            }
            else
            {
                unsigned int t_constraint_trials = a_Parameters.ConstraintTrials; // to prevent infinite loops

                do // - while the baby already exists somewhere in the new population or turned invalid in some way
                {
                    // this tells us if the baby is a result of mating
                    bool t_mated = false;

                    // There must be individuals there..
                    ASSERT(NumIndividuals() > 0);

                    // for a species of size 1 we can only mutate
                    // NOTE: but does it make sense since we know this is the champ?
                    if (NumIndividuals() == 1)
                    {
                        t_baby = GetIndividual(a_Parameters, a_RNG);
                        t_mated = false;
                    }
                    // else we can mate
                    else
                    {
                        // choose whether to mate at all
                        // Do not allow crossover when in simplifying phase
                        if ((a_RNG.RandFloat() < a_Parameters.CrossoverRate) && (a_Pop.GetSearchMode() != SIMPLIFYING))
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
                                    t_mom = GetIndividual(a_Parameters, a_RNG);
                                    t_dad = a_Pop.m_Species[t_diffspec].GetIndividual(a_Parameters, a_RNG);
                                    t_interspecies = true;
                                }
                                else
                                {
                                    continue;
                                }
                            }
                            else
                            {
                                // Mate within species
                                t_mom = GetIndividual(a_Parameters, a_RNG);
                                t_dad = GetIndividual(a_Parameters, a_RNG);

                                // The other parent should be a different one
                                // number of tries to find different parent
                                int t_tries = 32;
                                while (((t_mom.GetID() == t_dad.GetID())) && (t_tries--))
                                {
                                    t_mom = GetIndividual(a_Parameters, a_RNG);
                                    t_dad = GetIndividual(a_Parameters, a_RNG);
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
                        }
                        // don't mate - reproduce one individual asexually
                        else
                        {
                            t_baby = GetIndividual(a_Parameters, a_RNG);
                            t_mated = false;
                        }
                    }

                    // Mutate the baby
                    if ((!t_mated) || (a_RNG.RandFloat() < a_Parameters.OverallMutationRate))
                    {
                        MutateGenome(
                            false, a_Pop, t_baby, a_Parameters, a_RNG);
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
        double t_total_fitness = 0;
        int t_num_individuals = 0;

        // consider individuals that were evaluated only!
        for (unsigned int i = 0; i < m_Individuals.size(); i++)
        {
            if (m_Individuals[i].IsEvaluated())
            {
                double tf = m_Individuals[i].GetFitness();
                if (std::isinf(tf) || std::isnan(tf)) // nan/inf guard
                {
                    tf = 0.0;
                }
                t_total_fitness += tf;
                ++t_num_individuals;
            }
        }

        if (t_num_individuals > 0)
        {
            m_AverageFitness = t_total_fitness / static_cast<double>(t_num_individuals);
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

        do // - while the baby turned invalid in some way
        {
            t_baby = Genome(); // clear baby

            // this tells us if the baby is a result of mating
            bool t_mated = false;

            // There must be individuals there..
            ASSERT(NumIndividuals() > 0);

            // for a species of size 1 we can only mutate
            // NOTE: but does it make sense since we know this is the champ?
            if (NumIndividuals() == 1)
            {
                t_baby = GetIndividual(a_Parameters, a_RNG);
                t_mated = false;
            }
            // else we can mate
            else
            {
                // choose whether to mate at all
                // Do not allow crossover when in simplifying phase
                if ((a_RNG.RandFloat() < a_Parameters.CrossoverRate) && (a_Pop.GetSearchMode() != SIMPLIFYING))
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
                            t_mom = GetIndividual(a_Parameters, a_RNG);
                            t_dad = a_Pop.m_Species[t_diffspec].GetIndividual(a_Parameters, a_RNG);
                            t_interspecies = true;
                        }
                        else
                        {
                            continue;
                        }
                    }
                    else
                    {
                        // Mate within species
                        t_mom = GetIndividual(a_Parameters, a_RNG);
                        t_dad = GetIndividual(a_Parameters, a_RNG);

                        // The other parent should be a different one
                        // number of tries to find different parent
                        // we can mate the same mom and dad and still get different baby
                        int t_tries = 32;
                        while (((t_mom.GetID() == t_dad.GetID())) && (t_tries--))
                        {
                            t_mom = GetIndividual(a_Parameters, a_RNG);
                            t_dad = GetIndividual(a_Parameters, a_RNG);
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
                }
                // don't mate - reproduce one individual asexually
                else
                {
                    t_baby = GetIndividual(a_Parameters, a_RNG);
                    t_mated = false;
                }
            }

            // Mutate the baby
            if ((!t_mated) || (a_RNG.RandFloat() < a_Parameters.OverallMutationRate))
            {
                MutateGenome(
                    false, a_Pop, t_baby, a_Parameters, a_RNG);
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
        Species::MutateGenome(bool t_baby_is_clone, Population& a_Pop, Genome& t_baby, Parameters& a_Parameters, RNG& a_RNG)
    {
        // We will perform roulette wheel selection to choose the type of mutation and will mutate the baby
        // This method guarantees that the baby will be mutated at least with one mutation
        enum MutationTypes
        {
            ADD_NODE = 0, ADD_LINK, REMOVE_NODE, REMOVE_LINK, CHANGE_ACTIVATION_FUNCTION,
            MUTATE_WEIGHTS, MUTATE_ACTIVATION_A, MUTATE_ACTIVATION_B, MUTATE_TIMECONSTS, MUTATE_BIASES,
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

        // Some structural mutations can be impossible for a particular genome.
        // Bound retries so invalid configurations fail instead of hanging forever.
        for (int mutation_attempt = 0;
             mutation_attempt < 512 && !t_mutation_success;
             ++mutation_attempt)
        {
            int ChosenMutation = a_RNG.Roulette(t_mut_probs);

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
            throw std::runtime_error(
                "No enabled mutation could be applied to the genome");
        }
    }

std::string Species::Serialize() const
{
    std::ostringstream output;
    output.precision(std::numeric_limits<double>::max_digits10);
    output << "SpeciesStart\n";
    output << "SpeciesFormat 2\n";
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
    output << "SpeciesEnd\n";
    return output.str();
}

Species Species::Deserialize(const std::string &data)
{
    std::istringstream input(data);
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
        if (format_version != 2)
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

    input >> token;
    if (token != "SpeciesEnd")
        throw std::runtime_error(
            "Species::Deserialize: missing SpeciesEnd marker.");
    if (format_version == 1 && !species.m_Individuals.empty())
        species.m_BestGenome = species.GetLeader();
    return species;
}


} // namespace NEAT
