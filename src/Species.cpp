#include <algorithm>

#include "Genome.h"
#include "Species.h"
#include "Random.h"
#include "Population.h"
#include "Utils.h"
#include "Parameters.h"
#include "assert.h"

namespace NEAT
{
    RNG global_rng;

    bool genome_greater(Genome& ls, Genome& rs)
    {
        return (ls.GetFitness() > rs.GetFitness());
    }

    bool idxfitnesspair_greater(std::pair<int, double>& ls, std::pair<int, double>& rs)
    {
        return (ls.second > rs.second);
    }


    // initializes a species with a representative genome and an ID number
    Species::Species(const Genome& a_Genome, const Parameters& a_Parameters, int a_ID)
    {
        m_ID = a_ID;

        // copy the initializing genome locally.
        m_BestGenome = a_Genome;

        // add the first and only one individual
        m_Individuals.push_back(a_Genome);

        m_AgeGenerations = 0;
        m_AgeEvaluations = 0;
        m_GensNoImprovement = 0;
        m_EvalsNoImprovement = 0;
        m_OffspringRqd = 0;
        m_BestFitness = a_Genome.GetFitness();
        m_BestSpecies = false;
        m_WorstSpecies = false;
        m_AverageFitness = 0;

        // Choose a random color
        m_R = static_cast<int>(global_rng.RandFloat() * 255);
        m_G = static_cast<int>(global_rng.RandFloat() * 255) + 100;
        if (m_G > 255) m_G = 255;
        m_B = static_cast<int>(global_rng.RandFloat() * 255);
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
            char s[256];
            sprintf(s, "Species::GetIndividual (ID:%d) - No individuals in species", m_ID);
            throw std::runtime_error(s);
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
            char s[256];
            sprintf(s, "Species::GetIndividual (ID:%d) - No evaluated individuals", m_ID);
            throw std::runtime_error(s);
        }
        if (t_Evaluated.size() == 1)
        {
            return (m_Individuals[t_Evaluated[0].first]);
        }

        // Warning!!!! The individuals must be sorted by best fitness for this to work
        int t_chosen_one = 0;

        // Truncation selection goes first if enabled
        if (a_Parameters.TruncationSelection)
        {
            int t_num_parents = (int)(a_Parameters.SurvivalRate * (double)(t_Evaluated.size()));

            if (t_num_parents >= t_Evaluated.size())
            {
                t_num_parents = t_Evaluated.size() - 1;
            }
            if (t_num_parents < 1)
            {
                t_num_parents = 1;
            }
            // do truncation here, and other extra selections can be applied below
            t_Evaluated.resize(t_num_parents);
        }

        if (a_Parameters.TournamentSelection && (!a_Parameters.RouletteWheelSelection)) // pure tournament without roulette
        {
            std::vector< std::pair<int, double> > t_picked;
            // choose N individuals at random
            for (int i = 0; i < a_Parameters.TournamentSize; i++)
            {
                int c = a_RNG.RandInt(0, t_Evaluated.size() - 1);
                t_picked.push_back(t_Evaluated[c]);
            }

            // Proper tournament selection: select the best individual in the pool
            std::sort(t_picked.begin(), t_picked.end(), idxfitnesspair_greater);
            t_chosen_one = t_picked[0].first;
        }
        else if (a_Parameters.TournamentSelection && a_Parameters.RouletteWheelSelection) // tournament with roulette applied on the picked
        {
            std::vector< std::pair<int, double> > t_picked;
            // choose N individuals at random
            for (int i = 0; i < a_Parameters.TournamentSize; i++)
            {
                int c = a_RNG.RandInt(0, t_Evaluated.size() - 1);
                t_picked.push_back(t_Evaluated[c]);
            }

            // do a roulette on the picked
            std::vector<double> probs;
            for (auto p : t_picked)
            {
                probs.push_back(p.second);
            }
            t_chosen_one = t_picked[a_RNG.Roulette(probs)].first;
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
            int t_num_parents = t_Evaluated.size();
            std::vector<double> t_probs;
            for (unsigned int i = 0; i < t_num_parents; i++)
            {
                t_probs.push_back(t_Evaluated[i].second);
            }
            t_chosen_one = t_Evaluated[a_RNG.Roulette(t_probs)].first;
        }
        else
        {
            // default is pure truncation or just random search if truncation is off - the array has been resized already
            t_chosen_one = t_Evaluated[a_RNG.RandInt(0, t_Evaluated.size()-1)].first;
        }

        return (m_Individuals[t_chosen_one]);
    }


    // returns a completely random individual
    Genome& Species::GetRandomIndividual(RNG& a_RNG) 
    {
        if (m_Individuals.size() == 0) // no members yet, return representative
        {
            char s[256];
            sprintf(s, "Attempted GetRandomIndividual() but no individuals in species ID %d\n", m_ID);
            throw std::runtime_error(s);
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
            char s[256];
            sprintf(s, "Attempted GetLeader() but no individuals in species ID %d\n", m_ID);
            throw std::runtime_error(s);
        }

        double t_max_fitness = std::numeric_limits<double>::min();
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
            char s[256];
            sprintf(s, "Attempted GetRepresentative() but no individuals in species ID %d\n", m_ID);
            throw std::runtime_error(s);
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
        ASSERT(m_Individuals.size() > 0);

        double total_fitness=0;

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
                m_BestFitness = t_fitness;
                m_GensNoImprovement = 0;
            }

            // the fitness must be positive
            ASSERT(t_fitness >= 0.0);

            // this prevents the fitness to be below zero
            if (t_fitness <= 0.0) t_fitness = 0.0000001;


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

            unsigned int ms = m_Individuals.size();
            ASSERT(ms > 0);
            if (ms == 0)
            {
                ms = 1;
            }

            // Compute the adjusted fitness for this member
            m_Individuals[i].SetAdjFitness(t_fitness / (double)(ms));
        }
    }


    void Species::SortIndividuals()
    {
        std::sort(m_Individuals.begin(), m_Individuals.end(), genome_greater);
    }


    // Removes an individual from the species by its index within the species
    void Species::RemoveIndividual(unsigned int a_idx)
    {
        ASSERT(a_idx < m_Individuals.size());
        m_Individuals.erase(m_Individuals.begin() + a_idx);
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
                                double allp = 0;
                                for (int i = 0; i < a_Pop.m_Species.size(); i++)
                                {
                                    if ((a_Pop.m_Species[i].m_ID == m_ID))
                                    {
                                        probs.push_back(0.0);
                                    }
                                    else
                                    {
                                        probs.push_back(a_Pop.m_Species[i].GetLeader().GetAdjFitness()); // use the best's adj fitness to ensure positive
                                    }
                                    allp += probs[probs.size() - 1];
                                }
                                if (allp > 0)
                                {
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

                            // OK we have both mom and dad so mate them
                            // Choose randomly one of two types of crossover
                            if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                            {
                                t_baby = t_mom.Mate(t_dad, false, t_interspecies, a_RNG, a_Parameters);
                            }
                            else
                            {
                                t_baby = t_mom.Mate(t_dad, true, t_interspecies, a_RNG, a_Parameters);
                            }

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
                    bool dummy = false;
                    if ((!t_mated) || (a_RNG.RandFloat() < a_Parameters.OverallMutationRate))
                    {
                        MutateGenome(dummy, a_Pop, t_baby, a_Parameters, a_RNG);
                    }

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
                                if (t_baby.IsIdenticalTo(a_Pop.m_TempSpecies[i].m_Individuals[j]))
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
                            if (t_baby.IsIdenticalTo(a_Pop.m_GenomeArchive[i]))
                            {
                                t_baby_exists_in_pop = true;
                                break;
                            }
                        }
                    }
                } while ((t_baby_exists_in_pop || (t_baby.FailsConstraints(a_Parameters))) && (t_constraint_trials--)); // end do
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

            bool t_found = false;
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

                t_found = false;
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
            }
            t_num_individuals++;
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
        bool t_baby_is_clone = false;
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
                        double allp = 0;
                        for (int i = 0; i < a_Pop.m_Species.size(); i++)
                        {
                            if ((a_Pop.m_Species[i].m_ID == m_ID) || (a_Pop.m_Species[i].NumEvaluated() == 0))
                            {
                                probs.push_back(0.0);
                            }
                            else
                            {
                                probs.push_back(a_Pop.m_Species[i].m_AverageFitness);
                            }
                            allp += probs[probs.size() - 1];
                        }
                        if (allp > 0)
                        {
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

                    // OK we have both mom and dad so mate them
                    // Choose randomly one of two types of crossover
                    if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                    {
                        t_baby = t_mom.Mate(t_dad, false, t_interspecies, a_RNG, a_Parameters);
                    }
                    else
                    {
                        t_baby = t_mom.Mate(t_dad, true, t_interspecies, a_RNG, a_Parameters);
                    }

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
            t_baby_is_clone = false;
            bool dummy = false;
            if ((!t_mated) || (a_RNG.RandFloat() < a_Parameters.OverallMutationRate))
            {
                MutateGenome(dummy, a_Pop, t_baby, a_Parameters, a_RNG);
#ifdef VDEBUG
                std::cout << "mutated baby\n";
#endif
            }

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
                        if (t_baby.IsIdenticalTo(a_Pop.m_Species[i].m_Individuals[j]))
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
                        if (t_baby.IsIdenticalTo(a_Pop.m_GenomeArchive[i]))
                    {
                        t_baby_exists_in_pop = true;
                        break;
                    }
                }
            }
        } while ((t_baby_exists_in_pop || t_baby.FailsConstraints(a_Parameters)) && (t_constraint_trials--)); // end do


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
        std::vector<int> t_muts;
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

        bool t_mutation_success = false;

        // repeat until successful
        while (t_mutation_success == false)
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
                        break; // give up
                    }

                    t_saved_baby = t_baby;
                    t_mutation_success = t_saved_baby.Mutate_RemoveLink(a_RNG);

                    t_no_links = t_has_dead_ends = false;

                    if (t_saved_baby.NumLinks() == 0)
                        t_no_links = true;

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
    }

    // The Serialize method writes a header block plus the number of individuals
// and then calls Genome::Serialize() for each genome.
std::string Species::Serialize() const {
    std::ostringstream oss;
    // Use markers so that when unpickling we know where the species block starts/ends.
    oss << "SpeciesStart\n";
    // Write header fields: id, best species flag, worst species flag,
    // age in generations, age in evaluations, required offspring,
    // best fitness, generations & evaluations without improvement, and the color and average fitness.
    oss << m_ID << " " << m_BestSpecies << " " << m_WorstSpecies << " " 
        << m_AgeGenerations << " " << m_AgeEvaluations << " " << m_OffspringRqd << " "
        << m_BestFitness << " " << m_GensNoImprovement << " " << m_EvalsNoImprovement << " "
        << m_R << " " << m_G << " " << m_B << " " << m_AverageFitness << "\n";
    // Write how many genomes are contained.
    oss << m_Individuals.size() << "\n";
    // For each genome use its own serialization.
    for (const auto &genome : m_Individuals) {
         oss << genome.Serialize() << "\n";
    }
    oss << "SpeciesEnd\n";
    return oss.str();
}

// The Deserialize method reads the header in order and then reads the given number of genomes.
Species Species::Deserialize(const std::string &data) {
    std::istringstream iss(data);
    std::string token;

    iss >> token;
    if (token != "SpeciesStart")
         throw std::runtime_error("Species::Deserialize: missing SpeciesStart marker.");
    
    Species s;
    // Read header values in order.
    iss >> s.m_ID
        >> s.m_BestSpecies
        >> s.m_WorstSpecies
        >> s.m_AgeGenerations
        >> s.m_AgeEvaluations
        >> s.m_OffspringRqd
        >> s.m_BestFitness
        >> s.m_GensNoImprovement
        >> s.m_EvalsNoImprovement
        >> s.m_R >> s.m_G >> s.m_B >> s.m_AverageFitness;
    
    size_t numIndividuals;
    iss >> numIndividuals;
    
    // Consume any leftover newline.
    std::string dummy;
    std::getline(iss, dummy);
    
    s.m_Individuals.clear();
    // For each genome, assume Genome::Serialize produced a one‐line string.
    for (size_t i = 0; i < numIndividuals; i++) {
         std::string genomeBlock;
         std::getline(iss, genomeBlock);
         // In case of empty lines, skip
         while(genomeBlock.empty() && std::getline(iss, genomeBlock)) {}
         NEAT::Genome g = NEAT::Genome::Deserialize(genomeBlock);
         s.m_Individuals.push_back(g);
    }
    
    iss >> token;
    if (token != "SpeciesEnd")
         throw std::runtime_error("Species::Deserialize: missing SpeciesEnd marker.");
    
    return s;
}


} // namespace NEAT
