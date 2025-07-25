#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <sstream>

#include "Genome.h"
#include "Species.h"
#include "Random.h"
#include "Population.h"
#include "Utils.h"
#include "Assert.h"


namespace NEAT
{

// The constructor
Population::Population(const Genome& a_Seed, const Parameters& a_Parameters,
		               bool a_RandomizeWeights, double a_RandomizationRange, int a_RNG_seed)
{
    m_RNG.Seed(a_RNG_seed);
    m_BestFitnessEver = 0.0;
    m_Parameters = a_Parameters;

    m_Generation = 0;
    m_NumEvaluations = 0;
    m_NextGenomeID = m_Parameters.PopulationSize;
    m_NextSpeciesID = 1;
    m_GensSinceBestFitnessLastChanged = 0;
    m_GensSinceMPCLastChanged = 0;

    // Spawn the population
    for(unsigned int i=0; i<m_Parameters.PopulationSize; i++)
    {
        Genome t_clone = a_Seed;
        t_clone.SetID(i);
        m_Genomes.push_back( t_clone );
    }
        
    // Now now initialize each genome's weights
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        if (a_RandomizeWeights)
        {
            bool is_invalid = true;
            while (is_invalid)
            {
                m_Genomes[i].Randomize_LinkWeights(a_Parameters, m_RNG);
                // randomize the traits as well
                m_Genomes[i].Randomize_Traits(a_Parameters, m_RNG);
                // and mutate nodes one initial time
                m_Genomes[i].Mutate_NeuronActivations_A(a_Parameters, m_RNG);
                m_Genomes[i].Mutate_NeuronActivations_B(a_Parameters, m_RNG);
                m_Genomes[i].Mutate_NeuronActivation_Type(a_Parameters, m_RNG);
                m_Genomes[i].Mutate_NeuronTimeConstants(a_Parameters, m_RNG);
                m_Genomes[i].Mutate_NeuronBiases(a_Parameters, m_RNG);
                    
                // check in the population if there is a clone of that genome
                is_invalid = false;
                if (!m_Parameters.AllowClones)
                {
                    for(unsigned int j=0; j<m_Genomes.size(); j++)
                    {
                        if (i != j) // don't compare the same genome
                        {
                            if (m_Genomes[i].IsIdenticalTo(m_Genomes[j])) // equal genomes?
                            {
                                is_invalid = true;
                                break;
                            }
                        }
                    }
                }
                
                // Also don't let any genome to fail the constraints
                if (!is_invalid) // doesn't make sense to do the test if already failed
                {
                    if (m_Genomes[i].FailsConstraints(a_Parameters))
                    {
                        is_invalid = true;
                    }
                }
            }
        }
    }
    // Speciate
    Speciate();

    // set these phased search variables now since used in MutateGenome
    if (m_Parameters.PhasedSearching)
    {
        m_SearchMode = COMPLEXIFYING;
    }
    else
    {
        m_SearchMode = BLENDED;
    }

    // Initialize the innovation database
    m_InnovationDatabase.Init(a_Seed);

    m_BestGenome = m_Species[0].m_Individuals[0];
    
    m_ID = 0;

    // Set up the rest of the phased search variables
    CalculateMPC();
    m_BaseMPC = m_CurrentMPC;
    m_OldMPC = m_BaseMPC;
    
    // Reset IDs to be sure
    int cid=0;
    for(int i=0; i<(int)m_Species.size(); i++)
    {
        for(int j=0; j<(int)m_Species[i].m_Individuals.size(); j++)
        {
            m_Species[i].m_Individuals[j].SetID(cid);
            cid++;
        }
    }

    m_InnovationDatabase.m_Innovations.reserve(50000);
}


Population::Population(const std::string a_sFileName)
{

    auto a_FileName = a_sFileName.c_str();
    m_BestFitnessEver = 0.0;

    m_Generation = 0;
    m_NumEvaluations = 0;
    m_NextSpeciesID = 1;
    m_ID = 0;
    m_GensSinceBestFitnessLastChanged = 0;
    m_GensSinceMPCLastChanged = 0;

    std::ifstream t_DataFile(a_FileName);
    if (!t_DataFile.is_open()) throw std::exception();
    std::string t_str;

    // Load the parameters
    m_Parameters.Load(t_DataFile);

    // Load the innovation database
    m_InnovationDatabase.Init(t_DataFile);

    // Load all genomes
    for(unsigned int i=0; i<m_Parameters.PopulationSize; i++)
    {
        Genome t_genome(t_DataFile);
        m_Genomes.push_back( t_genome );
    }
    t_DataFile.close();

    m_NextGenomeID = 0;
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        if (m_Genomes[i].GetID() > m_NextGenomeID)
        {
            m_NextGenomeID = m_Genomes[i].GetID();
        }
    }
    m_NextGenomeID++;

    // Initialize
    Speciate();
    m_BestGenome = m_Species[0].GetLeader();

    // Set up the phased search variables
    CalculateMPC();
    m_BaseMPC = m_CurrentMPC;
    m_OldMPC = m_BaseMPC;
    if (m_Parameters.PhasedSearching)
    {
        m_SearchMode = COMPLEXIFYING;
    }
    else
    {
        m_SearchMode = BLENDED;
    }
}


// Save a whole population to a file
void Population::Save(const char* a_FileName)
{
    FILE* t_file = fopen(a_FileName, "w");

    // Save the parameters
    m_Parameters.Save(t_file);

    // Save the innovation database
    m_InnovationDatabase.Save(t_file);

    // Save each genome
    for(unsigned i=0; i<m_Species.size(); i++)
    {
        for(unsigned j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            m_Species[i].m_Individuals[j].Save(t_file);
        }
    }

    // bye
    fclose(t_file);
}


// Calculates the current mean population complexity
void Population::CalculateMPC()
{
    m_CurrentMPC = 0;

    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        m_CurrentMPC += AccessGenomeByIndex(i).NumLinks();
    }

    m_CurrentMPC /= m_Genomes.size();
}


// Separates the population into species
// also adjusts the compatibility treshold if this feature is enabled
void Population::Speciate()
{
    // iterate through the genome list and speciate
    // at least 1 genome must be present
    ASSERT(m_Genomes.size() > 0);

    // first clear out the species
    m_Species.clear();


    bool t_added = false;

    // NOTE: we are comparing the new generation's genomes to the representatives from species creation time!
    //
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        t_added = false;

        // iterate through each species and check if compatible. If compatible, then add to the species.
        // if not compatible, create a new species.
        for(unsigned int j=0; j<m_Species.size(); j++)
        {
            if (m_Species[j].NumIndividuals() > 0)
            {
                if (m_Genomes[i].IsCompatibleWith(m_Species[j].GetRepresentative(), m_Parameters))
                {
                    // Compatible, add to species
                    m_Species[j].AddIndividual(m_Genomes[i]);
                    t_added = true;
        
                    break;
                }
            }
        }

        if (!t_added)
        {
            // didn't find compatible species, create new species
            m_Species.push_back( Species(m_Genomes[i], m_Parameters, m_NextSpeciesID));
            m_NextSpeciesID++;
        }
    }

    // Remove all empty species (cleanup routine for every case..)
    ClearEmptySpecies();
}

// Adjust the fitness of all species
void Population::AdjustFitness()
{
    ASSERT(m_Genomes.size() > 0);
    ASSERT(m_Species.size() > 0);

    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].AdjustFitness(m_Parameters);
    }
}

// Calculates how many offspring each genome should have
void Population::CountOffspring()
{
    ASSERT(m_Genomes.size() > 0);
    ASSERT(m_Genomes.size() == m_Parameters.PopulationSize);

    double t_total_adjusted_fitness = 0.0;
    double t_average_adjusted_fitness = 0.0;
    Genome t_t;

    // get the total adjusted fitness for all individuals
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            t_total_adjusted_fitness += m_Species[i].m_Individuals[j].GetAdjFitness();
        }
    }

    // must be above 0
    ASSERT(t_total_adjusted_fitness > 0.0);

    t_average_adjusted_fitness = t_total_adjusted_fitness / static_cast<double>(m_Parameters.PopulationSize);
    if (t_average_adjusted_fitness == 0.0)
    {
        t_average_adjusted_fitness = 1.0;
    }
    
    // Calculate how much offspring each individual should have
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            m_Species[i].m_Individuals[j].SetOffspringAmount( m_Species[i].m_Individuals[j].GetAdjFitness() / t_average_adjusted_fitness);
        }
    }

    // Now count how many offpring each species should have
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].CountOffspring();
    }
}


// This little tool function helps ordering the genomes by fitness
bool species_greater(Species& ls, Species& rs)
{
    return ((ls.GetActualBestFitness()) > (rs.GetActualBestFitness()));
}

void Population::Sort()
{
    ASSERT(m_Species.size() > 0);

    // Step through each species and sort its members by fitness
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        ASSERT(m_Species[i].NumIndividuals() > 0);
        m_Species[i].SortIndividuals();
    }

    // Now sort the species by fitness (best first)
    std::sort(m_Species.begin(), m_Species.end(), species_greater);
}



// Updates the species
void Population::UpdateSpecies()
{
    // search for the current best species ID if not at generation #0
    int t_oldbestid = -1, t_newbestid = -1;
    int t_oldbestidx = -1;
    if (m_Generation > 0)
    {
        for(unsigned int i=0; i<m_Species.size(); i++)
        {
            if (m_Species[i].IsBestSpecies())
            {
                t_oldbestid  = m_Species[i].ID();
                t_oldbestidx = i;
            }
        }
        ASSERT(t_oldbestid  != -1);
        ASSERT(t_oldbestidx != -1);
    }

    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].SetBestSpecies(false);
    }

    bool t_marked = false; // new best species marked?

    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        // Reset the species and update its age
        m_Species[i].IncreaseAgeGens();
        m_Species[i].IncreaseGensNoImprovement();
        m_Species[i].SetOffspringRqd(0);

        // Mark the best species so it is guaranteed to survive
        // Only one species will be marked - in case several species
        // have equally best fitness
        if ((m_Species[i].GetBestFitness() >= m_BestFitnessEver) && (!t_marked))
        {
            m_Species[i].SetBestSpecies(true);
            t_marked = true;
            t_newbestid = m_Species[i].ID();
        }
    }

    // This prevents the previous best species from sudden death
    // If the best species happened to be another one, reset the old
    // species age so it still will have a chance of survival and improvement
    // if it grows old and stagnates again, it is no longer the best one
    // so it will die off anyway.
    if ((t_oldbestid != t_newbestid) && (t_oldbestid != -1))
    {
        m_Species[t_oldbestidx].ResetAgeGens();
    }
}



// the epoch method - the heart of the GA
void Population::Epoch()
{   
    // So, all genomes are evaluated..
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            m_Species[i].m_Individuals[j].SetEvaluated();
        }
    }

    // Sort the population first
    Sort(); 
        
    // Update species stagnation info & stuff
    UpdateSpecies();
    
    ///////////////////
    // Preparation
    ///////////////////

    // Adjust the species's fitness
    AdjustFitness();
    
    // Count the offspring of each individual and species
    CountOffspring();
    
    // Incrementing the global stagnation counter, we can check later for global stagnation
    m_GensSinceBestFitnessLastChanged++;
    // Find and save the best genome and fitness
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        // Update best genome info
        m_Species[i].m_BestGenome = m_Species[i].GetLeader();

        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            const double t_Fitness = m_Species[i].m_Individuals[j].GetFitness();
            if (t_Fitness > m_BestFitnessEver)
            {
                // Reset the stagnation counter only if the fitness jump is greater or equal to the delta.
                if (fabs(t_Fitness - m_BestFitnessEver) >= m_Parameters.StagnationDelta)
                {
                    m_GensSinceBestFitnessLastChanged = 0;
                }

                m_BestFitnessEver = t_Fitness;
                m_BestGenomeEver = m_Species[i].m_Individuals[j];
            }
        }
    }
    
    // Find and save the current best genome
    double t_bestf = std::numeric_limits<double>::min();
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            if (m_Species[i].m_Individuals[j].GetFitness() > t_bestf)
            {
                t_bestf = m_Species[i].m_Individuals[j].GetFitness();
                m_BestGenome = m_Species[i].m_Individuals[j];
            }
        }
    }
    
    // adjust the compatibility threshold
    if (m_Parameters.DynamicCompatibility == true)
    {
        if ((m_Generation % m_Parameters.CompatTreshChangeInterval_Generations) == 0)
        {
            if (m_Species.size() > m_Parameters.MaxSpecies)
            {
                m_Parameters.CompatTreshold += m_Parameters.CompatTresholdModifier;
            }
            else if (m_Species.size() < m_Parameters.MinSpecies)
            {
                m_Parameters.CompatTreshold -= m_Parameters.CompatTresholdModifier;
            }
        }

        if (m_Parameters.CompatTreshold < m_Parameters.MinCompatTreshold) m_Parameters.CompatTreshold = m_Parameters.MinCompatTreshold;
    }
    
    // A special case for global stagnation.
    // Delta coding - if there is a global stagnation
    // for dropoff age + 10 generations, focus the search on the top 2 species,
    // in case there are more than 2, of course
    if (m_Parameters.DeltaCoding)
    {
        if (m_GensSinceBestFitnessLastChanged > (m_Parameters.SpeciesMaxStagnation + 10))
        {
            // make the top 2 reproduce by 50% individuals
            // and the rest - no offspring
            if (m_Species.size() > 2)
            {
                // The first two will reproduce
                m_Species[0].SetOffspringRqd( m_Parameters.PopulationSize/2 );
                m_Species[1].SetOffspringRqd( m_Parameters.PopulationSize/2 );

                // The rest will not
                for (unsigned int i=2; i<m_Species.size(); i++)
                {
                    m_Species[i].SetOffspringRqd( 0 );
                }

                // Now reset the stagnation counter and species age
                m_Species[0].ResetAgeGens();
                m_Species[1].ResetAgeGens();
                m_GensSinceBestFitnessLastChanged = 0;
            }
        }
    }
    
    //////////////////////////////////
    // Phased searching core logic
    //////////////////////////////////
    // Update the current MPC
    CalculateMPC();
    if (m_Parameters.PhasedSearching)
    {
        // Keep track of complexity when in simplifying phase
        if (m_SearchMode == SIMPLIFYING)
        {
            // The MPC has lowered?
            if (m_CurrentMPC < m_OldMPC)
            {
                // reset that
                m_GensSinceMPCLastChanged = 0;
                m_OldMPC = m_CurrentMPC;
            }
            else
            {
                m_GensSinceMPCLastChanged++;
            }
        }


        // At complexifying phase?
        if (m_SearchMode == COMPLEXIFYING)
        {
            // Need to begin simplification?
            if (m_CurrentMPC > (m_BaseMPC + m_Parameters.SimplifyingPhaseMPCTreshold))
            {
                // Do this only if the whole population is stagnating
                if (m_GensSinceBestFitnessLastChanged > m_Parameters.SimplifyingPhaseStagnationTreshold)
                {
                    // Change the current search mode
                    m_SearchMode = SIMPLIFYING;

                    // Reset variables for simplifying mode
                    m_GensSinceMPCLastChanged = 0;
                    m_OldMPC = std::numeric_limits<double>::max(); // Really big one

                    // reset the age of species
                    for(unsigned int i=0; i<m_Species.size(); i++)
                    {
                        m_Species[i].ResetAgeGens();
                    }
                }
            }
        }
        else if (m_SearchMode == SIMPLIFYING)
            // At simplifying phase?
        {
            // The MPC reached its floor level?
            if (m_GensSinceMPCLastChanged > m_Parameters.ComplexityFloorGenerations)
            {
                // Re-enter complexifying phase
                m_SearchMode = COMPLEXIFYING;

                // Set the base MPC with the current MPC
                m_BaseMPC = m_CurrentMPC;

                // reset the age of species
                for(unsigned int i=0; i<m_Species.size(); i++)
                {
                    m_Species[i].ResetAgeGens();
                }
            }
        }
    }
    
    /////////////////////////////
    // Reproduction
    /////////////////////////////
    
    // Perform reproduction for each species
    m_TempSpecies.clear();
    m_TempSpecies = m_Species;
    for(unsigned int i=0; i<m_TempSpecies.size(); i++)
    {
        m_TempSpecies[i].Clear();
        m_TempSpecies[i].AddIndividual(m_Species[i].m_Individuals[0]);
    }

    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].Reproduce(*this, m_Parameters, m_RNG);
    }
    for(unsigned int i=0; i<m_TempSpecies.size(); i++)
    {
        m_TempSpecies[i].RemoveIndividual(0);
    }
    m_Species = m_TempSpecies;
    
    // Remove all empty species (cleanup routine for every case..)
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        if (m_Species[i].m_Individuals.size() == 0)
        {
            m_Species.erase(m_Species.begin() + i);
            i--;
        }
    }
    
    
    // If the total amount of genomes reproduced is less than the population size,
    // due to some floating point rounding error,
    // we will add some bonus clones of the first species's leader to it

    unsigned int t_total_genomes = 0;
    for(unsigned int i=0; i<m_Species.size(); i++)
        t_total_genomes += static_cast<unsigned int>(m_Species[i].m_Individuals.size());

    if (t_total_genomes < m_Parameters.PopulationSize)
    {
        int t_nts = m_Parameters.PopulationSize - t_total_genomes;

        while(t_nts--)
        {
            ASSERT(m_Species.size() > 0);
            Genome t_tg = m_Species[0].m_Individuals[0];
            m_Species[0].AddIndividual(t_tg);
        }
    }
    
    // Increase generation number
    m_Generation++;

    // At this point we may also empty our innovation database
    // This is the place where we control whether we want to
    // keep innovation numbers forever or not.
    if (!m_Parameters.InnovationsForever)
    {
        m_InnovationDatabase.Flush();
    }
}





Genome g_dummy; // empty genome

// ----------- FIXED: AccessGenomeByIndex uses total population size in all species ----------
Genome& Population::AccessGenomeByIndex(int const a_idx)
{
    // Instead of checking a_idx < m_Genomes.size(), check the actual total population size:
    unsigned int totalPop = 0;
    for (auto &sp : m_Species)
    {
        totalPop += sp.m_Individuals.size();
    }
    ASSERT(a_idx < totalPop);

    int t_counter = 0;
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            if (t_counter == a_idx)
            {
                return m_Species[i].m_Individuals[j];
            }
            t_counter++;
        }
    }
    throw std::runtime_error("Population::AccessGenomeByIndex: index not found in species");
}

Genome& Population::AccessGenomeByID(int const a_id)
{
    for (unsigned int i=0; i<m_Species.size(); i++)
    {
        for (unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            if (m_Species[i].m_Individuals[j].GetID() == a_id)// reached the ID?
            {
                return m_Species[i].m_Individuals[j];
            }
        }
    }
    
    char s[256];
    sprintf(s, "No such ID in population - %d\n", a_id);
    
    // not found?!
    throw std::runtime_error(s);
}


void Population::SameGenomeIDCheck()
{
    // count how much each ID found has occured
    std::map<int, int> ids;
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++)
        {
            ids[m_Species[i].m_Individuals[j].GetID()] = 0;
        }
    }
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++)
        {
            ids[m_Species[i].m_Individuals[j].GetID()] += 1;
        }
    }
    
    for(auto it = ids.begin(); it != ids.end(); it++)
    {
        if (it->second > 1)
        {
            char s[256];
            sprintf(s, "Genome ID %d appears %d times in the population\n", it->first, it->second);
            throw std::runtime_error(s);
        }
    }
}


// Removes worst member of the whole population that has been around for a minimum amount of time
// returns the genome that was just deleted (may be useful)
Genome Population::RemoveWorstIndividual()
{
    unsigned int t_worst_idx=0; // within the species
    unsigned int t_worst_species_idx=0; // within the population
    double       t_worst_fitness = std::numeric_limits<double>::max();
    int numev=0;

    Genome t_genome;
    
    bool found=false;

    // Find and kill the individual with the worst *adjusted* fitness
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        if (m_Species[i].m_Individuals.size() > 0)
        {
            double adjinv = 1.0 / static_cast<double>(m_Species[i].m_Individuals.size());
            for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++)
            {
                // only evaluated individuals can be removed
                if (m_Species[i].m_Individuals[j].IsEvaluated())
                {
                    numev++;
                    double t_adjusted_fitness = m_Species[i].m_Individuals[j].GetFitness() * adjinv;
                    if (std::isnan(t_adjusted_fitness) || std::isinf(t_adjusted_fitness))
                    {
                        t_adjusted_fitness = 0;
                    }
            
                    if (t_adjusted_fitness < t_worst_fitness)
                    {
                        t_worst_fitness = t_adjusted_fitness;
                        t_worst_idx = j;
                        t_worst_species_idx = i;
                        found = true;
                    }
                }
            }
        }
    }
    
    if (found)
    {
        t_genome = m_Species[t_worst_species_idx].m_Individuals[t_worst_idx];
        
        // make sure this isn't the only evaluated individual
        if (numev <= 1)
        {
            return t_genome;
        }

        // The individual is now removed
        m_Species[t_worst_species_idx].RemoveIndividual(t_worst_idx);
    
        // If the species becomes empty, remove the species as well
        if (m_Species[t_worst_species_idx].m_Individuals.size() == 0)
        {
            m_Species.erase(m_Species.begin() + t_worst_species_idx);
        }
    }
    else
    {
        // set ID of -1 to indicate nothing was removed
        t_genome.SetID(-1);
#ifdef VDEBUG
        std::cout << "RemoveWorst did not remove anything.\n";
#endif
    }
    
    return t_genome;
}


unsigned int Population::ChooseParentSpecies()
{
    ASSERT(m_Species.size() > 0);

    unsigned int t_curspecies = 0;
    //do
    std::vector<double> probs;
    for(int i=0; i<(int)m_Species.size(); i++)
    {
        if ((m_Species[i].NumEvaluated() == 0) || (m_Species[i].NumIndividuals() == 0))
        {
            probs.push_back(0.0);
        }
        else
        {
            probs.push_back(m_Species[i].m_AverageFitness);
        }
    }
    t_curspecies = m_RNG.Roulette(probs);

    return t_curspecies;
}


void Population::ReassignSpecies(int a_genome_idx)
{
    //ASSERT(a_genome_idx < m_Genomes.size());

    // first remember where is this genome exactly
    int t_species_idx = 0, t_genome_rel_idx = 0;
    int t_counter = 0;

    // to keep the genome
    Genome t_genome;

    // search for it
    t_species_idx = 0;
    for(int i=0; i<(int)m_Species.size(); i++)
    {
        t_genome_rel_idx = 0;
        if ((t_counter + (int)m_Species[i].m_Individuals.size()) > a_genome_idx)
        {
            // it's here
            t_genome_rel_idx = a_genome_idx - t_counter;
            break;
        }
        else
        {
            t_counter += (int)m_Species[i].m_Individuals.size();
            t_species_idx++;
        }
    }
    
    // save the individual
    t_genome = m_Species[t_species_idx].m_Individuals[t_genome_rel_idx];
    
    // Remove it from its species
    m_Species[t_species_idx].RemoveIndividual(t_genome_rel_idx);

    // Find a new species for this genome
    bool t_found = false;
    auto t_cur_species = m_Species.begin();

    // No species yet?
    if (t_cur_species == m_Species.end())
    {
        // create the first species and place the baby there
        m_Species.push_back( Species(t_genome, m_Parameters, GetNextSpeciesID()));
        IncrementNextSpeciesID();
    }
    else
    {
        // try to find a compatible species
        Genome& t_to_compare = t_cur_species->GetRepresentative(); 

        t_found = false;
        while((t_cur_species != m_Species.end()) && (!t_found))
        {
            if (t_genome.IsCompatibleWith( t_to_compare, m_Parameters ))
            {
                // found a compatible species
                t_cur_species->AddIndividual(t_genome);
                t_found = true; // the search is over
            }
            else
            {
                // keep searching for a matching non-empty species
                while(1)
                {
                    t_cur_species++;
                    if (t_cur_species == m_Species.end())
                    {
                        break;
                    }
                    if (t_cur_species->NumIndividuals() > 0)
                    {
                        // reassign t_to_compare for next iteration
                        break;
                    }
                }
            }
        }

        // if couldn't find a match, make a new species
        if (!t_found)
        {
            m_Species.push_back( Species(t_genome, m_Parameters, GetNextSpeciesID()));
            IncrementNextSpeciesID();
        }
    }
}

void Population::ClearEmptySpecies()
{
    auto t_cs = m_Species.begin();
    while(t_cs != m_Species.end())
    {
        if (t_cs->NumIndividuals() == 0)
        {
            // remove the dead species
            t_cs = m_Species.erase(t_cs );

            if (t_cs != m_Species.begin()) // in case the first species are dead
                t_cs--;
        }

        t_cs++;
    }
}


Genome* Population::Tick(Genome& a_deleted_genome)
{
    // Make sure at least one individual is evaluated
    int ne=0;
    for(int i=0; i<(int)m_Species.size(); i++)
    {
        ne += m_Species[i].NumEvaluated();
    }
    if (ne==0)
    {
        throw std::runtime_error("Called Tick() on population with no evaluated individuals.\n");
    }
    
#ifdef VDEBUG
    std::cout << "tracking stuff\n";
#endif

    m_NumEvaluations++;

    // Find and save the best genome and fitness
    m_EvalsSinceBestFitnessLastChanged++;
    for(int i=0; i<(int)m_Species.size(); i++)
    {
        for(int j=0; j<(int)m_Species[i].m_Individuals.size(); j++)
        {
            double t_fitness = m_Species[i].m_Individuals[j].GetFitness();
            if (std::isnan(t_fitness) || std::isinf(t_fitness))
            {
                t_fitness = 0;
            }
            
            if (t_fitness > m_BestFitnessEver)
            {
                // Reset the stagnation counter only if the fitness jump is greater or equal to the delta.
                if (fabs(t_fitness - m_BestFitnessEver) >= m_Parameters.StagnationDelta)
                {
                    m_EvalsSinceBestFitnessLastChanged = 0;
                }

                m_BestFitnessEver = t_fitness;
                m_BestGenomeEver = m_Species[i].m_Individuals[j];
            }
        }
    }

    double t_f = std::numeric_limits<double>::min();
    for(int i=0; i<(int)m_Species.size(); i++)
    {
        for(int j=0; j<(int)m_Species[i].m_Individuals.size(); j++)
        {
            if (m_Species[i].m_Individuals[j].GetFitness() > t_f)
            {
                t_f = m_Species[i].m_Individuals[j].GetFitness();
                m_BestGenome = m_Species[i].m_Individuals[j];
            }

            if (m_Species[i].m_Individuals[j].GetFitness() > m_Species[i].GetBestFitness())
            {
                m_Species[i].m_BestFitness = m_Species[i].m_Individuals[j].GetFitness();
                m_Species[i].m_EvalsNoImprovement = 0;
            }
        }
    }


    // adjust the compatibility treshold
    bool t_changed = false;
    if (m_Parameters.DynamicCompatibility == true)
    {
        double t_oldcompat = m_Parameters.CompatTreshold;
        if ((m_NumEvaluations % m_Parameters.CompatTreshChangeInterval_Evaluations) == 0)
        {
            if (m_Species.size() > m_Parameters.MaxSpecies)
            {
                m_Parameters.CompatTreshold += m_Parameters.CompatTresholdModifier;
            }
            else if (m_Species.size() < m_Parameters.MinSpecies)
            {
                m_Parameters.CompatTreshold -= m_Parameters.CompatTresholdModifier;
            }

            if (m_Parameters.CompatTreshold < m_Parameters.MinCompatTreshold)
                m_Parameters.CompatTreshold = m_Parameters.MinCompatTreshold;
            
            if (m_Parameters.CompatTreshold != t_oldcompat)
            {
                t_changed = true;
            }
        }
    }
    
    // If the compatibility treshold was changed, reassign all individuals by species
    if (t_changed)
    {        
        m_Genomes.clear();
        for(unsigned int i=0; i<m_Species.size(); i++)
        {
            for (unsigned int j = 0; j<m_Species[i].m_Individuals.size(); j++)
            {
                m_Genomes.push_back(m_Species[i].m_Individuals[j]);
            }
        }
        
        Speciate();
    }
    
#ifdef VDEBUG
    SameGenomeIDCheck();
#endif

#ifdef VDEBUG
    std::cout << "remove worst\n";
#endif
    // Remove the worst individual
    a_deleted_genome = RemoveWorstIndividual();


#ifdef VDEBUG
    std::cout << "calc avg fitness\n";
#endif
    // Recalculate all averages for each species
    // If the average species fitness of a species is 0,
    // then there are no evaluated individuals in it.
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].CalculateAverageFitness();
    }

#ifdef VDEBUG
    std::cout << "choose parents\n";
#endif
    // Now spawn the new offspring
    unsigned int t_parent_species_index = ChooseParentSpecies();

    Genome t_baby = m_Species[t_parent_species_index].ReproduceOne(*this, m_Parameters, m_RNG);
    ASSERT(t_baby.NumInputs() > 0);
    ASSERT(t_baby.NumOutputs() > 0);
    Genome* t_to_return = NULL;

#ifdef VDEBUG
    std::cout << "placing baby in species\n";
#endif

    // Add the baby to its proper species
    bool t_found = false;
    auto t_cur_species = m_Species.begin();

    // No species yet?
    if (t_cur_species == m_Species.end())
    {
        // create the first species and place the baby there
        m_Species.push_back( Species(t_baby, m_Parameters, GetNextSpeciesID()) );
        t_to_return = &(m_Species[ m_Species.size()-1 ].m_Individuals[ m_Species[ m_Species.size()-1 ].m_Individuals.size() - 1]);
        IncrementNextSpeciesID();

#ifdef VDEBUG
        std::cout << "made new species\n";
#endif
    }
    else
    {
        // try to find a compatible species
        Genome t_to_compare = t_cur_species->GetRepresentative();

        t_found = false;
        while((t_cur_species != m_Species.end()) && (!t_found))
        {
            if (t_baby.IsCompatibleWith( t_to_compare, m_Parameters ))
            {
                // found a compatible species
                t_cur_species->AddIndividual(t_baby);
                t_to_return = &(t_cur_species->m_Individuals[ t_cur_species->m_Individuals.size() - 1]);
                t_found = true; // the search is over

                // increase the evals counter for the new species
                t_cur_species->IncreaseEvalsNoImprovement();

#ifdef VDEBUG
                std::cout << "found compatible species\n";
#endif
            }
            else
            {
                // keep searching for a matching species
                while(1)
                {
                    t_cur_species++;
                    if (t_cur_species == m_Species.end())
                    {
                        break;
                    }
                    if (t_cur_species->NumIndividuals() > 0)
                    {
                        t_to_compare = t_cur_species->GetRepresentative();
                        break;
                    }
                };
            }
        }

        // if couldn't find a match, make a new species
        if (!t_found)
        {
            m_Species.push_back( Species(t_baby, m_Parameters, GetNextSpeciesID()) );
            t_to_return = &(m_Species[ m_Species.size()-1 ].m_Individuals[ m_Species[ m_Species.size()-1 ].m_Individuals.size() - 1]);
            IncrementNextSpeciesID();

#ifdef VDEBUG
            std::cout << "made new species\n";
#endif
        }
    }
    
#ifdef VDEBUG
    std::cout << "\n";
#endif
    
    ASSERT(t_to_return != NULL);

    return t_to_return;
}


bool Population::NoveltySearchTick(Genome& a_SuccessfulGenome)
{
    // Recompute the sparseness/fitness for all individuals in the population
    // This will introduce the constant pressure to do something new
    if ((m_NumEvaluations % m_Parameters.NoveltySearch_Recompute_Sparseness_Each)==0)
    {
        for(unsigned int i=0; i<m_Species.size(); i++)
        {
            for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
            {
                m_Species[i].m_Individuals[j].SetFitness(ComputeSparseness(m_Species[i].m_Individuals[j]));
            }
        }
    }

    // OK now get the new baby
    Genome  t_temp_genome;
    Genome* t_new_baby = Tick(t_temp_genome);

    // replace the new individual's behavior to point to the dead one's
    t_new_baby->m_PhenotypeBehavior = t_temp_genome.m_PhenotypeBehavior;

    // Now it is time to acquire the new behavior from the baby
    bool t_success = t_new_baby->m_PhenotypeBehavior->Acquire( t_new_baby );


    // if found a successful one, just copy it and return true
    if (t_success)
    {
        a_SuccessfulGenome = *t_new_baby;
        return true;
    }

    // We have the new behavior, now let's calculate the sparseness of
    // the point in behavior space
    double t_sparseness = ComputeSparseness(*t_new_baby);

    // OK now we have the sparseness for this behavior
    // if the sparseness is above Pmin, add this behavior to the archive
    m_GensSinceLastArchiving++;
    if (t_sparseness > m_Parameters.NoveltySearch_P_min )
    {
        // check to see if this behavior is already present in the archive
        bool present = false;

        if (!present)
        {
            m_BehaviorArchive->push_back( *(t_new_baby->m_PhenotypeBehavior) );
            m_GensSinceLastArchiving = 0;
            m_QuickAddCounter++;
        }
    }
    else
    {
        // no addition to the archive
        m_QuickAddCounter = 0;
    }


    // dynamic Pmin
    if (m_Parameters.NoveltySearch_Dynamic_Pmin)
    {
        // too many generations without adding to the archive?
        if (m_GensSinceLastArchiving > m_Parameters.NoveltySearch_No_Archiving_Stagnation_Treshold)
        {
            m_Parameters.NoveltySearch_P_min *= m_Parameters.NoveltySearch_Pmin_lowering_multiplier;
            if (m_Parameters.NoveltySearch_P_min < m_Parameters.NoveltySearch_Pmin_min)
            {
                m_Parameters.NoveltySearch_P_min = m_Parameters.NoveltySearch_Pmin_min;
            }
        }

        // too much additions to the archive (one after another)?
        if (m_QuickAddCounter > m_Parameters.NoveltySearch_Quick_Archiving_Min_Evaluations)
        {
            m_Parameters.NoveltySearch_P_min *= m_Parameters.NoveltySearch_Pmin_raising_multiplier;
        }
    }

    // Now we assign a fitness score based on the sparseness
    t_new_baby->SetFitness( t_sparseness );

    a_SuccessfulGenome = *t_new_baby;

    return t_new_baby->m_PhenotypeBehavior->Successful();
}

double Population::ComputeSparseness(Genome& genome)
{
    std::vector<double> distances;
    distances.clear();
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            distances.push_back( genome.m_PhenotypeBehavior->Distance_To( m_Species[i].m_Individuals[j].m_PhenotypeBehavior ) );
        }
    }
    if(m_BehaviorArchive)
    {
        for(unsigned int i=0; i<m_BehaviorArchive->size(); i++)
        {
            distances.push_back( genome.m_PhenotypeBehavior->Distance_To( &((*m_BehaviorArchive)[i]) ) );
        }
    }
    
    if(distances.empty())
        return 0.0;
    
    // Remove the self-distance (assumed to be the smallest—usually zero)
    auto selfIt = std::min_element(distances.begin(), distances.end());
    if(selfIt != distances.end()){
        distances.erase(selfIt);
    }
    
    int k = m_Parameters.NoveltySearch_K;
    if(distances.size() < static_cast<size_t>(k))
        k = distances.size();
    
    std::nth_element(distances.begin(), distances.begin() + k, distances.end());
    double sum = 0.0;
    for (int i = 0; i < k; i++){
        sum += distances[i];
    }
    return sum / k;
}


std::string Population::Serialize() const {
    std::ostringstream oss;
    // Use markers for clarity.
    oss << "PopulationStart\n";
    oss << m_Generation << " " << m_NumEvaluations << " " 
        << m_NextGenomeID << " " << m_NextSpeciesID << " " 
        << m_BestFitnessEver << "\n";
    // Write the number of species.
    oss << m_Species.size() << "\n";
    // For each species, write its serialization
    for (const auto &spec : m_Species) {
         oss << spec.Serialize() << "\n";
    }
    oss << "PopulationEnd\n";
    return oss.str();
}

Population Population::Deserialize(const std::string &data) {
    std::istringstream iss(data);
    std::string token;
    iss >> token;
    if (token != "PopulationStart")
        throw std::runtime_error("Population::Deserialize: missing PopulationStart marker.");
    
    Population pop;
    iss >> pop.m_Generation >> pop.m_NumEvaluations >> pop.m_NextGenomeID 
        >> pop.m_NextSpeciesID >> pop.m_BestFitnessEver;
    
    size_t numSpecies = 0;
    iss >> numSpecies;
    
    // Consume the rest of the line.
    std::string dummy;
    std::getline(iss, dummy);
    
    pop.m_Species.clear();
    // For each species we have a block (which might span multiple lines) delimited by the markers.
    for (size_t i = 0; i < numSpecies; i++) {
         std::ostringstream speciesStream;
         std::string l;
         // Keep reading until we encounter "SpeciesEnd" in a line.
         // First, we expect a line beginning with "SpeciesStart".
         while (std::getline(iss, l)) {
             if (l.find("SpeciesStart") != std::string::npos) {
                 speciesStream << l << "\n";
                 break;
             }
         }
         // Now read until the "SpeciesEnd" marker.
         while (std::getline(iss, l)) {
             speciesStream << l << "\n";
             if (l.find("SpeciesEnd") != std::string::npos) {
                 break;
             }
         }
         NEAT::Species s = NEAT::Species::Deserialize(speciesStream.str());
         pop.m_Species.push_back(s);
    }
    
    iss >> token;
    if (token != "PopulationEnd")
         throw std::runtime_error("Population::Deserialize: missing PopulationEnd marker.");
    
    return pop;
}



} // namespace NEAT

