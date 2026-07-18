#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <set>
#include <sstream>
#include <stdexcept>

#include "Genome.h"
#include "Species.h"
#include "Random.h"
#include "Population.h"
#include "Utils.h"
#include "Assert.h"
#include "FileIO.h"


namespace NEAT
{

bool Population::Validate(std::string* error) const
{
    const auto fail = [error](const std::string& message)
    {
        if (error != nullptr)
        {
            *error = message;
        }
        return false;
    };

    if (m_Species.empty())
    {
        return NumGenomes() == 0
            ? true
            : fail("Population has genomes but no species");
    }
    if (NumGenomes() != m_Parameters.PopulationSize)
    {
        return fail(
            "Population genome count does not match Parameters::PopulationSize");
    }
    if (!std::isfinite(m_BestFitnessEver) ||
        !std::isfinite(m_CurrentMPC) ||
        !std::isfinite(m_OldMPC) ||
        !std::isfinite(m_BaseMPC))
    {
        return fail("Population fitness and complexity state must be finite");
    }

    std::set<int> genome_ids;
    std::set<int> species_ids;
    int maximum_genome_id = -1;
    int maximum_species_id = 0;
    for (const auto& species : m_Species)
    {
        if (species.NumIndividuals() == 0)
            return fail("Population contains an empty species");
        if (species.ID() <= 0 ||
            !species_ids.insert(species.ID()).second)
            return fail("Population species IDs must be positive and unique");
        maximum_species_id = std::max(maximum_species_id, species.ID());

        for (const auto& genome : species.m_Individuals)
        {
            std::string genome_error;
            if (!genome.Validate(&genome_error))
                return fail("Population contains an invalid genome: " +
                            genome_error);
            if (genome.GetID() < 0 ||
                !genome_ids.insert(genome.GetID()).second)
                return fail(
                    "Population genome IDs must be non-negative and unique");
            maximum_genome_id =
                std::max(maximum_genome_id, genome.GetID());
        }
    }
    if (m_NextGenomeID <= static_cast<unsigned int>(maximum_genome_id))
        return fail("Population next genome ID would reuse an existing ID");
    if (m_NextSpeciesID <= static_cast<unsigned int>(maximum_species_id))
        return fail("Population next species ID would reuse an existing ID");
    return true;
}

// The constructor
Population::Population(const Genome& a_Seed, const Parameters& a_Parameters,
		               bool a_RandomizeWeights, double a_RandomizationRange, int a_RNG_seed)
{
    if (a_Parameters.PopulationSize == 0)
    {
        throw std::invalid_argument("PopulationSize must be greater than zero");
    }
    if (a_RandomizationRange < 0.0 || !std::isfinite(a_RandomizationRange))
    {
        throw std::invalid_argument("Randomization range must be finite and non-negative");
    }

    m_RNG.Seed(a_RNG_seed);
    m_BestFitnessEver = std::numeric_limits<double>::lowest();
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
            const int max_attempts = std::max(1, a_Parameters.ConstraintTrials);
            for (int attempt = 0; attempt < max_attempts && is_invalid; ++attempt)
            {
                Parameters initialization_parameters = a_Parameters;
                initialization_parameters.MinWeight =
                    std::max(a_Parameters.MinWeight, -a_RandomizationRange);
                initialization_parameters.MaxWeight =
                    std::min(a_Parameters.MaxWeight, a_RandomizationRange);
                if (initialization_parameters.MinWeight >
                    initialization_parameters.MaxWeight)
                {
                    throw std::invalid_argument(
                        "Randomization range does not overlap the configured weight range");
                }
                m_Genomes[i].Randomize_LinkWeights(initialization_parameters, m_RNG);
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
            if (is_invalid)
            {
                throw std::runtime_error(
                    "Unable to initialize a valid, non-cloning genome within ConstraintTrials");
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
    m_BestFitnessEver = std::numeric_limits<double>::lowest();

    m_Generation = 0;
    m_NumEvaluations = 0;
    m_NextSpeciesID = 1;
    m_ID = 0;
    m_GensSinceBestFitnessLastChanged = 0;
    m_GensSinceMPCLastChanged = 0;

    std::ifstream t_DataFile(a_FileName, std::ios::binary);
    if (!t_DataFile.is_open())
        throw std::runtime_error("Cannot open population file");

    std::string first_token;
    t_DataFile >> first_token;
    t_DataFile.clear();
    t_DataFile.seekg(0);
    if (first_token == "PopulationStart")
    {
        std::ostringstream checkpoint;
        checkpoint << t_DataFile.rdbuf();
        *this = Deserialize(checkpoint.str());
        return;
    }

    // Load the parameters
    if (m_Parameters.Load(t_DataFile) != 0 ||
        m_Parameters.PopulationSize == 0)
    {
        throw std::runtime_error(
            "Population file contains invalid parameters");
    }

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
        if (m_Genomes[i].GetID() >= 0 &&
            static_cast<unsigned int>(m_Genomes[i].GetID()) > m_NextGenomeID)
        {
            m_NextGenomeID =
                static_cast<unsigned int>(m_Genomes[i].GetID());
        }
    }
    m_NextGenomeID++;

    // Initialize
    Speciate();
    m_BestGenome = m_Species[0].GetLeader();
    m_BestGenomeEver = m_BestGenome;
    m_BestFitnessEver = m_BestGenome.GetFitness();

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
    if (a_FileName == nullptr)
    {
        throw std::invalid_argument("Population filename is null");
    }
    FILE* t_file = detail::OpenFile(a_FileName, "w");
    if (t_file == nullptr)
    {
        throw std::runtime_error("Cannot open population file for writing");
    }

    try
    {
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
    }
    catch (...)
    {
        fclose(t_file);
        throw;
    }

    if (fclose(t_file) != 0)
    {
        throw std::runtime_error("Failed to close population file");
    }
}


// Calculates the current mean population complexity
void Population::CalculateMPC()
{
    m_CurrentMPC = 0;

    const unsigned int genome_count = NumGenomes();
    if (genome_count == 0)
    {
        return;
    }

    for (unsigned int i = 0; i < genome_count; ++i)
    {
        m_CurrentMPC += AccessGenomeByIndex(static_cast<int>(i)).NumLinks();
    }

    m_CurrentMPC /= static_cast<double>(genome_count);
}


// Separates the population into species
// also adjusts the compatibility treshold if this feature is enabled
void Population::Speciate()
{
    // iterate through the genome list and speciate
    // at least 1 genome must be present
    ASSERT(m_Genomes.size() > 0);

    if (m_Genomes.empty())
    {
        throw std::runtime_error("Cannot speciate an empty population");
    }

    // first clear out the species
    m_Species.clear();

    if (!m_Parameters.Speciation)
    {
        m_Species.emplace_back(
            m_Genomes.front(), m_Parameters, m_NextSpeciesID++);
        for (std::size_t i = 1; i < m_Genomes.size(); ++i)
        {
            m_Species.front().AddIndividual(m_Genomes[i]);
        }
        return;
    }


    // NOTE: we are comparing the new generation's genomes to the representatives from species creation time!
    //
    for(unsigned int i=0; i<m_Genomes.size(); i++)
    {
        bool t_added = false;

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
    if (m_Species.empty() || NumGenomes() == 0)
    {
        throw std::runtime_error(
            "Cannot adjust fitness for an empty population");
    }

    double minimum_fitness = 0.0;
    bool found_finite = false;
    for (const auto &species : m_Species)
    {
        for (const auto &genome : species.m_Individuals)
        {
            if (std::isfinite(genome.GetFitness()))
            {
                minimum_fitness = found_finite
                    ? std::min(minimum_fitness, genome.GetFitness())
                    : genome.GetFitness();
                found_finite = true;
            }
        }
    }
    const double offset =
        !found_finite || minimum_fitness <= 0.0
            ? -minimum_fitness + 1.0e-7
            : 0.0;

    for (auto &species : m_Species)
    {
        species.AdjustFitness(m_Parameters, offset);
    }
}

// Calculates how many offspring each genome should have
void Population::CountOffspring()
{
    const unsigned int population_size = NumGenomes();
    if (population_size == 0)
    {
        throw std::runtime_error(
            "Cannot count offspring for an empty population");
    }
    if (population_size != m_Parameters.PopulationSize)
    {
        throw std::runtime_error(
            "Population size does not match Parameters::PopulationSize");
    }

    double t_total_adjusted_fitness = 0.0;
    double t_average_adjusted_fitness = 0.0;

    // get the total adjusted fitness for all individuals
    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
        {
            t_total_adjusted_fitness += m_Species[i].m_Individuals[j].GetAdjFitness();
        }
    }

    t_average_adjusted_fitness =
        t_total_adjusted_fitness / static_cast<double>(population_size);
    if (!std::isfinite(t_average_adjusted_fitness) ||
        t_average_adjusted_fitness <= 0.0)
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

void Population::SaveState(const char* a_FileName) const
{
    if (a_FileName == nullptr)
    {
        throw std::invalid_argument("Population checkpoint filename is null");
    }

    std::ofstream output(a_FileName, std::ios::binary | std::ios::trunc);
    if (!output.is_open())
    {
        throw std::runtime_error(
            "Cannot open population checkpoint for writing");
    }
    output << Serialize();
    output.close();
    if (!output)
    {
        throw std::runtime_error("Failed to write population checkpoint");
    }
}

void Population::ResetSpecies()
{
    for (auto &species : m_Species)
    {
        species.ClearIndividuals();
        species.SetOffspringRqd(0.0);
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
    if (m_Species.empty() || NumGenomes() == 0)
    {
        throw std::runtime_error("Cannot run Epoch on an empty population");
    }
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
    double t_bestf = std::numeric_limits<double>::lowest();
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
        if (m_Parameters.CompatTreshChangeInterval_Generations > 0 &&
            (m_Generation %
             m_Parameters.CompatTreshChangeInterval_Generations) == 0)
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

    // Convert per-species fractional requirements into an exact population
    // total before reproduction. The old post-reproduction padding cloned a
    // leader (and its ID), violating both ID uniqueness and AllowClones.
    std::vector<unsigned int> offspring_counts(m_Species.size(), 0);
    unsigned int assigned_offspring = 0;
    for (std::size_t i = 0; i < m_Species.size(); ++i)
    {
        offspring_counts[i] = Rounded(m_Species[i].GetOffspringRqd());
        assigned_offspring += offspring_counts[i];
    }
    if (assigned_offspring < m_Parameters.PopulationSize)
    {
        offspring_counts.front() +=
            m_Parameters.PopulationSize - assigned_offspring;
    }
    else
    {
        unsigned int excess =
            assigned_offspring - m_Parameters.PopulationSize;
        for (std::size_t i = offspring_counts.size();
             i > 0 && excess > 0;
             --i)
        {
            const std::size_t index = i - 1;
            const unsigned int reduction =
                std::min(excess, offspring_counts[index]);
            offspring_counts[index] -= reduction;
            excess -= reduction;
        }
        if (excess != 0)
        {
            throw std::runtime_error(
                "Unable to reconcile species offspring counts");
        }
    }
    for (std::size_t i = 0; i < m_Species.size(); ++i)
    {
        m_Species[i].SetOffspringRqd(
            static_cast<double>(offspring_counts[i]));
    }
    
    // Perform reproduction for each species
    m_TempSpecies.clear();
    m_TempSpecies = m_Species;
    const std::size_t existing_species_count = m_TempSpecies.size();
    for(unsigned int i=0; i<m_TempSpecies.size(); i++)
    {
        m_TempSpecies[i].Clear();
        m_TempSpecies[i].AddIndividual(m_Species[i].m_Individuals[0]);
    }

    for(unsigned int i=0; i<m_Species.size(); i++)
    {
        m_Species[i].Reproduce(*this, m_Parameters, m_RNG);
    }
    // Only the original species contain the representative placeholder.
    // Species created during reproduction contain real offspring at index 0.
    for (std::size_t i = 0; i < existing_species_count; ++i)
    {
        m_TempSpecies[i].RemoveIndividual(0);
    }
    m_Species = m_TempSpecies;
    
    // Remove all empty species (cleanup routine for every case..)
    ClearEmptySpecies();
    
    
    unsigned int t_total_genomes = 0;
    for(unsigned int i=0; i<m_Species.size(); i++)
        t_total_genomes += static_cast<unsigned int>(m_Species[i].m_Individuals.size());

    if (t_total_genomes != m_Parameters.PopulationSize)
    {
        throw std::runtime_error(
            "Reproduction did not preserve the configured population size");
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
    if (a_idx < 0)
    {
        throw std::out_of_range("Population genome index cannot be negative");
    }

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
    throw std::out_of_range("Population genome index is out of range");
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
    
    throw std::out_of_range(
        "No genome with ID " + std::to_string(a_id) + " exists in the population");
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
            throw std::runtime_error(
                "Genome ID " + std::to_string(it->first) + " appears " +
                std::to_string(it->second) + " times in the population");
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
            t_genome.SetID(-1);
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
    if (m_Species.empty())
    {
        throw std::runtime_error("Cannot choose a parent from an empty population");
    }

    std::vector<double> probs;
    double minimum = 0.0;
    bool have_eligible = false;
    for (const auto &species : m_Species)
    {
        if (species.NumEvaluated() == 0 || species.NumIndividuals() == 0)
        {
            probs.push_back(0.0);
        }
        else
        {
            const double fitness =
                std::isfinite(species.m_AverageFitness) ? species.m_AverageFitness : 0.0;
            probs.push_back(fitness);
            minimum = have_eligible ? std::min(minimum, fitness) : fitness;
            have_eligible = true;
        }
    }

    if (!have_eligible)
    {
        throw std::runtime_error("No evaluated species is available for reproduction");
    }
    if (minimum < 0.0)
    {
        for (std::size_t i = 0; i < probs.size(); ++i)
        {
            if (m_Species[i].NumEvaluated() > 0 &&
                m_Species[i].NumIndividuals() > 0)
            {
                probs[i] -= minimum;
            }
        }
    }

    return static_cast<unsigned int>(m_RNG.Roulette(probs));
}


void Population::ReassignSpecies(int a_genome_idx)
{
    if (a_genome_idx < 0 ||
        static_cast<unsigned int>(a_genome_idx) >= NumGenomes())
    {
        throw std::out_of_range("Population genome index is out of range");
    }

    int counter = 0;
    std::size_t source_species = 0;
    std::size_t source_genome = 0;
    for (; source_species < m_Species.size(); ++source_species)
    {
        const int species_size =
            static_cast<int>(m_Species[source_species].m_Individuals.size());
        if (a_genome_idx < counter + species_size)
        {
            source_genome = static_cast<std::size_t>(a_genome_idx - counter);
            break;
        }
        counter += species_size;
    }

    Genome genome = m_Species[source_species].m_Individuals[source_genome];
    m_Species[source_species].RemoveIndividual(
        static_cast<unsigned int>(source_genome));

    bool found = false;
    for (auto &species : m_Species)
    {
        if (species.NumIndividuals() > 0 &&
            genome.IsCompatibleWith(species.GetRepresentative(), m_Parameters))
        {
            species.AddIndividual(genome);
            found = true;
            break;
        }
    }

    if (!found)
    {
        m_Species.emplace_back(genome, m_Parameters, GetNextSpeciesID());
        IncrementNextSpeciesID();
    }
    ClearEmptySpecies();
}

void Population::ClearEmptySpecies()
{
    m_Species.erase(
        std::remove_if(
            m_Species.begin(), m_Species.end(),
            [](const Species &species) { return species.NumIndividuals() == 0; }),
        m_Species.end());
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

    double t_f = std::numeric_limits<double>::lowest();
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
        if (m_Parameters.CompatTreshChangeInterval_Evaluations > 0 &&
            (m_NumEvaluations %
             m_Parameters.CompatTreshChangeInterval_Evaluations) == 0)
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
    if (a_deleted_genome.GetID() < 0)
    {
        throw std::runtime_error(
            "Tick requires at least two evaluated individuals so one can be replaced");
    }


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

        bool t_found = false;
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


void Population::InitPhenotypeBehaviorData(
    std::vector<PhenotypeBehavior>* a_population,
    std::vector<PhenotypeBehavior>* a_archive)
{
    if (a_population == nullptr || a_archive == nullptr)
    {
        throw std::invalid_argument(
            "Novelty search behavior containers cannot be null");
    }

    a_population->clear();
    a_population->resize(NumGenomes());
    m_BehaviorArchive = a_archive;
    m_BehaviorArchive->clear();

    std::size_t counter = 0;
    for (auto &species : m_Species)
    {
        for (auto &genome : species.m_Individuals)
        {
            genome.m_PhenotypeBehavior = &a_population->at(counter++);
            genome.SetFitness(0.0);
        }
    }
}

void Population::InitPhenotypeBehaviorData(
    const std::vector<std::shared_ptr<PhenotypeBehavior>>& a_population)
{
    if (a_population.size() != NumGenomes())
    {
        throw std::invalid_argument(
            "Novelty search requires one behavior object per genome");
    }
    if (std::any_of(
            a_population.begin(),
            a_population.end(),
            [](const std::shared_ptr<PhenotypeBehavior>& behavior)
            {
                return behavior == nullptr;
            }))
    {
        throw std::invalid_argument(
            "Novelty search behavior objects cannot be null");
    }

    m_OwnedBehaviorData = a_population;
    m_OwnedBehaviorArchive.clear();
    m_BehaviorArchive = &m_OwnedBehaviorArchive;

    std::size_t counter = 0;
    for (auto& species : m_Species)
    {
        for (auto& genome : species.m_Individuals)
        {
            genome.m_PhenotypeBehavior =
                m_OwnedBehaviorData.at(counter++).get();
            genome.SetFitness(0.0);
        }
    }
}

const std::vector<PhenotypeBehavior>& Population::GetBehaviorArchive() const
{
    if (m_BehaviorArchive == nullptr)
    {
        throw std::runtime_error(
            "Novelty search behavior data has not been initialized");
    }
    return *m_BehaviorArchive;
}

bool Population::NoveltySearchTick(Genome& a_SuccessfulGenome)
{
    if (m_BehaviorArchive == nullptr)
    {
        throw std::runtime_error(
            "Novelty search behavior data has not been initialized");
    }

    // Recompute the sparseness/fitness for all individuals in the population
    // This will introduce the constant pressure to do something new
    if (m_Parameters.NoveltySearch_Recompute_Sparseness_Each > 0 &&
        (m_NumEvaluations %
         m_Parameters.NoveltySearch_Recompute_Sparseness_Each) == 0)
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
    if (t_new_baby->m_PhenotypeBehavior == nullptr)
    {
        throw std::runtime_error("Novelty search encountered an uninitialized behavior");
    }

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
        // Do not archive the same behavior characterization repeatedly.
        const auto present = std::find(
            m_BehaviorArchive->begin(),
            m_BehaviorArchive->end(),
            *t_new_baby->m_PhenotypeBehavior);
        if (present == m_BehaviorArchive->end())
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
    
    std::size_t k = std::min<std::size_t>(
        distances.size(), m_Parameters.NoveltySearch_K);
    if (k == 0)
        return 0.0;
    if (k < distances.size())
        std::nth_element(
            distances.begin(), distances.begin() + k, distances.end());
    double sum = 0.0;
    for (std::size_t i = 0; i < k; i++){
        sum += distances[i];
    }
    return sum / static_cast<double>(k);
}


namespace
{
std::string ReadDelimitedBlock(std::istream& input,
                               const std::string& start,
                               const std::string& end)
{
    std::string token;
    input >> token;
    if (token != start)
        throw std::runtime_error(
            "Population::Deserialize: missing " + start + " marker.");

    std::ostringstream block;
    block << start;
    std::string line;
    std::getline(input, line);
    block << line << '\n';
    while (std::getline(input, line))
    {
        block << line << '\n';
        if (line == end)
            return block.str();
    }
    throw std::runtime_error(
        "Population::Deserialize: missing " + end + " marker.");
}
}

std::string Population::Serialize() const
{
    std::string validation_error;
    if (!Validate(&validation_error))
    {
        throw std::runtime_error(
            "Population::Serialize: " + validation_error);
    }
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "PopulationStart\n";
    output << "PopulationFormat 2\n";
    output << "PopulationState " << m_Generation << ' ' << m_NumEvaluations
           << ' ' << m_NextGenomeID << ' ' << m_NextSpeciesID << ' '
           << m_BestFitnessEver << ' ' << m_ID << ' '
           << m_GensSinceBestFitnessLastChanged << ' '
           << m_EvalsSinceBestFitnessLastChanged << ' '
           << m_GensSinceMPCLastChanged << ' '
           << static_cast<int>(m_SearchMode) << ' ' << m_CurrentMPC << ' '
           << m_OldMPC << ' ' << m_BaseMPC << ' '
           << m_GensSinceLastArchiving << ' ' << m_QuickAddCounter << '\n';
    output << "RNG " << std::quoted(m_RNG.Serialize()) << '\n';
    output << "Parameters\n" << m_Parameters.Serialize();
    output << "InnovationDatabase\n" << m_InnovationDatabase.Serialize();
    output << "BestGenome\n" << m_BestGenome.Serialize();
    output << "BestGenomeEver\n" << m_BestGenomeEver.Serialize();
    output << "GenomeArchive " << m_GenomeArchive.size() << '\n';
    for (const auto& genome : m_GenomeArchive)
        output << genome.Serialize();
    output << "Species " << m_Species.size() << '\n';
    for (const auto &species : m_Species)
        output << species.Serialize();
    output << "PopulationEnd\n";
    return output.str();
}

Population Population::Deserialize(const std::string &data)
{
    std::istringstream input(data);
    std::string token;
    input >> token;
    if (token != "PopulationStart")
        throw std::runtime_error(
            "Population::Deserialize: missing PopulationStart marker.");

    Population population;
    input >> token;
    if (token != "PopulationFormat")
    {
        // Legacy format: the token is the generation number.
        try
        {
            population.m_Generation =
                static_cast<unsigned int>(std::stoul(token));
        }
        catch (const std::exception&)
        {
            throw std::runtime_error(
                "Population::Deserialize: malformed population header.");
        }
        input >> population.m_NumEvaluations >> population.m_NextGenomeID
              >> population.m_NextSpeciesID >> population.m_BestFitnessEver;
        std::size_t species_count = 0;
        input >> species_count;
        population.m_Species.clear();
        for (std::size_t i = 0; i < species_count; ++i)
        {
            population.m_Species.push_back(Species::Deserialize(
                ReadDelimitedBlock(input, "SpeciesStart", "SpeciesEnd")));
        }
    }
    else
    {
        int version = 0;
        input >> version;
        if (version != 2)
            throw std::runtime_error(
                "Population::Deserialize: unsupported format.");
        input >> token;
        if (token != "PopulationState")
            throw std::runtime_error(
                "Population::Deserialize: missing PopulationState marker.");

        int search_mode = 0;
        input >> population.m_Generation >> population.m_NumEvaluations
              >> population.m_NextGenomeID >> population.m_NextSpeciesID
              >> population.m_BestFitnessEver >> population.m_ID
              >> population.m_GensSinceBestFitnessLastChanged
              >> population.m_EvalsSinceBestFitnessLastChanged
              >> population.m_GensSinceMPCLastChanged >> search_mode
              >> population.m_CurrentMPC >> population.m_OldMPC
              >> population.m_BaseMPC >> population.m_GensSinceLastArchiving
              >> population.m_QuickAddCounter;
        if (search_mode < COMPLEXIFYING || search_mode > BLENDED)
            throw std::runtime_error(
                "Population::Deserialize: invalid search mode.");
        population.m_SearchMode = static_cast<SearchMode>(search_mode);

        std::string rng_state;
        input >> token >> std::quoted(rng_state);
        if (token != "RNG")
            throw std::runtime_error(
                "Population::Deserialize: missing RNG marker.");
        population.m_RNG.Deserialize(rng_state);

        input >> token;
        if (token != "Parameters")
            throw std::runtime_error(
                "Population::Deserialize: missing Parameters marker.");
        population.m_Parameters = Parameters::Deserialize(
            ReadDelimitedBlock(
                input, "NEAT_ParametersStart", "NEAT_ParametersEnd"));

        input >> token;
        if (token != "InnovationDatabase")
            throw std::runtime_error(
                "Population::Deserialize: missing innovation marker.");
        population.m_InnovationDatabase = InnovationDatabase::Deserialize(
            ReadDelimitedBlock(
                input, "InnovationDatabaseStart", "InnovationDatabaseEnd"));

        input >> token;
        if (token != "BestGenome")
            throw std::runtime_error(
                "Population::Deserialize: missing BestGenome marker.");
        population.m_BestGenome = Genome(input);

        input >> token;
        if (token != "BestGenomeEver")
            throw std::runtime_error(
                "Population::Deserialize: missing BestGenomeEver marker.");
        population.m_BestGenomeEver = Genome(input);

        std::size_t archive_count = 0;
        input >> token >> archive_count;
        if (token != "GenomeArchive")
            throw std::runtime_error(
                "Population::Deserialize: missing GenomeArchive marker.");
        population.m_GenomeArchive.clear();
        population.m_GenomeArchive.reserve(archive_count);
        for (std::size_t i = 0; i < archive_count; ++i)
            population.m_GenomeArchive.emplace_back(input);

        std::size_t species_count = 0;
        input >> token >> species_count;
        if (token != "Species")
            throw std::runtime_error(
                "Population::Deserialize: missing Species marker.");
        population.m_Species.clear();
        population.m_Species.reserve(species_count);
        for (std::size_t i = 0; i < species_count; ++i)
        {
            population.m_Species.push_back(Species::Deserialize(
                ReadDelimitedBlock(input, "SpeciesStart", "SpeciesEnd")));
        }
    }

    input >> token;
    if (token != "PopulationEnd")
        throw std::runtime_error(
            "Population::Deserialize: missing PopulationEnd marker.");

    population.m_Genomes.clear();
    for (const auto& species : population.m_Species)
    {
        population.m_Genomes.insert(
            population.m_Genomes.end(),
            species.m_Individuals.begin(),
            species.m_Individuals.end());
    }
    population.m_TempSpecies.clear();
    population.m_BehaviorArchive = nullptr;
    population.m_OwnedBehaviorData.clear();
    population.m_OwnedBehaviorArchive.clear();
    if (!population.m_Species.empty() &&
        population.m_BestGenome.NumNeurons() == 0)
    {
        population.m_BestGenome = population.GetBestGenome();
    }
    std::string validation_error;
    if (!population.Validate(&validation_error))
    {
        throw std::runtime_error(
            "Population::Deserialize: " + validation_error);
    }
    return population;
}



} // namespace NEAT

