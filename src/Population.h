#ifndef _POPULATION_H
#define _POPULATION_H

#include <cmath>
#include <vector>
#include <float.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "Innovation.h"
#include "Genome.h"
#include "PhenotypeBehavior.h"
#include "Genes.h"
#include "Species.h"
#include "Parameters.h"
#include "Random.h"

namespace NEAT
{

//////////////////////////////////////////////
// The Population class
//////////////////////////////////////////////

enum SearchMode
{
    COMPLEXIFYING,
    SIMPLIFYING,
    BLENDED
};

class Species;

class Population
{
    /////////////////////
    // Members
    /////////////////////

private:

    // The innovation database
    InnovationDatabase m_InnovationDatabase;

    // next genome ID
    unsigned int m_NextGenomeID = 0;

    // next species ID
    unsigned int m_NextSpeciesID = 0;

    ////////////////////////////
    // Phased searching members

    // The current mode of search
    SearchMode m_SearchMode = BLENDED;

    // The current Mean Population Complexity
    double m_CurrentMPC = 0.0;

    // The MPC from the previous generation (for comparison)
    double m_OldMPC = 0.0;

    // The base MPC (for switching between complexifying/simplifying phase)
    double m_BaseMPC = 0.0;

    // Separates the population into species based on compatibility distance
    void Speciate();

    // Adjusts each species's fitness
    void AdjustFitness();

    // Calculates how many offspring each genome should have
    void CountOffspring();

    // Empties all species
    void ResetSpecies();

    // Updates the species
    void UpdateSpecies();

    // Calculates the current mean population complexity
    void CalculateMPC();

    // best fitness ever achieved
    double m_BestFitnessEver = std::numeric_limits<double>::lowest();

    // Keep a local copy of the best ever genome found in the run
    Genome m_BestGenome;
    Genome m_BestGenomeEver;

    // Number of generations since the best fitness changed
    unsigned int m_GensSinceBestFitnessLastChanged = 0;

    // Number of evaluations since the best fitness changed
    unsigned int m_EvalsSinceBestFitnessLastChanged = 0;

    // How many generations passed until the last change of MPC
    unsigned int m_GensSinceMPCLastChanged = 0;

    // The initial list of genomes
    std::vector<Genome> m_Genomes;

public:

    // The archive
    std::vector<Genome> m_GenomeArchive;

    // Random number generator
    RNG m_RNG;

    // Evolution parameters
    Parameters m_Parameters;

    // Current generation
    unsigned int m_Generation = 0;

    // The list of species
    std::vector<Species> m_Species;
    
    int m_ID = 0;


    ////////////////////////////
    // Constructors
    ////////////////////////////

    // Initializes a population from a seed genome G. Then it initializes all weights
    // To small numbers between -R and R.
    // The population size is determined by GlobalParameters.PopulationSize
    Population(const Genome& a_G, const Parameters& a_Parameters,
    		   bool a_RandomizeWeights, double a_RandomRange, int a_RNG_seed);


    // Loads a population from a file.
    Population(const std::string a_FileName);

    Population() = default;

    ////////////////////////////
    // Destructor
    ////////////////////////////

    ////////////////////////////
    // Methods
    ////////////////////////////

    // Access
    SearchMode GetSearchMode() const { return m_SearchMode; }
    double GetCurrentMPC() const { return m_CurrentMPC; }
    double GetBaseMPC() const { return m_BaseMPC; }

    unsigned int NumGenomes() const
    {
    	unsigned int num=0;
    	for(unsigned int i=0; i<m_Species.size(); i++)
    	{
            num += static_cast<unsigned int>(
                m_Species[i].m_Individuals.size());
    	}
    	return num;
    }

    unsigned int GetGeneration() const { return m_Generation; }
    double GetBestFitnessEver() const { return m_BestFitnessEver; }
    Genome GetBestGenome() const
    {
        if (m_Species.empty())
            throw std::runtime_error(
                "Population::GetBestGenome: population is empty.");

        double best = std::numeric_limits<double>::lowest();
        int idx_species = 0;
        int idx_genome = 0;
        bool found = false;
        for(unsigned int i=0; i<m_Species.size(); i++)
        {
            for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
            {
                const Genome& genome =
                    m_Species[i].m_Individuals[j];
                if (!genome.IsEvaluated() ||
                    !std::isfinite(genome.GetFitness()))
                {
                    continue;
                }
                if (!found || genome.GetFitness() > best)
                {
                    best = genome.GetFitness();
                    idx_species = i;
                    idx_genome = j;
                    found = true;
                }
            }
        }

        if (!found)
        {
            for (const auto& species : m_Species)
            {
                if (!species.m_Individuals.empty())
                    return species.m_Individuals.front();
            }
            throw std::runtime_error(
                "Population::GetBestGenome: population has no genomes.");
        }
        return m_Species[idx_species].m_Individuals[idx_genome];
    }

    unsigned int GetStagnation() const { return m_GensSinceBestFitnessLastChanged; }
    unsigned int GetMPCStagnation() const { return m_GensSinceMPCLastChanged; }

    unsigned int GetNextGenomeID() const { return m_NextGenomeID; }
    unsigned int GetNextSpeciesID() const { return m_NextSpeciesID; }
    void IncrementNextGenomeID()
    {
        if (m_NextGenomeID ==
            static_cast<unsigned int>(
                std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("Genome ID space is exhausted");
        }
        ++m_NextGenomeID;
    }
    void IncrementNextSpeciesID()
    {
        if (m_NextSpeciesID ==
            static_cast<unsigned int>(
                std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("Species ID space is exhausted");
        }
        ++m_NextSpeciesID;
    }
    
    
    // Make sure no same genome IDs exist in the population
    void SameGenomeIDCheck();
    Genome& AccessGenomeByIndex(int const a_idx);
    Genome& AccessGenomeByID(int const a_id);

    InnovationDatabase& AccessInnovationDatabase() { return m_InnovationDatabase; }

    // Checks population-wide structural and ID invariants.
    bool Validate(std::string* error = nullptr) const;

    // Sorts each species's genomes by fitness
    void Sort();

    // Performs one generation and reproduces the genomes
    void Epoch();

    // Saves the whole population to a file
    void Save(const char* a_FileName);

    // Saves a complete resumable checkpoint. Save() retains the historical
    // parameters/innovations/genomes format for existing consumers.
    void SaveState(const char* a_FileName) const;

    //////////////////////
    std::vector<Species> m_TempSpecies; // useful in reproduction


    //////////////////////
    // Real-Time methods

    // Choose the parent species that will reproduce
    // This is a real-time version of fitness sharing
    // Returns the species index
    unsigned int ChooseParentSpecies();

    // Removes worst member of the whole population that has been around for a minimum amount of time
    // returns the genome that was just deleted (may be useful)
    Genome RemoveWorstIndividual();
    
    void ClearEmptySpecies();
    
    // The main reaitime tick. Analog to Epoch(). Replaces the worst evaluated individual with a new one.
    // Returns a pointer to the new baby.
    // and copies the genome that was deleted to a_geleted_genome
    Genome* Tick(Genome& a_deleted_genome);

    // Takes an individual and puts it in its apropriate species
    // Useful in realtime when the compatibility treshold changes
    void ReassignSpecies(int a_genome_idx);

    unsigned int m_NumEvaluations = 0;



    ///////////////////////////////
    // Novelty search

    // A pointer to the archive of PhenotypeBehaviors
    // Necessary to contain derived custom classes.
    std::vector< PhenotypeBehavior >* m_BehaviorArchive = nullptr;

    // Call this function to allocate memory for your custom
    // behaviors. This initializes everything.
    void InitPhenotypeBehaviorData(std::vector< PhenotypeBehavior >* a_population, 
                                   std::vector< PhenotypeBehavior >* a_archive);

    // Ownership-safe overload for language bindings and modern C++ callers.
    void InitPhenotypeBehaviorData(
        const std::vector<std::shared_ptr<PhenotypeBehavior>>& a_population);
    const std::vector<PhenotypeBehavior>& GetBehaviorArchive() const;

    // This is the main method performing novelty search.
    // Performs one reproduction and assigns novelty scores
    // based on the current population and the archive.
    // If a successful behavior was encountered, returns true
    // and the genome a_SuccessfulGenome is overwritten with the
    // genome generating the successful behavior
    bool NoveltySearchTick(Genome& a_SuccessfulGenome);

    double ComputeSparseness(Genome& genome);

    // counters for archive stagnation
    unsigned int m_GensSinceLastArchiving = 0;
    unsigned int m_QuickAddCounter = 0;

    std::string Serialize() const;
    static Population Deserialize(const std::string &data);

private:
    std::vector<std::shared_ptr<PhenotypeBehavior>> m_OwnedBehaviorData;
    std::vector<PhenotypeBehavior> m_OwnedBehaviorArchive;
        
    };

} // namespace NEAT

#endif

