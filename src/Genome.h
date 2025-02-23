#ifndef _GENOME_H
#define _GENOME_H

#include <vector>
#include <queue>

#include "NeuralNetwork.h"
#include "Substrate.h"
#include "Innovation.h"
#include "Genes.h"
#include "Assert.h"
#include "PhenotypeBehavior.h"
#include "Random.h"

namespace NEAT
{

    //////////////////////////////////////////////
    // The Genome class
    //////////////////////////////////////////////

    // forward
    class Innovation;

    class InnovationDatabase;

    class PhenotypeBehavior;

    extern ActivationFunction GetRandomActivation(Parameters &a_Parameters, RNG &a_RNG);

	enum GenomeSeedType
	{
		PERCEPTRON = 0,
		LAYERED = 1
	};

	class GenomeInitStruct
	{
	public:
		int NumInputs;
		int NumHidden; // ignored for seed_type == 0, specifies number of hidden units if seed_type == 1
		int NumOutputs;
		bool FS_NEAT;
		ActivationFunction OutputActType;
		ActivationFunction HiddenActType;
		GenomeSeedType SeedType;
		int NumLayers;
		int FS_NEAT_links;

		GenomeInitStruct()
		{
			NumInputs = 1;
			NumHidden = 0;
			NumOutputs = 1;
			FS_NEAT = 0;
			FS_NEAT_links = 1;
			HiddenActType = UNSIGNED_SIGMOID;
			OutputActType = UNSIGNED_SIGMOID;
			SeedType = GenomeSeedType::PERCEPTRON;
			NumLayers = 0;
		}
	};


    class Genome
    {
        /////////////////////
        // Members
        /////////////////////
    private:

        // ID of genome
        int m_ID;
        
        // How many inputs/outputs
        int m_NumInputs;
        int m_NumOutputs;

        // The genome's fitness score
        double m_Fitness;

        // The genome's adjusted fitness score
        double m_AdjustedFitness;

        // The depth of the network
        int m_Depth;

        // how many individuals this genome should spawn
        double m_OffspringAmount;

        ////////////////////
        // Private methods

        // Returns true if the specified neuron ID is present in the genome
        bool HasNeuronID(int a_id) const;

        // Returns true if the specified link is present in the genome
        bool HasLink(int a_n1id, int a_n2id) const;

        // Returns true if the specified link is present in the genome
        bool HasLinkByInnovID(int a_id) const;

        // Removes the link with the specified innovation ID
        void RemoveLinkGene(int a_innovid);

        // Remove node
        // Links connected to this node are also removed
        void RemoveNeuronGene(int a_id);

        // Returns the count of links inputting from the specified neuron ID
        int LinksInputtingFrom(int a_id) const;

        // Returns the count of links outputting to the specified neuron ID
        int LinksOutputtingTo(int a_id) const;

        // A recursive function returning the max depth from the specified neuron to the inputs
        unsigned int NeuronDepth(int a_NeuronID, unsigned int a_Depth);

        // Returns true is the specified neuron ID is a dead end or isolated
        bool IsDeadEndNeuron(int a_id) const;

    public:

        // The two lists of genes
        std::vector<NeuronGene> m_NeuronGenes;
        std::vector<LinkGene> m_LinkGenes;

        // To have traits that belong to the genome itself
        Gene m_GenomeGene;

        // tells whether this genome was evaluated already
        // used in steady state evolution
        bool m_Evaluated;

        // the initial genome complexity
        int m_initial_num_neurons;
        int m_initial_num_links;

        // A pointer to a class representing the phenotype's behavior
        // Used in novelty searches
        PhenotypeBehavior *m_PhenotypeBehavior;
        // A Python object behavior
#ifdef USE_BOOST_PYTHON
        py::object m_behavior;
#endif

        ////////////////////////////
        // Constructors
        ////////////////////////////

        Genome();

        // copy constructor
        Genome(const Genome &a_g);

        // assignment operator
        Genome &operator=(const Genome &a_g);

        bool operator==(Genome const &other) const
        {
            return m_ID == other.m_ID;
        }

        // Builds this genome from a file
        Genome(const char *a_filename);

        // Builds this genome from an opened file
        Genome(std::ifstream &a_DataFile);

        // This creates a standart minimal genome - perceptron-like structure
        Genome(const Parameters &a_Parameters,
			   const GenomeInitStruct &init_struct);

        /////////////
        // Other possible constructors for different types of networks go here


        ////////////////////////////
        // Destructor
        ////////////////////////////

        ////////////////////////////
        // Methods
        ////////////////////////////

        ////////////////////
        // Accessor methods

        NeuronGene GetNeuronByID(int a_ID) const;

        NeuronGene GetNeuronByIndex(int a_idx) const;

        LinkGene GetLinkByInnovID(int a_ID) const;

        LinkGene GetLinkByIndex(int a_idx) const;

        // A little helper function to find the index of a neuron, given its ID
        int GetNeuronIndex(int a_id) const;

        // A little helper function to find the index of a link, given its innovation ID
        int GetLinkIndex(int a_innovid) const;

        unsigned int NumNeurons() const
        { return static_cast<unsigned int>(m_NeuronGenes.size()); }

        unsigned int NumLinks() const
        { return static_cast<unsigned int>(m_LinkGenes.size()); }

        unsigned int NumInputs() const
        { return m_NumInputs; }

        unsigned int NumOutputs() const
        { return m_NumOutputs; }

        void SetNeuronXY(unsigned int a_idx, int a_x, int a_y);

        void SetNeuronX(unsigned int a_idx, int a_x);

        void SetNeuronY(unsigned int a_idx, int a_y);

        double GetFitness() const;

        double GetAdjFitness() const;

        void SetFitness(double a_f);

        void SetAdjFitness(double a_af);

        int GetID() const;
        
        void SetID(int a_id);

        unsigned int GetDepth() const;

        void SetDepth(unsigned int a_d);

        // Returns true if there is any dead end in the network
        bool HasDeadEnds() const;

        // Returns true if there is any looping path in the network
        bool HasLoops();

        bool FailsConstraints(const Parameters &a_Parameters)
        {
            bool fails = false;

            if (HasDeadEnds() || (NumLinks() == 0))
            {
                return true; // no reason to continue
            }


            if ((HasLoops() && (a_Parameters.AllowLoops == false)))
            {
                return true;
            }

            // Custom constraints
            if (a_Parameters.CustomConstraints != NULL)
            {
                if (a_Parameters.CustomConstraints(*this))
                {
                    return true;
                }
            }

            // add more constraints here
            return false;
        }

        double GetOffspringAmount() const;

        void SetOffspringAmount(double a_oa);

        // This builds a fastnetwork structure out from the genome
        void BuildPhenotype(NeuralNetwork &net);

        // Projects the phenotype's weights back to the genome
        void DerivePhenotypicChanges(NeuralNetwork &a_Net);

        ////////////
        // Other possible methods for building a phenotype go here
        // Like CPPN/HyperNEAT stuff
        ////////////
        void BuildHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst);

        // Saves this genome to a file
        void Save(const char *a_filename);

        // Saves this genome to an already opened file for writing
        void Save(FILE *a_fstream);

        void PrintTraits(std::map< std::string, Trait>& traits);
        void PrintAllTraits();

        // returns the max neuron ID
        int GetLastNeuronID() const;

        // returns the max innovation Id
        int GetLastInnovationID() const;

        // Sorts the genes of the genome
        // The neurons by IDs and the links by innovation numbers.
        void SortGenes();

        // overload '<' used for sorting. From fittest to poorest.
        friend bool operator<(const Genome &a_lhs, const Genome &a_rhs)
        {
            return (a_lhs.m_Fitness > a_rhs.m_Fitness);
        }

        // Returns true if this genome and a_G are compatible (belong in the same species)
        bool IsCompatibleWith(Genome &a_G, Parameters &a_Parameters);

        // returns the absolute compatibility distance between this genome and a_G
        double CompatibilityDistance(Genome &a_G, Parameters &a_Parameters);

        // Calculates the network depth
        void CalculateDepth();

        ////////////
        // Mutation
        ////////////

        // Adds a new neuron to the genome
        // returns true if succesful
        bool Mutate_AddNeuron(InnovationDatabase &a_Innovs, Parameters &a_Parameters, RNG &a_RNG);

        // Adds a new link to the genome
        // returns true if succesful
        bool Mutate_AddLink(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG);

        // Remove a random link from the genome
        // A cleanup procedure is invoked so any dead-ends or stranded neurons are also deleted
        // returns true if succesful
        bool Mutate_RemoveLink(RNG &a_RNG);

        // Removes a hidden neuron having only one input and only one output with
        // a direct link between them.
        bool Mutate_RemoveSimpleNeuron(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG);

        // Perturbs the weights
        bool Mutate_LinkWeights(const Parameters &a_Parameters, RNG &a_RNG);

        // Set all link weights to random values between [-R .. R]
        void Randomize_LinkWeights(const Parameters &a_Parameters, RNG &a_RNG);

        // Set all traits to random values
        void Randomize_Traits(const Parameters& a_Parameters, RNG &a_RNG);

        // Perturbs the A parameters of the neuron activation functions
        bool Mutate_NeuronActivations_A(const Parameters &a_Parameters, RNG &a_RNG);

        // Perturbs the B parameters of the neuron activation functions
        bool Mutate_NeuronActivations_B(const Parameters &a_Parameters, RNG &a_RNG);

        // Changes the activation function type for a random neuron
        bool Mutate_NeuronActivation_Type(const Parameters &a_Parameters, RNG &a_RNG);

        // Perturbs the neuron time constants
        bool Mutate_NeuronTimeConstants(const Parameters &a_Parameters, RNG &a_RNG);

        // Perturbs the neuron biases
        bool Mutate_NeuronBiases(const Parameters &a_Parameters, RNG &a_RNG);

        // Perturbs the neuron traits
        bool Mutate_NeuronTraits(const Parameters &a_Parameters, RNG &a_RNG);

        // Perturbs the link traits
        bool Mutate_LinkTraits(const Parameters &a_Parameters, RNG &a_RNG);

        // Perturbs the genome traits
        bool Mutate_GenomeTraits(const Parameters &a_Parameters, RNG &a_RNG);

        ///////////
        // Mating
        ///////////


        // Mate this genome with dad and return the baby
        // If this is multipoint mating, genes are inherited randomly
        // If the a_averagemating bool is true, then the genes are averaged
        // Disjoint and excess genes are inherited from the fittest parent
        // If fitness is equal, the smaller genome is assumed to be the better one
        Genome Mate(Genome &a_dad, bool a_averagemating, bool a_interspecies, RNG &a_RNG, Parameters &a_Parameters);


        //////////
        // Utility
        //////////

        // Search the genome for isolated structure and clean it up
        // Returns true is something was removed
        bool Cleanup();

        ////////////////////
        // new stuff
        bool IsEvaluated() const;

        void SetEvaluated();

        void ResetEvaluated();

    };

#define DBG(x) { std::cerr << x << std::endl; }


} // namespace NEAT

#endif
