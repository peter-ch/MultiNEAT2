// Include pybind11 headers and STL bindings
#define assert(x) (true)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
namespace py = pybind11;

// Include ALL the MultiNEAT headers
#include "Assert.h"
#include "Genes.h"             
#include "Genome.h"            
#include "Innovation.h"        
#include "NeuralNetwork.h"     
#include "Parameters.h"        
#include "PhenotypeBehavior.h" 
#include "Population.h"        
#include "Species.h"           
#include "Random.h"            
#include "Substrate.h"         
#include "Traits.h"            
#include "Utils.h"             


// Create a pybind11 module called "pymultineat"
PYBIND11_MODULE(pymultineat, m) {
    m.doc() = "MultiNEAT - All classes exposed via pybind11";

    // Expose enums

    py::enum_<NEAT::NeuronType>(m, "NeuronType")
        .value("NONE", NEAT::NONE)
        .value("INPUT", NEAT::INPUT)
        .value("BIAS", NEAT::BIAS)
        .value("HIDDEN", NEAT::HIDDEN)
        .value("OUTPUT", NEAT::OUTPUT)
        .export_values();

    py::enum_<NEAT::ActivationFunction>(m, "ActivationFunction")
        .value("SIGNED_SIGMOID", NEAT::SIGNED_SIGMOID)
        .value("UNSIGNED_SIGMOID", NEAT::UNSIGNED_SIGMOID)
        .value("TANH", NEAT::TANH)
        .value("TANH_CUBIC", NEAT::TANH_CUBIC)
        .value("SIGNED_STEP", NEAT::SIGNED_STEP)
        .value("UNSIGNED_STEP", NEAT::UNSIGNED_STEP)
        .value("SIGNED_GAUSS", NEAT::SIGNED_GAUSS)
        .value("UNSIGNED_GAUSS", NEAT::UNSIGNED_GAUSS)
        .value("ABS", NEAT::ABS)
        .value("SIGNED_SINE", NEAT::SIGNED_SINE)
        .value("UNSIGNED_SINE", NEAT::UNSIGNED_SINE)
        .value("LINEAR", NEAT::LINEAR)
        .value("RELU", NEAT::RELU)
        .value("SOFTPLUS", NEAT::SOFTPLUS)
        .export_values();

    py::enum_<NEAT::GenomeSeedType>(m, "GenomeSeedType")
        .value("PERCEPTRON", NEAT::PERCEPTRON)
        .value("LAYERED", NEAT::LAYERED)
        .export_values();

    py::enum_<NEAT::InnovationType>(m, "InnovationType")
        .value("NEW_NEURON", NEAT::NEW_NEURON)
        .value("NEW_LINK", NEAT::NEW_LINK)
        .export_values();


    // ========================
    // Bindings for traits-related classes
    // ========================

    py::class_<NEAT::intsetelement>(m, "intsetelement")
        .def(py::init<>())
        .def_readwrite("value", &NEAT::intsetelement::value);

    py::class_<NEAT::floatsetelement>(m, "floatsetelement")
        .def(py::init<>())
        .def_readwrite("value", &NEAT::floatsetelement::value);

    py::class_<NEAT::IntTraitParameters>(m, "IntTraitParameters")
        .def(py::init<>())
        .def_readwrite("min", &NEAT::IntTraitParameters::min)
        .def_readwrite("max", &NEAT::IntTraitParameters::max)
        .def_readwrite("mut_power", &NEAT::IntTraitParameters::mut_power)
        .def_readwrite("mut_replace_prob", &NEAT::IntTraitParameters::mut_replace_prob);

    py::class_<NEAT::FloatTraitParameters>(m, "FloatTraitParameters")
        .def(py::init<>())
        .def_readwrite("min", &NEAT::FloatTraitParameters::min)
        .def_readwrite("max", &NEAT::FloatTraitParameters::max)
        .def_readwrite("mut_power", &NEAT::FloatTraitParameters::mut_power)
        .def_readwrite("mut_replace_prob", &NEAT::FloatTraitParameters::mut_replace_prob);

    py::class_<NEAT::StringTraitParameters>(m, "StringTraitParameters")
        .def(py::init<>())
        .def_readwrite("set", &NEAT::StringTraitParameters::set)
        .def_readwrite("probs", &NEAT::StringTraitParameters::probs);

    py::class_<NEAT::IntSetTraitParameters>(m, "IntSetTraitParameters")
        .def(py::init<>())
        .def_readwrite("set", &NEAT::IntSetTraitParameters::set)
        .def_readwrite("probs", &NEAT::IntSetTraitParameters::probs);

    py::class_<NEAT::FloatSetTraitParameters>(m, "FloatSetTraitParameters")
        .def(py::init<>())
        .def_readwrite("set", &NEAT::FloatSetTraitParameters::set)
        .def_readwrite("probs", &NEAT::FloatSetTraitParameters::probs);

    py::class_<NEAT::TraitParameters>(m, "TraitParameters")
        .def(py::init<>())
        .def_readwrite("m_ImportanceCoeff", &NEAT::TraitParameters::m_ImportanceCoeff)
        .def_readwrite("m_MutationProb", &NEAT::TraitParameters::m_MutationProb)
        .def_readwrite("type", &NEAT::TraitParameters::type)
        // Note: m_Details is a variant – binding it directly may require additional work.
        .def_readwrite("dep_key", &NEAT::TraitParameters::dep_key)
        .def_readwrite("dep_values", &NEAT::TraitParameters::dep_values);

    py::class_<NEAT::Trait>(m, "Trait")
        .def(py::init<>())
        .def_readwrite("value", &NEAT::Trait::value)
        .def_readwrite("dep_key", &NEAT::Trait::dep_key)
        .def_readwrite("dep_values", &NEAT::Trait::dep_values);


    // ========================
    // Bindings for Genes and derived classes
    // ========================

    py::class_<NEAT::Gene>(m, "Gene")
        .def(py::init<>())
        .def("InitTraits", &NEAT::Gene::InitTraits)
        .def("MateTraits", &NEAT::Gene::MateTraits)
        .def("MutateTraits", &NEAT::Gene::MutateTraits)
        .def("GetTraitDistances", &NEAT::Gene::GetTraitDistances)
        .def_readwrite("m_Traits", &NEAT::Gene::m_Traits);

    py::class_<NEAT::LinkGene, NEAT::Gene>(m, "LinkGene")
        .def(py::init<>())
        .def(py::init<int, int, int, double, bool>(),
             py::arg("a_InID"), py::arg("a_OutID"), py::arg("a_InnovID"), py::arg("a_Wgt"), py::arg("a_Recurrent")=false)
        .def("GetWeight", &NEAT::LinkGene::GetWeight)
        .def("SetWeight", &NEAT::LinkGene::SetWeight)
        .def("FromNeuronID", &NEAT::LinkGene::FromNeuronID)
        .def("ToNeuronID", &NEAT::LinkGene::ToNeuronID)
        .def("InnovationID", &NEAT::LinkGene::InnovationID)
        .def("IsRecurrent", &NEAT::LinkGene::IsRecurrent)
        .def("IsLoopedRecurrent", &NEAT::LinkGene::IsLoopedRecurrent)
        .def_readwrite("m_FromNeuronID", &NEAT::LinkGene::m_FromNeuronID)
        .def_readwrite("m_ToNeuronID", &NEAT::LinkGene::m_ToNeuronID)
        .def_readwrite("m_InnovationID", &NEAT::LinkGene::m_InnovationID)
        .def_readwrite("m_Weight", &NEAT::LinkGene::m_Weight)
        .def_readwrite("m_IsRecurrent", &NEAT::LinkGene::m_IsRecurrent);

    py::class_<NEAT::NeuronGene, NEAT::Gene>(m, "NeuronGene")
        .def(py::init<>())
        .def(py::init<NEAT::NeuronType, int, double>(),
             py::arg("a_type"), py::arg("a_id"), py::arg("a_splity"))
        .def("ID", &NEAT::NeuronGene::ID)
        .def("Type", &NEAT::NeuronGene::Type)
        .def("SplitY", &NEAT::NeuronGene::SplitY)
        .def("Init", &NEAT::NeuronGene::Init)
        .def_readwrite("m_ID", &NEAT::NeuronGene::m_ID)
        .def_readwrite("m_Type", &NEAT::NeuronGene::m_Type)
        .def_readwrite("x", &NEAT::NeuronGene::x)
        .def_readwrite("y", &NEAT::NeuronGene::y)
        .def_readwrite("m_SplitY", &NEAT::NeuronGene::m_SplitY)
        .def_readwrite("m_A", &NEAT::NeuronGene::m_A)
        .def_readwrite("m_B", &NEAT::NeuronGene::m_B)
        .def_readwrite("m_TimeConstant", &NEAT::NeuronGene::m_TimeConstant)
        .def_readwrite("m_Bias", &NEAT::NeuronGene::m_Bias)
        .def_readwrite("m_ActFunction", &NEAT::NeuronGene::m_ActFunction);


    // ========================
    // Bindings for Genome and GenomeInitStruct
    // ========================

    py::class_<NEAT::Genome>(m, "Genome")
        .def(py::init<>())
        .def(py::init<const NEAT::Parameters&, const NEAT::GenomeInitStruct&>())
        .def(py::init<const char*>())
        .def(py::init<std::ifstream&>())
        .def("GetNeuronByID", &NEAT::Genome::GetNeuronByID)
        .def("GetNeuronByIndex", &NEAT::Genome::GetNeuronByIndex)
        .def("GetLinkByInnovID", &NEAT::Genome::GetLinkByInnovID)
        .def("GetLinkByIndex", &NEAT::Genome::GetLinkByIndex)
        .def("GetNeuronIndex", &NEAT::Genome::GetNeuronIndex)
        .def("GetLinkIndex", &NEAT::Genome::GetLinkIndex)
        .def("SetNeuronXY", &NEAT::Genome::SetNeuronXY)
        .def("SetNeuronX", &NEAT::Genome::SetNeuronX)
        .def("SetNeuronY", &NEAT::Genome::SetNeuronY)
        .def("GetFitness", &NEAT::Genome::GetFitness)
        .def("GetAdjFitness", &NEAT::Genome::GetAdjFitness)
        .def("SetFitness", &NEAT::Genome::SetFitness)
        .def("SetAdjFitness", &NEAT::Genome::SetAdjFitness)
        .def("SetEvaluated", &NEAT::Genome::SetEvaluated)
        .def("GetID", &NEAT::Genome::GetID)
        .def("GetDepth", &NEAT::Genome::GetDepth)
        .def("HasDeadEnds", &NEAT::Genome::HasDeadEnds)
        .def("GetLastNeuronID", &NEAT::Genome::GetLastNeuronID)
        .def("GetLastInnovationID", &NEAT::Genome::GetLastInnovationID)
        .def("BuildPhenotype", &NEAT::Genome::BuildPhenotype)
        .def("DerivePhenotypicChanges", &NEAT::Genome::DerivePhenotypicChanges)
        .def("CompatibilityDistance", &NEAT::Genome::CompatibilityDistance)
        .def("IsCompatibleWith", &NEAT::Genome::IsCompatibleWith)
        .def("Mutate_LinkWeights", &NEAT::Genome::Mutate_LinkWeights)
        .def("Randomize_LinkWeights", &NEAT::Genome::Randomize_LinkWeights)
        .def("Randomize_Traits", &NEAT::Genome::Randomize_Traits)
        .def("Mutate_NeuronActivations_A", &NEAT::Genome::Mutate_NeuronActivations_A)
        .def("Mutate_NeuronActivations_B", &NEAT::Genome::Mutate_NeuronActivations_B)
        .def("Mutate_NeuronActivation_Type", &NEAT::Genome::Mutate_NeuronActivation_Type)
        .def("Mutate_NeuronTimeConstants", &NEAT::Genome::Mutate_NeuronTimeConstants)
        .def("Mutate_NeuronBiases", &NEAT::Genome::Mutate_NeuronBiases)
        .def("Mutate_NeuronTraits", &NEAT::Genome::Mutate_NeuronTraits)
        .def("Mutate_LinkTraits", &NEAT::Genome::Mutate_LinkTraits)
        .def("Mutate_GenomeTraits", &NEAT::Genome::Mutate_GenomeTraits)
        .def("Mutate_AddNeuron", &NEAT::Genome::Mutate_AddNeuron)
        .def("Mutate_AddLink", &NEAT::Genome::Mutate_AddLink)
        .def("Mutate_RemoveLink", &NEAT::Genome::Mutate_RemoveLink)
        .def("Mutate_RemoveSimpleNeuron", &NEAT::Genome::Mutate_RemoveSimpleNeuron)
        .def("Cleanup", &NEAT::Genome::Cleanup)
        .def("Mate", &NEAT::Genome::Mate)
        .def("SortGenes", &NEAT::Genome::SortGenes)
        .def_readwrite("m_NeuronGenes", &NEAT::Genome::m_NeuronGenes)
        .def_readwrite("m_LinkGenes", &NEAT::Genome::m_LinkGenes)
        .def_readwrite("m_GenomeGene", &NEAT::Genome::m_GenomeGene)
        .def_readwrite("m_Evaluated", &NEAT::Genome::m_Evaluated)
        .def_readwrite("m_PhenotypeBehavior", &NEAT::Genome::m_PhenotypeBehavior)
        .def_readwrite("m_initial_num_neurons", &NEAT::Genome::m_initial_num_neurons)
        .def_readwrite("m_initial_num_links", &NEAT::Genome::m_initial_num_links)
        .def(py::pickle(
            // __getstate__: returns a string with the serialized genome.
            [](const NEAT::Genome &g) -> std::string {
                return g.Serialize();  
            },
            // __setstate__: creates a genome from the serialized string.
            [](const std::string &s) {
                return NEAT::Genome::Deserialize(s);
            }
        ));
        

    py::class_<NEAT::GenomeInitStruct>(m, "GenomeInitStruct")
        .def(py::init<>())
        .def_readwrite("NumInputs", &NEAT::GenomeInitStruct::NumInputs)
        .def_readwrite("NumHidden", &NEAT::GenomeInitStruct::NumHidden)
        .def_readwrite("NumOutputs", &NEAT::GenomeInitStruct::NumOutputs)
        .def_readwrite("FS_NEAT", &NEAT::GenomeInitStruct::FS_NEAT)
        .def_readwrite("OutputActType", &NEAT::GenomeInitStruct::OutputActType)
        .def_readwrite("HiddenActType", &NEAT::GenomeInitStruct::HiddenActType)
        .def_readwrite("SeedType", &NEAT::GenomeInitStruct::SeedType)
        .def_readwrite("NumLayers", &NEAT::GenomeInitStruct::NumLayers)
        .def_readwrite("FS_NEAT_links", &NEAT::GenomeInitStruct::FS_NEAT_links);


    // ========================
    // Bindings for Innovation
    // ========================

    py::class_<NEAT::Innovation>(m, "Innovation")
        .def(py::init<int, NEAT::InnovationType, int, int, NEAT::NeuronType, int>(),
             py::arg("a_ID"), py::arg("a_InnovType"), py::arg("a_From"), py::arg("a_To"),
             py::arg("a_NType"), py::arg("a_NID"))
        .def("ID", &NEAT::Innovation::ID)
        .def("InnovType", &NEAT::Innovation::InnovType)
        .def("FromNeuronID", &NEAT::Innovation::FromNeuronID)
        .def("ToNeuronID", &NEAT::Innovation::ToNeuronID)
        .def("NeuronID", &NEAT::Innovation::NeuronID)
        .def("GetNeuronType", &NEAT::Innovation::GetNeuronType);

    py::class_<NEAT::InnovationDatabase>(m, "InnovationDatabase")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("a_LastInnovationNum"), py::arg("a_LastNeuronID"))
        .def("Init", (void (NEAT::InnovationDatabase::*)(int, int)) &NEAT::InnovationDatabase::Init)
        .def("InitFromGenome", (void (NEAT::InnovationDatabase::*)(const NEAT::Genome&)) &NEAT::InnovationDatabase::Init)
        .def("InitFromFile", (void (NEAT::InnovationDatabase::*)(std::ifstream&)) &NEAT::InnovationDatabase::Init)
        .def("CheckInnovation", &NEAT::InnovationDatabase::CheckInnovation)
        .def("CheckLastInnovation", &NEAT::InnovationDatabase::CheckLastInnovation)
        .def("CheckAllInnovations", &NEAT::InnovationDatabase::CheckAllInnovations)
        .def("FindNeuronID", &NEAT::InnovationDatabase::FindNeuronID)
        .def("FindLastNeuronID", &NEAT::InnovationDatabase::FindLastNeuronID)
        .def("AddLinkInnovation", &NEAT::InnovationDatabase::AddLinkInnovation)
        .def("AddNeuronInnovation", &NEAT::InnovationDatabase::AddNeuronInnovation)
        .def("Flush", &NEAT::InnovationDatabase::Flush);

    // ========================
    // Bindings for NeuralNetwork (and its inner classes)
    // ========================

    py::class_<NEAT::Connection>(m, "Connection")
        .def(py::init<>())
        .def_readwrite("m_source_neuron_idx", &NEAT::Connection::m_source_neuron_idx)
        .def_readwrite("m_target_neuron_idx", &NEAT::Connection::m_target_neuron_idx)
        .def_readwrite("m_weight", &NEAT::Connection::m_weight)
        .def_readwrite("m_signal", &NEAT::Connection::m_signal)
        .def_readwrite("m_recur_flag", &NEAT::Connection::m_recur_flag)
        .def_readwrite("m_hebb_rate", &NEAT::Connection::m_hebb_rate)
        .def_readwrite("m_hebb_pre_rate", &NEAT::Connection::m_hebb_pre_rate);

    py::class_<NEAT::Neuron>(m, "Neuron")
        .def(py::init<>())
        .def_readwrite("m_activesum", &NEAT::Neuron::m_activesum)
        .def_readwrite("m_activation", &NEAT::Neuron::m_activation)
        .def_readwrite("m_a", &NEAT::Neuron::m_a)
        .def_readwrite("m_b", &NEAT::Neuron::m_b)
        .def_readwrite("m_timeconst", &NEAT::Neuron::m_timeconst)
        .def_readwrite("m_bias", &NEAT::Neuron::m_bias)
        .def_readwrite("m_membrane_potential", &NEAT::Neuron::m_membrane_potential)
        .def_readwrite("m_activation_function_type", &NEAT::Neuron::m_activation_function_type)
        .def_readwrite("m_x", &NEAT::Neuron::m_x)
        .def_readwrite("m_y", &NEAT::Neuron::m_y)
        .def_readwrite("m_z", &NEAT::Neuron::m_z)
        .def_readwrite("m_sx", &NEAT::Neuron::m_sx)
        .def_readwrite("m_sy", &NEAT::Neuron::m_sy)
        .def_readwrite("m_sz", &NEAT::Neuron::m_sz)
        .def_readwrite("m_substrate_coords", &NEAT::Neuron::m_substrate_coords)
        .def_readwrite("m_split_y", &NEAT::Neuron::m_split_y)
        .def_readwrite("m_type", &NEAT::Neuron::m_type)
        .def_readwrite("m_sensitivity_matrix", &NEAT::Neuron::m_sensitivity_matrix);

        py::class_<NEAT::NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<bool>(), py::arg("a_Minimal")=false)
        .def(py::init<>())
        .def("ActivateFast", &NEAT::NeuralNetwork::ActivateFast)
        .def("Activate", &NEAT::NeuralNetwork::Activate)
        .def("ActivateUseInternalBias", &NEAT::NeuralNetwork::ActivateUseInternalBias)
        .def("ActivateLeaky", &NEAT::NeuralNetwork::ActivateLeaky)
        .def("Flush", &NEAT::NeuralNetwork::Flush)
        .def("FlushCube", &NEAT::NeuralNetwork::FlushCube)
        .def("Input", &NEAT::NeuralNetwork::Input)
        .def("Output", &NEAT::NeuralNetwork::Output)
        .def("Save", (void (NEAT::NeuralNetwork::*)(const char*)) &NEAT::NeuralNetwork::Save)
        .def("Load", (bool (NEAT::NeuralNetwork::*)(const char*)) &NEAT::NeuralNetwork::Load)
        .def_readwrite("m_num_inputs", &NEAT::NeuralNetwork::m_num_inputs)
        .def_readwrite("m_num_outputs", &NEAT::NeuralNetwork::m_num_outputs)
        .def_readwrite("m_connections", &NEAT::NeuralNetwork::m_connections)
        .def_readwrite("m_neurons", &NEAT::NeuralNetwork::m_neurons);

        py::class_<NEAT::Parameters>(m, "Parameters")
            .def(py::init<>())
            // Methods
            .def("Load", (int (NEAT::Parameters::*)(const char*)) &NEAT::Parameters::Load, py::arg("filename"))
            .def("LoadFromStream", (int (NEAT::Parameters::*)(std::ifstream&)) &NEAT::Parameters::Load, py::arg("dataFile"))
            .def("Save", (void (NEAT::Parameters::*)(const char*)) &NEAT::Parameters::Save, py::arg("filename"))
            .def("SaveToStream", (void (NEAT::Parameters::*)(FILE*)) &NEAT::Parameters::Save, py::arg("fstream"))
            .def("Reset", &NEAT::Parameters::Reset)
            // Public members – Basic parameters
            .def_readwrite("PopulationSize", &NEAT::Parameters::PopulationSize)
            .def_readwrite("Speciation", &NEAT::Parameters::Speciation)
            .def_readwrite("DynamicCompatibility", &NEAT::Parameters::DynamicCompatibility)
            .def_readwrite("MinSpecies", &NEAT::Parameters::MinSpecies)
            .def_readwrite("MaxSpecies", &NEAT::Parameters::MaxSpecies)
            .def_readwrite("InnovationsForever", &NEAT::Parameters::InnovationsForever)
            .def_readwrite("AllowClones", &NEAT::Parameters::AllowClones)
            .def_readwrite("ArchiveEnforcement", &NEAT::Parameters::ArchiveEnforcement)
            .def_readwrite("NormalizeGenomeSize", &NEAT::Parameters::NormalizeGenomeSize)
            .def_readwrite("CustomConstraints", &NEAT::Parameters::CustomConstraints)
            // GA Parameters
            .def_readwrite("YoungAgeTreshold", &NEAT::Parameters::YoungAgeTreshold)
            .def_readwrite("YoungAgeFitnessBoost", &NEAT::Parameters::YoungAgeFitnessBoost)
            .def_readwrite("SpeciesMaxStagnation", &NEAT::Parameters::SpeciesMaxStagnation)
            .def_readwrite("StagnationDelta", &NEAT::Parameters::StagnationDelta)
            .def_readwrite("OldAgeTreshold", &NEAT::Parameters::OldAgeTreshold)
            .def_readwrite("OldAgePenalty", &NEAT::Parameters::OldAgePenalty)
            .def_readwrite("DetectCompetetiveCoevolutionStagnation", &NEAT::Parameters::DetectCompetetiveCoevolutionStagnation)
            .def_readwrite("KillWorstSpeciesEach", &NEAT::Parameters::KillWorstSpeciesEach)
            .def_readwrite("KillWorstAge", &NEAT::Parameters::KillWorstAge)
            .def_readwrite("SurvivalRate", &NEAT::Parameters::SurvivalRate)
            .def_readwrite("CrossoverRate", &NEAT::Parameters::CrossoverRate)
            .def_readwrite("OverallMutationRate", &NEAT::Parameters::OverallMutationRate)
            .def_readwrite("InterspeciesCrossoverRate", &NEAT::Parameters::InterspeciesCrossoverRate)
            .def_readwrite("MultipointCrossoverRate", &NEAT::Parameters::MultipointCrossoverRate)
            .def_readwrite("PreferFitterParentRate", &NEAT::Parameters::PreferFitterParentRate)
            .def_readwrite("RouletteWheelSelection", &NEAT::Parameters::RouletteWheelSelection)
            .def_readwrite("TournamentSelection", &NEAT::Parameters::TournamentSelection)
            .def_readwrite("TournamentSize", &NEAT::Parameters::TournamentSize)
            .def_readwrite("EliteFraction", &NEAT::Parameters::EliteFraction)
            // Phased Search parameters
            .def_readwrite("PhasedSearching", &NEAT::Parameters::PhasedSearching)
            .def_readwrite("DeltaCoding", &NEAT::Parameters::DeltaCoding)
            .def_readwrite("SimplifyingPhaseMPCTreshold", &NEAT::Parameters::SimplifyingPhaseMPCTreshold)
            .def_readwrite("SimplifyingPhaseStagnationTreshold", &NEAT::Parameters::SimplifyingPhaseStagnationTreshold)
            .def_readwrite("ComplexityFloorGenerations", &NEAT::Parameters::ComplexityFloorGenerations)
            // Novelty Search parameters
            .def_readwrite("NoveltySearch_K", &NEAT::Parameters::NoveltySearch_K)
            .def_readwrite("NoveltySearch_P_min", &NEAT::Parameters::NoveltySearch_P_min)
            .def_readwrite("NoveltySearch_Dynamic_Pmin", &NEAT::Parameters::NoveltySearch_Dynamic_Pmin)
            .def_readwrite("NoveltySearch_No_Archiving_Stagnation_Treshold", &NEAT::Parameters::NoveltySearch_No_Archiving_Stagnation_Treshold)
            .def_readwrite("NoveltySearch_Pmin_lowering_multiplier", &NEAT::Parameters::NoveltySearch_Pmin_lowering_multiplier)
            .def_readwrite("NoveltySearch_Pmin_min", &NEAT::Parameters::NoveltySearch_Pmin_min)
            .def_readwrite("NoveltySearch_Quick_Archiving_Min_Evaluations", &NEAT::Parameters::NoveltySearch_Quick_Archiving_Min_Evaluations)
            .def_readwrite("NoveltySearch_Pmin_raising_multiplier", &NEAT::Parameters::NoveltySearch_Pmin_raising_multiplier)
            .def_readwrite("NoveltySearch_Recompute_Sparseness_Each", &NEAT::Parameters::NoveltySearch_Recompute_Sparseness_Each)
            // Mutation parameters
            .def_readwrite("MutateAddNeuronProb", &NEAT::Parameters::MutateAddNeuronProb)
            .def_readwrite("SplitRecurrent", &NEAT::Parameters::SplitRecurrent)
            .def_readwrite("SplitLoopedRecurrent", &NEAT::Parameters::SplitLoopedRecurrent)
            .def_readwrite("NeuronTries", &NEAT::Parameters::NeuronTries)
            .def_readwrite("MutateAddLinkProb", &NEAT::Parameters::MutateAddLinkProb)
            .def_readwrite("MutateAddLinkFromBiasProb", &NEAT::Parameters::MutateAddLinkFromBiasProb)
            .def_readwrite("MutateRemLinkProb", &NEAT::Parameters::MutateRemLinkProb)
            .def_readwrite("MutateRemSimpleNeuronProb", &NEAT::Parameters::MutateRemSimpleNeuronProb)
            .def_readwrite("LinkTries", &NEAT::Parameters::LinkTries)
            .def_readwrite("MaxLinks", &NEAT::Parameters::MaxLinks)
            .def_readwrite("MaxNeurons", &NEAT::Parameters::MaxNeurons)
            .def_readwrite("RecurrentProb", &NEAT::Parameters::RecurrentProb)
            .def_readwrite("RecurrentLoopProb", &NEAT::Parameters::RecurrentLoopProb)
            .def_readwrite("MutateWeightsProb", &NEAT::Parameters::MutateWeightsProb)
            .def_readwrite("MutateWeightsSevereProb", &NEAT::Parameters::MutateWeightsSevereProb)
            .def_readwrite("WeightMutationRate", &NEAT::Parameters::WeightMutationRate)
            .def_readwrite("WeightReplacementRate", &NEAT::Parameters::WeightReplacementRate)
            .def_readwrite("WeightMutationMaxPower", &NEAT::Parameters::WeightMutationMaxPower)
            .def_readwrite("WeightReplacementMaxPower", &NEAT::Parameters::WeightReplacementMaxPower)
            .def_readwrite("MaxWeight", &NEAT::Parameters::MaxWeight)
            .def_readwrite("MinWeight", &NEAT::Parameters::MinWeight)
            .def_readwrite("MutateActivationAProb", &NEAT::Parameters::MutateActivationAProb)
            .def_readwrite("MutateActivationBProb", &NEAT::Parameters::MutateActivationBProb)
            .def_readwrite("ActivationAMutationMaxPower", &NEAT::Parameters::ActivationAMutationMaxPower)
            .def_readwrite("ActivationBMutationMaxPower", &NEAT::Parameters::ActivationBMutationMaxPower)
            .def_readwrite("TimeConstantMutationMaxPower", &NEAT::Parameters::TimeConstantMutationMaxPower)
            .def_readwrite("BiasMutationMaxPower", &NEAT::Parameters::BiasMutationMaxPower)
            .def_readwrite("MinActivationA", &NEAT::Parameters::MinActivationA)
            .def_readwrite("MaxActivationA", &NEAT::Parameters::MaxActivationA)
            .def_readwrite("MinActivationB", &NEAT::Parameters::MinActivationB)
            .def_readwrite("MaxActivationB", &NEAT::Parameters::MaxActivationB)
            .def_readwrite("MutateNeuronActivationTypeProb", &NEAT::Parameters::MutateNeuronActivationTypeProb)
            .def_readwrite("ActivationFunction_SignedSigmoid_Prob", &NEAT::Parameters::ActivationFunction_SignedSigmoid_Prob)
            .def_readwrite("ActivationFunction_UnsignedSigmoid_Prob", &NEAT::Parameters::ActivationFunction_UnsignedSigmoid_Prob)
            .def_readwrite("ActivationFunction_Tanh_Prob", &NEAT::Parameters::ActivationFunction_Tanh_Prob)
            .def_readwrite("ActivationFunction_TanhCubic_Prob", &NEAT::Parameters::ActivationFunction_TanhCubic_Prob)
            .def_readwrite("ActivationFunction_SignedStep_Prob", &NEAT::Parameters::ActivationFunction_SignedStep_Prob)
            .def_readwrite("ActivationFunction_UnsignedStep_Prob", &NEAT::Parameters::ActivationFunction_UnsignedStep_Prob)
            .def_readwrite("ActivationFunction_SignedGauss_Prob", &NEAT::Parameters::ActivationFunction_SignedGauss_Prob)
            .def_readwrite("ActivationFunction_UnsignedGauss_Prob", &NEAT::Parameters::ActivationFunction_UnsignedGauss_Prob)
            .def_readwrite("ActivationFunction_Abs_Prob", &NEAT::Parameters::ActivationFunction_Abs_Prob)
            .def_readwrite("ActivationFunction_SignedSine_Prob", &NEAT::Parameters::ActivationFunction_SignedSine_Prob)
            .def_readwrite("ActivationFunction_UnsignedSine_Prob", &NEAT::Parameters::ActivationFunction_UnsignedSine_Prob)
            .def_readwrite("ActivationFunction_Linear_Prob", &NEAT::Parameters::ActivationFunction_Linear_Prob)
            .def_readwrite("ActivationFunction_Relu_Prob", &NEAT::Parameters::ActivationFunction_Relu_Prob)
            .def_readwrite("ActivationFunction_Softplus_Prob", &NEAT::Parameters::ActivationFunction_Softplus_Prob)
            .def_readwrite("MutateNeuronTimeConstantsProb", &NEAT::Parameters::MutateNeuronTimeConstantsProb)
            .def_readwrite("MutateNeuronBiasesProb", &NEAT::Parameters::MutateNeuronBiasesProb)
            .def_readwrite("MinNeuronTimeConstant", &NEAT::Parameters::MinNeuronTimeConstant)
            .def_readwrite("MaxNeuronTimeConstant", &NEAT::Parameters::MaxNeuronTimeConstant)
            .def_readwrite("MinNeuronBias", &NEAT::Parameters::MinNeuronBias)
            .def_readwrite("MaxNeuronBias", &NEAT::Parameters::MaxNeuronBias)
            // Speciation parameters
            .def_readwrite("DisjointCoeff", &NEAT::Parameters::DisjointCoeff)
            .def_readwrite("ExcessCoeff", &NEAT::Parameters::ExcessCoeff)
            .def_readwrite("ActivationADiffCoeff", &NEAT::Parameters::ActivationADiffCoeff)
            .def_readwrite("ActivationBDiffCoeff", &NEAT::Parameters::ActivationBDiffCoeff)
            .def_readwrite("WeightDiffCoeff", &NEAT::Parameters::WeightDiffCoeff)
            .def_readwrite("TimeConstantDiffCoeff", &NEAT::Parameters::TimeConstantDiffCoeff)
            .def_readwrite("BiasDiffCoeff", &NEAT::Parameters::BiasDiffCoeff)
            .def_readwrite("ActivationFunctionDiffCoeff", &NEAT::Parameters::ActivationFunctionDiffCoeff)
            .def_readwrite("CompatTreshold", &NEAT::Parameters::CompatTreshold)
            .def_readwrite("MinCompatTreshold", &NEAT::Parameters::MinCompatTreshold)
            .def_readwrite("CompatTresholdModifier", &NEAT::Parameters::CompatTresholdModifier)
            .def_readwrite("CompatTreshChangeInterval_Generations", &NEAT::Parameters::CompatTreshChangeInterval_Generations)
            .def_readwrite("CompatTreshChangeInterval_Evaluations", &NEAT::Parameters::CompatTreshChangeInterval_Evaluations)
            .def_readwrite("MinDeltaCompatEqualGenomes", &NEAT::Parameters::MinDeltaCompatEqualGenomes)
            .def_readwrite("ConstraintTrials", &NEAT::Parameters::ConstraintTrials)
            // Genome properties params
            .def_readwrite("DontUseBiasNeuron", &NEAT::Parameters::DontUseBiasNeuron)
            .def_readwrite("AllowLoops", &NEAT::Parameters::AllowLoops)
            // ES HyperNEAT params
            .def_readwrite("DivisionThreshold", &NEAT::Parameters::DivisionThreshold)
            .def_readwrite("VarianceThreshold", &NEAT::Parameters::VarianceThreshold)
            .def_readwrite("BandThreshold", &NEAT::Parameters::BandThreshold)
            .def_readwrite("InitialDepth", &NEAT::Parameters::InitialDepth)
            .def_readwrite("MaxDepth", &NEAT::Parameters::MaxDepth)
            .def_readwrite("IterationLevel", &NEAT::Parameters::IterationLevel)
            .def_readwrite("CPPN_Bias", &NEAT::Parameters::CPPN_Bias)
            .def_readwrite("Width", &NEAT::Parameters::Width)
            .def_readwrite("Height", &NEAT::Parameters::Height)
            .def_readwrite("Qtree_X", &NEAT::Parameters::Qtree_X)
            .def_readwrite("Qtree_Y", &NEAT::Parameters::Qtree_Y)
            .def_readwrite("Leo", &NEAT::Parameters::Leo)
            .def_readwrite("LeoThreshold", &NEAT::Parameters::LeoThreshold)
            .def_readwrite("LeoSeed", &NEAT::Parameters::LeoSeed)
            .def_readwrite("GeometrySeed", &NEAT::Parameters::GeometrySeed)
            // Universal traits
            .def_readwrite("NeuronTraits", &NEAT::Parameters::NeuronTraits)
            .def_readwrite("LinkTraits", &NEAT::Parameters::LinkTraits)
            .def_readwrite("GenomeTraits", &NEAT::Parameters::GenomeTraits)
            .def_readwrite("MutateNeuronTraitsProb", &NEAT::Parameters::MutateNeuronTraitsProb)
            .def_readwrite("MutateLinkTraitsProb", &NEAT::Parameters::MutateLinkTraitsProb)
            .def_readwrite("MutateGenomeTraitsProb", &NEAT::Parameters::MutateGenomeTraitsProb)
            ;
    //};

    // ========================
    // Bindings for PhenotypeBehavior
    // ========================

    py::class_<NEAT::PhenotypeBehavior, std::shared_ptr<NEAT::PhenotypeBehavior>>(m, "PhenotypeBehavior")
        .def(py::init<>())
        .def("Acquire", &NEAT::PhenotypeBehavior::Acquire)
        .def("Distance_To", &NEAT::PhenotypeBehavior::Distance_To)
        .def("Successful", &NEAT::PhenotypeBehavior::Successful)
        .def_readwrite("m_Data", &NEAT::PhenotypeBehavior::m_Data);

    // ========================
    // Bindings for Population
    // ========================

    py::class_<NEAT::Population>(m, "Population")
        .def(py::init<const NEAT::Genome&, const NEAT::Parameters&, bool, double, int>(),
            py::arg("genome"), py::arg("parameters"), py::arg("randomizeWeights"), py::arg("randomizationRange"), py::arg("rng_seed"))
        .def(py::init<const std::string>())
        .def("GetGeneration", &NEAT::Population::GetGeneration)
        .def("GetBestFitnessEver", &NEAT::Population::GetBestFitnessEver)
        .def("GetBestGenome", &NEAT::Population::GetBestGenome)
        .def("GetStagnation", &NEAT::Population::GetStagnation)
        .def("GetMPCStagnation", &NEAT::Population::GetMPCStagnation)
        .def("GetNextGenomeID", &NEAT::Population::GetNextGenomeID)
        .def("GetNextSpeciesID", &NEAT::Population::GetNextSpeciesID)
        .def("SameGenomeIDCheck", &NEAT::Population::SameGenomeIDCheck)
        .def("AccessGenomeByIndex", &NEAT::Population::AccessGenomeByIndex, py::return_value_policy::reference)
        .def("AccessGenomeByID", &NEAT::Population::AccessGenomeByID, py::return_value_policy::reference)
        .def("Epoch", &NEAT::Population::Epoch)
        .def("Save", &NEAT::Population::Save)
        .def("Tick", &NEAT::Population::Tick, py::return_value_policy::reference)
        .def("NoveltySearchTick", &NEAT::Population::NoveltySearchTick)
        .def_readwrite("m_GenomeArchive", &NEAT::Population::m_GenomeArchive)
        .def_readwrite("m_RNG", &NEAT::Population::m_RNG)
        .def_readwrite("m_Parameters", &NEAT::Population::m_Parameters)
        .def_readwrite("m_Generation", &NEAT::Population::m_Generation)
        .def_readwrite("m_Species", &NEAT::Population::m_Species)
        .def_readwrite("m_ID", &NEAT::Population::m_ID)
        .def_readwrite("m_NumEvaluations", &NEAT::Population::m_NumEvaluations);

    // ========================
    // Bindings for Species
    // ========================

    py::class_<NEAT::Species>(m, "Species")
        .def(py::init<const NEAT::Genome&, const NEAT::Parameters&, int>(),
             py::arg("seed"), py::arg("parameters"), py::arg("id"))
        .def("GetBestFitness", &NEAT::Species::GetBestFitness)
        .def("NumIndividuals", &NEAT::Species::NumIndividuals)
        .def("ID", &NEAT::Species::ID)
        .def("GensNoImprovement", &NEAT::Species::GensNoImprovement)
        .def("EvalsNoImprovement", &NEAT::Species::EvalsNoImprovement)
        .def("AgeGens", &NEAT::Species::AgeGens)
        .def("AgeEvals", &NEAT::Species::AgeEvals)
        .def("GetIndividualByIdx", &NEAT::Species::GetIndividualByIdx)
        .def("IsBestSpecies", &NEAT::Species::IsBestSpecies)
        .def("IsWorstSpecies", &NEAT::Species::IsWorstSpecies)
        .def("NumEvaluated", &NEAT::Species::NumEvaluated)
        .def("GetLeader", &NEAT::Species::GetLeader, py::return_value_policy::reference)
        .def("GetRepresentative", &NEAT::Species::GetRepresentative, py::return_value_policy::reference)
        .def("GetIndividual", &NEAT::Species::GetIndividual, py::return_value_policy::reference)
        .def("GetRandomIndividual", &NEAT::Species::GetRandomIndividual, py::return_value_policy::reference)
        .def_readwrite("m_BestGenome", &NEAT::Species::m_BestGenome)
        .def_readwrite("m_GensNoImprovement", &NEAT::Species::m_GensNoImprovement)
        .def_readwrite("m_EvalsNoImprovement", &NEAT::Species::m_EvalsNoImprovement)
        .def_readwrite("m_R", &NEAT::Species::m_R)
        .def_readwrite("m_G", &NEAT::Species::m_G)
        .def_readwrite("m_B", &NEAT::Species::m_B)
        .def_readwrite("m_AverageFitness", &NEAT::Species::m_AverageFitness)
        .def_readwrite("m_Individuals", &NEAT::Species::m_Individuals);

    // ========================
    // Bindings for RNG
    // ========================

    py::class_<NEAT::RNG>(m, "RNG")
        .def(py::init<>())
        .def("Seed", &NEAT::RNG::Seed)
        .def("TimeSeed", &NEAT::RNG::TimeSeed)
        .def("RandPosNeg", &NEAT::RNG::RandPosNeg)
        .def("RandInt", &NEAT::RNG::RandInt)
        .def("RandFloat", &NEAT::RNG::RandFloat)
        .def("RandFloatSigned", &NEAT::RNG::RandFloatSigned)
        .def("RandGaussSigned", &NEAT::RNG::RandGaussSigned)
        .def("Roulette", &NEAT::RNG::Roulette);

    // ========================
    // Bindings for Substrate
    // ========================

    py::class_<NEAT::Substrate>(m, "Substrate")
        .def(py::init<>())
        .def(py::init<std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, std::vector<std::vector<double>>&>())
        .def("SetCustomConnectivity", &NEAT::Substrate::SetCustomConnectivity)
        .def("ClearCustomConnectivity", &NEAT::Substrate::ClearCustomConnectivity)
        .def("GetMinCPPNInputs", &NEAT::Substrate::GetMinCPPNInputs)
        .def("GetMinCPPNOutputs", &NEAT::Substrate::GetMinCPPNOutputs)
        .def("GetMaxDims", &NEAT::Substrate::GetMaxDims)
        .def("PrintInfo", &NEAT::Substrate::PrintInfo)
        .def_readwrite("m_input_coords", &NEAT::Substrate::m_input_coords)
        .def_readwrite("m_hidden_coords", &NEAT::Substrate::m_hidden_coords)
        .def_readwrite("m_output_coords", &NEAT::Substrate::m_output_coords)
        .def_readwrite("m_leaky", &NEAT::Substrate::m_leaky)
        .def_readwrite("m_with_distance", &NEAT::Substrate::m_with_distance)
        .def_readwrite("m_allow_input_hidden_links", &NEAT::Substrate::m_allow_input_hidden_links)
        .def_readwrite("m_allow_input_output_links", &NEAT::Substrate::m_allow_input_output_links)
        .def_readwrite("m_allow_hidden_hidden_links", &NEAT::Substrate::m_allow_hidden_hidden_links)
        .def_readwrite("m_allow_hidden_output_links", &NEAT::Substrate::m_allow_hidden_output_links)
        .def_readwrite("m_allow_output_hidden_links", &NEAT::Substrate::m_allow_output_hidden_links)
        .def_readwrite("m_allow_output_output_links", &NEAT::Substrate::m_allow_output_output_links)
        .def_readwrite("m_allow_looped_hidden_links", &NEAT::Substrate::m_allow_looped_hidden_links)
        .def_readwrite("m_allow_looped_output_links", &NEAT::Substrate::m_allow_looped_output_links)
        .def_readwrite("m_custom_connectivity", &NEAT::Substrate::m_custom_connectivity)
        .def_readwrite("m_custom_conn_obeys_flags", &NEAT::Substrate::m_custom_conn_obeys_flags)
        .def_readwrite("m_query_weights_only", &NEAT::Substrate::m_query_weights_only)
        .def_readwrite("m_hidden_nodes_activation", &NEAT::Substrate::m_hidden_nodes_activation)
        .def_readwrite("m_output_nodes_activation", &NEAT::Substrate::m_output_nodes_activation)
        .def_readwrite("m_max_weight_and_bias", &NEAT::Substrate::m_max_weight_and_bias)
        .def_readwrite("m_min_time_const", &NEAT::Substrate::m_min_time_const)
        .def_readwrite("m_max_time_const", &NEAT::Substrate::m_max_time_const);
}; 


