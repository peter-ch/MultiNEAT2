// Must include Python.h first on Linux
#include <cassert>
#include <Python.h>

// Then include pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>

#include <utility>

namespace py = pybind11;

// Then include MultiNEAT headers
#include "Assert.h"
#include "Genes.h"             
#include "Genome.h"            
#include "Innovation.h"        
#include "NeuralNetwork.h"     
#include "Parameters.h"        
#include "PhenotypeBehavior.h" 
#include "Population.h"        
#include "Species.h"           
#include "SpikingLearning.h"
#include "Random.h"            
#include "Substrate.h"         
#include "Traits.h"            
#include "Utils.h"             

namespace
{
class PyPhenotypeBehavior : public NEAT::PhenotypeBehavior
{
public:
    using NEAT::PhenotypeBehavior::PhenotypeBehavior;

    bool Acquire(NEAT::Genome* genome) override
    {
        PYBIND11_OVERRIDE(bool, NEAT::PhenotypeBehavior, Acquire, genome);
    }

    double Distance_To(NEAT::PhenotypeBehavior* other) override
    {
        PYBIND11_OVERRIDE(
            double, NEAT::PhenotypeBehavior, Distance_To, other);
    }

    bool Successful() override
    {
        PYBIND11_OVERRIDE(bool, NEAT::PhenotypeBehavior, Successful);
    }
};
}


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
        .value("SPIKING_LIF", NEAT::SPIKING_LIF)
        .value(
            "SPIKING_ADAPTIVE_LIF",
            NEAT::SPIKING_ADAPTIVE_LIF)
        .value(
            "SPIKING_IZHIKEVICH",
            NEAT::SPIKING_IZHIKEVICH)
        .export_values();

    py::enum_<NEAT::SpikingInputMode>(m, "SpikingInputMode")
        .value("CURRENT_INPUT", NEAT::CURRENT_INPUT)
        .value("BINARY_SPIKE_INPUT", NEAT::BINARY_SPIKE_INPUT)
        .value("POISSON_RATE_INPUT", NEAT::POISSON_RATE_INPUT)
        .export_values();

    py::enum_<NEAT::SpikingOutputMode>(m, "SpikingOutputMode")
        .value("SPIKE_OUTPUT", NEAT::SPIKE_OUTPUT)
        .value("FIRING_RATE_OUTPUT", NEAT::FIRING_RATE_OUTPUT)
        .value(
            "FILTERED_SPIKE_OUTPUT",
            NEAT::FILTERED_SPIKE_OUTPUT)
        .value(
            "MEMBRANE_POTENTIAL_OUTPUT",
            NEAT::MEMBRANE_POTENTIAL_OUTPUT)
        .export_values();

    py::enum_<NEAT::EPropOptimizer>(m, "EPropOptimizer")
        .value("EPROP_ADAMW", NEAT::EPROP_ADAMW)
        .value("EPROP_SGD", NEAT::EPROP_SGD)
        .export_values();

    py::enum_<NEAT::EPropFeedbackMode>(
        m, "EPropFeedbackMode")
        .value(
            "EPROP_RANDOM_FEEDBACK",
            NEAT::EPROP_RANDOM_FEEDBACK)
        .value(
            "EPROP_SYMMETRIC_FEEDBACK",
            NEAT::EPROP_SYMMETRIC_FEEDBACK)
        .value(
            "EPROP_UNIFORM_FEEDBACK",
            NEAT::EPROP_UNIFORM_FEEDBACK)
        .export_values();

    py::enum_<NEAT::EPropSurrogate>(m, "EPropSurrogate")
        .value(
            "EPROP_FAST_SIGMOID",
            NEAT::EPROP_FAST_SIGMOID)
        .value(
            "EPROP_TRIANGULAR",
            NEAT::EPROP_TRIANGULAR)
        .value("EPROP_ARCTAN", NEAT::EPROP_ARCTAN)
        .export_values();

    py::enum_<NEAT::EPropLoss>(m, "EPropLoss")
        .value(
            "EPROP_MEAN_SQUARED_ERROR",
            NEAT::EPROP_MEAN_SQUARED_ERROR)
        .value(
            "EPROP_HUBER_LOSS",
            NEAT::EPROP_HUBER_LOSS)
        .export_values();
    m.def(
        "IsSpikingActivation",
        &NEAT::IsSpikingActivation,
        py::arg("activation"));

    py::enum_<NEAT::GenomeSeedType>(m, "GenomeSeedType")
        .value("PERCEPTRON", NEAT::PERCEPTRON)
        .value("LAYERED", NEAT::LAYERED)
        .export_values();

    py::enum_<NEAT::InnovationType>(m, "InnovationType")
        .value("NEW_NEURON", NEAT::NEW_NEURON)
        .value("NEW_LINK", NEAT::NEW_LINK)
        .export_values();

    py::enum_<NEAT::SearchMode>(m, "SearchMode")
        .value("COMPLEXIFYING", NEAT::COMPLEXIFYING)
        .value("SIMPLIFYING", NEAT::SIMPLIFYING)
        .value("BLENDED", NEAT::BLENDED)
        .export_values();

    py::enum_<NEAT::SelectionMode>(m, "SelectionMode")
        .value("LEGACY_SELECTION", NEAT::LEGACY_SELECTION)
        .value("TRUNCATION", NEAT::TRUNCATION)
        .value("ROULETTE", NEAT::ROULETTE)
        .value("RANK_LINEAR", NEAT::RANK_LINEAR)
        .value("RANK_EXP", NEAT::RANK_EXP)
        .value("TOURNAMENT", NEAT::TOURNAMENT)
        .value("STOCHASTIC", NEAT::STOCHASTIC)
        .value("BOLTZMANN", NEAT::BOLTZMANN)
        .export_values();

    py::enum_<NEAT::CrossoverMode>(m, "CrossoverMode")
        .value("MULTIPOINT", NEAT::MULTIPOINT)
        .value("AVERAGE", NEAT::AVERAGE)
        .value("SINGLE_POINT", NEAT::SINGLE_POINT)
        .value("BLEND", NEAT::BLEND)
        .value("SIMULATED_BINARY", NEAT::SIMULATED_BINARY)
        .export_values();

    py::enum_<NEAT::WeightMutationMode>(m, "WeightMutationMode")
        .value("UNIFORM_MUTATION", NEAT::UNIFORM_MUTATION)
        .value("GAUSSIAN_MUTATION", NEAT::GAUSSIAN_MUTATION)
        .value("CAUCHY_MUTATION", NEAT::CAUCHY_MUTATION)
        .value("POLYNOMIAL_MUTATION", NEAT::POLYNOMIAL_MUTATION)
        .export_values();

    py::enum_<NEAT::SpeciesRepresentativeMode>(
        m, "SpeciesRepresentativeMode")
        .value("FIRST_REPRESENTATIVE", NEAT::FIRST_REPRESENTATIVE)
        .value("RANDOM_REPRESENTATIVE", NEAT::RANDOM_REPRESENTATIVE)
        .value("BEST_REPRESENTATIVE", NEAT::BEST_REPRESENTATIVE)
        .value("MEDOID_REPRESENTATIVE", NEAT::MEDOID_REPRESENTATIVE)
        .export_values();

    py::enum_<NEAT::OffspringAllocationMode>(
        m, "OffspringAllocationMode")
        .value("LARGEST_REMAINDER", NEAT::LARGEST_REMAINDER)
        .value("STOCHASTIC_REMAINDER", NEAT::STOCHASTIC_REMAINDER)
        .export_values();

    py::enum_<NEAT::CompatibilityThresholdMode>(
        m, "CompatibilityThresholdMode")
        .value(
            "LEGACY_COMPATIBILITY_THRESHOLD",
            NEAT::LEGACY_COMPATIBILITY_THRESHOLD)
        .value(
            "PROPORTIONAL_COMPATIBILITY_THRESHOLD",
            NEAT::PROPORTIONAL_COMPATIBILITY_THRESHOLD)
        .export_values();

    py::enum_<NEAT::FitnessScalingMode>(m, "FitnessScalingMode")
        .value(
            "SHIFTED_FITNESS_SCALING",
            NEAT::SHIFTED_FITNESS_SCALING)
        .value(
            "LINEAR_RANK_FITNESS_SCALING",
            NEAT::LINEAR_RANK_FITNESS_SCALING)
        .value(
            "SIGMA_FITNESS_SCALING",
            NEAT::SIGMA_FITNESS_SCALING)
        .value(
            "BOLTZMANN_FITNESS_SCALING",
            NEAT::BOLTZMANN_FITNESS_SCALING)
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
        .def_readwrite("m_Details", &NEAT::TraitParameters::m_Details)
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
        .def("GetTraitDistances",
             [](const NEAT::Gene& gene,
                const std::map<std::string, NEAT::Trait>& other) {
                 return gene.GetTraitDistances(other);
             })
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
        .def_readwrite("m_IsRecurrent", &NEAT::LinkGene::m_IsRecurrent)
        .def_readwrite(
            "m_SynapticDelay", &NEAT::LinkGene::m_SynapticDelay)
        .def_readwrite(
            "m_SynapticTimeConstant",
            &NEAT::LinkGene::m_SynapticTimeConstant)
        .def_readwrite(
            "m_STDPEnabled", &NEAT::LinkGene::m_STDPEnabled)
        .def_readwrite("m_STDPPlus", &NEAT::LinkGene::m_STDPPlus)
        .def_readwrite("m_STDPMinus", &NEAT::LinkGene::m_STDPMinus)
        .def_readwrite(
            "m_STDPTauPlus", &NEAT::LinkGene::m_STDPTauPlus)
        .def_readwrite(
            "m_STDPTauMinus", &NEAT::LinkGene::m_STDPTauMinus)
        .def_readwrite(
            "m_STDPMinWeight", &NEAT::LinkGene::m_STDPMinWeight)
        .def_readwrite(
            "m_STDPMaxWeight", &NEAT::LinkGene::m_STDPMaxWeight);

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
        .def_readwrite("m_ActFunction", &NEAT::NeuronGene::m_ActFunction)
        .def_readwrite(
            "m_SpikeThreshold", &NEAT::NeuronGene::m_SpikeThreshold)
        .def_readwrite(
            "m_ResetPotential", &NEAT::NeuronGene::m_ResetPotential)
        .def_readwrite(
            "m_RestingPotential", &NEAT::NeuronGene::m_RestingPotential)
        .def_readwrite(
            "m_RefractoryPeriod", &NEAT::NeuronGene::m_RefractoryPeriod)
        .def_readwrite(
            "m_MembraneResistance",
            &NEAT::NeuronGene::m_MembraneResistance)
        .def_readwrite(
            "m_AdaptationTimeConstant",
            &NEAT::NeuronGene::m_AdaptationTimeConstant)
        .def_readwrite(
            "m_AdaptationIncrement",
            &NEAT::NeuronGene::m_AdaptationIncrement)
        .def_readwrite(
            "m_RateTimeConstant",
            &NEAT::NeuronGene::m_RateTimeConstant)
        .def_readwrite(
            "m_IzhikevichA", &NEAT::NeuronGene::m_IzhikevichA)
        .def_readwrite(
            "m_IzhikevichB", &NEAT::NeuronGene::m_IzhikevichB)
        .def_readwrite(
            "m_IzhikevichC", &NEAT::NeuronGene::m_IzhikevichC)
        .def_readwrite(
            "m_IzhikevichD", &NEAT::NeuronGene::m_IzhikevichD);


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
        .def("NumNeurons", &NEAT::Genome::NumNeurons)
        .def("NumLinks", &NEAT::Genome::NumLinks)
        .def("NumInputs", &NEAT::Genome::NumInputs)
        .def("NumOutputs", &NEAT::Genome::NumOutputs)
        .def("SetNeuronXY", &NEAT::Genome::SetNeuronXY)
        .def("SetNeuronX", &NEAT::Genome::SetNeuronX)
        .def("SetNeuronY", &NEAT::Genome::SetNeuronY)
        .def("GetFitness", &NEAT::Genome::GetFitness)
        .def("GetAdjFitness", &NEAT::Genome::GetAdjFitness)
        .def("SetFitness", &NEAT::Genome::SetFitness)
        .def("SetAdjFitness", &NEAT::Genome::SetAdjFitness)
        .def("IsEvaluated", &NEAT::Genome::IsEvaluated)
        .def("SetEvaluated", &NEAT::Genome::SetEvaluated)
        .def("ResetEvaluated", &NEAT::Genome::ResetEvaluated)
        .def("GetID", &NEAT::Genome::GetID)
        .def("SetID", &NEAT::Genome::SetID)
        .def("GetDepth", &NEAT::Genome::GetDepth)
        .def("SetDepth", &NEAT::Genome::SetDepth)
        .def("CalculateDepth", &NEAT::Genome::CalculateDepth)
        .def("HasDeadEnds", &NEAT::Genome::HasDeadEnds)
        .def("HasLoops", &NEAT::Genome::HasLoops)
        .def("FailsConstraints", &NEAT::Genome::FailsConstraints)
        .def("IsIdenticalTo", &NEAT::Genome::IsIdenticalTo)
        .def("Validate",
             [](const NEAT::Genome &genome) {
                 std::string error;
                 return py::make_tuple(genome.Validate(&error), error);
             })
        .def("GetOffspringAmount", &NEAT::Genome::GetOffspringAmount)
        .def("SetOffspringAmount", &NEAT::Genome::SetOffspringAmount)
        .def("GetLastNeuronID", &NEAT::Genome::GetLastNeuronID)
        .def("GetLastInnovationID", &NEAT::Genome::GetLastInnovationID)
        .def("BuildPhenotype", &NEAT::Genome::BuildPhenotype)
        .def("BuildHyperNEATPhenotype",
             &NEAT::Genome::BuildHyperNEATPhenotype)
        .def("BuildESHyperNEATPhenotype",
             &NEAT::Genome::BuildESHyperNEATPhenotype)
        .def("DerivePhenotypicChanges", &NEAT::Genome::DerivePhenotypicChanges)
        .def("CompatibilityDistance", &NEAT::Genome::CompatibilityDistance)
        .def("IsCompatibleWith", &NEAT::Genome::IsCompatibleWith)
        .def("Mutate_LinkWeights", &NEAT::Genome::Mutate_LinkWeights)
        .def("Randomize_LinkWeights", &NEAT::Genome::Randomize_LinkWeights)
        .def("Randomize_Traits", &NEAT::Genome::Randomize_Traits)
        .def(
            "Randomize_SpikingParameters",
            &NEAT::Genome::Randomize_SpikingParameters)
        .def("Mutate_NeuronActivations_A", &NEAT::Genome::Mutate_NeuronActivations_A)
        .def("Mutate_NeuronActivations_B", &NEAT::Genome::Mutate_NeuronActivations_B)
        .def("Mutate_NeuronActivation_Type", &NEAT::Genome::Mutate_NeuronActivation_Type)
        .def("Mutate_NeuronTimeConstants", &NEAT::Genome::Mutate_NeuronTimeConstants)
        .def("Mutate_NeuronBiases", &NEAT::Genome::Mutate_NeuronBiases)
        .def(
            "Mutate_NeuronSpikingParameters",
            &NEAT::Genome::Mutate_NeuronSpikingParameters)
        .def(
            "Mutate_LinkSpikingParameters",
            &NEAT::Genome::Mutate_LinkSpikingParameters)
        .def("Mutate_NeuronTraits", &NEAT::Genome::Mutate_NeuronTraits)
        .def("Mutate_LinkTraits", &NEAT::Genome::Mutate_LinkTraits)
        .def("Mutate_GenomeTraits", &NEAT::Genome::Mutate_GenomeTraits)
        .def("Mutate_AddNeuron", &NEAT::Genome::Mutate_AddNeuron)
        .def("Mutate_AddLink", &NEAT::Genome::Mutate_AddLink)
        .def("Mutate_RemoveLink", &NEAT::Genome::Mutate_RemoveLink)
        .def("Mutate_RemoveSimpleNeuron", &NEAT::Genome::Mutate_RemoveSimpleNeuron)
        .def("Cleanup", &NEAT::Genome::Cleanup)
        .def("Mate", &NEAT::Genome::Mate)
        .def("MateWithMode", &NEAT::Genome::MateWithMode)
        .def("SortGenes", &NEAT::Genome::SortGenes)
        .def("Save",
             py::overload_cast<const char*>(&NEAT::Genome::Save))
        .def("Serialize", &NEAT::Genome::Serialize)
        .def_static("Deserialize", &NEAT::Genome::Deserialize)
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
        .def(py::init<>())
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
        .def("Flush", &NEAT::InnovationDatabase::Flush)
        .def("RebuildIndex", &NEAT::InnovationDatabase::RebuildIndex)
        .def("GetInnovationByIdx",
             &NEAT::InnovationDatabase::GetInnovationByIdx)
        .def_readwrite("m_Innovations",
                       &NEAT::InnovationDatabase::m_Innovations)
        .def("Serialize", &NEAT::InnovationDatabase::Serialize)
        .def_static(
            "Deserialize", &NEAT::InnovationDatabase::Deserialize)
        .def(py::pickle(
            [](const NEAT::InnovationDatabase &database) {
                return database.Serialize();
            },
            [](const std::string &state) {
                return NEAT::InnovationDatabase::Deserialize(state);
            }
        ));

    // ========================
    // Bindings for NeuralNetwork (and its inner classes)
    // ========================

    py::class_<NEAT::SpikeEvent>(m, "SpikeEvent")
        .def(py::init<>())
        .def_readwrite("time", &NEAT::SpikeEvent::time)
        .def_readwrite(
            "neuron_index", &NEAT::SpikeEvent::neuron_index)
        .def_readwrite("amplitude", &NEAT::SpikeEvent::amplitude)
        .def_readwrite("input", &NEAT::SpikeEvent::input);

    py::class_<NEAT::PendingSynapticEvent>(
        m, "PendingSynapticEvent")
        .def(py::init<>())
        .def_readwrite(
            "delivery_time",
            &NEAT::PendingSynapticEvent::delivery_time)
        .def_readwrite(
            "amplitude",
            &NEAT::PendingSynapticEvent::amplitude)
        .def_readwrite(
            "source_amplitude",
            &NEAT::PendingSynapticEvent::source_amplitude);

    py::class_<NEAT::Connection>(m, "Connection")
        .def(py::init<>())
        .def_readwrite("m_source_neuron_idx", &NEAT::Connection::m_source_neuron_idx)
        .def_readwrite("m_target_neuron_idx", &NEAT::Connection::m_target_neuron_idx)
        .def_readwrite("m_weight", &NEAT::Connection::m_weight)
        .def_readwrite("m_signal", &NEAT::Connection::m_signal)
        .def_readwrite("m_source_activation",
                       &NEAT::Connection::m_source_activation)
        .def_readwrite("m_recur_flag", &NEAT::Connection::m_recur_flag)
        .def_readwrite("m_hebb_rate", &NEAT::Connection::m_hebb_rate)
        .def_readwrite("m_hebb_pre_rate", &NEAT::Connection::m_hebb_pre_rate)
        .def_readwrite(
            "m_synaptic_delay",
            &NEAT::Connection::m_synaptic_delay)
        .def_readwrite(
            "m_synaptic_time_constant",
            &NEAT::Connection::m_synaptic_time_constant)
        .def_readwrite(
            "m_presynaptic_signal",
            &NEAT::Connection::m_presynaptic_signal)
        .def_readwrite(
            "m_synaptic_current",
            &NEAT::Connection::m_synaptic_current)
        .def_readwrite(
            "m_stdp_enabled", &NEAT::Connection::m_stdp_enabled)
        .def_readwrite("m_stdp_plus", &NEAT::Connection::m_stdp_plus)
        .def_readwrite(
            "m_stdp_minus", &NEAT::Connection::m_stdp_minus)
        .def_readwrite(
            "m_stdp_tau_plus",
            &NEAT::Connection::m_stdp_tau_plus)
        .def_readwrite(
            "m_stdp_tau_minus",
            &NEAT::Connection::m_stdp_tau_minus)
        .def_readwrite(
            "m_stdp_pre_trace",
            &NEAT::Connection::m_stdp_pre_trace)
        .def_readwrite(
            "m_stdp_post_trace",
            &NEAT::Connection::m_stdp_post_trace)
        .def_readwrite(
            "m_stdp_min_weight",
            &NEAT::Connection::m_stdp_min_weight)
        .def_readwrite(
            "m_stdp_max_weight",
            &NEAT::Connection::m_stdp_max_weight)
        .def_readwrite(
            "m_pending_events",
            &NEAT::Connection::m_pending_events);

    py::class_<NEAT::Neuron>(m, "Neuron")
        .def(py::init<>())
        .def_readwrite("m_activesum", &NEAT::Neuron::m_activesum)
        .def_readwrite("m_activation", &NEAT::Neuron::m_activation)
        .def_readwrite("m_a", &NEAT::Neuron::m_a)
        .def_readwrite("m_b", &NEAT::Neuron::m_b)
        .def_readwrite("m_timeconst", &NEAT::Neuron::m_timeconst)
        .def_readwrite("m_bias", &NEAT::Neuron::m_bias)
        .def_readwrite("m_membrane_potential", &NEAT::Neuron::m_membrane_potential)
        .def_readwrite("m_last_input", &NEAT::Neuron::m_last_input)
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
        .def_readwrite("m_sensitivity_matrix", &NEAT::Neuron::m_sensitivity_matrix)
        .def_readwrite(
            "m_spike_threshold", &NEAT::Neuron::m_spike_threshold)
        .def_readwrite(
            "m_reset_potential", &NEAT::Neuron::m_reset_potential)
        .def_readwrite(
            "m_resting_potential", &NEAT::Neuron::m_resting_potential)
        .def_readwrite(
            "m_refractory_period", &NEAT::Neuron::m_refractory_period)
        .def_readwrite(
            "m_refractory_remaining",
            &NEAT::Neuron::m_refractory_remaining)
        .def_readwrite(
            "m_membrane_resistance",
            &NEAT::Neuron::m_membrane_resistance)
        .def_readwrite(
            "m_adaptation_time_constant",
            &NEAT::Neuron::m_adaptation_time_constant)
        .def_readwrite(
            "m_adaptation_increment",
            &NEAT::Neuron::m_adaptation_increment)
        .def_readwrite(
            "m_adaptation", &NEAT::Neuron::m_adaptation)
        .def_readwrite(
            "m_izhikevich_a", &NEAT::Neuron::m_izhikevich_a)
        .def_readwrite(
            "m_izhikevich_b", &NEAT::Neuron::m_izhikevich_b)
        .def_readwrite(
            "m_izhikevich_c", &NEAT::Neuron::m_izhikevich_c)
        .def_readwrite(
            "m_izhikevich_d", &NEAT::Neuron::m_izhikevich_d)
        .def_readwrite(
            "m_izhikevich_recovery",
            &NEAT::Neuron::m_izhikevich_recovery)
        .def_readwrite("m_spike", &NEAT::Neuron::m_spike)
        .def_readwrite(
            "m_spike_count", &NEAT::Neuron::m_spike_count)
        .def_readwrite(
            "m_last_spike_time", &NEAT::Neuron::m_last_spike_time)
        .def_readwrite(
            "m_rate_trace", &NEAT::Neuron::m_rate_trace)
        .def_readwrite(
            "m_rate_time_constant",
            &NEAT::Neuron::m_rate_time_constant);

    py::class_<NEAT::EPropConfig>(m, "EPropConfig")
        .def(py::init<>())
        .def_readwrite(
            "learning_rate",
            &NEAT::EPropConfig::learning_rate)
        .def_readwrite(
            "optimizer",
            &NEAT::EPropConfig::optimizer)
        .def_readwrite(
            "feedback_mode",
            &NEAT::EPropConfig::feedback_mode)
        .def_readwrite(
            "surrogate",
            &NEAT::EPropConfig::surrogate)
        .def_readwrite("loss", &NEAT::EPropConfig::loss)
        .def_readwrite(
            "surrogate_scale",
            &NEAT::EPropConfig::surrogate_scale)
        .def_readwrite(
            "surrogate_dampening",
            &NEAT::EPropConfig::surrogate_dampening)
        .def_readwrite(
            "gradient_clip_norm",
            &NEAT::EPropConfig::gradient_clip_norm)
        .def_readwrite(
            "weight_decay",
            &NEAT::EPropConfig::weight_decay)
        .def_readwrite(
            "adam_beta1",
            &NEAT::EPropConfig::adam_beta1)
        .def_readwrite(
            "adam_beta2",
            &NEAT::EPropConfig::adam_beta2)
        .def_readwrite(
            "adam_epsilon",
            &NEAT::EPropConfig::adam_epsilon)
        .def_readwrite(
            "huber_delta",
            &NEAT::EPropConfig::huber_delta)
        .def_readwrite(
            "min_weight",
            &NEAT::EPropConfig::min_weight)
        .def_readwrite(
            "max_weight",
            &NEAT::EPropConfig::max_weight)
        .def_readwrite(
            "update_interval",
            &NEAT::EPropConfig::update_interval)
        .def_readwrite(
            "random_seed",
            &NEAT::EPropConfig::random_seed)
        .def_readwrite(
            "train_input_connections",
            &NEAT::EPropConfig::train_input_connections)
        .def_readwrite(
            "train_hidden_connections",
            &NEAT::EPropConfig::train_hidden_connections)
        .def_readwrite(
            "train_output_connections",
            &NEAT::EPropConfig::train_output_connections)
        .def_readwrite(
            "train_recurrent_connections",
            &NEAT::EPropConfig::train_recurrent_connections)
        .def_readwrite(
            "allow_stdp",
            &NEAT::EPropConfig::allow_stdp);

    py::class_<NEAT::EPropConnectionState>(
        m, "EPropConnectionState")
        .def(py::init<>())
        .def_readonly(
            "synaptic_trace",
            &NEAT::EPropConnectionState::synaptic_trace)
        .def_readonly(
            "voltage_eligibility",
            &NEAT::EPropConnectionState::voltage_eligibility)
        .def_readonly(
            "adaptation_eligibility",
            &NEAT::EPropConnectionState::
                adaptation_eligibility)
        .def_readonly(
            "readout_eligibility",
            &NEAT::EPropConnectionState::readout_eligibility)
        .def_readonly(
            "gradient",
            &NEAT::EPropConnectionState::gradient)
        .def_readonly(
            "first_moment",
            &NEAT::EPropConnectionState::first_moment)
        .def_readonly(
            "second_moment",
            &NEAT::EPropConnectionState::second_moment);

    py::class_<NEAT::EPropStepResult>(
        m, "EPropStepResult")
        .def(py::init<>())
        .def_readonly(
            "outputs", &NEAT::EPropStepResult::outputs)
        .def_readonly("loss", &NEAT::EPropStepResult::loss)
        .def_readonly(
            "gradient_norm",
            &NEAT::EPropStepResult::gradient_norm)
        .def_readonly(
            "updated_connections",
            &NEAT::EPropStepResult::updated_connections)
        .def_readonly(
            "update_applied",
            &NEAT::EPropStepResult::update_applied);

    py::class_<NEAT::EPropSequenceResult>(
        m, "EPropSequenceResult")
        .def(py::init<>())
        .def_readonly(
            "outputs",
            &NEAT::EPropSequenceResult::outputs)
        .def_readonly(
            "losses", &NEAT::EPropSequenceResult::losses)
        .def_readonly(
            "mean_loss",
            &NEAT::EPropSequenceResult::mean_loss)
        .def_readonly(
            "final_gradient_norm",
            &NEAT::EPropSequenceResult::final_gradient_norm)
        .def_readonly(
            "optimizer_updates",
            &NEAT::EPropSequenceResult::optimizer_updates)
        .def_readonly(
            "updated_connections",
            &NEAT::EPropSequenceResult::updated_connections);

        py::class_<NEAT::NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<bool>(), py::arg("a_Minimal"))
        .def(py::init<>())
        .def("InitRTRLMatrix", &NEAT::NeuralNetwork::InitRTRLMatrix)
        .def("InitSparseRTRLMatrix",
             &NEAT::NeuralNetwork::InitSparseRTRLMatrix)
        .def("ActivateFast", &NEAT::NeuralNetwork::ActivateFast)
        .def("Activate", &NEAT::NeuralNetwork::Activate)
        .def(
            "ActivateSteps",
            &NEAT::NeuralNetwork::ActivateSteps,
            py::arg("steps"),
            py::arg("fast") = true)
        .def("ActivateUseInternalBias", &NEAT::NeuralNetwork::ActivateUseInternalBias)
        .def("ActivateLeaky", &NEAT::NeuralNetwork::ActivateLeaky)
        .def("RTRL_update_gradients",
             &NEAT::NeuralNetwork::RTRL_update_gradients)
        .def("RTRL_update_gradients_sparse",
             &NEAT::NeuralNetwork::RTRL_update_gradients_sparse)
        .def("RTRL_update_error",
             py::overload_cast<double>(
                 &NEAT::NeuralNetwork::RTRL_update_error))
        .def("RTRL_update_error",
             py::overload_cast<const std::vector<double>&, double>(
                 &NEAT::NeuralNetwork::RTRL_update_error),
             py::arg("targets"),
             py::arg("learning_rate") = 0.0001)
        .def(
            "RTRL_update_error_sparse",
            py::overload_cast<double, double>(
                &NEAT::NeuralNetwork::RTRL_update_error_sparse),
            py::arg("target"),
            py::arg("learning_rate") = 0.0001)
        .def(
            "RTRL_update_error_sparse",
            py::overload_cast<const std::vector<double>&, double>(
                &NEAT::NeuralNetwork::RTRL_update_error_sparse),
            py::arg("targets"),
            py::arg("learning_rate") = 0.0001)
        .def("RTRL_update_weights",
             &NEAT::NeuralNetwork::RTRL_update_weights)
        .def("Adapt", &NEAT::NeuralNetwork::Adapt)
        .def("ConnectionExists",
             &NEAT::NeuralNetwork::ConnectionExists)
        .def("Flush", &NEAT::NeuralNetwork::Flush)
        .def("FlushCube", &NEAT::NeuralNetwork::FlushCube)
        .def("Input", &NEAT::NeuralNetwork::Input)
        .def("InputExact", &NEAT::NeuralNetwork::InputExact)
        .def("Output", &NEAT::NeuralNetwork::Output)
        .def(
            "ActivateBatch",
            &NEAT::NeuralNetwork::ActivateBatch,
            py::arg("inputs"),
            py::arg("steps") = 1,
            py::arg("use_internal_bias") = false)
        .def(
            "StepSpiking",
            &NEAT::NeuralNetwork::StepSpiking,
            py::arg("inputs"),
            py::arg("time_step") = -1.0)
        .def(
            "SimulateSpiking",
            &NEAT::NeuralNetwork::SimulateSpiking,
            py::arg("inputs"),
            py::arg("time_step") = -1.0,
            py::arg("reset") = false)
        .def("OutputSpikes", &NEAT::NeuralNetwork::OutputSpikes)
        .def("OutputRates", &NEAT::NeuralNetwork::OutputRates)
        .def(
            "OutputFilteredSpikes",
            &NEAT::NeuralNetwork::OutputFilteredSpikes)
        .def(
            "OutputMembranePotentials",
            &NEAT::NeuralNetwork::OutputMembranePotentials)
        .def(
            "OutputDecoded",
            &NEAT::NeuralNetwork::OutputDecoded)
        .def("IsSpiking", &NEAT::NeuralNetwork::IsSpiking)
        .def("SpikingTime", &NEAT::NeuralNetwork::SpikingTime)
        .def(
            "SpikingTimeStep",
            &NEAT::NeuralNetwork::SpikingTimeStep)
        .def(
            "SetSpikingTimeStep",
            &NEAT::NeuralNetwork::SetSpikingTimeStep)
        .def(
            "SetSpikingInputMode",
            &NEAT::NeuralNetwork::SetSpikingInputMode)
        .def(
            "GetSpikingInputMode",
            &NEAT::NeuralNetwork::GetSpikingInputMode)
        .def(
            "SetSpikingOutputMode",
            &NEAT::NeuralNetwork::SetSpikingOutputMode)
        .def(
            "GetSpikingOutputMode",
            &NEAT::NeuralNetwork::GetSpikingOutputMode)
        .def(
            "SeedSpiking",
            &NEAT::NeuralNetwork::SeedSpiking)
        .def(
            "EnableSpikeRecording",
            &NEAT::NeuralNetwork::EnableSpikeRecording,
            py::arg("enabled"),
            py::arg("max_events") = 100000)
        .def(
            "GetSpikeHistory",
            [](const NEAT::NeuralNetwork& network)
            {
                return network.GetSpikeHistory();
            })
        .def(
            "ClearSpikeHistory",
            &NEAT::NeuralNetwork::ClearSpikeHistory)
        .def("EnableSTDP", &NEAT::NeuralNetwork::EnableSTDP)
        .def(
            "SparseRTRLStateSize",
            &NEAT::NeuralNetwork::SparseRTRLStateSize)
        .def("AddNeuron", &NEAT::NeuralNetwork::AddNeuron)
        .def("AddConnection", &NEAT::NeuralNetwork::AddConnection)
        .def("GetNeuronByIndex", &NEAT::NeuralNetwork::GetNeuronByIndex)
        .def("GetConnectionByIndex", &NEAT::NeuralNetwork::GetConnectionByIndex)
        .def("SetInputOutputDimentions",
             &NEAT::NeuralNetwork::SetInputOutputDimentions)
        .def("SetInputOutputDimensions",
             &NEAT::NeuralNetwork::SetInputOutputDimensions)
        .def("NumInputs", &NEAT::NeuralNetwork::NumInputs)
        .def("NumOutputs", &NEAT::NeuralNetwork::NumOutputs)
        .def("GetConnectionLenght",
             &NEAT::NeuralNetwork::GetConnectionLenght)
        .def("GetConnectionLength",
             &NEAT::NeuralNetwork::GetConnectionLength)
        .def("GetTotalConnectionLength",
             &NEAT::NeuralNetwork::GetTotalConnectionLength)
        .def("Save", (void (NEAT::NeuralNetwork::*)(const char*)) &NEAT::NeuralNetwork::Save)
        .def("Load", (bool (NEAT::NeuralNetwork::*)(const char*)) &NEAT::NeuralNetwork::Load)
        .def("Serialize", &NEAT::NeuralNetwork::Serialize)
        .def_static("Deserialize", &NEAT::NeuralNetwork::Deserialize)
        .def_readwrite("m_num_inputs", &NEAT::NeuralNetwork::m_num_inputs)
        .def_readwrite("m_num_outputs", &NEAT::NeuralNetwork::m_num_outputs)
        .def_readwrite("m_connections", &NEAT::NeuralNetwork::m_connections)
        .def_readwrite("m_neurons", &NEAT::NeuralNetwork::m_neurons)
        .def(py::pickle(
            [](const NEAT::NeuralNetwork &nn) -> std::string {
                return nn.Serialize();
            },
            [](const std::string &state) {
                return NEAT::NeuralNetwork::Deserialize(state);
            }
        ));

        py::class_<NEAT::EPropLearner>(m, "EPropLearner")
        .def(py::init<>())
        .def(
            py::init<const NEAT::EPropConfig&>(),
            py::arg("config"))
        .def(
            "Initialize",
            &NEAT::EPropLearner::Initialize,
            py::arg("network"))
        .def(
            "RefreshFeedback",
            &NEAT::EPropLearner::RefreshFeedback,
            py::arg("network"))
        .def(
            "IsInitialized",
            &NEAT::EPropLearner::IsInitialized)
        .def(
            "ResetEligibility",
            &NEAT::EPropLearner::ResetEligibility)
        .def(
            "ResetOptimizer",
            &NEAT::EPropLearner::ResetOptimizer)
        .def(
            "ZeroGradients",
            &NEAT::EPropLearner::ZeroGradients)
        .def(
            "TrainStep",
            &NEAT::EPropLearner::TrainStep,
            py::arg("network"),
            py::arg("inputs"),
            py::arg("targets"),
            py::arg("time_step") = -1.0)
        .def(
            "TrainStepWithSignals",
            &NEAT::EPropLearner::TrainStepWithSignals,
            py::arg("network"),
            py::arg("inputs"),
            py::arg("learning_signals"),
            py::arg("time_step") = -1.0)
        .def(
            "TrainSequence",
            &NEAT::EPropLearner::TrainSequence,
            py::arg("network"),
            py::arg("inputs"),
            py::arg("targets"),
            py::arg("time_step") = -1.0,
            py::arg("reset_network") = true,
            py::arg("apply_final_update") = true)
        .def(
            "AccumulateLearningSignals",
            &NEAT::EPropLearner::AccumulateLearningSignals,
            py::arg("network"),
            py::arg("learning_signals"),
            py::arg("time_step") = -1.0)
        .def(
            "ApplyGradients",
            &NEAT::EPropLearner::ApplyGradients,
            py::arg("network"))
        .def(
            "ConnectionStates",
            &NEAT::EPropLearner::ConnectionStates)
        .def(
            "FeedbackMatrix",
            &NEAT::EPropLearner::FeedbackMatrix)
        .def(
            "OptimizerStep",
            &NEAT::EPropLearner::OptimizerStep)
        .def(
            "AccumulatedSteps",
            &NEAT::EPropLearner::AccumulatedSteps)
        .def("Serialize", &NEAT::EPropLearner::Serialize)
        .def_static(
            "Deserialize",
            &NEAT::EPropLearner::Deserialize)
        .def_readwrite(
            "m_config",
            &NEAT::EPropLearner::m_config)
        .def(py::pickle(
            [](const NEAT::EPropLearner& learner)
            {
                return learner.Serialize();
            },
            [](const std::string& state)
            {
                return NEAT::EPropLearner::Deserialize(state);
            }));

        py::class_<NEAT::Parameters>(m, "Parameters")
            .def(py::init<>())
            // Methods
            .def("Load", (int (NEAT::Parameters::*)(const char*)) &NEAT::Parameters::Load, py::arg("filename"))
            .def("LoadFromStream", (int (NEAT::Parameters::*)(std::ifstream&)) &NEAT::Parameters::Load, py::arg("dataFile"))
            .def("Save", (void (NEAT::Parameters::*)(const char*)) &NEAT::Parameters::Save, py::arg("filename"))
            .def("SaveToStream", (void (NEAT::Parameters::*)(FILE*)) &NEAT::Parameters::Save, py::arg("fstream"))
            .def("Reset", &NEAT::Parameters::Reset)
            .def(
                "ConfigureSpiking",
                &NEAT::Parameters::ConfigureSpiking,
                py::arg("enable_stdp") = false)
            .def("Serialize", &NEAT::Parameters::Serialize)
            .def_static("Deserialize", &NEAT::Parameters::Deserialize)
            .def("Validate",
                 [](const NEAT::Parameters& parameters) {
                     std::string error;
                     return py::make_tuple(
                         parameters.Validate(&error), error);
                 })
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
                .def_property(
                    "CustomConstraints",
                    [](const NEAT::Parameters &parameters) {
                        return parameters.GetCustomConstraintsFunction();
                    },
                    [](NEAT::Parameters &parameters,
                       std::function<bool(NEAT::Genome&)> callback) {
                        parameters.SetCustomConstraintsFunction(
                            std::move(callback));
                    })
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
            .def_readwrite("TruncationSelection", &NEAT::Parameters::TruncationSelection)
            .def_readwrite("RouletteWheelSelection", &NEAT::Parameters::RouletteWheelSelection)
            .def_readwrite("TournamentSelection", &NEAT::Parameters::TournamentSelection)
            .def_readwrite("TournamentSize", &NEAT::Parameters::TournamentSize)
            .def_readwrite("EliteFraction", &NEAT::Parameters::EliteFraction)
            .def_readwrite("Elitism", &NEAT::Parameters::Elitism)
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
            .def_readwrite("ActivationFunction_SpikingLIF_Prob", &NEAT::Parameters::ActivationFunction_SpikingLIF_Prob)
            .def_readwrite("ActivationFunction_SpikingAdaptiveLIF_Prob", &NEAT::Parameters::ActivationFunction_SpikingAdaptiveLIF_Prob)
            .def_readwrite("ActivationFunction_SpikingIzhikevich_Prob", &NEAT::Parameters::ActivationFunction_SpikingIzhikevich_Prob)
            .def_readwrite("MutateNeuronTimeConstantsProb", &NEAT::Parameters::MutateNeuronTimeConstantsProb)
            .def_readwrite("MutateNeuronBiasesProb", &NEAT::Parameters::MutateNeuronBiasesProb)
            .def_readwrite("MinNeuronTimeConstant", &NEAT::Parameters::MinNeuronTimeConstant)
            .def_readwrite("MaxNeuronTimeConstant", &NEAT::Parameters::MaxNeuronTimeConstant)
            .def_readwrite("MinNeuronBias", &NEAT::Parameters::MinNeuronBias)
            .def_readwrite("MaxNeuronBias", &NEAT::Parameters::MaxNeuronBias)
            .def_readwrite("MutateNeuronSpikingParametersProb", &NEAT::Parameters::MutateNeuronSpikingParametersProb)
            .def_readwrite("MutateLinkSpikingParametersProb", &NEAT::Parameters::MutateLinkSpikingParametersProb)
            .def_readwrite("SpikingParameterMutationRate", &NEAT::Parameters::SpikingParameterMutationRate)
            .def_readwrite("SpikingParameterMutationPower", &NEAT::Parameters::SpikingParameterMutationPower)
            .def_readwrite("MinSpikingTimeConstant", &NEAT::Parameters::MinSpikingTimeConstant)
            .def_readwrite("MaxSpikingTimeConstant", &NEAT::Parameters::MaxSpikingTimeConstant)
            .def_readwrite("MinSpikeThreshold", &NEAT::Parameters::MinSpikeThreshold)
            .def_readwrite("MaxSpikeThreshold", &NEAT::Parameters::MaxSpikeThreshold)
            .def_readwrite("MinResetPotential", &NEAT::Parameters::MinResetPotential)
            .def_readwrite("MaxResetPotential", &NEAT::Parameters::MaxResetPotential)
            .def_readwrite("MinRestingPotential", &NEAT::Parameters::MinRestingPotential)
            .def_readwrite("MaxRestingPotential", &NEAT::Parameters::MaxRestingPotential)
            .def_readwrite("MinRefractoryPeriod", &NEAT::Parameters::MinRefractoryPeriod)
            .def_readwrite("MaxRefractoryPeriod", &NEAT::Parameters::MaxRefractoryPeriod)
            .def_readwrite("MinMembraneResistance", &NEAT::Parameters::MinMembraneResistance)
            .def_readwrite("MaxMembraneResistance", &NEAT::Parameters::MaxMembraneResistance)
            .def_readwrite("MinAdaptationTimeConstant", &NEAT::Parameters::MinAdaptationTimeConstant)
            .def_readwrite("MaxAdaptationTimeConstant", &NEAT::Parameters::MaxAdaptationTimeConstant)
            .def_readwrite("MinAdaptationIncrement", &NEAT::Parameters::MinAdaptationIncrement)
            .def_readwrite("MaxAdaptationIncrement", &NEAT::Parameters::MaxAdaptationIncrement)
            .def_readwrite("MinSpikeRateTimeConstant", &NEAT::Parameters::MinSpikeRateTimeConstant)
            .def_readwrite("MaxSpikeRateTimeConstant", &NEAT::Parameters::MaxSpikeRateTimeConstant)
            .def_readwrite("MinIzhikevichA", &NEAT::Parameters::MinIzhikevichA)
            .def_readwrite("MaxIzhikevichA", &NEAT::Parameters::MaxIzhikevichA)
            .def_readwrite("MinIzhikevichThreshold", &NEAT::Parameters::MinIzhikevichThreshold)
            .def_readwrite("MaxIzhikevichThreshold", &NEAT::Parameters::MaxIzhikevichThreshold)
            .def_readwrite("MinIzhikevichB", &NEAT::Parameters::MinIzhikevichB)
            .def_readwrite("MaxIzhikevichB", &NEAT::Parameters::MaxIzhikevichB)
            .def_readwrite("MinIzhikevichC", &NEAT::Parameters::MinIzhikevichC)
            .def_readwrite("MaxIzhikevichC", &NEAT::Parameters::MaxIzhikevichC)
            .def_readwrite("MinIzhikevichD", &NEAT::Parameters::MinIzhikevichD)
            .def_readwrite("MaxIzhikevichD", &NEAT::Parameters::MaxIzhikevichD)
            .def_readwrite("MinSynapticDelay", &NEAT::Parameters::MinSynapticDelay)
            .def_readwrite("MaxSynapticDelay", &NEAT::Parameters::MaxSynapticDelay)
            .def_readwrite("MinSynapticTimeConstant", &NEAT::Parameters::MinSynapticTimeConstant)
            .def_readwrite("MaxSynapticTimeConstant", &NEAT::Parameters::MaxSynapticTimeConstant)
            .def_readwrite("InitialSTDPEnabledProb", &NEAT::Parameters::InitialSTDPEnabledProb)
            .def_readwrite("MinSTDPPlus", &NEAT::Parameters::MinSTDPPlus)
            .def_readwrite("MaxSTDPPlus", &NEAT::Parameters::MaxSTDPPlus)
            .def_readwrite("MinSTDPMinus", &NEAT::Parameters::MinSTDPMinus)
            .def_readwrite("MaxSTDPMinus", &NEAT::Parameters::MaxSTDPMinus)
            .def_readwrite("MinSTDPTau", &NEAT::Parameters::MinSTDPTau)
            .def_readwrite("MaxSTDPTau", &NEAT::Parameters::MaxSTDPTau)
            // Speciation parameters
            .def_readwrite("DisjointCoeff", &NEAT::Parameters::DisjointCoeff)
            .def_readwrite("ExcessCoeff", &NEAT::Parameters::ExcessCoeff)
            .def_readwrite("ActivationADiffCoeff", &NEAT::Parameters::ActivationADiffCoeff)
            .def_readwrite("ActivationBDiffCoeff", &NEAT::Parameters::ActivationBDiffCoeff)
            .def_readwrite("WeightDiffCoeff", &NEAT::Parameters::WeightDiffCoeff)
            .def_readwrite("TimeConstantDiffCoeff", &NEAT::Parameters::TimeConstantDiffCoeff)
            .def_readwrite("BiasDiffCoeff", &NEAT::Parameters::BiasDiffCoeff)
            .def_readwrite("ActivationFunctionDiffCoeff", &NEAT::Parameters::ActivationFunctionDiffCoeff)
            .def_readwrite("SpikingNeuronDiffCoeff", &NEAT::Parameters::SpikingNeuronDiffCoeff)
            .def_readwrite("SpikingLinkDiffCoeff", &NEAT::Parameters::SpikingLinkDiffCoeff)
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
            .def_readwrite("ParentSelectionMode", &NEAT::Parameters::ParentSelectionMode)
            .def_readwrite("RankSelectionPressure", &NEAT::Parameters::RankSelectionPressure)
            .def_readwrite("RankSelectionExponent", &NEAT::Parameters::RankSelectionExponent)
            .def_readwrite("BoltzmannTemperature", &NEAT::Parameters::BoltzmannTemperature)
            .def_readwrite("SinglePointCrossoverRate", &NEAT::Parameters::SinglePointCrossoverRate)
            .def_readwrite("BlendCrossoverRate", &NEAT::Parameters::BlendCrossoverRate)
            .def_readwrite("SimulatedBinaryCrossoverRate", &NEAT::Parameters::SimulatedBinaryCrossoverRate)
            .def_readwrite("CrossoverBlendAlpha", &NEAT::Parameters::CrossoverBlendAlpha)
            .def_readwrite("CrossoverSBXEta", &NEAT::Parameters::CrossoverSBXEta)
            .def_readwrite("WeightMutationDistribution", &NEAT::Parameters::WeightMutationDistribution)
            .def_readwrite("WeightMutationSigma", &NEAT::Parameters::WeightMutationSigma)
            .def_readwrite("WeightMutationCauchyScale", &NEAT::Parameters::WeightMutationCauchyScale)
            .def_readwrite("WeightMutationPolynomialEta", &NEAT::Parameters::WeightMutationPolynomialEta)
            .def_readwrite("SpeciesRepresentativeSelection", &NEAT::Parameters::SpeciesRepresentativeSelection)
            .def_readwrite("RepresentativeSelectionCandidates", &NEAT::Parameters::RepresentativeSelectionCandidates)
            .def_readwrite("OffspringAllocation", &NEAT::Parameters::OffspringAllocation)
            .def_readwrite("MinSpeciesSize", &NEAT::Parameters::MinSpeciesSize)
            .def_readwrite("SpeciesElitism", &NEAT::Parameters::SpeciesElitism)
            .def_readwrite("StagnationPenalty", &NEAT::Parameters::StagnationPenalty)
            .def_readwrite("CompatibilityThresholdControl", &NEAT::Parameters::CompatibilityThresholdControl)
            .def_readwrite("TargetSpecies", &NEAT::Parameters::TargetSpecies)
            .def_readwrite("CompatibilityThresholdGain", &NEAT::Parameters::CompatibilityThresholdGain)
            .def_readwrite("MaxCompatTreshold", &NEAT::Parameters::MaxCompatTreshold)
            .def_readwrite("RequireEvaluatedGenomes", &NEAT::Parameters::RequireEvaluatedGenomes)
            .def_readwrite("RejectNonFiniteFitness", &NEAT::Parameters::RejectNonFiniteFitness)
            .def_readwrite("MutationOperatorsPerOffspring", &NEAT::Parameters::MutationOperatorsPerOffspring)
            .def_readwrite("AdaptiveMutationStart", &NEAT::Parameters::AdaptiveMutationStart)
            .def_readwrite("AdaptiveMutationRate", &NEAT::Parameters::AdaptiveMutationRate)
            .def_readwrite("AdaptiveMutationMaxFactor", &NEAT::Parameters::AdaptiveMutationMaxFactor)
            .def_readwrite("FitnessScaling", &NEAT::Parameters::FitnessScaling)
            .def_readwrite("FitnessRankPressure", &NEAT::Parameters::FitnessRankPressure)
            .def_readwrite("FitnessSigmaScale", &NEAT::Parameters::FitnessSigmaScale)
            .def_readwrite("FitnessBoltzmannTemperature", &NEAT::Parameters::FitnessBoltzmannTemperature)
            .def(py::pickle(
                [](const NEAT::Parameters &parameters) {
                    return parameters.Serialize();
                },
                [](const std::string &state) {
                    return NEAT::Parameters::Deserialize(state);
                }
            ));
    //};

    // ========================
    // Bindings for PhenotypeBehavior
    // ========================

    py::class_<
        NEAT::PhenotypeBehavior,
        PyPhenotypeBehavior,
        std::shared_ptr<NEAT::PhenotypeBehavior>>(m, "PhenotypeBehavior")
        .def(py::init<>())
        .def("Acquire", &NEAT::PhenotypeBehavior::Acquire)
        .def("Distance_To", &NEAT::PhenotypeBehavior::Distance_To)
        .def("Successful", &NEAT::PhenotypeBehavior::Successful)
        .def_readwrite("m_Data", &NEAT::PhenotypeBehavior::m_Data);

    // ========================
    // Bindings for Population
    // ========================

    py::class_<NEAT::Population>(m, "Population")
        .def(py::init<>())
        .def(py::init<const NEAT::Genome&, const NEAT::Parameters&, bool, double, int>(),
            py::arg("genome"), py::arg("parameters"), py::arg("randomizeWeights"), py::arg("randomizationRange"), py::arg("rng_seed"))
        .def(py::init<const std::string>())
        .def("GetGeneration", &NEAT::Population::GetGeneration)
        .def("NumGenomes", &NEAT::Population::NumGenomes)
        .def("GetSearchMode", &NEAT::Population::GetSearchMode)
        .def("GetCurrentMPC", &NEAT::Population::GetCurrentMPC)
        .def("GetBaseMPC", &NEAT::Population::GetBaseMPC)
        .def("GetBestFitnessEver", &NEAT::Population::GetBestFitnessEver)
        .def("GetBestGenome", &NEAT::Population::GetBestGenome)
        .def("Validate",
             [](const NEAT::Population& population) {
                 std::string error;
                 return py::make_tuple(
                     population.Validate(&error), error);
             })
        .def("GetStagnation", &NEAT::Population::GetStagnation)
        .def("GetMPCStagnation", &NEAT::Population::GetMPCStagnation)
        .def("GetNextGenomeID", &NEAT::Population::GetNextGenomeID)
        .def("GetNextSpeciesID", &NEAT::Population::GetNextSpeciesID)
        .def("SameGenomeIDCheck", &NEAT::Population::SameGenomeIDCheck)
        .def("AccessGenomeByIndex", &NEAT::Population::AccessGenomeByIndex, py::return_value_policy::reference)
        .def("AccessGenomeByID", &NEAT::Population::AccessGenomeByID, py::return_value_policy::reference)
        .def("Epoch", &NEAT::Population::Epoch)
        .def("Save", &NEAT::Population::Save)
        .def("SaveState", &NEAT::Population::SaveState)
        .def("Tick", &NEAT::Population::Tick, py::return_value_policy::reference)
        .def("NoveltySearchTick", &NEAT::Population::NoveltySearchTick)
        .def(
            "InitPhenotypeBehaviorData",
            py::overload_cast<
                const std::vector<
                    std::shared_ptr<NEAT::PhenotypeBehavior>>&>(
                &NEAT::Population::InitPhenotypeBehaviorData))
        .def(
            "GetBehaviorArchive",
            &NEAT::Population::GetBehaviorArchive)
        .def("Serialize", &NEAT::Population::Serialize)
        .def_static("Deserialize", &NEAT::Population::Deserialize)
        .def_readwrite("m_GenomeArchive", &NEAT::Population::m_GenomeArchive)
        .def_readwrite("m_RNG", &NEAT::Population::m_RNG)
        .def_readwrite("m_Parameters", &NEAT::Population::m_Parameters)
        .def_readwrite("m_Generation", &NEAT::Population::m_Generation)
        .def_readwrite("m_Species", &NEAT::Population::m_Species)
        .def_readwrite("m_ID", &NEAT::Population::m_ID)
        .def_readwrite("m_NumEvaluations", &NEAT::Population::m_NumEvaluations)
        .def(py::pickle(
            [](const NEAT::Population &pop) -> std::string { return pop.Serialize(); },
            [](const std::string &state) { return NEAT::Population::Deserialize(state); }
        ));

    // ========================
    // Bindings for Species
    // ========================

    py::class_<NEAT::Species>(m, "Species")
        .def(py::init<const NEAT::Genome&, const NEAT::Parameters&, int>(),
             py::arg("seed"), py::arg("parameters"), py::arg("id"))
        .def("GetBestFitness", &NEAT::Species::GetBestFitness)
        .def("NumIndividuals",
             py::overload_cast<>(&NEAT::Species::NumIndividuals))
        .def("ID", py::overload_cast<>(&NEAT::Species::ID))
        .def("GensNoImprovement", &NEAT::Species::GensNoImprovement)
        .def("EvalsNoImprovement", &NEAT::Species::EvalsNoImprovement)
        .def("AgeGens", &NEAT::Species::AgeGens)
        .def("AgeEvals", &NEAT::Species::AgeEvals)
        .def("GetIndividualByIdx", &NEAT::Species::GetIndividualByIdx)
        .def("IsBestSpecies", &NEAT::Species::IsBestSpecies)
        .def("IsWorstSpecies", &NEAT::Species::IsWorstSpecies)
        .def("NumEvaluated",
             py::overload_cast<>(&NEAT::Species::NumEvaluated))
        .def("GetLeader", &NEAT::Species::GetLeader, py::return_value_policy::reference)
        .def("GetRepresentative", &NEAT::Species::GetRepresentative, py::return_value_policy::reference)
        .def("GetIndividual", &NEAT::Species::GetIndividual, py::return_value_policy::reference)
        .def("GetRandomIndividual", &NEAT::Species::GetRandomIndividual, py::return_value_policy::reference)
        .def("Serialize", &NEAT::Species::Serialize)
        .def_static("Deserialize", &NEAT::Species::Deserialize)
        .def_readwrite("m_BestGenome", &NEAT::Species::m_BestGenome)
        .def_readwrite("m_GensNoImprovement", &NEAT::Species::m_GensNoImprovement)
        .def_readwrite("m_EvalsNoImprovement", &NEAT::Species::m_EvalsNoImprovement)
        .def_readwrite("m_R", &NEAT::Species::m_R)
        .def_readwrite("m_G", &NEAT::Species::m_G)
        .def_readwrite("m_B", &NEAT::Species::m_B)
        .def_readwrite("m_AverageFitness", &NEAT::Species::m_AverageFitness)
        .def_readwrite("m_Individuals", &NEAT::Species::m_Individuals)
        .def(py::pickle(
            [](const NEAT::Species &s) -> std::string { return s.Serialize(); },
            [](const std::string &state) { return NEAT::Species::Deserialize(state); }
        ));

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
        .def(
            "RandNormal",
            &NEAT::RNG::RandNormal,
            py::arg("mean") = 0.0,
            py::arg("standard_deviation") = 1.0)
        .def(
            "RandCauchy",
            &NEAT::RNG::RandCauchy,
            py::arg("location") = 0.0,
            py::arg("scale") = 1.0)
        .def("Roulette", &NEAT::RNG::Roulette)
        .def("Serialize", &NEAT::RNG::Serialize)
        .def("Deserialize", &NEAT::RNG::Deserialize)
        .def(py::pickle(
            [](const NEAT::RNG &rng) { return rng.Serialize(); },
            [](const std::string &state) {
                NEAT::RNG rng;
                rng.Deserialize(state);
                return rng;
            }
        ));

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
