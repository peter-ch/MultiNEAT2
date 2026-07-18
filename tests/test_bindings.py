import pickle
import tempfile
from pathlib import Path

import pymultineat as neat


def make_genome(parameters: neat.Parameters) -> neat.Genome:
    init = neat.GenomeInitStruct()
    init.NumInputs = 3
    init.NumOutputs = 1
    return neat.Genome(parameters, init)


parameters = neat.Parameters()
parameters.PopulationSize = 4
parameters.MutateAddNeuronProb = 0.42
parameters.DivisionThreshold = 0.77
parameters.Elitism = 0.125
assert parameters.EliteFraction == 0.125
restored_parameters = pickle.loads(pickle.dumps(parameters))
assert restored_parameters.PopulationSize == 4
assert restored_parameters.MutateAddNeuronProb == 0.42
assert restored_parameters.DivisionThreshold == 0.77
assert parameters.Validate() == (True, "")

advanced = neat.Parameters()
advanced.ParentSelectionMode = neat.SelectionMode.RANK_LINEAR
advanced.RankSelectionPressure = 1.8
advanced.MultipointCrossoverRate = 0.25
advanced.SinglePointCrossoverRate = 0.25
advanced.BlendCrossoverRate = 0.25
advanced.SimulatedBinaryCrossoverRate = 0.25
advanced.WeightMutationDistribution = neat.WeightMutationMode.GAUSSIAN_MUTATION
advanced.SpeciesRepresentativeSelection = neat.MEDOID_REPRESENTATIVE
advanced.RepresentativeSelectionCandidates = 4
advanced.OffspringAllocation = neat.STOCHASTIC_REMAINDER
advanced.MinSpeciesSize = 2
advanced.SpeciesElitism = 2
advanced.CompatibilityThresholdControl = neat.PROPORTIONAL_COMPATIBILITY_THRESHOLD
advanced.TargetSpecies = 4
advanced.MutationOperatorsPerOffspring = 2.0
advanced.FitnessScaling = neat.SIGMA_FITNESS_SCALING
advanced.FitnessRankPressure = 1.7
advanced.FitnessSigmaScale = 2.5
advanced.FitnessBoltzmannTemperature = 0.75
advanced_round_trip = neat.Parameters.Deserialize(advanced.Serialize())
assert advanced_round_trip.ParentSelectionMode == neat.RANK_LINEAR
assert advanced_round_trip.SinglePointCrossoverRate == 0.25
assert advanced_round_trip.WeightMutationDistribution == neat.GAUSSIAN_MUTATION
assert advanced_round_trip.SpeciesRepresentativeSelection == neat.MEDOID_REPRESENTATIVE
assert advanced_round_trip.OffspringAllocation == neat.STOCHASTIC_REMAINDER
assert advanced_round_trip.MutationOperatorsPerOffspring == 2.0
assert advanced_round_trip.FitnessScaling == neat.SIGMA_FITNESS_SCALING
assert advanced_round_trip.FitnessSigmaScale == 2.5
assert advanced.Validate() == (True, "")

genome = make_genome(parameters)
genome.SetFitness(-7.5)
genome.SetEvaluated()
genome.SetNeuronXY(0, 11, 22)
restored_genome = pickle.loads(pickle.dumps(genome))
assert restored_genome.GetFitness() == -7.5
assert restored_genome.IsEvaluated()
assert restored_genome.GetNeuronByIndex(0).x == 11

species = neat.Species(genome, parameters, 7)
restored_species = pickle.loads(pickle.dumps(species))
assert restored_species.ID() == 7
assert restored_species.NumIndividuals() == 1

population = neat.Population(genome, parameters, True, 1.0, 123)
population.m_Species[0].m_Individuals[0].SetFitness(-3.0)
restored_population = pickle.loads(pickle.dumps(population))
assert len(restored_population.m_Species) == len(population.m_Species)
assert restored_population.GetBestGenome().GetFitness() == -3.0
assert neat.Population.Deserialize(population.Serialize()).NumGenomes() == 4
with tempfile.TemporaryDirectory() as temporary_directory:
    checkpoint = Path(temporary_directory) / "population.state"
    population.SaveState(str(checkpoint))
    assert neat.Population(str(checkpoint)).NumGenomes() == 4

network = neat.NeuralNetwork()
assert neat.NeuralNetwork().m_neurons == []
assert len(neat.NeuralNetwork(False).m_neurons) == 5
input_neuron = neat.Neuron()
input_neuron.m_type = neat.NeuronType.INPUT
output_neuron = neat.Neuron()
output_neuron.m_type = neat.NeuronType.OUTPUT
output_neuron.m_activation_function_type = neat.ActivationFunction.LINEAR
connection = neat.Connection()
connection.m_source_neuron_idx = 0
connection.m_target_neuron_idx = 1
connection.m_weight = 2.0
network.AddNeuron(input_neuron)
network.AddNeuron(output_neuron)
network.AddConnection(connection)
network.SetInputOutputDimentions(1, 1)
network.Input([3.0])
network.Activate()
assert network.Output() == [6.0]
network.Flush()
network.Input([3.0])
network.ActivateFast()
assert network.Output() == [6.0]
network.Flush()
network.Input([3.0])
network.ActivateSteps(2, False)
assert network.Output() == [6.0]
assert pickle.loads(pickle.dumps(network)).Output() == [6.0]
assert network.ActivateBatch([[1.0], [2.0], [-1.0]]) == [
    [2.0],
    [4.0],
    [-2.0],
]

trainable = neat.NeuralNetwork()
input_neuron.m_activation = 0.0
output_neuron.m_a = 2.0
connection.m_weight = 0.5
trainable.AddNeuron(input_neuron)
trainable.AddNeuron(output_neuron)
trainable.AddConnection(connection)
trainable.SetInputOutputDimensions(1, 1)
trainable.InitRTRLMatrix()
trainable.Input([1.0])
trainable.Activate()
trainable.RTRL_update_gradients()
trainable.RTRL_update_error([1.0], 0.25)
old_weight = trainable.m_connections[0].m_weight
trainable.RTRL_update_weights()
assert trainable.m_connections[0].m_weight != old_weight
trainable.InitSparseRTRLMatrix()
trainable.Flush()
trainable.InputExact([1.0])
trainable.Activate()
trainable.RTRL_update_gradients_sparse()
trainable.RTRL_update_error_sparse([1.0], 0.25)
old_weight = trainable.m_connections[0].m_weight
trainable.RTRL_update_weights()
assert trainable.m_connections[0].m_weight != old_weight
assert trainable.SparseRTRLStateSize() == (
    len(trainable.m_neurons) * len(trainable.m_connections)
)
assert (
    pickle.loads(pickle.dumps(trainable)).SparseRTRLStateSize()
    == trainable.SparseRTRLStateSize()
)

dad = make_genome(parameters)
dad.SetID(genome.GetID() + 1)
genome.SetFitness(2.0)
dad.SetFitness(1.0)
mom_links = genome.m_LinkGenes
dad_links = dad.m_LinkGenes
for link in mom_links:
    link.SetWeight(2.0)
for link in dad_links:
    link.SetWeight(-2.0)
genome.m_LinkGenes = mom_links
dad.m_LinkGenes = dad_links
rng = neat.RNG()
rng.Seed(42)
for mode in (
    neat.SINGLE_POINT,
    neat.BLEND,
    neat.SIMULATED_BINARY,
):
    child = genome.MateWithMode(dad, mode, False, rng, parameters)
    assert child.Validate() == (True, "")

for mode in (
    neat.UNIFORM_MUTATION,
    neat.GAUSSIAN_MUTATION,
    neat.CAUCHY_MUTATION,
    neat.POLYNOMIAL_MUTATION,
):
    mutation_parameters = neat.Parameters()
    mutation_parameters.WeightMutationDistribution = mode
    mutation_parameters.MutateWeightsSevereProb = 0.0
    mutation_parameters.WeightMutationRate = 1.0
    mutation_parameters.WeightReplacementRate = 0.0
    mutation_parameters.WeightMutationMaxPower = 0.5
    candidate = make_genome(mutation_parameters)
    assert candidate.Mutate_LinkWeights(mutation_parameters, rng)
    assert all(
        mutation_parameters.MinWeight
        <= link.GetWeight()
        <= mutation_parameters.MaxWeight
        for link in candidate.m_LinkGenes
    )

assert isinstance(rng.RandNormal(), float)
assert isinstance(rng.RandCauchy(), float)

es_parameters = neat.Parameters()
es_parameters.InitialDepth = 1
es_parameters.MaxDepth = 1
es_parameters.IterationLevel = 0
es_parameters.DivisionThreshold = 0.0
es_parameters.VarianceThreshold = 10.0
es_parameters.BandThreshold = 0.25
es_init = neat.GenomeInitStruct()
es_init.NumInputs = 5
es_init.NumOutputs = 1
es_init.OutputActType = neat.ActivationFunction.LINEAR
es_cppn = neat.Genome(es_parameters, es_init)
es_neurons = es_cppn.m_NeuronGenes
es_links = es_cppn.m_LinkGenes
for link in es_links:
    link.SetWeight(
        1.0 if link.FromNeuronID() in {es_neurons[0].ID(), es_neurons[2].ID()} else 0.0
    )
es_cppn.m_LinkGenes = es_links
es_substrate = neat.Substrate()
es_substrate.m_input_coords = [[-1.0, 0.0], [0.0, 0.0]]
es_substrate.m_output_coords = [[1.0, 0.0]]
es_substrate.m_query_weights_only = True
es_substrate.m_max_weight_and_bias = 1.0
es_network = neat.NeuralNetwork()
es_cppn.BuildESHyperNEATPhenotype(es_network, es_substrate, es_parameters)
assert len(es_network.m_neurons) == 7
assert len(es_network.m_connections) == 12

first = neat.Parameters()
second = neat.Parameters()
first.CustomConstraints = lambda candidate: True
second.CustomConstraints = lambda candidate: False
assert first.CustomConstraints(genome) is True
assert second.CustomConstraints(genome) is False


class Behavior(neat.PhenotypeBehavior):
    def __init__(self, value: float):
        super().__init__()
        self.acquire_calls = 0
        self.m_Data = [[value]]

    def Acquire(self, candidate: neat.Genome) -> bool:
        self.acquire_calls += 1
        self.m_Data = [[float(candidate.GetID())]]
        return False

    def Distance_To(self, other: neat.PhenotypeBehavior) -> float:
        if not self.m_Data or not other.m_Data:
            return 0.0
        return abs(self.m_Data[0][0] - other.m_Data[0][0])


behaviors = [Behavior(float(index)) for index in range(population.NumGenomes())]
population.InitPhenotypeBehaviorData(behaviors)
for index in range(population.NumGenomes()):
    candidate = population.AccessGenomeByIndex(index)
    candidate.SetFitness(float(index + 1))
    candidate.SetEvaluated()
population.m_Parameters.AllowClones = True
population.NoveltySearchTick(neat.Genome())
assert sum(behavior.acquire_calls for behavior in behaviors) == 1
