"""Public Python API ownership, loading, and online-learning regressions."""

import gc
import math
from pathlib import Path
import pickle
import subprocess
import tempfile
import unittest
import weakref

import pymultineat as neat


def network():
    net = neat.NeuralNetwork()
    neuron = neat.Neuron()
    neuron.m_type = neat.INPUT
    net.AddNeuron(neuron)
    net.AddNeuron(neuron)
    neuron.m_type = neat.OUTPUT
    neuron.m_activation_function_type = neat.SPIKING_LIF
    net.AddNeuron(neuron)
    link = neat.Connection()
    link.m_source_neuron_idx = 0
    link.m_target_neuron_idx = 2
    link.m_weight = 0.1
    net.AddConnection(link)
    net.SetInputOutputDimensions(2, 1)
    net.SetSpikingInputMode(neat.POISSON_RATE_INPUT)
    net.Flush()
    return net


class RuntimeSafetyTests(unittest.TestCase):
    def test_borrowed_genomes_keep_their_population_alive(self):
        for method in ("AccessGenomeByIndex", "AccessGenomeByID"):
            with self.subTest(method=method):
                params = neat.Parameters()
                params.PopulationSize = 4
                init = neat.GenomeInitStruct()
                init.NumInputs = 2
                seed = neat.Genome(params, init)
                pop = neat.Population(seed, params, True, 1.0, 42)
                identifier = 0 if method.endswith("Index") else pop.AccessGenomeByIndex(0).GetID()
                candidate = getattr(pop, method)(identifier)
                owner = weakref.ref(pop)
                del pop
                gc.collect()
                self.assertIsNotNone(owner())
                self.assertTrue(candidate.Validate()[0])
                del candidate
                gc.collect()
                self.assertIsNone(owner())

    def test_borrowed_genomes_keep_their_species_alive(self):
        for method in ("GetLeader", "GetRepresentative"):
            with self.subTest(method=method):
                params = neat.Parameters()
                init = neat.GenomeInitStruct()
                init.NumInputs = 2
                seed = neat.Genome(params, init)
                species = neat.Species(seed, params, 1)
                candidate = getattr(species, method)()
                owner = weakref.ref(species)
                del species
                gc.collect()
                self.assertIsNotNone(owner())
                self.assertTrue(candidate.Validate()[0])
                del candidate
                gc.collect()
                self.assertIsNone(owner())

    def test_failed_load_preserves_running_network(self):
        net = network()
        net.StepSpiking([100.0, 100.0])
        before = net.Serialize()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.txt"
            for data in ("", "NNstart 1", "NNstart 1 1\nneuron 1",
                         "NNstart 1 1\nNNend"):
                with self.subTest(data=data):
                    path.write_text(data, encoding="utf-8")
                    self.assertFalse(net.Load(str(path)))
                    self.assertEqual(net.Serialize(), before)

    def test_failed_rng_load_preserves_random_stream(self):
        rng = neat.RNG()
        rng.Seed(42)
        before = rng.Serialize()
        for data in ("1 2 3", before + " trailing"):
            with self.subTest(data=data[:30]):
                with self.assertRaises(ValueError):
                    rng.Deserialize(data)
                self.assertEqual(rng.Serialize(), before)

    def test_rejected_spiking_inputs_preserve_state_and_random_stream(self):
        net = network()
        before = net.Serialize()
        for invalid in (math.nan, math.inf, -1.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    net.StepSpiking([1000.0, invalid])
                self.assertEqual(net.Serialize(), before)

    def test_rejected_learning_signals_preserve_network_and_learner(self):
        net = network()
        learner = neat.EPropLearner()
        learner.Initialize(net)
        before = net.Serialize(), learner.Serialize()
        for signals in ([], [math.nan], [math.inf], [0.0] * 4):
            with self.subTest(signals=signals):
                with self.assertRaises(ValueError):
                    learner.TrainStepWithSignals(net, [100.0, 100.0], signals)
                self.assertEqual((net.Serialize(), learner.Serialize()), before)

    def test_rejected_sequence_does_not_apply_partial_training(self):
        net = network()
        learner = neat.EPropLearner()
        learner.Initialize(net)
        learner.TrainStep(net, [100.0, 100.0], [1.0])
        before = net.Serialize(), learner.Serialize()
        for inputs, targets in (([[100.0, 100.0], [100.0]], [[1.0], [1.0]]),
                                ([[100.0, 100.0]] * 2, [[1.0], [math.nan]]),
                                ([[100.0, 100.0], [100.0, -1.0]], [[1.0]] * 2)):
            with self.subTest(inputs=inputs, targets=targets):
                with self.assertRaises(ValueError):
                    learner.TrainSequence(net, inputs, targets)
                self.assertEqual((net.Serialize(), learner.Serialize()), before)


def small_genome(params=None, inputs=2):
    params = params or neat.Parameters()
    init = neat.GenomeInitStruct()
    init.NumInputs = inputs
    init.NumOutputs = 1
    return neat.Genome(params, init)


def float_schema(minimum=0.0, maximum=1.0):
    schema = neat.TraitParameters()
    details = neat.FloatTraitParameters()
    details.min = minimum
    details.max = maximum
    details.mut_power = 0.1
    details.mut_replace_prob = 0.1
    schema.m_Details = details
    schema.m_MutationProb = 1.0
    schema.m_ImportanceCoeff = 1.0
    return schema


def int_schema(minimum=0, maximum=3):
    schema = neat.TraitParameters()
    details = neat.IntTraitParameters()
    details.min = minimum
    details.max = maximum
    details.mut_power = 1
    details.mut_replace_prob = 0.1
    schema.m_Details = details
    schema.m_MutationProb = 1.0
    schema.m_ImportanceCoeff = 1.0
    return schema


class BindingOwnershipTests(unittest.TestCase):
    """Container/variant members are returned by value, never as views."""

    def test_evaluation_samples_are_independent_copies(self):
        genome = small_genome()
        sample = neat.EvaluationSample()
        sample.case_id = 1
        sample.score = 2.5
        genome.AddEvaluationSample(sample)
        samples = genome.GetEvaluationSamples()
        for case in range(1000):
            extra = neat.EvaluationSample()
            extra.case_id = case + 2
            extra.score = 1.0
            genome.AddEvaluationSample(extra)
        self.assertEqual(samples[0].score, 2.5)
        # Writing through the returned object must not bypass validation.
        samples[0].score = math.nan
        self.assertEqual(genome.GetEvaluationSamples()[0].score, 2.5)

    def test_neuron_and_link_genes_survive_structural_mutation(self):
        params = neat.Parameters()
        genome = small_genome(params)
        neurons = genome.m_NeuronGenes
        links = genome.m_LinkGenes
        ids = [neuron.m_ID for neuron in neurons]
        innovations = [link.m_InnovationID for link in links]
        database = neat.InnovationDatabase()
        database.InitFromGenome(genome)
        rng = neat.RNG()
        rng.Seed(3)
        for _ in range(20):
            genome.Mutate_AddNeuron(database, params, rng)
        self.assertGreater(genome.NumNeurons(), len(ids))
        self.assertEqual([neuron.m_ID for neuron in neurons], ids)
        self.assertEqual([link.m_InnovationID for link in links], innovations)
        # Whole-value assignment still round-trips edits.
        edited = genome.m_LinkGenes
        edited[0].SetWeight(0.625)
        genome.m_LinkGenes = edited
        self.assertEqual(genome.m_LinkGenes[0].GetWeight(), 0.625)

    def test_network_neurons_survive_network_growth(self):
        net = network()
        neurons = net.m_neurons
        connections = net.m_connections
        for _ in range(200):
            net.AddNeuron(neat.Neuron())
        self.assertEqual(neurons[2].m_type, neat.OUTPUT)
        self.assertEqual(connections[0].m_weight, 0.1)

    def test_trait_details_variant_is_a_copy(self):
        schema = neat.TraitParameters()
        details = schema.m_Details
        schema.m_Details = neat.FloatTraitParameters()
        details.min = 5  # writes to an independent copy, not freed storage
        self.assertEqual(schema.type, "float")
        self.assertIsInstance(schema.m_Details, neat.FloatTraitParameters)
        self.assertEqual(schema.m_Details.min, 0.0)
        schema.m_Details = neat.IntSetTraitParameters()
        self.assertEqual(schema.type, "intset")

    def test_trait_value_variant_is_a_copy(self):
        trait = neat.Trait()
        element = neat.intsetelement()
        element.value = 4
        trait.value = element
        held = trait.value
        trait.value = "replaced"
        held.value = 99
        self.assertEqual(trait.value, "replaced")

    def test_set_trait_parameters_return_copies(self):
        details = neat.IntSetTraitParameters()
        element = neat.intsetelement()
        element.value = 1
        details.set = [element]
        stored = details.set
        details.set = []
        stored[0].value = 7
        self.assertEqual(details.set, [])

    def test_parameter_trait_maps_return_copies(self):
        params = neat.Parameters()
        params.NeuronTraits = {"gain": float_schema()}
        view = params.NeuronTraits
        params.NeuronTraits = {}
        view["gain"].m_ImportanceCoeff = 3.0
        view["gain"].m_Details = neat.IntTraitParameters()
        self.assertEqual(params.NeuronTraits, {})

    def test_gene_trait_map_returns_copies(self):
        params = neat.Parameters()
        params.NeuronTraits = {"gain": float_schema()}
        genome = small_genome(params)
        neuron = genome.m_NeuronGenes[0]
        traits = neuron.m_Traits
        neuron.m_Traits = {}
        traits["gain"].value = 0.5
        self.assertEqual(neuron.m_Traits, {})

    def test_qd_archive_is_a_copy(self):
        params = neat.Parameters()
        params.PopulationSize = 4
        pop = neat.Population(small_genome(params), params, True, 1.0, 1)
        self.assertIsInstance(pop.GetQDArchive(), list)

    def test_phenotype_behavior_pointer_is_read_only(self):
        genome = small_genome()
        self.assertIsNone(genome.m_PhenotypeBehavior)
        with self.assertRaises(AttributeError):
            genome.m_PhenotypeBehavior = neat.PhenotypeBehavior()

    def test_innovation_reassignment_rebuilds_index(self):
        database = neat.InnovationDatabase()
        first = database.AddLinkInnovation(1, 2)
        second = database.AddLinkInnovation(3, 4)
        self.assertEqual(database.CheckInnovation(1, 2, neat.NEW_LINK), first)
        database.m_Innovations = [
            neat.Innovation(first, neat.NEW_LINK, 5, 6, neat.NONE, -1),
            neat.Innovation(second, neat.NEW_LINK, 7, 8, neat.NONE, -1),
        ]
        self.assertEqual(database.CheckInnovation(1, 2, neat.NEW_LINK), -1)
        self.assertEqual(database.CheckInnovation(5, 6, neat.NEW_LINK), first)

    def test_uncallable_stream_bindings_are_removed(self):
        self.assertFalse(hasattr(neat.Parameters, "LoadFromStream"))
        self.assertFalse(hasattr(neat.Parameters, "SaveToStream"))
        self.assertFalse(hasattr(neat.InnovationDatabase, "InitFromFile"))


class TraitTypeTests(unittest.TestCase):
    def test_python_int_assigned_to_float_trait_stays_float(self):
        trait = neat.Trait()
        trait.value = 0.25
        trait.value = 1
        self.assertIsInstance(trait.value, float)
        element = neat.floatsetelement()
        element.value = 0.5
        trait.value = element
        trait.value = 2
        self.assertIsInstance(trait.value, neat.floatsetelement)
        self.assertEqual(trait.value.value, 2.0)

    def test_int_assigned_through_gene_traits_does_not_break_mutation(self):
        params = neat.Parameters()
        params.NeuronTraits = {"gain": float_schema(0.0, 10.0)}
        params.LinkTraits = {"gain": float_schema(0.0, 10.0)}
        genome = small_genome(params)
        neurons = genome.m_NeuronGenes
        edited = 0
        for neuron in neurons:
            traits = neuron.m_Traits
            if "gain" not in traits:
                continue
            edited += 1
            raw = dict(traits)
            replacement = neat.Trait()
            replacement.value = 3  # plain Python int
            raw["gain"] = replacement
            neuron.m_Traits = raw
            self.assertIsInstance(neuron.m_Traits["gain"].value, float)
        self.assertGreater(edited, 0)
        genome.m_NeuronGenes = neurons
        rng = neat.RNG()
        rng.Seed(11)
        for _ in range(10):
            genome.Mutate_NeuronTraits(params, rng)

    def test_dependency_values_are_normalized_to_dependency_type(self):
        params = neat.Parameters()
        dependent = float_schema()
        dependent.dep_key = "mode"
        dependent.dep_values = [1.0, 2.0]
        params.GenomeTraits = {"mode": int_schema(), "gain": dependent}
        self.assertEqual(params.Validate(), (True, ""))
        values = params.GenomeTraits["gain"].dep_values
        self.assertEqual(values, [1, 2])
        self.assertTrue(all(isinstance(value, int) for value in values))

    def test_unmatchable_dependency_values_fail_validation(self):
        params = neat.Parameters()
        dependent = float_schema()
        dependent.dep_key = "mode"
        dependent.dep_values = [1.5]
        params.GenomeTraits = {"mode": int_schema(), "gain": dependent}
        ok, message = params.Validate()
        self.assertFalse(ok)
        self.assertIn("never match", message)
        dependent.dep_values = ["on"]
        params.GenomeTraits = {"mode": int_schema(), "gain": dependent}
        ok, message = params.Validate()
        self.assertFalse(ok)
        self.assertIn("dependency", message)


class ParameterPersistenceTests(unittest.TestCase):
    def test_failed_save_does_not_truncate_existing_file(self):
        params = neat.Parameters()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.txt"
            params.Save(str(path))
            before = path.read_bytes()
            self.assertGreater(len(before), 0)
            broken = float_schema()
            broken.type = "int"  # m_Details still holds FloatTraitParameters
            params.NeuronTraits = {"gain": broken}
            with self.assertRaisesRegex(ValueError, "gain"):
                params.Save(str(path))
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                pickle.dumps(params)

    def test_load_preserves_custom_constraints(self):
        params = neat.Parameters()
        params.PopulationSize = 33
        genome = small_genome()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.txt"
            params.Save(str(path))
            target = neat.Parameters()
            target.CustomConstraints = lambda candidate: True
            self.assertEqual(target.Load(str(path)), 0)
            self.assertEqual(target.PopulationSize, 33)
            self.assertIsNotNone(target.CustomConstraints)
            self.assertTrue(target.CustomConstraints(genome))

    def test_non_finite_doubles_round_trip(self):
        params = neat.Parameters()
        params.LeoThreshold = math.inf
        params.NeuronTraits = {"gain": float_schema()}
        restored = pickle.loads(pickle.dumps(params))
        self.assertEqual(restored.LeoThreshold, math.inf)
        params.LeoThreshold = -math.inf
        self.assertEqual(
            neat.Parameters.Deserialize(params.Serialize()).LeoThreshold,
            -math.inf)

    def test_malformed_scalars_are_rejected(self):
        base = neat.Parameters().Serialize()
        for key, value in (("PopulationSize", "150.5"),
                           ("PopulationSize", "-1"),
                           ("PopulationSize", "99999999999"),
                           ("NeuronTries", "12abc"),
                           ("MutateAddNeuronProb", "0.1x")):
            with self.subTest(key=key, value=value):
                lines = [
                    f"{key} {value}" if line.split(" ", 1)[0] == key else line
                    for line in base.splitlines()
                ]
                with self.assertRaises(Exception):
                    neat.Parameters.Deserialize("\n".join(lines) + "\n")
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "params.txt"
                    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    target = neat.Parameters()
                    target.PopulationSize = 7
                    self.assertEqual(target.Load(str(path)), 1)
                    self.assertEqual(target.PopulationSize, 7)


class SeedTests(unittest.TestCase):
    def draws(self, seed):
        rng = neat.RNG()
        rng.Seed(seed)
        return [rng.RandFloat() for _ in range(4)]

    def test_wide_seeds_are_accepted(self):
        for seed in (2 ** 31, 2 ** 32, 2 ** 63 - 1, 2 ** 64 - 1, -(2 ** 63)):
            with self.subTest(seed=seed):
                self.assertEqual(self.draws(seed), self.draws(seed))
        self.assertNotEqual(self.draws(2 ** 32 + 1), self.draws(2 ** 33 + 1))
        with self.assertRaises((ValueError, OverflowError)):
            neat.RNG().Seed(2 ** 64)
        with self.assertRaises(TypeError):
            neat.RNG().Seed(1.5)

    def test_32_bit_seeds_keep_historical_mapping(self):
        # Historically the seed was cast to the engine's 32-bit result type,
        # so -1 and 2**32 - 1 select the same stream.
        self.assertEqual(self.draws(-1), self.draws(2 ** 32 - 1))
        self.assertEqual(self.draws(42), self.draws(42))
        # std::mt19937 seeded with 5489 (its default seed) is standardized.
        rng = neat.RNG()
        rng.Seed(5489)
        state = rng.Serialize().split()
        self.assertEqual(state[0], "5489")


class SeededGenomeTests(unittest.TestCase):
    def test_rng_constructor_is_reproducible(self):
        params = neat.Parameters()
        params.NeuronTraits = {"gain": float_schema(0.0, 10.0)}
        init = neat.GenomeInitStruct()
        init.NumInputs = 4
        init.NumOutputs = 2
        init.FS_NEAT = True
        init.FS_NEAT_links = 3

        def build(seed):
            rng = neat.RNG()
            rng.Seed(seed)
            return neat.Genome(params, init, rng).Serialize()

        self.assertEqual(build(17), build(17))


class CommandLineTests(unittest.TestCase):
    def test_demo_executable_rejects_malformed_integers(self):
        directory = Path(neat.__file__).resolve().parent
        candidates = [directory / "multineat_exe.exe",
                      directory / "multineat_exe"]
        executable = next((path for path in candidates if path.exists()), None)
        if executable is None:
            self.skipTest("multineat_exe was not built next to the module")
        for value in ("abc", "12abc", "", "99999999999999999999", "1.5"):
            with self.subTest(value=value):
                result = subprocess.run(
                    [str(executable), "--smoke", "--seed", value],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--seed", result.stderr)


if __name__ == "__main__":
    unittest.main()
