import os
import sys
import tempfile
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib.pyplot as plt
    import networkx  # noqa: F401
    import numpy  # noqa: F401
except ImportError:
    # Visualization dependencies are deliberately optional for core users.
    raise SystemExit(77)

import pymultineat as neat
import neattools


parameters = neat.Parameters()
parameters.PopulationSize = 6
initial = neat.GenomeInitStruct()
initial.NumInputs = 3
initial.NumOutputs = 1
genome = neat.Genome(parameters, initial)
genome.SetFitness(1.25)

graph = neattools.Genome2NX(genome)
assert graph.number_of_nodes() == len(genome.m_NeuronGenes)
assert graph.number_of_edges() == len(genome.m_LinkGenes)
assert len(neattools.compute_node_positions(genome)) == len(genome.m_NeuronGenes)
assert neattools.get_layered_nodes(genome)
assert neattools.get_topologically_sorted_nodes(genome)
assert neattools.genome_summary(genome)["links"] == len(genome.m_LinkGenes)
assert neattools.genome_summary(genome)["feed_forward_depth"] >= 1
assert neattools.genome_summary(genome)["density"] >= 0.0
assert neattools.compare_genomes(genome, genome)["left_only_innovations"] == []

neattools.DrawGenome(genome, show=False)
neattools.DrawGenomes([genome, genome], show=False)
comparison_genome = neat.Genome.Deserialize(genome.Serialize())
comparison_links = comparison_genome.m_LinkGenes
comparison_links[0].SetWeight(comparison_links[0].GetWeight() + 1.0)
comparison_genome.m_LinkGenes = comparison_links
comparison = neattools.compare_genomes(genome, comparison_genome)
assert comparison["changed_weights"]
neattools.DrawGenomeComparison(
    genome,
    comparison_genome,
    with_edge_labels=True,
    show=False,
)

population = neat.Population(genome, parameters, True, 1.0, 123)
for index in range(population.NumGenomes()):
    population.AccessGenomeByIndex(index).SetFitness(float(index))
neattools.DrawPopulation(population, show=False)
population_metrics = neattools.population_summary(population)
assert population_metrics["population_size"] == parameters.PopulationSize
assert population_metrics["species"] == len(population.m_Species)
assert population_metrics["effective_species"] >= 1.0
assert neattools.species_summary(population.m_Species[0])["size"] > 0

tracker = neattools.EvolutionTracker()
tracker.record(population, 0)
tracker.draw(show=False)

with tempfile.TemporaryDirectory() as directory:
    json_path = neattools.export_genome_graph(genome, Path(directory) / "genome.json")
    svg_path = neattools.export_genome_graph(genome, Path(directory) / "genome.svg")
    assert json_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0
    assert tracker.save(Path(directory) / "evolution.json").stat().st_size > 0
    assert tracker.save(Path(directory) / "evolution.csv").stat().st_size > 0

try:
    import plotly  # noqa: F401
except ImportError:
    pass
else:
    assert len(neattools.InteractiveGenome(genome).data) > 0
    assert len(neattools.InteractivePopulation(population).data) >= 5
    assert len(tracker.interactive().data) >= 3

plt.close("all")
