#!/usr/bin/env python
"""
multineat2.py
=============

This module provides helper functions for working with MultiNEAT genomes
(via the pnt module) using NetworkX and matplotlib.

The following main functions are provided:
  • Genome2NX(genome)
      Converts a MultiNEAT Genome (pnt.Genome instance) to a networkx.DiGraph,
      adding node and edge attributes (including traits) to the graph.

  • DrawGenome(genome, …)
      Draws the genome as a neural network diagram:
         • Input nodes are arranged as a row across the top,
         • Output nodes are arranged as a row across the bottom,
         • Hidden nodes are placed in between according to a “split_y” value
           (or simply evenly if not provided).
      Arrowheads indicate edge direction.

  • narrate_traits(genome)
      Prints out (narrates) the traits in the genome. It prints the genome’s own traits,
      then for every neuron gene and every link gene the trait information is printed.

  • get_layered_nodes(genome)
      Computes node positions from the genome in a similar way as DrawGenome() and groups the node IDs
      by their “layer” (top to bottom), so you can see which nodes lie in the input, hidden, and output layers.
  
  • export_genome_graph(genome, filename)
      Exports the genome’s graph (obtained via Genome2NX()) in Graphviz DOT format.
      
  • print_genome_summary(genome)
      Prints a simple summary of the genome.

Additional utility functions (such as topological sorting) are also provided.

Note:
  This module assumes that pnt has been built with pybind11 so that the C++ members
  (such as m_NeuronGenes, m_LinkGenes, and the neuron/link/tratis members) are accessible.

Usage example:
  from multineat2 import Genome2NX, DrawGenome, narrate_traits, get_layered_nodes, print_genome_summary
  from pnt import Genome, Parameters, GenomeInitStruct, INPUT, OUTPUT, HIDDEN, UNSIGNED_SIGMOID
  params = Parameters()
  init = GenomeInitStruct()
  init.NumInputs = 3
  init.NumOutputs = 1
  init.NumHidden = 0
  init.SeedType = 0   
  init.HiddenActType = UNSIGNED_SIGMOID
  init.OutputActType = UNSIGNED_SIGMOID
  seed_genome = Genome(params, init)
  print_genome_summary(seed_genome)
  narrate_traits(seed_genome)
  DrawGenome(seed_genome)
  layers = get_layered_nodes(seed_genome)
  print("Layers (top-to-bottom):", layers)
  export_genome_graph(seed_genome, "genome.dot")
"""

import pymultineat as pnt
import networkx as nx
import matplotlib.pyplot as plt

# Import common neuron type names from pnt.
INPUT  = pnt.INPUT
OUTPUT = pnt.OUTPUT
HIDDEN = pnt.HIDDEN


def Genome2NX(genome):
    """
    Convert a MultiNEAT Genome (pnt.Genome) into a NetworkX directed graph.
    For each neuron gene, the node is added with attributes such as type, x, y, split_y,
    activation parameters, and traits.
    For each link gene, the edge is added with attributes such as innovation id, weight,
    recurrent flag, and traits.
    """
    G = nx.DiGraph()
    
    # Add neuron genes as nodes.
    for neuron in genome.m_NeuronGenes:
        node_id = neuron.m_ID  # assuming m_ID is unique
        attr = {
            "type": neuron.m_Type,        # typically INPUT, OUTPUT, HIDDEN (as int enums)
            "x": neuron.x,
            "y": neuron.y,
            "split_y": neuron.m_SplitY,
            "a": neuron.m_A,
            "b": neuron.m_B,
            "time_constant": neuron.m_TimeConstant,
            "bias": neuron.m_Bias,
            "act_function": neuron.m_ActFunction,
            "traits": neuron.m_Traits
        }
        G.add_node(node_id, **attr)
        
    # Add link genes as directed edges.
    for link in genome.m_LinkGenes:
        source = link.m_FromNeuronID
        target = link.m_ToNeuronID
        attr = {
            "innovation_id": link.m_InnovationID,
            "weight": link.m_Weight,
            "is_recurrent": link.m_IsRecurrent,
            "traits": link.m_Traits
        }
        G.add_edge(source, target, **attr)
        
    return G


def compute_node_positions(genome):
    """
    Computes positions for nodes in the genome based on a simple layer layout:
      - Input nodes are placed at y = 1.0 (top)
      - Output nodes are placed at y = 0.0 (bottom)
      - Hidden nodes are placed based on their m_SplitY (if available) so that
        computed y = 1 - m_SplitY; if not available, default to y = 0.5.
    Within each group, nodes are spaced evenly in x.
    Returns a dictionary mapping node ID to (x, y).
    """
    pos = {}
    # Partition genes into input, output and hidden.
    input_nodes = [n for n in genome.m_NeuronGenes if n.m_Type == INPUT]
    output_nodes = [n for n in genome.m_NeuronGenes if n.m_Type == OUTPUT]
    hidden_nodes = [n for n in genome.m_NeuronGenes if n.m_Type not in (INPUT, OUTPUT)]
    
    # Sort by m_ID for consistency
    input_nodes.sort(key=lambda n: n.m_ID)
    output_nodes.sort(key=lambda n: n.m_ID)
    # For hidden nodes, sort by m_SplitY if available. (Lower m_SplitY means it appeared later in evolution.)
    hidden_nodes.sort(key=lambda n: getattr(n, "m_SplitY", 0.5))
    
    n_in = len(input_nodes)
    n_out = len(output_nodes)
    n_hidden = len(hidden_nodes)
    
    # Position input nodes evenly along top.
    for i, n in enumerate(input_nodes):
        pos[n.m_ID] = ((i + 1) / (n_in + 1), 1.0)
        
    # Position output nodes evenly along bottom.
    for i, n in enumerate(output_nodes):
        pos[n.m_ID] = ((i + 1) / (n_out + 1), 0.0)
    
    # Position hidden nodes.
    for i, n in enumerate(hidden_nodes):
        # Try to use m_SplitY to determine vertical position:
        try:
            sy = n.m_SplitY
        except AttributeError:
            sy = 0.5
        # In DrawGenome we used: y = 1 - (split_y) so that lower split_y means lower position.
        y = 1.0 - sy
        pos[n.m_ID] = ((i + 1) / (n_hidden + 1), y)
        
    return pos


def get_layered_nodes(genome):
    """
    Computes node positions (using compute_node_positions) and groups node IDs by layer.
    For grouping, the y value (from 0 to 1) is quantized (rounded to 1 decimal place).
    Returns a dictionary mapping layer (y-level) to a sorted list of node IDs.
    Layers are returned sorted descending (so that the top layer is first).
    """
    pos = compute_node_positions(genome)
    layers = {}
    for nid, (x, y) in pos.items():
        # Quantize layer by rounding y to one decimal place.
        layer = round(y, 1)
        layers.setdefault(layer, []).append((nid, x))
    # Sort nodes in each layer by increasing x.
    for layer in layers:
        layers[layer].sort(key=lambda tup: tup[1])
        # Now extract only node IDs.
        layers[layer] = [nid for nid, x in layers[layer]]
    
    # Return layers sorted by descending layer (top = highest y)
    sorted_layers = dict(sorted(layers.items(), key=lambda item: item[0], reverse=True))
    return sorted_layers


def get_topologically_sorted_nodes(genome):
    """
    Tries to return a list of node IDs from the genome that is topologically sorted
    (using networkx.topological_sort on the graph produced by Genome2NX).
    If the graph is not a DAG, falls back to sorting nodes by their computed y position (top to bottom).
    """
    try:
        G = Genome2NX(genome)
        sorted_nodes = list(nx.topological_sort(G))
        return sorted_nodes
    except Exception as e:
        print("Topological sort failed (graph not a DAG). Falling back to sorting by y position.")
        pos = compute_node_positions(genome)
        sorted_nodes = sorted(pos.keys(), key=lambda nid: pos[nid][1], reverse=True)
        return sorted_nodes


def DrawGenome(genome, figsize=(12, 8), node_size=600):
    """
    Draws the given MultiNEAT Genome as a neural network diagram.
    The layout is computed so that:
      • Input nodes are arranged as a row on the top (y = 1),
      • Output nodes as a row on the bottom (y = 0),
      • Hidden nodes are placed between according to their m_SplitY value.
    Arrowed edges are drawn to show edge direction.
    """
    G = Genome2NX(genome)
    
    # Partition nodes by type.
    input_nodes  = [n for n, d in G.nodes(data=True) if d.get("type") == INPUT]
    output_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == OUTPUT]
    hidden_nodes = [n for n, d in G.nodes(data=True) if d.get("type") not in (INPUT, OUTPUT)]
    
    pos = {}
    # Compute positions: inputs evenly spaced at y=1, outputs at y=0.
    if input_nodes:
        n = len(input_nodes)
        for i, node in enumerate(sorted(input_nodes)):
            pos[node] = ((i + 1) / (n + 1), 1.0)
    if output_nodes:
        n = len(output_nodes)
        for i, node in enumerate(sorted(output_nodes)):
            pos[node] = ((i + 1) / (n + 1), 0.0)
    if hidden_nodes:
        n = len(hidden_nodes)
        # Use the "split_y" attribute if available.
        for i, node in enumerate(sorted(hidden_nodes)):
            data = G.nodes[node]
            split_y = data.get("split_y", 0.5)
            y = 1 - split_y  # so that lower split_y gives lower position
            pos[node] = ((i + 1) / (n + 1), y)
    
    plt.figure(figsize=figsize)
    
    # Draw nodes.
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size)
    
    # Create labels for nodes.
    labels = {}
    for node, data in G.nodes(data=True):
        node_type = data.get("type")
        if node_type == INPUT:
            ttype = "Input"
        elif node_type == OUTPUT:
            ttype = "Output"
        elif node_type == HIDDEN:
            ttype = "Hidden"
        else:
            ttype = str(node_type)
        labels[node] = f"{node}\n{ttype}"
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # Draw edges with arrowheads.
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=10, edge_color='gray')
    
    plt.title("Genome Network")
    plt.axis("off")
    plt.show()


def narrate_traits(genome):
    """
    Prints (narrates) all trait information in the genome.
    This includes the genome’s own traits, as well as the traits for each neuron and each link.
    """
    print("===== Genome Traits =====", flush=True)
    try:
        # Genome-level traits.
        print("GenomeGene Traits:", flush=True)
        for key, value in genome.m_GenomeGene.m_Traits.items():
            print(f"  {key}: {value}", flush=True)
    except AttributeError:
        print("  [No GenomeGene traits found]", flush=True)
    
    print("\n-- Neuron Traits --")
    for neuron in genome.m_NeuronGenes:
        print(f"Neuron {neuron.m_ID} (type: {neuron.m_Type}):", flush=True)
        if hasattr(neuron, "m_Traits") and neuron.m_Traits:
            for key, value in neuron.m_Traits.items():
                print(f"  {key}: {value}", flush=True)
        else:
            print("  [No traits]", flush=True)
    
    print("\n-- Link Traits --", flush=True)
    for link in genome.m_LinkGenes:
        print(f"Link Innovation {link.m_InnovationID} (from {link.m_FromNeuronID} to {link.m_ToNeuronID}):", flush=True)
        if hasattr(link, "m_Traits") and link.m_Traits:
            for key, value in link.m_Traits.items():
                print(f"  {key}: {value}", flush=True)
        else:
            print("  [No traits]", flush=True)


def export_genome_graph(genome, filename):
    """
    Exports the genome as a Graphviz DOT file.
    
    Requires pydot. Install via pip if needed.
    """
    try:
        from networkx.drawing.nx_pydot import write_dot
    except ImportError:
        raise ImportError("pydot is required to export the genome to DOT format (pip install pydot).")
    
    G = Genome2NX(genome)
    write_dot(G, filename)
    print(f"Genome exported in DOT format to {filename}", flush=True)


def print_genome_summary(genome):
    """
    Prints a brief summary of the genome:
      - Genome ID
      - Number of neuron genes
      - Number of link genes
      - Fitness value
    """
    num_neurons = len(genome.m_NeuronGenes)
    num_links = len(genome.m_LinkGenes)
    fitness = genome.GetFitness()
    print("===== Genome Summary =====", flush=True)
    print(f"ID: {genome.GetID()}", flush=True)
    print(f"Number of Neurons: {num_neurons}", flush=True)
    print(f"Number of Links: {num_links}", flush=True)
    print(f"Fitness: {fitness}", flush=True)


if __name__ == "__main__":
    # When run as main, demonstrate functionality with a seed genome.

    # Set up parameters.
    params = pnt.Parameters()
    params.PopulationSize = 150
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = False
    params.WeightDiffCoeff = 0.1
    params.CompatTreshold = 2.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 15
    params.OldAgeTreshold = 35
    params.MinSpecies = 2
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.0
    params.OverallMutationRate = 0.3
    params.ArchiveEnforcement = False
    params.MutateWeightsProb = 0.25
    params.WeightMutationMaxPower = 0.5
    params.WeightReplacementMaxPower = 8.0
    params.MutateWeightsSevereProb = 0.0
    params.WeightMutationRate = 0.85
    params.WeightReplacementRate = 0.2
    params.MaxWeight = 8.0
    params.MutateAddNeuronProb = 0.01
    params.MutateAddLinkProb = 0.1
    params.MutateRemLinkProb = 0.0
    params.MutateRemSimpleNeuronProb = 0.0
    params.NeuronTries = 64
    params.MutateAddLinkFromBiasProb = 0.0
    params.CrossoverRate = 0.0
    params.MultipointCrossoverRate = 0.0
    params.SurvivalRate = 0.2
    params.MutateNeuronTraitsProb = 0.0
    params.MutateLinkTraitsProb = 0.0
    params.AllowLoops = False
    params.AllowClones = False

    init = pnt.GenomeInitStruct()
    init.NumInputs = 3
    init.NumOutputs = 1
    init.NumHidden = 0
    init.SeedType = pnt.GenomeSeedType.PERCEPTRON    
    init.HiddenActType = pnt.UNSIGNED_SIGMOID
    init.OutputActType = pnt.UNSIGNED_SIGMOID
    
    seed_genome = pnt.Genome(params, init)
    
    print_genome_summary(seed_genome)
    narrate_traits(seed_genome)
    
    # Draw the genome.
    DrawGenome(seed_genome)
    
    # Generate and print layered node grouping.
    layers = get_layered_nodes(seed_genome)
    print("Layered nodes (by quantized y):", flush=True)
    for layer, nodes in layers.items():
        print(f"Layer {layer}: {nodes}", flush=True)
    
    # Topologically sorted nodes, if possible.
    sorted_nodes = get_topologically_sorted_nodes(seed_genome)
    print("Topologically sorted node IDs (if graph is a DAG):", flush=True)
    print(sorted_nodes)
    
    # Export the genome graph to DOT (requires pydot).
    # Uncomment the following line if you wish to export.
    # export_genome_graph(seed_genome, "genome.dot")
    
    # Also convert to networkx graph and print basic info.
    G = Genome2NX(seed_genome)
    print(f"Converted genome contains {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.", flush=True)