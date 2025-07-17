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
      The title of the diagram now reflects the individual’s ID and fitness.
      Arrowheads indicate edge direction.

  • DrawGenomes(genomes, …)
      Draws a list of genomes in a grid of subplots, automatically determining
      how many rows and columns to use so that the overall layout is approximately 
      in a 5:3 aspect ratio (e.g. 15 genomes will be drawn in 3 rows of 5 columns).

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
  from multineat2 import Genome2NX, DrawGenome, DrawGenomes, narrate_traits, get_layered_nodes, print_genome_summary
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
  DrawGenomes([seed_genome, ...])
  layers = get_layered_nodes(seed_genome)
  print("Layers (top-to-bottom):", layers)
  export_genome_graph(seed_genome, "genome.dot")
"""

import pymultineat as pnt
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np

# Import common neuron type names from pnt.
INPUT  = pnt.INPUT
BIAS   = pnt.BIAS
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
      - Bias nodes are placed at the far right of the input row (y = 1.0, x = 1.0).
    Within each group, nodes are spaced evenly in x.
    Returns a dictionary mapping node ID to (x, y).
    """
    pos = {}
    # Partition genes into input, output, bias, and hidden.
    input_nodes = [n for n in genome.m_NeuronGenes if n.m_Type == INPUT]
    output_nodes = [n for n in genome.m_NeuronGenes if n.m_Type == OUTPUT]
    bias_nodes = [n for n in genome.m_NeuronGenes if n.m_Type == BIAS]
    hidden_nodes = [n for n in genome.m_NeuronGenes if n.m_Type not in (INPUT, OUTPUT, BIAS)]
    
    # Sort by m_ID for consistency
    input_nodes.sort(key=lambda n: n.m_ID)
    output_nodes.sort(key=lambda n: n.m_ID)
    bias_nodes.sort(key=lambda n: n.m_ID)
    # For hidden nodes, sort by m_SplitY if available.
    hidden_nodes.sort(key=lambda n: getattr(n, "m_SplitY", 0.5))
    
    n_in = len(input_nodes)
    n_out = len(output_nodes)
    n_hidden = len(hidden_nodes)
    
    # Position input nodes evenly along top, leaving space for the bias node.
    for i, n in enumerate(input_nodes):
        pos[n.m_ID] = ((i + 1) / (n_in + 1), 1.0)  # +2 to account for bias node
    
    # Position bias node at the far right of the input row.
    if bias_nodes:
        pos[bias_nodes[0].m_ID] = (1.0, 1.0)
        
    # Position output nodes evenly along bottom.
    for i, n in enumerate(output_nodes):
        pos[n.m_ID] = ((i + 1) / (n_out + 1), 0.0)
    
    # Position hidden nodes.
    for i, n in enumerate(hidden_nodes):
        try:
            sy = n.m_SplitY
        except AttributeError:
            sy = 0.5
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
        layer = round(y, 1)
        layers.setdefault(layer, []).append((nid, x))
    for layer in layers:
        layers[layer].sort(key=lambda tup: tup[1])
        layers[layer] = [nid for nid, x in layers[layer]]
    
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


def DrawGenome(genome, ax=None, node_size=100, with_edge_labels=False):
    # If no axis is provided, create one.
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        own_fig = True

    # Get the networkx graph and positions.
    G = Genome2NX(genome)
    pos = compute_node_positions(genome)
    
    # Partition nodes by type.
    input_nodes  = [n for n, d in G.nodes(data=True) if d.get("type") == INPUT]
    bias_nodes   = [n for n, d in G.nodes(data=True) if d.get("type") == BIAS]
    output_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == OUTPUT]
    hidden_nodes = [n for n, d in G.nodes(data=True) if d.get("type") not in (INPUT, OUTPUT, BIAS)]
    
    # Draw nodes with different shapes and colors.
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='lightgreen', node_shape='s',
                           node_size=node_size, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=bias_nodes, node_color='yellow', node_shape='s',
                           node_size=node_size, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='lightblue', node_shape='o',
                           node_size=node_size, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='salmon', node_shape='D',
                           node_size=node_size, ax=ax)
    
    # Prepare edge groups based on recurrence.
    normal_edges = []
    normal_edge_colors = []
    normal_edge_widths = []
    
    recurrent_edges = []
    recurrent_edge_colors = []
    recurrent_edge_widths = []
    
    for u, v, edata in G.edges(data=True):
        weight = edata.get("weight", 1)
        if weight > 0:
            color = "green"
        elif weight < 0:
            color = "red"
        else:
            color = "gray"
        www = np.clip(abs(weight), 0.0, 5.0)
        width = max(1, www)
        
        if edata.get("is_recurrent", False):
            recurrent_edges.append((u, v))
            recurrent_edge_colors.append(color)
            recurrent_edge_widths.append(width)
        else:
            normal_edges.append((u, v))
            normal_edge_colors.append(color)
            normal_edge_widths.append(width)
    
    # Draw normal edges.
    if normal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=normal_edge_widths,
                               edge_color=normal_edge_colors, arrows=True,
                               arrowstyle='-|>', arrowsize=10, ax=ax)
    # Draw recurrent edges in dashed style.
    if recurrent_edges:
        nx.draw_networkx_edges(G, pos, edgelist=recurrent_edges, width=recurrent_edge_widths,
                               edge_color=recurrent_edge_colors, style='dashed', arrows=True,
                               arrowstyle='-|>', arrowsize=10, ax=ax)
    
    # Create labels for nodes (showing node ID and type).
    labels = {}
    for node, data in G.nodes(data=True):
        node_type = data.get("type")
        if node_type == INPUT:
            ttype = "Input"
        elif node_type == OUTPUT:
            ttype = "Output"
        elif node_type == HIDDEN:
            ttype = "Hidden"
        elif node_type == BIAS:
            ttype = "Bias"
        else:
            ttype = str(node_type)
        labels[node] = f"{node}"
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    
    if with_edge_labels:
        edge_labels = {}
        for u, v, edata in G.edges(data=True):
            edge_labels[(u, v)] = f"{edata.get('weight', 0):.2f}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    
    # Set the plot title to include the genome's ID and fitness.
    try:
        individual_id = genome.GetID()
    except AttributeError:
        individual_id = "N/A"
    try:
        fitness = genome.GetFitness()
    except AttributeError:
        fitness = "N/A"
    ax.set_title(f"ID: {individual_id} | Fitness: {fitness}")
    
    ax.axis("off")
    if own_fig:
        plt.tight_layout()
        plt.show()


def DrawGenomes(genomes, node_size=600, with_edge_labels=False):
    """
    Draws a list of genomes as neural network diagrams in subplots.

    The function automatically determines the subplots grid so that the overall
    figure has an approximate 5:3 (width:height) aspect ratio. For example, if there
    are 15 genomes in the list, the grid will be 3 rows x 5 columns.

    Each genome is drawn by calling DrawGenome() on its respective subplot.
    """
    n = len(genomes)
    if n == 0:
        print("No genomes to draw.")
        return

    # Determine optimal grid dimensions.
    target_ratio = 5 / 3  # desired ratio columns/rows
    best_diff = None
    best_rows, best_cols = 1, n
    for rows in range(1, n + 1):
        cols = math.ceil(n / rows)
        ratio = cols / rows
        diff = abs(ratio - target_ratio)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_rows, best_cols = rows, cols

    # Create subplots. Each subplot is assigned approximate width=5 and height=3.
    fig, axes = plt.subplots(best_rows, best_cols, figsize=(best_cols * 5, best_rows * 3))
    # Flatten axes in case there is more than one.
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Draw each genome in its corresponding subplot.
    for ax, genome in zip(axes, genomes):
        DrawGenome(genome, ax=ax, node_size=node_size, with_edge_labels=with_edge_labels)
    
    # Turn off any extra subplots that don't have a genome.
    for ax in axes[len(genomes):]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()


def narrate_traits(genome):
    """
    Prints (narrates) all trait information in the genome.
    This includes the genome’s own traits, as well as the traits for each neuron and each link.
    """
    print("===== Genome Traits =====", flush=True)
    try:
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
    
    # Draw the genome in its own figure.
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
    
    # Demonstration: if you have a list of genomes, you can draw them all together.
    # For demonstration, we will use a list containing several copies of seed_genome.
    demo_genomes = [seed_genome for _ in range(15)]
    DrawGenomes(demo_genomes)