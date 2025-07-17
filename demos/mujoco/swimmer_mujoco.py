#!/usr/bin/env python3
# swimmer_neat.py

import gymnasium as gym
import pymultineat as pnt
import numpy as np
import time
import multiprocessing
from tqdm import tqdm
import pygame  # For key press detection
import matplotlib.pyplot as plt
import queue
import networkx as nx

# Worker initialization function for multiprocessing
def init_worker():
    global worker_env
    worker_env = gym.make('Swimmer-v5')

# Define the evaluation function for a genome
def evaluate_genome(genome, env=None, render=False, max_steps=1000):
    # Use worker environment if none provided
    if env is None:
        env = worker_env
        
    nn = pnt.NeuralNetwork()
    genome.BuildPhenotype(nn)
    
    # Get initial observation
    try:
        observation_data = env.reset()
    except pygame.error:
        # Environment was closed, create a new one
        if render:
            env = gym.make('Swimmer-v5', render_mode='human')
        else:
            env = gym.make('Swimmer-v5')
        observation_data = env.reset()
    
    # Handle different return types from env.reset()
    if isinstance(observation_data, tuple):
        observation = observation_data[0]  # (observation, info) format
    else:
        observation = observation_data
        
    total_reward = 0
    min_reward = 0
    step_count = 0
    
    while True:
        # Check for ESC key press if rendering
        if render:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return total_reward + abs(min_reward) + 1  # Return current fitness
        
        # Prepare inputs: convert observation to list and add bias
        inputs = observation.tolist() if hasattr(observation, 'tolist') else list(observation)
        inputs.append(1.0)  # Add bias
        
        nn.Input(inputs)
        nn.Activate()  # Activate only once per timestep
        outputs = nn.Output()
        
        # Scale outputs to [-1, 1] range (tanh already does this)
        action = outputs
        
        # Handle both old (4 return values) and new (5 return values) Gym API
        step_result = env.step(action)
        if len(step_result) == 4:
            observation, reward, done, _ = step_result
        else:
            observation, reward, done, _, _ = step_result
        total_reward += reward
        min_reward = min(min_reward, reward)
        step_count += 1
        
        if render:
            try:
                env.render()
                env.unwrapped.mujoco_renderer.viewer._hide_overlay = True
                env.unwrapped.mujoco_renderer.viewer._hide_menu = True
            except pygame.error:
                # Display was closed, skip rendering
                pass
            
        if done or step_count >= max_steps:
            break
            
    # Adjust for negative rewards (NEAT requires non-negative fitness)
    fitness = total_reward + abs(min_reward) + 1
    return fitness

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
    # For hidden nodes, sort by m_SplitY if available.
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
        try:
            sy = n.m_SplitY
        except AttributeError:
            sy = 0.5
        y = 1.0 - sy
        pos[n.m_ID] = ((i + 1) / (n_hidden + 1), y)
        
    return pos

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
    output_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == OUTPUT]
    hidden_nodes = [n for n, d in G.nodes(data=True) if d.get("type") not in (INPUT, OUTPUT)]
    
    # Draw nodes with different shapes and colors.
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='lightgreen', node_shape='s',
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
        width = max(1, abs(weight))
        
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

import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Swimmer NEAT')
    parser.add_argument('--serial', action='store_true', help='Use serial evaluation instead of parallel')
    args = parser.parse_args()
    
    # Initialize pygame for key detection
    pygame.init()
    pygame.display.set_mode((1, 1))  # Create a tiny window for event handling
    
    # Set up matplotlib figure for fitness tracking and genome visualization
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('NEAT Training Progress')
    
    # Fitness plot
    ax1.set_title('Best Fitness per Generation')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    line, = ax1.plot([], [], 'b-')  # Create an empty line
    best_fitness_history = []
    
    # Genome visualization plot
    ax2.set_title('Best Genome')
    ax2.axis("off")
    
    # Create a temporary environment for serial evaluation and rendering
    temp_env = gym.make('Swimmer-v5')
    
    # Create and customize MultiNEAT parameters
    params = pnt.Parameters()
    params.PopulationSize = 150
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = False
    params.WeightDiffCoeff = 0.02
    params.CompatTreshold = 1.0  
    params.YoungAgeTreshold = 10
    params.SpeciesMaxStagnation = 12
    params.OldAgeTreshold = 30
    params.MinSpecies = 2
    params.MaxSpecies = 6
    params.RouletteWheelSelection = False
    params.TournamentSelection = False
    params.TournamentSize = 4
    params.RecurrentProb = 0.2 
    params.OverallMutationRate = 0.8
    params.ArchiveEnforcement = False
    params.MutateWeightsProb = 0.5
    params.WeightMutationMaxPower = 1.0
    params.WeightReplacementMaxPower = 4.0
    params.MutateWeightsSevereProb = 0.2
    params.WeightMutationRate = 0.25
    params.WeightReplacementRate = 0.1
    params.MaxWeight = 16
    params.MutateAddNeuronProb = 0.01
    params.MutateAddLinkProb = 0.05     
    params.MutateRemLinkProb = 0.05
    params.MinActivationA = 5.0
    params.MaxActivationA = 5.0
    params.ActivationFunction_SignedSigmoid_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
    params.ActivationFunction_Tanh_Prob = 0.0  
    params.ActivationFunction_SignedStep_Prob = 0.0
    params.CrossoverRate = 0.7
    params.MultipointCrossoverRate = 0.6
    params.SurvivalRate = 0.2
    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0
    params.AllowLoops = True
    params.AllowClones = False
    params.EliteFraction = 0.02

    # Create a GenomeInitStruct
    # 8 inputs (observations) + 1 bias = 9 inputs, 2 outputs (actions)
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 9  # 8 observations + 1 bias
    init_struct.NumOutputs = 2  # 2 actions (torques for rotors)
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.HiddenActType = pnt.TANH
    init_struct.OutputActType = pnt.TANH

    # Create a prototype genome
    genome_prototype = pnt.Genome(params, init_struct)

    # Create the initial population
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))

    # Create process pool once if using parallel processing
    pool = None
    if not args.serial:
        print("Creating process pool...")
        pool = multiprocessing.Pool(processes=16, initializer=init_worker)
    
    generations = 2500
    
    try:
        for gen in tqdm(range(1, generations), desc="Generations"):
            best_fitness = -float('inf')
            best_genome = None
            
            # Evaluate all genomes
            if args.serial:
                # Serial evaluation
                for species in pop.m_Species:
                    for individual in species.m_Individuals:
                        fitness = evaluate_genome(individual, temp_env)
                        individual.SetFitness(fitness)
                        
                        # Track best genome
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_genome = individual
            else:
                # Parallel evaluation - reuse existing pool
                genomes = [individual for species in pop.m_Species for individual in species.m_Individuals]
                fitnesses = pool.map(evaluate_genome, genomes)
                
                # Assign fitness scores back to genomes
                idx = 0
                for species in pop.m_Species:
                    for individual in species.m_Individuals:
                        individual.SetFitness(fitnesses[idx])
                        # Track best genome
                        if fitnesses[idx] > best_fitness:
                            best_fitness = fitnesses[idx]
                            best_genome = individual
                        idx += 1
            
            # Store best fitness for progress tracking
            best_fitness_history.append(best_fitness)
            
            # Update the fitness plot
            line.set_xdata(range(len(best_fitness_history)))
            line.set_ydata(best_fitness_history)
            ax1.relim()
            ax1.autoscale_view()
            
            # Update the genome visualization
            ax2.clear()
            if best_genome:
                DrawGenome(best_genome, ax=ax2, node_size=100, with_edge_labels=False)
            ax2.set_title(f"Best Genome (Gen {gen})")
            ax2.axis("off")
            
            plt.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Print generation stats
            print(f"\nGeneration {gen}: Best Fitness = {best_fitness:.2f}")
            
            # Render best individual every N generations
            if best_genome and gen % 100 == 0:
                print(f"\nRendering best individual from generation {gen}...")
                for i in range(10):
                    print(f"Episode {i+1} (Press ESC to skip remaining episodes)")
                    # Create a fresh render environment for each episode
                    env_render = gym.make('Swimmer-v5', render_mode='human')
                    try:
                        evaluate_genome(best_genome, env_render, render=True, max_steps=1000)
                    finally:
                        env_render.close()
            
            # Advance to next generation
            pop.Epoch()
    finally:
        # Clean up resources
        if pool:
            print("Closing process pool...")
            pool.close()
            pool.join()
    
    # Keep the plot window open after training completes
    plt.ioff()
    plt.show()
    
    print("\nTraining completed. Best fitness progression:")
    for gen, fitness in enumerate(best_fitness_history):
        print(f"Gen {gen}: {fitness:.2f}")

if __name__ == "__main__":
    main()