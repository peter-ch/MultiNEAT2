#!/usr/bin/env python3
# ant_neat.py

import gymnasium as gym
import pymultineat as pnt
import numpy as np
import time
import multiprocessing
from tqdm import tqdm
import pygame  # For key press detection
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import rgb2hex

# Fitness shift constant to ensure all fitness values are positive
FITNESS_SHIFT = 1000.0

# Worker initialization function for multiprocessing
def init_worker():
    global worker_env
    worker_env = gym.make('Ant-v5')

# Define the evaluation function for a genome
def evaluate_genome(genome, env=None, render=False, max_steps=500):
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
            env = gym.make('Ant-v5', render_mode='human')
        else:
            env = gym.make('Ant-v5')
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
                    return total_reward + abs(min_reward) + 1 + FITNESS_SHIFT  # Return shifted fitness
        
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
            
    # Adjust for negative rewards and add fitness shift
    fitness = total_reward + abs(min_reward) + 1 + FITNESS_SHIFT
    return fitness

import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Ant NEAT')
    parser.add_argument('--serial', action='store_true', help='Use serial evaluation instead of parallel')
    args = parser.parse_args()
    
    # Initialize pygame for key detection
    pygame.init()
    pygame.display.set_mode((1, 1))  # Create a tiny window for event handling
    
    # Set up matplotlib figures
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Figure 1: Best Fitness per Generation
    ax1.set_title('Best Fitness per Generation')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    line, = ax1.plot([], [], 'b-')  # Create an empty line
    best_fitness_history = []
    
    # Figure 2: Population Visualization
    ax2.set_title('Population Fitness by Species')
    ax2.set_xlabel('Individual')
    ax2.set_ylabel('Fitness')
    ax2.set_ylim(0, 2000)  # Adjust based on expected fitness range
    population_bars = None
    
    # Statistics text box
    stats_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, verticalalignment='top')
    
    # Create a temporary environment for serial evaluation and rendering
    temp_env = gym.make('Ant-v5')
    
    # Create and customize MultiNEAT parameters
    params = pnt.Parameters()
    params.PopulationSize = 150
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = False
    params.WeightDiffCoeff = 0.05
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
    params.SplitRecurrent = True 
    params.SplitLoopedRecurrent = True
    params.MinActivationA = 1.0
    params.MaxActivationA = 8.0
    params.MutateActivationAProb = 0.25
    params.ActivationAMutationMaxPower = 2.0
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
    params.AllowClones = True
    params.EliteFraction = 0.02

    # Create a GenomeInitStruct
    # 105 inputs (observations) + 1 bias = 106 inputs, 8 outputs (actions)
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 106
    init_struct.NumOutputs = 8
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.HiddenActType = pnt.UNSIGNED_SIGMOID
    init_struct.OutputActType = pnt.TANH
    init_struct.FS_NEAT = True # start with only a few links
    init_struct.FS_NEAT_links = 32

    # Create a prototype genome
    genome_prototype = pnt.Genome(params, init_struct)

    # Create the initial population
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))

    # Create persistent process pool if using parallel
    pool = None
    if not args.serial:
        pool = multiprocessing.Pool(processes=16, initializer=init_worker)

    generations = 2500
    
    try:
        for gen in tqdm(range(1,generations), desc="Generations"):
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
                # Parallel evaluation using persistent pool
                genomes = [individual for species in pop.m_Species for individual in species.m_Individuals]
                
                # Evaluate genomes in parallel
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
            
            # Update the population visualization
            if population_bars is not None:
                for bar in population_bars:
                    bar.remove()
            
            # Collect fitness and species data
            fitness_data = []
            species_data = []
            neurons_data = []
            links_data = []
            for species in pop.m_Species:
                for individual in species.m_Individuals:
                    fitness_data.append(individual.GetFitness())
                    species_data.append(species.ID())
                    nn = pnt.NeuralNetwork()
                    individual.BuildPhenotype(nn)
                    neurons_data.append(len(nn.m_neurons))  # Number of neurons
                    links_data.append(len(nn.m_connections))  # Number of links
            
            # Assign colors to species
            unique_species = list(set(species_data))
            cmap = cm.get_cmap('tab20', len(unique_species))
            species_colors = {s: rgb2hex(cmap(i)[:3]) for i, s in enumerate(unique_species)}
            colors = [species_colors[s] for s in species_data]
            
            # Plot population bars
            population_bars = ax2.bar(range(len(fitness_data)), fitness_data, color=colors)
            
            # Update statistics
            stats = f"Population Stats:\n"
            stats += f"Max Neurons: {max(neurons_data)}\n"
            stats += f"Min Neurons: {min(neurons_data)}\n"
            stats += f"Max Links: {max(links_data)}\n"
            stats += f"Min Links: {min(links_data)}\n"
            stats += f"Species Count: {len(unique_species)}"
            stats_text.set_text(stats)
            
            # Redraw figures
            plt.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Print generation stats
            print(f"\nGeneration {gen}: Best Fitness = {best_fitness:.2f}")
            
            # Render best individual every 50 generations
            if best_genome and gen % 50 == 0:
                print(f"\nRendering best individual from generation {gen}...")
                for i in range(10):
                    print(f"Episode {i+1} (Press ESC to skip remaining episodes)")
                    # Create a fresh render environment for each episode
                    env_render = gym.make('Ant-v5', render_mode='human')
                    try:
                        evaluate_genome(best_genome, env_render, render=True, max_steps=500)
                    finally:
                        env_render.close()
            
            # Advance to next generation
            pop.Epoch()
        
    finally:
        # Clean up pool resources
        if pool is not None:
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