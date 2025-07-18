#!/usr/bin/env python3
# walker2d_neat.py

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
from neattools import DrawGenome  # Import the DrawGenome function

# Constant to ensure fitness is always positive
FITNESS_SHIFT = 250.0

MAX_TIMESTEPS = 800
NUM_TRIALS = 1

# Worker initialization function for multiprocessing
def init_worker():
    global worker_env
    worker_env = gym.make('Walker2d-v5')

# Define the evaluation function for a genome
def evaluate_genome(args):
    genome, render, max_steps, num_trials = args
    
    # Handle environment creation based on context
    if render:
        # Create a new rendering environment in the main process
        env = gym.make('Walker2d-v5', render_mode='human')
    else:
        try:
            # Use worker environment if available
            env = worker_env
        except NameError:
            # Create a new environment in the main process for serial evaluation
            env = gym.make('Walker2d-v5')
    
    nn = pnt.NeuralNetwork()
    genome.BuildPhenotype(nn)
    
    total_reward = 0.0
    
    for _ in range(num_trials):
        nn.Flush()

        # Get initial observation
        try:
            observation_data = env.reset()
        except pygame.error:
            # Environment was closed, create a new one
            if render:
                env = gym.make('Walker2d-v5', render_mode='human')
            else:
                env = gym.make('Walker2d-v5')
            observation_data = env.reset()

        
        # Handle different return types from env.reset()
        if isinstance(observation_data, tuple):
            observation = observation_data[0]  # (observation, info) format
        else:
            observation = observation_data
            
        trial_reward = 0
        step_count = 0
        
        while True:
            # Check for ESC key press if rendering
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        env.close()
                        return (total_reward + trial_reward) / num_trials 
            
            # Prepare inputs: convert observation to list and add bias
            # observation = np.tanh(observation) # squash
            inputs = observation.tolist() if hasattr(observation, 'tolist') else list(observation)
            inputs.append(1.0)  # Add bias
            
            nn.Input(inputs)
            nn.Activate()  # Activate only once per timestep
            outputs = nn.Output()
            
            action = np.clip((np.array(outputs)-0.5)*2, -1, 1)
            
            # Handle both old (4 return values) and new (5 return values) Gym API
            step_result = env.step(action)
            if len(step_result) == 4:
                observation, reward, done, _ = step_result
            else:
                observation, reward, done, _, _ = step_result
            trial_reward += reward 
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
        
        total_reward += trial_reward 

    env.close()
    
    # Shift fitness to ensure it's always positive while preserving ranks
    fitness = (total_reward / num_trials) 
    return fitness + FITNESS_SHIFT

import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Walker2D NEAT')
    parser.add_argument('--serial', action='store_true', help='Use serial evaluation instead of parallel')
    args = parser.parse_args()
    
    # Initialize pygame for key detection
    pygame.init()
    pygame.display.set_mode((1, 1))  # Create a tiny window for event handling
    
    # Set up matplotlib figures
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Best Fitness per Generation
    ax2 = fig.add_subplot(gs[1, 0])  # Population Visualization
    ax3 = fig.add_subplot(gs[:, 1])  # Best Genome Visualization
    
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
    # ax2.set_ylim(0, 2000)  # Adjust based on expected fitness range
    population_bars = None
    
    # Statistics text box
    stats_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, verticalalignment='top')
    
    # Create a temporary environment for serial evaluation and rendering
    temp_env = gym.make('Walker2d-v5')
    
    params = pnt.Parameters()
    params.PopulationSize = 300
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = False
    params.WeightDiffCoeff = 0.01
    params.CompatTreshold = 2.5  
    params.YoungAgeTreshold = 25
    params.SpeciesMaxStagnation = 60
    params.OldAgeTreshold = 90
    params.MinSpecies = 4
    params.MaxSpecies = 10
    params.TruncationSelection = True
    params.RouletteWheelSelection = True
    params.TournamentSelection = False
    params.TournamentSize = 5
    params.RecurrentProb = 0.2
    params.OverallMutationRate = 0.8

    params.MutateWeightsProb = 0.75
    params.MutateWeightsSevereProb = 0.2
    params.WeightMutationMaxPower = 0.6
    params.WeightReplacementMaxPower = 1.2
    params.WeightMutationRate = 0.5
    params.WeightReplacementRate = 0.1
    params.MinWeight = -5
    params.MaxWeight = 5

    params.MutateAddNeuronProb = 0.01
    params.MutateAddLinkProb = 0.06    
    params.MutateRemLinkProb = 0.02
    params.SplitRecurrent = True 
    params.SplitLoopedRecurrent = True

    params.MinActivationA = 4.9
    params.MaxActivationA = 4.9
    params.MutateActivationAProb = 0.0
    params.ActivationAMutationMaxPower = 1.0

    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
    params.ActivationFunction_Tanh_Prob = 0.0  
    params.ActivationFunction_Relu_Prob = 0.0
    params.ActivationFunction_Softplus_Prob = 0.0
    params.ActivationFunction_Linear_Prob = 0.0
    params.MutateNeuronActivationTypeProb = 0

    params.CrossoverRate = 0.6
    params.MultipointCrossoverRate = 0.4
    params.InterspeciesCrossoverRate = 0.001
    params.SurvivalRate = 0.2

    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0
    params.AllowLoops = True
    params.AllowClones = True
    params.EliteFraction = 0.01
    params.ArchiveEnforcement = False


    # Create a GenomeInitStruct
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 18
    init_struct.NumOutputs = 6
    init_struct.NumHidden = 2
    init_struct.NumLayers = 1
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.HiddenActType = pnt.UNSIGNED_SIGMOID # regular sigmoids internally
    init_struct.OutputActType = pnt.UNSIGNED_SIGMOID 
    
    init_struct.FS_NEAT = False 
    init_struct.FS_NEAT_links = 6

    # Create a prototype genome
    genome_prototype = pnt.Genome(params, init_struct)

    # Create the initial population
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))

    generations = 2500
    
    # Create persistent process pool if using parallel evaluation
    pool = None
    if not args.serial:
        pool = multiprocessing.Pool(processes=16, initializer=init_worker)
    
    for gen in tqdm(range(1, generations), desc="Generations"):
        best_fitness = -float('inf')
        best_genome = None
        
        # Evaluate all genomes
        if args.serial:
            # Serial evaluation
            for species in pop.m_Species:
                for individual in species.m_Individuals:
                    fitness = evaluate_genome((individual, False, MAX_TIMESTEPS, NUM_TRIALS))
                    individual.SetFitness(fitness)
                    
                    # Track best genome
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_genome = individual
        else:
            # Parallel evaluation with persistent pool
            genomes = [individual for species in pop.m_Species for individual in species.m_Individuals]
            # Prepare arguments for evaluation
            args_list = [(genome, False, MAX_TIMESTEPS, NUM_TRIALS) for genome in genomes]
            
            # Evaluate genomes in parallel using persistent pool
            fitnesses = pool.map(evaluate_genome, args_list)
            
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
        
        # Draw the best genome in the third subplot
        if best_genome:
            ax3.clear()
            DrawGenome(best_genome, ax=ax3)
            ax3.set_title(f"Best Genome (Gen {gen})")
        
        # Redraw figures
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Print generation stats
        print(f"\nGeneration {gen}: Best Fitness = {best_fitness:.2f}")
        
        # Render best individual every N generations
        if best_genome and gen % 100 == 0:
            print(f"\nRendering best individual from generation {gen}...")
            for i in range(5):
                print(f"Episode {i+1} (Press ESC to skip remaining episodes)")
                # Evaluate with rendering in the main process
                evaluate_genome((best_genome, True, MAX_TIMESTEPS, 1))
        
        # Advance to next generation
        pop.Epoch()
    
    # Clean up process pool
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