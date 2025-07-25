#!/usr/bin/env python3
# hopper_neat.py

import gymnasium as gym
import pymultineat as pnt
import numpy as np
import time
import multiprocessing
from tqdm import tqdm
import pygame  # For key press detection
import matplotlib.pyplot as plt
import argparse
from neattools import DrawGenome  # Import the DrawGenome function

# Constant to ensure all fitness values are positive
FITNESS_SHIFT = 1000.0

# Worker initialization function for multiprocessing
def init_worker():
    global worker_env
    worker_env = gym.make('Hopper-v5')

# Function to measure min/max values for each observation feature
def measure_observation_bounds(env, num_episodes=10, max_steps=100):
    min_vals = None
    max_vals = None
    
    for _ in range(num_episodes):
        observation_data = env.reset()
        if isinstance(observation_data, tuple):
            observation = observation_data[0]  # (observation, info) format
        else:
            observation = observation_data
        
        if min_vals is None:
            min_vals = np.array(observation)
            max_vals = np.array(observation)
        else:
            min_vals = np.minimum(min_vals, observation)
            max_vals = np.maximum(max_vals, observation)
        
        for _ in range(max_steps):
            action = env.action_space.sample()  # Random action
            step_result = env.step(action)
            if len(step_result) == 4:
                observation, _, done, _ = step_result
            else:
                observation, _, done, _, _ = step_result
            
            min_vals = np.minimum(min_vals, observation)
            max_vals = np.maximum(max_vals, observation)
            
            if done:
                break
    
    # Avoid division by zero in case min == max for any feature
    for i in range(len(min_vals)):
        if min_vals[i] == max_vals[i]:
            max_vals[i] += 1e-6  # Small offset
    
    return min_vals, max_vals

# Define the evaluation function for a genome
def evaluate_genome(genome, env=None, render=False, max_steps=300, obs_min=None, obs_max=None):
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
            env = gym.make('Hopper-v5', render_mode='human')
        else:
            env = gym.make('Hopper-v5')
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
                    return max(0, total_reward + FITNESS_SHIFT)  # Ensure fitness is non-negative
        
        # Normalize observation (excluding the bias term)
        observation_list = observation.tolist() if hasattr(observation, 'tolist') else list(observation)
        normalized_obs = []
        for i, val in enumerate(observation_list):
            # Clip to avoid extreme values (optional, but safer)
            clipped_val = np.clip(val, obs_min[i], obs_max[i])
            # Scale to [-1, 1]
            normalized_val = (clipped_val - obs_min[i]) / (obs_max[i] - obs_min[i]) * 2 - 1
            normalized_obs.append(normalized_val)
        
        # Add bias (1.0) without normalization
        inputs = normalized_obs + [1.0]
        
        nn.Input(inputs)
        nn.Activate()  # Activate only once per timestep
        outputs = nn.Output()
        
        # Scale outputs to [-1, 1] range (tanh already does this)
        action = (np.array(outputs)-0.5)*2.0
        
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
            
    # Ensure fitness is non-negative by adding the shift constant
    fitness = max(0, total_reward + FITNESS_SHIFT)
    return fitness

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Hopper NEAT')
    parser.add_argument('--serial', action='store_true', help='Use serial evaluation instead of parallel')
    args = parser.parse_args()
    
    # Initialize pygame for key detection
    pygame.init()
    pygame.display.set_mode((1, 1))  # Create a tiny window for event handling
    
    # Set up matplotlib figure for fitness tracking and genome visualization
    plt.ion()  # Turn on interactive mode
    fig, (ax_fitness, ax_genome) = plt.subplots(1, 2, figsize=(15, 5))
    ax_fitness.set_title('Best Fitness per Generation')
    ax_fitness.set_xlabel('Generation')
    ax_fitness.set_ylabel('Fitness')
    line, = ax_fitness.plot([], [], 'b-')  # Create an empty line
    best_fitness_history = []
    
    # Create a temporary environment for measuring observation bounds
    temp_env = gym.make('Hopper-v5')
    obs_min, obs_max = measure_observation_bounds(temp_env)
    temp_env.close()
    
    print(f"Measured observation bounds (min): {obs_min}")
    print(f"Measured observation bounds (max): {obs_max}")
    
    # Create a temporary environment for serial evaluation and rendering
    temp_env = gym.make('Hopper-v5')
    
    # Create and customize MultiNEAT parameters
    params = pnt.Parameters()
    params.PopulationSize = 250
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = False
    params.WeightDiffCoeff = 0.035
    params.CompatTreshold = 1.5  
    params.YoungAgeTreshold = 10
    params.SpeciesMaxStagnation = 12
    params.OldAgeTreshold = 30
    params.MinSpecies = 2
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False
    params.TournamentSelection = True
    params.TournamentSize = 5
    params.RecurrentProb = 0.2 
    params.OverallMutationRate = 0.333
    params.MutateWeightsProb = 0.75
    params.WeightMutationMaxPower = 1.5
    params.WeightReplacementMaxPower = 4.0
    params.MutateWeightsSevereProb = 0.2
    params.WeightMutationRate = 0.75
    params.WeightReplacementRate = 0.2
    params.MinWeight = -4.0
    params.MaxWeight = 4.0
    params.MutateAddNeuronProb = 0.005
    params.MutateAddLinkProb = 0.05    
    params.MutateRemLinkProb = 0.02
    params.SplitRecurrent = True 
    params.SplitLoopedRecurrent = True
    params.MinActivationA = 4.0
    params.MaxActivationA = 4.0
    params.MutateActivationAProb = 0.0
    params.ActivationAMutationMaxPower = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
    params.ActivationFunction_Tanh_Prob = 0.0  
    params.ActivationFunction_Relu_Prob = 0.0
    params.ActivationFunction_Softplus_Prob = 0.0
    params.ActivationFunction_Linear_Prob = 0.0
    params.MutateNeuronActivationTypeProb = 0
    params.CrossoverRate = 0.4
    params.MultipointCrossoverRate = 0.4
    params.InterspeciesCrossoverRate = 0.05
    params.SurvivalRate = 0.5
    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0
    params.AllowLoops = True
    params.AllowClones = True
    params.EliteFraction = 0.02
    params.ArchiveEnforcement = False

    # Create a GenomeInitStruct
    # 11 inputs (observations) + 1 bias = 12 inputs, 3 outputs (actions)
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 12
    init_struct.NumOutputs = 3
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.HiddenActType = pnt.UNSIGNED_SIGMOID
    init_struct.OutputActType = pnt.UNSIGNED_SIGMOID

    # Create a prototype genome
    genome_prototype = pnt.Genome(params, init_struct)

    # Create the initial population
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))

    generations = 2500
    
    # Create persistent process pool for parallel evaluation
    pool = None
    if not args.serial:
        print("Creating persistent process pool with 16 workers...")
        pool = multiprocessing.Pool(processes=16, initializer=init_worker)
    
    try:
        for gen in tqdm(range(1, generations), desc="Generations"):
            best_fitness = -float('inf')
            best_genome = None
            
            # Evaluate all genomes
            if args.serial:
                # Serial evaluation
                for species in pop.m_Species:
                    for individual in species.m_Individuals:
                        fitness = evaluate_genome(individual, temp_env, obs_min=obs_min, obs_max=obs_max)
                        individual.SetFitness(fitness)
                        
                        # Track best genome
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_genome = individual
            else:
                # Parallel evaluation using persistent pool
                genomes = [individual for species in pop.m_Species for individual in species.m_Individuals]
                
                # Evaluate genomes in parallel using starmap_async for better performance
                results = pool.starmap_async(
                    evaluate_genome, 
                    [(g, None, False, 300, obs_min, obs_max) for g in genomes]
                )
                fitnesses = results.get()
                
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
            ax_fitness.relim()
            ax_fitness.autoscale_view()
            
            # Visualize the best genome
            if best_genome:
                ax_genome.clear()
                DrawGenome(best_genome, ax=ax_genome)
            
            plt.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Print generation stats
            print(f"\nGeneration {gen}: Best Fitness = {best_fitness:.2f}")
            
            # Render best individual every N generations
            if best_genome and gen % 150 == 0:
                print(f"\nRendering best individual from generation {gen}...")
                for i in range(5):
                    print(f"Episode {i+1} (Press ESC to skip remaining episodes)")
                    # Create a fresh render environment for each episode
                    env_render = gym.make('Hopper-v5', render_mode='human')
                    try:
                        evaluate_genome(best_genome, env_render, render=True, max_steps=300, obs_min=obs_min, obs_max=obs_max)
                    finally:
                        env_render.close()
            
            # Advance to next generation
            pop.Epoch()
    finally:
        # Clean up resources
        temp_env.close()
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