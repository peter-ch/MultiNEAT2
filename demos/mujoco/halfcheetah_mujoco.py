#!/usr/bin/env python3
# half_cheetah_neat.py

import gymnasium as gym
import pymultineat as pnt
import numpy as np
import time
import multiprocessing
from tqdm import tqdm
import pygame  # For key press detection
import matplotlib.pyplot as plt

# Worker initialization function for multiprocessing
def init_worker():
    global worker_env
    worker_env = gym.make('HalfCheetah-v5')

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
            env = gym.make('HalfCheetah-v5', render_mode='human')
        else:
            env = gym.make('HalfCheetah-v5')
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

import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Half Cheetah NEAT')
    parser.add_argument('--serial', action='store_true', help='Use serial evaluation instead of parallel')
    args = parser.parse_args()
    
    # Initialize pygame for key detection
    pygame.init()
    pygame.display.set_mode((1, 1))  # Create a tiny window for event handling
    
    # Set up matplotlib figure for fitness tracking
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Best Fitness per Generation')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    line, = ax.plot([], [], 'b-')  # Create an empty line
    best_fitness_history = []
    
    # Create a temporary environment for serial evaluation and rendering
    temp_env = gym.make('HalfCheetah-v5')
    
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
    # 17 inputs (observations) + 1 bias = 18 inputs, 6 outputs (actions)
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 18  # 17 observations + 1 bias
    init_struct.NumOutputs = 6  # 6 actions (torques for joints)
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.HiddenActType = pnt.TANH
    init_struct.OutputActType = pnt.TANH
    init_struct.FS_NEAT = True # start with only a few links
    init_struct.FS_NEAT_links = 5    

    # Create a prototype genome
    genome_prototype = pnt.Genome(params, init_struct)

    # Create the initial population
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))

    generations = 2500
    
    # Create process pool once at the beginning if using parallel mode
    pool = None
    if not args.serial:
        pool = multiprocessing.Pool(processes=16, initializer=init_worker)
    
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
                # Parallel evaluation with persistent pool
                genomes = [individual for species in pop.m_Species for individual in species.m_Individuals]
                
                # Evaluate genomes in parallel using the existing pool
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
            
            # Update the plot
            line.set_xdata(range(len(best_fitness_history)))
            line.set_ydata(best_fitness_history)
            ax.relim()
            ax.autoscale_view()
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
                    env_render = gym.make('HalfCheetah-v5', render_mode='human')
                    try:
                        evaluate_genome(best_genome, env_render, render=True, max_steps=500)
                    finally:
                        env_render.close()
            
            # Advance to next generation
            pop.Epoch()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
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