#!/usr/bin/env python3
# bipedal_walker_neat.py

import gymnasium as gym
import pymultineat as pnt
import numpy as np
import time
import multiprocessing
from tqdm import tqdm
import pygame  # For key press detection

# Worker initialization function for multiprocessing
def init_worker():
    global worker_env
    worker_env = gym.make('BipedalWalker-v3')

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
            env = gym.make('BipedalWalker-v3', render_mode='human')
        else:
            env = gym.make('BipedalWalker-v3')
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
    parser = argparse.ArgumentParser(description='Bipedal Walker NEAT')
    parser.add_argument('--serial', action='store_true', help='Use serial evaluation instead of parallel')
    args = parser.parse_args()
    
    # Initialize pygame for key detection
    pygame.init()
    pygame.display.set_mode((1, 1))  # Create a tiny window for event handling
    
    # Training environment is now created per worker process
    
    # Create a temporary environment for serial evaluation and rendering
    temp_env = gym.make('BipedalWalker-v3')
    
    # Create and customize MultiNEAT parameters
    params = pnt.Parameters()
    params.PopulationSize = 240
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = False
    params.WeightDiffCoeff = 0.1
    params.CompatTreshold = 3.0  
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 20
    params.OldAgeTreshold = 35
    params.MinSpecies = 3
    params.MaxSpecies = 12
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.3  
    params.OverallMutationRate = 0.4
    params.ArchiveEnforcement = False
    params.MutateWeightsProb = 0.25
    params.WeightMutationMaxPower = 0.5
    params.WeightReplacementMaxPower = 8.0
    params.MutateWeightsSevereProb = 0.0
    params.WeightMutationRate = 0.85
    params.WeightReplacementRate = 0.2
    params.MaxWeight = 8
    params.MutateAddNeuronProb = 0.02  
    params.MutateAddLinkProb = 0.15     
    params.MutateRemLinkProb = 0.02
    params.MinActivationA = 4.9
    params.MaxActivationA = 4.9
    params.ActivationFunction_SignedSigmoid_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
    params.ActivationFunction_Tanh_Prob = 1.0  # Use Tanh for symmetric outputs
    params.ActivationFunction_SignedStep_Prob = 0.0
    params.CrossoverRate = 0.0
    params.MultipointCrossoverRate = 0.0
    params.SurvivalRate = 0.25
    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0
    params.AllowLoops = True
    params.AllowClones = False

    # Create a GenomeInitStruct
    # 24 inputs (observations) + 1 bias = 25 inputs, 4 outputs
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 25
    init_struct.NumOutputs = 4
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.HiddenActType = pnt.TANH
    init_struct.OutputActType = pnt.TANH

    # Create a prototype genome
    genome_prototype = pnt.Genome(params, init_struct)

    # Create the initial population
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))

    generations = 250
    best_fitness_history = []
    
    for gen in tqdm(range(generations), desc="Generations"):
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
            # Parallel evaluation
            genomes = [individual for species in pop.m_Species for individual in species.m_Individuals]
            
            # Create process pool with worker initialization
            with multiprocessing.Pool(processes=16, initializer=init_worker) as pool:
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
        
        # Print generation stats
        print(f"\nGeneration {gen}: Best Fitness = {best_fitness:.2f}")
        
        # Render best individual every 50 generations
        if best_genome and gen % 50 == 0:
            print(f"\nRendering best individual from generation {gen}...")
            for i in range(10):
                print(f"Episode {i+1} (Press ESC to skip remaining episodes)")
                # Create a fresh render environment for each episode
                env_render = gym.make('BipedalWalker-v3', render_mode='human')
                try:
                    evaluate_genome(best_genome, env_render, render=True, max_steps=1000)
                finally:
                    env_render.close()
        
        # Advance to next generation
        pop.Epoch()
    
    print("\nTraining completed. Best fitness progression:")
    for gen, fitness in enumerate(best_fitness_history):
        print(f"Gen {gen}: {fitness:.2f}")

if __name__ == "__main__":
    main()