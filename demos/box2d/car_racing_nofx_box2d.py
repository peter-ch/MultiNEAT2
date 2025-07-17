#!/usr/bin/env python3
# car_racing_neat_simple.py

import gymnasium as gym
import pymultineat as pnt
import numpy as np
import time
import multiprocessing
from tqdm import tqdm
import pygame
from PIL import Image
import argparse

# Define all functions at module level that need to be pickled
def preprocess_image(image):
    """Resize image to 10x10 with minimal info loss and normalize for neural network"""
    pil_image = Image.fromarray(image)
    
    # Use Lanczos resampling (best for downsampling)
    pil_image = pil_image.resize((10, 10), Image.Resampling.LANCZOS)
    
    # Convert to grayscale only if needed (preserve more info for color images)
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    
    # Normalize to 0-1 range while preserving more precision
    img_array = np.array(pil_image, dtype=np.float32) / 255.0
    
    return img_array.flatten()

def evaluate_genome(genome, env=None, render=False, max_steps=200):
    nn = pnt.NeuralNetwork()
    genome.BuildPhenotype(nn)
    
    try:
        observation_data = env.reset()
    except (pygame.error, gym.error.Error):
        if render:
            env = gym.make('CarRacing-v2', render_mode='human')
        else:
            env = gym.make('CarRacing-v2')
        observation_data = env.reset()
    
    observation = observation_data[0] if isinstance(observation_data, tuple) else observation_data
    total_reward = 0
    min_reward = 0
    step_count = 0
    
    while True:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return total_reward + abs(min_reward) + 1
        
        pixels = preprocess_image(observation)
        inputs = pixels.tolist() + [1.0]  # Add bias
        
        nn.Input(inputs)
        nn.Activate()
        outputs = nn.Output()
        action = outputs[:3]  # Steering, gas, brake
        
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
                pygame.time.delay(10)
            except (pygame.error, gym.error.Error):
                pass
            
        if done or step_count >= max_steps:
            break
            
    return total_reward + abs(min_reward) + 1

# Worker function needs to be defined at module level
def evaluate_genome_worker(args):
    genome, render, max_steps = args
    env = gym.make('CarRacing-v2', render_mode='human' if render else None)
    try:
        return evaluate_genome(genome, env, render, max_steps)
    finally:
        env.close()

def main():
    parser = argparse.ArgumentParser(description='Car Racing NEAT')
    parser.add_argument('--serial', action='store_true', help='Use serial evaluation instead of parallel')
    args = parser.parse_args()
    
    pygame.init()
    pygame.display.set_mode((1, 1))
    

    params = pnt.Parameters()
    params.PopulationSize = 120  # Reduced population size due to computational cost
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = False
    params.WeightDiffCoeff = 0.1
    params.CompatTreshold = 3.0  
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 20
    params.OldAgeTreshold = 35
    params.MinSpecies = 3
    params.MaxSpecies = 8
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.5  # Higher recurrent probability for temporal dependencies
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
    
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 101
    init_struct.NumOutputs = 3
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.HiddenActType = pnt.TANH
    init_struct.OutputActType = pnt.TANH

    genome_prototype = pnt.Genome(params, init_struct)
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))

    # Render initial individual
    print("\nRendering initial random individual...")
    initial_genome = pop.m_Species[0].m_Individuals[0]
    evaluate_genome_worker((initial_genome, True, 200))
    
    generations = 100
    best_fitness_history = []
    
    for gen in tqdm(range(generations), desc="Generations"):
        best_fitness = -float('inf')
        best_genome = None
        
        if args.serial:
            for species in pop.m_Species:
                for individual in species.m_Individuals:
                    fitness = evaluate_genome_worker((individual, False, 200))
                    individual.SetFitness(fitness)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_genome = individual
        else:
            genomes = [(individual, False, 200) for species in pop.m_Species 
                     for individual in species.m_Individuals]
            
            with multiprocessing.Pool(processes=16) as pool:
                fitnesses = pool.map(evaluate_genome_worker, genomes)
            
            idx = 0
            for species in pop.m_Species:
                for individual in species.m_Individuals:
                    individual.SetFitness(fitnesses[idx])
                    if fitnesses[idx] > best_fitness:
                        best_fitness = fitnesses[idx]
                        best_genome = individual
                    idx += 1
        
        best_fitness_history.append(best_fitness)
        print(f"\nGeneration {gen}: Best Fitness = {best_fitness:.2f}")
        
        if best_genome and gen % 5 == 0:
            print(f"\nRendering best individual from generation {gen}...")
            for i in range(3):
                print(f"Episode {i+1} (Press ESC to skip remaining episodes)")
                evaluate_genome_worker((best_genome, True, 200))
        
        pop.Epoch()
    
    print("\nTraining completed. Best fitness progression:")
    for gen, fitness in enumerate(best_fitness_history):
        print(f"Gen {gen}: {fitness:.2f}")

    pygame.quit()

if __name__ == "__main__":
    # Set start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()

