#!/usr/bin/env python3
# bipedal_walker_neat.py

import gym
import pymultineat as pnt
import numpy as np
import time
from tqdm import tqdm

# Define the evaluation function for a genome
def evaluate_genome(genome, env, render=False, max_steps=1000):
    nn = pnt.NeuralNetwork()
    genome.BuildPhenotype(nn)
    # Get initial observation
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
            env.render()
            
        if done or step_count >= max_steps:
            break
            
    # Adjust for negative rewards (NEAT requires non-negative fitness)
    fitness = total_reward + abs(min_reward) + 1
    return fitness

def main():
    # Create training environment (no rendering for faster evaluation)
    env_train = gym.make('BipedalWalker-v3')
    
    # Create rendering environment (only for demo purposes)
    env_render = gym.make('BipedalWalker-v3', render_mode='human')
    
    # Create and customize MultiNEAT parameters
    params = pnt.Parameters()
    params.PopulationSize = 300
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = False
    params.WeightDiffCoeff = 0.1
    params.CompatTreshold = 3.0  # Increased for more complex problem
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 20
    params.OldAgeTreshold = 35
    params.MinSpecies = 3
    params.MaxSpecies = 15
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.3  # Enable recurrence
    params.OverallMutationRate = 0.4
    params.ArchiveEnforcement = False
    params.MutateWeightsProb = 0.25
    params.WeightMutationMaxPower = 0.5
    params.WeightReplacementMaxPower = 8.0
    params.MutateWeightsSevereProb = 0.0
    params.WeightMutationRate = 0.85
    params.WeightReplacementRate = 0.2
    params.MaxWeight = 8
    params.MutateAddNeuronProb = 0.02  # Increased for complex task
    params.MutateAddLinkProb = 0.15     # Increased for complex task
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

    generations = 100
    best_fitness_history = []
    
    for gen in tqdm(range(generations), desc="Generations"):
        best_fitness = -float('inf')
        best_genome = None
        
        # Evaluate all genomes
        for species in pop.m_Species:
            for individual in species.m_Individuals:
                fitness = evaluate_genome(individual, env_train)
                individual.SetFitness(fitness)
                
                # Track best genome
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_genome = individual
        
        # Store best fitness for progress tracking
        best_fitness_history.append(best_fitness)
        
        # Print generation stats
        print(f"\nGeneration {gen}: Best Fitness = {best_fitness:.2f}")
        
        # Render best individual every 5 generations
        if best_genome and gen % 5 == 0:
            print(f"\nRendering best individual from generation {gen}...")
            for i in range(3):
                print(f"Episode {i+1}")
                evaluate_genome(best_genome, env_render, render=True)
        
        # Advance to next generation
        pop.Epoch()
    
    print("\nTraining completed. Best fitness progression:")
    for gen, fitness in enumerate(best_fitness_history):
        print(f"Gen {gen}: {fitness:.2f}")

if __name__ == "__main__":
    main()
