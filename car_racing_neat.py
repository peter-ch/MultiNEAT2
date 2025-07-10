#!/usr/bin/env python3
# car_racing_neat.py

import gym
import pymultineat as pnt
import numpy as np
import time
import multiprocessing
from tqdm import tqdm
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Feature extractor class using ResNet
class FeatureExtractor:
    def __init__(self):
        # Load pre-trained ResNet18
        self.model = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()  # Set to evaluation mode
        
        # Define image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(224),  # ResNet input size
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        """Extract features from a raw image array"""
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        # Preprocess and add batch dimension
        input_tensor = self.preprocess(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            features = self.model(input_tensor)
        
        # Flatten and reduce to 128 features
        features = features.squeeze().numpy()
        # If features are > 128, use PCA-like reduction
        if len(features) > 128:
            # Simple dimensionality reduction (in practice, use PCA or train a reducer)
            features = features[:128]
        return features

# Worker initialization function for multiprocessing
def init_worker():
    global worker_env, feature_extractor
    worker_env = gym.make('CarRacing-v2')
    feature_extractor = FeatureExtractor()

# Define the evaluation function for a genome
def evaluate_genome(genome, env=None, render=False, max_steps=1000):
    # Use worker environment if none provided
    if env is None:
        env = worker_env
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
        # Extract features from image
        features = feature_extractor.extract_features(observation)
        
        # Prepare inputs: features + bias
        inputs = features.tolist()
        inputs.append(1.0)  # Add bias
        
        nn.Input(inputs)
        nn.Activate()  # Activate only once per timestep
        outputs = nn.Output()
        
        # Continuous action space: use all outputs
        action = outputs[:3]  # Steering, gas, brake
        
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

import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Car Racing NEAT')
    parser.add_argument('--serial', action='store_true', help='Use serial evaluation instead of parallel')
    args = parser.parse_args()
    
    # Training environment is now created per worker process
    
    # Create rendering environment (only for demo purposes)
    env_render = gym.make('CarRacing-v2', render_mode='human')
    
    # Create a temporary environment for serial evaluation and rendering
    temp_env = gym.make('CarRacing-v2')
    
    # Create and customize MultiNEAT parameters
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

    # Create a GenomeInitStruct
    # 128 features + 1 bias = 129 inputs, 3 outputs
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 129
    init_struct.NumOutputs = 3
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.HiddenActType = pnt.TANH
    init_struct.OutputActType = pnt.TANH

    # Create a prototype genome
    genome_prototype = pnt.Genome(params, init_struct)

    # Create the initial population
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))

    generations = 100  # Fewer generations due to computational cost
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
            with multiprocessing.Pool(processes=8, initializer=init_worker) as pool:  # Fewer processes
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
        
        # Render best individual every 5 generations
        if best_genome and gen % 5 == 0:
            print(f"\nRendering best individual from generation {gen}...")
            for i in range(1):  # Fewer episodes due to rendering cost
                print(f"Episode {i+1}")
                evaluate_genome(best_genome, env_render, render=True, max_steps=1000)
        
        # Advance to next generation
        pop.Epoch()
    
    print("\nTraining completed. Best fitness progression:")
    for gen, fitness in enumerate(best_fitness_history):
        print(f"Gen {gen}: {fitness:.2f}")

if __name__ == "__main__":
    main()
