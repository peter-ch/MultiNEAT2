#!/usr/bin/env python3
# car_racing_neat_resnet_pca.py

import gymnasium as gym
import pymultineat as pnt
import numpy as np
import time
import multiprocessing
from tqdm import tqdm
import pygame
from PIL import Image
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import joblib
import os

# Global variables for feature extraction
model = None
transform = None
pca = None

def setup_model():
    """Initialize ResNet model and transforms"""
    # Load pretrained ResNet18 without classifier
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, transform

def extract_features(image):
    """Extract 32D features using ResNet18 and PCA"""
    global model, transform, pca
    pil_image = Image.fromarray(image)
    tensor_img = transform(pil_image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        features = model(tensor_img).flatten().numpy()
    
    return pca.transform(features.reshape(1, -1)).flatten()

def collect_frames_for_pca(num_frames=300):
    """Collect diverse frames from random policy for PCA training"""
    env = gym.make('CarRacing-v2', render_mode='human')
    frames = []
    ikk = 0
    obs, _ = env.reset()
    
    pbar = tqdm(total=num_frames, desc="Collecting frames for PCA")
    while len(frames) < num_frames:
        action = env.action_space.sample()
        next_obs, _, done, truncated, _ = env.step(action)
        
        # Only collect every 5th frame for diversity
        if ikk % 5 == 0:
            frames.append(next_obs)
            pbar.update(1)
        
        if done or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
        ikk += 1
    
    env.close()
    pbar.close()
    return frames

def train_pca(frames, n_components=32):
    """Train PCA on ResNet features extracted from frames"""
    global model, transform
    features_list = []
    
    for frame in tqdm(frames, desc="Extracting ResNet features"):
        pil_image = Image.fromarray(frame)
        tensor_img = transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            features = model(tensor_img).flatten().numpy()
        features_list.append(features)
    
    features_matrix = np.array(features_list)
    pca = PCA(n_components=n_components)
    pca.fit(features_matrix)
    print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")
    return pca

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
        
        # Extract 32D features using ResNet+PCA
        features = extract_features(observation)
        inputs = features.tolist() + [1.0]  # Add bias
        
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

def init_worker():
    """Initialize worker process with ResNet and PCA"""
    global model, transform, pca
    model, transform = setup_model()
    pca = joblib.load('pca_params.pkl')

def evaluate_genome_worker(args):
    genome, render, max_steps = args
    env = gym.make('CarRacing-v2', render_mode='human' if render else None)
    try:
        return evaluate_genome(genome, env, render, max_steps)
    finally:
        env.close()

def main():
    global model, transform, pca
    
    parser = argparse.ArgumentParser(description='Car Racing NEAT')
    parser.add_argument('--serial', action='store_true', help='Use serial evaluation instead of parallel')
    args = parser.parse_args()
    
    pygame.init()
    pygame.display.set_mode((1, 1))
    
    # Initialize model and transforms
    model, transform = setup_model()
    
    # PCA file management
    pca_file = 'pca_params.pkl'
    if not os.path.exists(pca_file):
        print("Training PCA...")
        frames = collect_frames_for_pca(300)
        pca = train_pca(frames, 32)
        joblib.dump(pca, pca_file)
        print("PCA trained and saved")
    else:
        print("Using existing PCA model")
        pca = joblib.load(pca_file)

    params = pnt.Parameters()
    params.PopulationSize = 120
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
    params.RecurrentProb = 0.5
    params.OverallMutationRate = 0.4
    params.ArchiveEnforcement = False
    params.MutateWeightsProb = 0.25
    params.WeightMutationMaxPower = 0.5
    params.WeightReplacementMaxPower = 8.0
    params.MutateWeightsSevereProb = 0.0
    params.WeightMutationRate = 0.85
    params.WeightReplacementRate = 0.2
    params.MinWeight = -4.0
    params.MaxWeight = 4.0
    params.MutateAddNeuronProb = 0.02  
    params.MutateAddLinkProb = 0.15     
    params.MutateRemLinkProb = 0.02
    params.MinActivationA = 4.9
    params.MaxActivationA = 4.9
    params.ActivationFunction_SignedSigmoid_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
    params.ActivationFunction_Tanh_Prob = 1.0
    params.ActivationFunction_SignedStep_Prob = 0.0
    params.CrossoverRate = 0.0
    params.MultipointCrossoverRate = 0.0
    params.SurvivalRate = 0.25
    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0
    params.AllowLoops = True
    params.AllowClones = False
    
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 33  # 32 PCA features + 1 bias
    init_struct.NumOutputs = 3
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.HiddenActType = pnt.TANH
    init_struct.OutputActType = pnt.TANH

    genome_prototype = pnt.Genome(params, init_struct)
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))
    
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
            
            with multiprocessing.Pool(processes=16, initializer=init_worker) as pool:
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
    multiprocessing.set_start_method('spawn', force=True)
    main()