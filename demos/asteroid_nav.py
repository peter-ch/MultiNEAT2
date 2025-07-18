#!/usr/bin/env python
"""
neat_asteroids.py

This experiment runs an Asteroids game implemented with PyGame where a ship (controlled by
a NEAT neural network from pymultineat) must avoid asteroids. The ship’s sensor rays (“whiskers”)
detect nearby asteroids and the genome (with 8 sensor inputs plus a bias, and three outputs)
produces control signals for turning left/right and thrust. The simulation ends when a collision
occurs, and fitness is measured by the number of timesteps survived.

Pressing the F key toggles “fast mode” so that evaluation is performed without rendering.
When fast mode is off the simulation shows only the best-ever individual being replayed as a demo.
"""

from __future__ import annotations
import pygame
import math
import time
import random
import sys
from typing import Optional, List, Tuple
import pymultineat as pnt

# Import numba and numpy for optimized routines.
import numba
import numpy as np

# -----------------------------
# Constants and configuration
# -----------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Global flag: when True, training is done in non‐rendered fast mode.
# When False, a best‐genome demo is run.
FAST_MODE = True

# Colors.
BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)
YELLOW = (255, 255, 0)
GRAY   = (180, 180, 180)

# Physics and ship constants.
SHIP_RADIUS = 10
ROTATION_MULTIPLIER = 3        # degrees per frame
FORWARD_ACCELERATION = 0.5     # acceleration per frame (for forward thrust)
FRICTION = 0.95                # damp the ship’s velocity
MAX_VELOCITY = 3.0

# Sensor configuration.
NUM_SENSORS = 16
MAX_SENSOR_RANGE = 150         # sensor range in pixels
# Evenly spaced sensor offsets (sensor 0 is in ship’s forward direction).
SENSOR_OFFSETS = [i * (360 / NUM_SENSORS) for i in range(NUM_SENSORS)]
# Make a numpy array version (for numba functions).
SENSOR_OFFSETS_NP = np.array(SENSOR_OFFSETS, dtype=np.float64)

# Asteroid settings.
NUM_ASTEROIDS = 50
ASTEROID_MIN_RADIUS = 15
ASTEROID_MAX_RADIUS = 30
ASTEROID_MIN_SPEED = 0.5
ASTEROID_MAX_SPEED = 2.0

# Trial settings.
MAX_TRIAL_TIME = 30000          

# -----------------------------
# Numba-accelerated helper functions
# -----------------------------
@numba.njit
def ray_circle_intersection_numba(ox, oy, dx, dy, cx, cy, r):
    # Computes intersection t along a ray (origin + t*direction)
    f0 = ox - cx
    f1 = oy - cy
    a = 1.0  # since direction is normalized
    b = 2.0 * (dx * f0 + dy * f1)
    c = f0 * f0 + f1 * f1 - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return -1.0  # no intersection
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    if t1 >= 0:
        return t1
    elif t2 >= 0:
        return t2
    else:
        return -1.0

@numba.njit
def get_sensor_readings_numba(ship_x, ship_y, ship_angle, sensor_offsets, asteroid_positions, asteroid_radii, max_sensor_range):
    num_sensors = sensor_offsets.shape[0]
    readings = np.empty(num_sensors, dtype=np.float64)
    endpoints = np.empty((num_sensors, 2), dtype=np.float64)
    for i in range(num_sensors):
        sensor_angle = (ship_angle + sensor_offsets[i]) * (math.pi / 180.0)
        dx = math.cos(sensor_angle)
        dy = math.sin(sensor_angle)
        min_t = max_sensor_range + 1.0
        for j in range(asteroid_positions.shape[0]):
            t = ray_circle_intersection_numba(ship_x, ship_y, dx, dy,
                                              asteroid_positions[j, 0],
                                              asteroid_positions[j, 1],
                                              asteroid_radii[j])
            if t >= 0 and t < min_t:
                min_t = t
        if min_t <= max_sensor_range:
            reading = (max_sensor_range - min_t) / max_sensor_range
            endpoints[i, 0] = ship_x + dx * min_t
            endpoints[i, 1] = ship_y + dy * min_t
        else:
            reading = 0.0
            endpoints[i, 0] = ship_x + dx * max_sensor_range
            endpoints[i, 1] = ship_y + dy * max_sensor_range
        readings[i] = reading
    return readings, endpoints

@numba.njit
def update_ship_numba(ship_x, ship_y, ship_vx, ship_vy, ship_angle, rotation, thrust,
                      max_velocity, friction, screen_width, screen_height):
    ship_angle = (ship_angle + rotation) % 360.0
    rad = ship_angle * (math.pi / 180.0)
    forward_x = math.cos(rad)
    forward_y = math.sin(rad)
    ship_vx += forward_x * thrust
    ship_vy += forward_y * thrust
    speed = math.sqrt(ship_vx * ship_vx + ship_vy * ship_vy)
    if speed > max_velocity:
        ship_vx = ship_vx / speed * max_velocity
        ship_vy = ship_vy / speed * max_velocity
    ship_x = ship_x + ship_vx
    ship_y = ship_y + ship_vy
    ship_x = ship_x % screen_width
    ship_y = ship_y % screen_height
    ship_vx *= friction
    ship_vy *= friction
    return ship_x, ship_y, ship_vx, ship_vy, ship_angle

@numba.njit
def update_asteroids_numba(asteroid_positions, asteroid_velocities, screen_width, screen_height):
    for i in range(asteroid_positions.shape[0]):
        asteroid_positions[i, 0] = (asteroid_positions[i, 0] + asteroid_velocities[i, 0]) % screen_width
        asteroid_positions[i, 1] = (asteroid_positions[i, 1] + asteroid_velocities[i, 1]) % screen_height
    return asteroid_positions

@numba.njit
def check_collision_numba(ship_x, ship_y, ship_radius, asteroid_positions, asteroid_radii):
    for i in range(asteroid_positions.shape[0]):
        dx = ship_x - asteroid_positions[i, 0]
        dy = ship_y - asteroid_positions[i, 1]
        total_radius = ship_radius + asteroid_radii[i]
        if dx * dx + dy * dy < total_radius * total_radius:
            return True
    return False

# -----------------------------
# Original helper functions (non-physics)
# -----------------------------
def wrap_position(pos: pygame.math.Vector2) -> None:
    pos.x %= SCREEN_WIDTH
    pos.y %= SCREEN_HEIGHT

# -----------------------------
# Game object classes
# -----------------------------
class Ship:
    def __init__(self, pos: Tuple[float, float], angle: float = -90) -> None:
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = angle

    def draw(self, screen: pygame.Surface) -> None:
        point1 = pygame.math.Vector2(0, -SHIP_RADIUS * 2)
        point2 = pygame.math.Vector2(-SHIP_RADIUS, SHIP_RADIUS)
        point3 = pygame.math.Vector2(SHIP_RADIUS, SHIP_RADIUS)
        points = []
        for point in [point1, point2, point3]:
            rotated_point = point.rotate(self.angle + 90)
            points.append((self.pos.x + rotated_point.x, self.pos.y + rotated_point.y))
        pygame.draw.polygon(screen, WHITE, points)

class Asteroid:
    def __init__(self, pos: Optional[pygame.math.Vector2] = None, radius: Optional[float] = None) -> None:
        if pos is None:
            margin = 50
            pos = pygame.math.Vector2(
                random.uniform(margin, SCREEN_WIDTH - margin),
                random.uniform(margin, SCREEN_HEIGHT - margin)
            )
        self.pos = pos
        if radius is None:
            self.radius = random.uniform(ASTEROID_MIN_RADIUS, ASTEROID_MAX_RADIUS)
        else:
            self.radius = radius
        angle = random.uniform(0, 360)
        speed = random.uniform(ASTEROID_MIN_SPEED, ASTEROID_MAX_SPEED)
        self.vel = pygame.math.Vector2(speed, 0).rotate(angle)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(screen, GRAY, (int(self.pos.x), int(self.pos.y)), int(self.radius), 2)

def spawn_asteroids(ship: Ship) -> List[Asteroid]:
    asteroids: List[Asteroid] = []
    max_attempts = 10000
    attempts = 0
    while len(asteroids) < NUM_ASTEROIDS and attempts < max_attempts:
        attempts += 1
        candidate_radius = random.uniform(ASTEROID_MIN_RADIUS, ASTEROID_MAX_RADIUS)
        candidate_pos = pygame.math.Vector2(
            random.uniform(candidate_radius, SCREEN_WIDTH - candidate_radius),
            random.uniform(candidate_radius, SCREEN_HEIGHT - candidate_radius)
        )
        if (candidate_pos - ship.pos).length() < 200:
            continue
        overlap = False
        for asteroid in asteroids:
            min_distance = candidate_radius + asteroid.radius
            if (candidate_pos - asteroid.pos).length() < min_distance:
                overlap = True
                break
        if overlap:
            continue
        new_ast = Asteroid(pos=candidate_pos, radius=candidate_radius)
        asteroids.append(new_ast)
    if len(asteroids) < NUM_ASTEROIDS:
        print("Warning: Could not place all asteroids without overlap after many attempts.", flush=True)
    return asteroids

# -----------------------------
# Asteroids Simulation Runner
# -----------------------------
class AsteroidsSimulation:
    def __init__(self, genome: pnt.Genome, screen: pygame.Surface) -> None:
        self.screen = screen
        self.simulation_steps = 0
        self.max_steps = MAX_TRIAL_TIME * FPS
        # Build the neural network phenotype.
        self.nn = pnt.NeuralNetwork()
        genome.BuildPhenotype(self.nn)
        self.ship = Ship(pos=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        self.asteroids = spawn_asteroids(self.ship)
        self.clock = pygame.time.Clock()
        self.sensor_endpoints: List[pygame.math.Vector2] = []
        self.last_status_print = 0

    def handle_events(self) -> None:
        global FAST_MODE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    # Toggling fast mode. When toggled, we want to switch between training and demo modes.
                    FAST_MODE = not FAST_MODE
                    mode_text = "TRAINING (fast mode)" if FAST_MODE else "DEMO (best individual)"
                    print(f"Toggled mode: {mode_text}", flush=True)

    def update(self) -> bool:
        self.handle_events()
        self.simulation_steps += 1

        n_ast = len(self.asteroids)
        asteroid_positions = np.empty((n_ast, 2), dtype=np.float64)
        asteroid_radii = np.empty(n_ast, dtype=np.float64)
        asteroid_velocities = np.empty((n_ast, 2), dtype=np.float64)
        for i, asteroid in enumerate(self.asteroids):
            asteroid_positions[i, 0] = asteroid.pos.x
            asteroid_positions[i, 1] = asteroid.pos.y
            asteroid_radii[i] = asteroid.radius
            asteroid_velocities[i, 0] = asteroid.vel.x
            asteroid_velocities[i, 1] = asteroid.vel.y

        ship_x = self.ship.pos.x
        ship_y = self.ship.pos.y
        ship_vx = self.ship.vel.x
        ship_vy = self.ship.vel.y
        ship_angle = self.ship.angle

        sensor_values, sensor_endpoints_arr = get_sensor_readings_numba(
            ship_x, ship_y, ship_angle, SENSOR_OFFSETS_NP,
            asteroid_positions, asteroid_radii, MAX_SENSOR_RANGE
        )
        if not FAST_MODE:
            self.sensor_endpoints = [
                pygame.math.Vector2(sensor_endpoints_arr[i, 0], sensor_endpoints_arr[i, 1])
                for i in range(sensor_endpoints_arr.shape[0])
            ]
        else:
            self.sensor_endpoints = []

        inputs = sensor_values.tolist() + [1.0]
        self.nn.Input(inputs)
        self.nn.Activate()
        outputs = self.nn.Output()

        rotation = (outputs[1] - outputs[0]) * ROTATION_MULTIPLIER

        # New thrust mapping: output[2] in [0,1] is scaled to [-1,1].
        # Forward thrust is full (FORWARD_ACCELERATION), reverse thrust is 40% as strong.
        thrust_signal = 2.0 * (outputs[2] - 0.5)
        if thrust_signal >= 0:
            thrust = thrust_signal * FORWARD_ACCELERATION
        else:
            thrust = thrust_signal * 0.4 * FORWARD_ACCELERATION

        ship_x, ship_y, ship_vx, ship_vy, ship_angle = update_ship_numba(
            ship_x, ship_y, ship_vx, ship_vy, ship_angle, rotation, thrust,
            MAX_VELOCITY, FRICTION, SCREEN_WIDTH, SCREEN_HEIGHT
        )
        self.ship.pos.x = ship_x
        self.ship.pos.y = ship_y
        self.ship.vel.x = ship_vx
        self.ship.vel.y = ship_vy
        self.ship.angle = ship_angle

        asteroid_positions = update_asteroids_numba(asteroid_positions, asteroid_velocities, SCREEN_WIDTH, SCREEN_HEIGHT)
        for i, asteroid in enumerate(self.asteroids):
            asteroid.pos.x = asteroid_positions[i, 0]
            asteroid.pos.y = asteroid_positions[i, 1]
            asteroid.vel.x = asteroid_velocities[i, 0]
            asteroid.vel.y = asteroid_velocities[i, 1]

        if check_collision_numba(ship_x, ship_y, SHIP_RADIUS, asteroid_positions, asteroid_radii):
            return False

        if self.simulation_steps >= self.max_steps:
            return False

        return True

    def draw(self) -> None:
        self.screen.fill(BLACK)
        for asteroid in self.asteroids:
            asteroid.draw(self.screen)
        for endpoint in self.sensor_endpoints:
            pygame.draw.line(self.screen, YELLOW, (self.ship.pos.x, self.ship.pos.y),
                             (endpoint.x, endpoint.y), 1)
        self.ship.draw(self.screen)
        font = pygame.font.SysFont("Arial", 18)
        sim_time = self.simulation_steps / FPS
        time_text = font.render(f"SimTime: {sim_time:.2f}s", True, WHITE)
        self.screen.blit(time_text, (10, 10))
        pygame.display.flip()

    def run(self) -> int:
        running = True
        global FAST_MODE
        while running:
            running = self.update()
            if not FAST_MODE:
                self.draw()
                self.clock.tick(FPS)
        return self.simulation_steps

# -----------------------------
# Main NEAT Loop and Mode Switch
# -----------------------------
def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NEAT Asteroids Experiment")

    # Setup MultiNEAT parameters.
    params = pnt.Parameters()
    params.PopulationSize = 500
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = False
    params.WeightDiffCoeff = 0.1
    params.CompatTreshold = 4.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 15
    params.OldAgeTreshold = 35
    params.MinSpecies = 2
    params.MaxSpecies = 12
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.0
    params.OverallMutationRate = 0.3
    params.ArchiveEnforcement = False
    params.MutateWeightsProb = 0.25
    params.WeightMutationMaxPower = 0.5
    params.WeightReplacementMaxPower = 8.0
    params.MutateWeightsSevereProb = 0.0
    params.WeightMutationRate = 0.85
    params.WeightReplacementRate = 0.2
    params.MaxWeight = 8.0
    params.MutateAddNeuronProb = 0.01
    params.MutateAddLinkProb = 0.1
    params.MutateRemLinkProb = 0.0
    params.MutateRemSimpleNeuronProb = 0.0
    params.NeuronTries = 64
    params.MutateAddLinkFromBiasProb = 0.0
    params.CrossoverRate = 0.75
    params.MultipointCrossoverRate = 0.0
    params.SurvivalRate = 0.2
    params.MutateNeuronTraitsProb = 0.0
    params.MutateLinkTraitsProb = 0.0
    params.AllowLoops = True
    params.AllowClones = True

    # Genome initialization: NUM_SENSORS (plus bias) inputs and 3 outputs.
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = NUM_SENSORS + 1
    init_struct.NumOutputs = 3
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON    
    init_struct.HiddenActType = pnt.UNSIGNED_SIGMOID
    init_struct.OutputActType = pnt.UNSIGNED_SIGMOID

    genome_prototype = pnt.Genome(params, init_struct)
    pop = pnt.Population(genome_prototype, params, True, 1.0, int(time.time()))

    GENERATIONS = 5000000
    gen = 0

    # Main loop switches between two modes:
    # When FAST_MODE is True, evolution runs in fast (non-rendered) training mode.
    # When FAST_MODE is False, we run a continuous demo of only the best individual.
    while True:
        global FAST_MODE
        if FAST_MODE:
            # Training mode: run one generation.
            total_fitness = 0.0
            num_evaluated = 0
            for species in pop.m_Species:
                for idx in range(len(species.m_Individuals)):
                    genome = species.m_Individuals[idx]
                    simulation = AsteroidsSimulation(genome, screen)
                    fitness = simulation.run()
                    # If fast mode is turned off mid-evaluation, break out immediately.
                    if not FAST_MODE:
                        break
                    genome.SetFitness(fitness)
                    genome.SetEvaluated()
                    total_fitness += fitness
                    num_evaluated += 1
                if not FAST_MODE:
                    break
            # Only complete generation if still in training mode.
            if FAST_MODE:
                bestGenome = pop.GetBestGenome()
                bestFitness = bestGenome.GetFitness()
                avgFitness = total_fitness / num_evaluated if num_evaluated > 0 else 0.0

                print("\n" + "="*60)
                print(f"Generation: {gen}")
                print(f"Overall Best Fitness: {bestFitness}")
                print(f"Average Fitness: {avgFitness:.2f}")
                print(f"Total Evaluated Individuals: {num_evaluated}")
                print(f"Number of Species: {len(pop.m_Species)}")
                for i, species in enumerate(pop.m_Species, start=1):
                    species_size = len(species.m_Individuals)
                    best_species_fitness = max((ind.GetFitness() for ind in species.m_Individuals), default=0.0)
                    print(f"  Species {i:2d}: Size = {species_size:3d}, Best Fitness = {best_species_fitness:.2f}")
                print(f"Compatibility Threshold: {params.CompatTreshold:.2f}")
                print("="*60 + "\n", flush=True)

                pop.Epoch()
                gen += 1
        else:
            # Demo mode: continuously replay the best genome.
            bestGenome = pop.GetBestGenome()
            print("Demo mode: Replaying best individual... (Press F to resume training)", flush=True)
            simulation = AsteroidsSimulation(bestGenome, screen)
            simulation.run()

if __name__ == "__main__":
    main()