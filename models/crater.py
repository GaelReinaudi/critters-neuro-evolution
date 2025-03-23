"""
Crater model for the crater simulation
"""
import random
import math
import copy
import pygame
import numpy as np
from craters.config import (
    WIDTH, HEIGHT, SENSOR_RANGE, NUM_SENSORS,
    DIRECTION_COLOR, ENERGY_TEXT_COLOR, INITIAL_ENERGY,
    MAX_ENERGY, ENERGY_DEPLETION_RATE, ENERGY_ROTATION_COST,
    FONT_SIZE, MAX_SPEED, ACCELERATION_FACTOR, FRICTION,
    MUTATION_RATE, MUTATION_SCALE, MATING_COLOR,
    MATING_ENERGY_THRESHOLD, MATING_PROBABILITY, MATING_DURATION,
    YOUNG_COLOR, ADULT_COLOR, MATURE_COLOR, ELDER_COLOR,
    AGE_YOUNG, AGE_ADULT, AGE_MATURE, SENSOR_UPDATE_FRAMES,
    DISTANCE_CUTOFF, PRECOMPUTE_ANGLES, USE_DEEP_NETWORK,
    NETWORK_HIDDEN_LAYERS, NETWORK_ACTIVATION, FOOD_DETECTION_RANGE,
    WALL_DETECTION_RANGE
)
from craters.models.neural_network import SimpleNeuralNetwork, DeepNeuralNetwork

# Precompute sensor angles if enabled
if PRECOMPUTE_ANGLES:
    SENSOR_ANGLES = [(i * (2 * math.pi / NUM_SENSORS)) for i in range(NUM_SENSORS)]
    SIN_COS_CACHE = {angle: (math.sin(angle), math.cos(angle)) for angle in SENSOR_ANGLES}
    SIN_COS_CACHE.update({angle + math.pi*2: SIN_COS_CACHE[angle] for angle in SENSOR_ANGLES})

class Crater:
    """
    Represents a crater entity with neural network-based behavior
    """
    def __init__(self, x=None, y=None, size=None, font=None, brain=None):
        """
        Initialize crater with random or specified attributes
        
        Args:
            x (float, optional): X coordinate. If None, a random position is used.
            y (float, optional): Y coordinate. If None, a random position is used.
            size (int, optional): Crater size. If None, a random size is used.
            font: Pygame font for energy display
            brain: Neural network to use. If None, a new one is created.
        """
        # Initialize crater with random values if not provided
        self.x = x if x is not None else random.randint(50, WIDTH-50)
        self.y = y if y is not None else random.randint(50, HEIGHT-50)
        self.size = size if size is not None else random.randint(10, 30)
        self.rotation = random.uniform(0, 2 * math.pi)
        self.font = font
        
        # Movement properties
        self.max_speed = MAX_SPEED
        self.speed = 0
        self.angular_velocity = 0
        
        # Energy system
        self.energy = INITIAL_ENERGY
        self.max_energy = MAX_ENERGY
        
        # Create the triangular shape
        self.points = []
        self.generate_shape()
        
        # Neural network for crater behavior
        # Inputs:
        # - For each sensor direction:
        #   - Wall distance
        #   - General crater detection
        #   - Mating crater detection
        #   - Detected crater energy level
        #   - Food detection
        # - Own energy level
        if brain is None:
            # 5 sensor readings per direction (wall, crater, mating, energy, food) + own energy
            input_size = NUM_SENSORS * 5 + 1
            output_size = 3  # forward, reverse, rotation
            
            if USE_DEEP_NETWORK:
                self.brain = DeepNeuralNetwork(
                    input_size=input_size,
                    hidden_layers=NETWORK_HIDDEN_LAYERS,
                    output_size=output_size,
                    activation=NETWORK_ACTIVATION
                )
            else:
                self.brain = SimpleNeuralNetwork(input_size, 12, output_size)
        else:
            self.brain = brain
        
        # For visualizing sensor rays and storing sensor data
        # Structure: [wall_dist, crater_dist, mating_dist, energy_level, food_dist] * NUM_SENSORS
        self.sensor_readings = [1.0] * NUM_SENSORS * 5
        
        # Evolution tracking
        self.age = 0  # Age in frames
        self.food_eaten = 0  # Number of food pellets consumed
        self.distance_traveled = 0  # Total distance traveled
        
        # Inactivity tracking
        self.inactive_frames = 0  # Count of frames with minimal movement
        self.inactivity_threshold = 100  # Number of frames before applying penalty
        self.movement_threshold = 0.2  # Minimum movement required to be considered active
        
        # Mating state
        self.is_mating = False
        self.mating_timer = 0
        
        # Optimization
        self.frames_since_sensor_update = 0
        self.cached_sensor_data = None
    
    def generate_shape(self):
        """Create the triangular shape for the crater"""
        # Define the three points of the triangle relative to center (x, y)
        self.points = []
        
        if PRECOMPUTE_ANGLES:
            # Use precomputed sin/cos values
            sin_rot, cos_rot = math.sin(self.rotation), math.cos(self.rotation)
            sin_rot_plus, cos_rot_plus = math.sin(self.rotation + 2.09), math.cos(self.rotation + 2.09)
            sin_rot_minus, cos_rot_minus = math.sin(self.rotation - 2.09), math.cos(self.rotation - 2.09)
            
            # Front point (pointing in direction of rotation)
            front_x = self.x + self.size * cos_rot
            front_y = self.y + self.size * sin_rot
            # Two back points
            left_x = self.x + self.size * cos_rot_plus
            left_y = self.y + self.size * sin_rot_plus
            right_x = self.x + self.size * cos_rot_minus
            right_y = self.y + self.size * sin_rot_minus
        else:
            # Original calculation
            # Front point (pointing in direction of rotation)
            front_x = self.x + self.size * math.cos(self.rotation)
            front_y = self.y + self.size * math.sin(self.rotation)
            # Two back points
            left_x = self.x + self.size * math.cos(self.rotation + 2.09)  # ~120 degrees
            left_y = self.y + self.size * math.sin(self.rotation + 2.09)
            right_x = self.x + self.size * math.cos(self.rotation - 2.09)
            right_y = self.y + self.size * math.sin(self.rotation - 2.09)
        
        self.points = [(front_x, front_y), (left_x, left_y), (right_x, right_y)]
    
    def sense_environment(self, craters, food_pellets, force_update=False):
        """
        Cast rays in different directions to detect walls, other craters, and food
        
        Args:
            craters (list): List of all craters in the environment
            food_pellets (list): List of all food pellets
            force_update (bool): Whether to force update even if not time yet
            
        Returns:
            list: Sensor readings (normalized distances and entity information)
        """
        # Use cached sensor data if available and not time to update
        if not force_update and self.cached_sensor_data is not None:
            if self.frames_since_sensor_update < SENSOR_UPDATE_FRAMES:
                self.frames_since_sensor_update += 1
                return self.cached_sensor_data
                
        # Reset sensor update counter and readings
        self.frames_since_sensor_update = 0
        self.sensor_readings = []
        
        # Cast rays in multiple directions
        for i in range(NUM_SENSORS):
            if PRECOMPUTE_ANGLES:
                angle = self.rotation + SENSOR_ANGLES[i]
            else:
                angle = self.rotation + (i * (2 * math.pi / NUM_SENSORS))
            
            # Wall distance (limited to WALL_DETECTION_RANGE but normalized to SENSOR_RANGE)
            wall_distance = self.get_wall_distance(angle)
            wall_reading = min(wall_distance / SENSOR_RANGE, 1.0)
            self.sensor_readings.append(wall_reading)
            
            # Detect all types of craters
            crater_info = self.get_crater_info(angle, craters)
            
            # Regular crater distance (any crater)
            crater_distance = crater_info['distance']
            crater_reading = min(crater_distance / SENSOR_RANGE, 1.0)
            self.sensor_readings.append(crater_reading)
            
            # Mating crater distance
            mating_distance = crater_info['mating_distance']
            mating_reading = min(mating_distance / SENSOR_RANGE, 1.0)
            self.sensor_readings.append(mating_reading)
            
            # Energy level of detected crater (normalized)
            crater_energy = crater_info['energy_level']
            self.sensor_readings.append(crater_energy)
            
            # Food distance
            food_distance = self.get_food_distance(angle, food_pellets)
            food_reading = min(food_distance / SENSOR_RANGE, 1.0)
            self.sensor_readings.append(food_reading)
        
        # Cache the sensor data
        self.cached_sensor_data = self.sensor_readings.copy()
        return self.sensor_readings
    
    def get_wall_distance(self, angle):
        """
        Calculate distance to the nearest wall in the given direction
        
        Args:
            angle (float): Angle of the ray in radians
            
        Returns:
            float: Distance to the nearest wall
        """
        # Ray starting point
        ray_x = self.x
        ray_y = self.y
        
        # Ray direction using precomputed values if available
        if PRECOMPUTE_ANGLES and angle % (2 * math.pi) in SIN_COS_CACHE:
            ray_dy, ray_dx = SIN_COS_CACHE[angle % (2 * math.pi)]
        else:
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)
        
        # Distance to walls
        dist_to_right = (WIDTH - ray_x) / ray_dx if ray_dx > 0 else float('inf')
        dist_to_left = -ray_x / ray_dx if ray_dx < 0 else float('inf')
        dist_to_bottom = (HEIGHT - ray_y) / ray_dy if ray_dy > 0 else float('inf')
        dist_to_top = -ray_y / ray_dy if ray_dy < 0 else float('inf')
        
        # Find minimum positive distance
        wall_dist = min(
            d for d in [dist_to_right, dist_to_left, dist_to_bottom, dist_to_top]
            if d > 0
        )
        
        # Limit to wall detection range (shorter than general sensor range)
        return min(wall_dist, WALL_DETECTION_RANGE)
    
    def get_crater_info(self, angle, craters):
        """
        Get detailed information about craters in a specific direction
        
        Args:
            angle (float): Angle of the ray in radians
            craters (list): List of all craters
            
        Returns:
            dict: Dictionary with distance, mating status, and energy level info
        """
        # Ray starting point
        ray_x = self.x
        ray_y = self.y
        
        # Get ray direction
        if PRECOMPUTE_ANGLES and angle % (2 * math.pi) in SIN_COS_CACHE:
            ray_dy, ray_dx = SIN_COS_CACHE[angle % (2 * math.pi)]
        else:
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)
        
        # Initialize with default values (max range, no crater detected)
        result = {
            'distance': SENSOR_RANGE,
            'mating_distance': SENSOR_RANGE,
            'energy_level': 0.0
        }
        
        # Track the nearest detected crater for energy reporting
        nearest_crater = None
        min_distance = SENSOR_RANGE
        
        # Only check craters that are potentially in range
        for crater in craters:
            if crater is self:
                continue
                
            # Quick distance check first to avoid unnecessary calculation
            dx = crater.x - ray_x
            dy = crater.y - ray_y
            distance_squared = dx*dx + dy*dy
            
            # Skip if crater is definitely too far away
            cutoff_squared = (DISTANCE_CUTOFF + crater.size) ** 2
            if distance_squared > cutoff_squared:
                continue
            
            # Early angle rejection - check if generally in the right direction
            # Calculate angle from ray to crater
            angle_to_crater = math.atan2(dy, dx)
            
            # Normalize angle difference to [-pi, pi]
            angle_diff = (angle_to_crater - angle) % (2 * math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
                
            # If not in ray direction (within a small cone), skip
            # Use wider threshold for initial check (about 60 degrees)
            if abs(angle_diff) > 1.0:
                continue
                
            # Distance to crater center
            distance_to_center = math.sqrt(distance_squared)
            if distance_to_center > SENSOR_RANGE + crater.size:
                continue
                
            # More precise angle check now that we know it's worth checking
            # If not in ray direction (within a small cone), skip
            if abs(angle_diff) > 0.5:  # About 30 degrees
                continue
                
            # Project distance
            distance = distance_to_center * math.cos(angle_diff) - crater.size
            distance = max(0, distance)
            
            # Update general crater distance if this is closer
            if distance < result['distance']:
                result['distance'] = distance
                nearest_crater = crater
            
            # Update mating crater distance if this crater is mating
            if hasattr(crater, 'is_mating') and crater.is_mating and distance < result['mating_distance']:
                result['mating_distance'] = distance
        
        # Get energy level from the nearest crater
        if nearest_crater and hasattr(nearest_crater, 'energy'):
            result['energy_level'] = nearest_crater.energy / nearest_crater.max_energy
        
        return result
    
    def get_food_distance(self, angle, food_pellets):
        """
        Calculate distance to the nearest food in the given direction
        
        Args:
            angle (float): Angle of the ray in radians
            food_pellets (list): List of all food pellets
            
        Returns:
            float: Distance to the nearest food
        """
        # Ray starting point
        ray_x = self.x
        ray_y = self.y
        
        # Get ray direction
        if PRECOMPUTE_ANGLES and angle % (2 * math.pi) in SIN_COS_CACHE:
            ray_dy, ray_dx = SIN_COS_CACHE[angle % (2 * math.pi)]
        else:
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)
        
        min_distance = SENSOR_RANGE
        
        # Only check food pellets that are potentially in range
        for food in food_pellets:
            if not food.active:
                continue
                
            # Quick distance check first to avoid unnecessary calculation
            dx = food.x - ray_x
            dy = food.y - ray_y
            distance_squared = dx*dx + dy*dy
            
            # Skip if food is definitely too far away
            # Use FOOD_DETECTION_RANGE instead of DISTANCE_CUTOFF for longer food detection
            cutoff_squared = (FOOD_DETECTION_RANGE + food.size) ** 2
            if distance_squared > cutoff_squared:
                continue
            
            # Early angle rejection - check if generally in the right direction
            # Calculate angle from ray to food
            angle_to_food = math.atan2(dy, dx)
            
            # Normalize angle difference to [-pi, pi]
            angle_diff = (angle_to_food - angle) % (2 * math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
                
            # If not in ray direction (within a wider cone for early rejection), skip
            if abs(angle_diff) > 1.0:  # About 60 degrees
                continue
                
            # Distance to food center
            distance_to_center = math.sqrt(distance_squared)
            if distance_to_center > SENSOR_RANGE + food.size:
                continue
            
            # More precise angle check now that we know it's worth checking
            # If not in ray direction (within a small cone), skip
            if abs(angle_diff) > 0.5:  # About 30 degrees
                continue
                
            # Project distance
            distance = distance_to_center * math.cos(angle_diff) - food.size
            
            if distance < min_distance:
                min_distance = max(0, distance)
        
        return min_distance
    
    def check_food_collision(self, food_pellets):
        """
        Check for collision with food pellets and absorb energy
        
        Args:
            food_pellets (list): List of all food pellets
            
        Returns:
            bool: True if food was consumed, False otherwise
        """
        for food in food_pellets:
            if not food.active:
                continue
                
            # Distance to food
            dx = food.x - self.x
            dy = food.y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If collision
            if distance < self.size + food.size:
                # Absorb energy
                self.energy = min(self.max_energy, self.energy + food.energy)
                food.active = False
                self.food_eaten += 1  # Track food consumption for fitness
                return True
        
        return False
    
    def check_mating_collision(self, craters):
        """
        Check for collision with other mating craters
        
        Args:
            craters (list): List of all craters in the environment
            
        Returns:
            Crater or None: The other crater if mating collision occurred, None otherwise
        """
        if not self.is_mating:
            return None
            
        for other in craters:
            # Skip self, non-craters, or non-mating craters
            if other is self or not isinstance(other, Crater) or not hasattr(other, 'is_mating') or not other.is_mating:
                continue
                
            # Calculate distance
            dx = other.x - self.x
            dy = other.y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If close enough
            if distance < self.size + other.size + 10:
                return other
                
        return None
    
    def update(self, craters, food_pellets):
        """
        Update crater based on neural network output and energy level
        
        Args:
            craters (list): List of all craters
            food_pellets (list): List of all food pellets
            
        Returns:
            Crater or None: Other crater if mating occurred, None otherwise
        """
        # Skip updates if too low on energy
        if self.energy <= 0:
            return None
            
        # Increment age
        self.age += 1
        
        # Update mating state
        if self.is_mating:
            self.mating_timer -= 1
            if self.mating_timer <= 0:
                self.is_mating = False
        else:
            # Force mating state when energy is above threshold
            if self.energy > MATING_ENERGY_THRESHOLD:
                self.is_mating = True
                self.mating_timer = MATING_DURATION
        
        # Check for mating with other craters
        other_crater = self.check_mating_collision(craters)
        if other_crater:
            return other_crater
        
        # Get sensor readings
        sensor_data = self.sense_environment(craters, food_pellets)
        
        # Feed sensor data to neural network (including normalized energy level)
        nn_inputs = sensor_data + [self.energy / self.max_energy]
        output = self.brain.forward(nn_inputs)
        
        # Interpret neural network output
        forward_thrust = (output[0] * 2) - 1  # Range: -1 to 1
        reverse_thrust = (output[1] * 2) - 1  # Range: -1 to 1
        rotation_force = (output[2] * 2) - 1  # Range: -1 to 1
        
        # Apply rotation (costs energy)
        old_rotation = self.rotation
        self.angular_velocity += rotation_force * 0.05
        self.angular_velocity *= 0.9  # Damping
        self.rotation += self.angular_velocity
        
        # Calculate rotation energy cost
        rotation_cost = abs(self.rotation - old_rotation) * ENERGY_ROTATION_COST
        
        # Apply thrust (costs energy)
        acceleration = forward_thrust * ACCELERATION_FACTOR - reverse_thrust * (ACCELERATION_FACTOR/2)
        self.speed += acceleration
        self.speed *= FRICTION  # Reduced friction
        self.speed = max(-self.max_speed, min(self.max_speed, self.speed))
        
        # Update position
        old_x, old_y = self.x, self.y
        self.x += self.speed * math.cos(self.rotation)
        self.y += self.speed * math.sin(self.rotation)
        
        # Calculate distance moved
        distance_moved = math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2)
        self.distance_traveled += distance_moved  # Track distance for fitness
        
        # Track inactivity and apply penalties if necessary
        if distance_moved < self.movement_threshold:
            self.inactive_frames += 1
            # Apply inactivity penalty after threshold is reached
            if self.inactive_frames > self.inactivity_threshold:
                # Increase energy cost as inactivity persists
                inactivity_penalty = 0.2 * (1 + (self.inactive_frames - self.inactivity_threshold) / 100)
                # Cap the penalty to avoid instant death
                inactivity_penalty = min(inactivity_penalty, 2.0)
                self.energy -= inactivity_penalty
        else:
            # Reset inactivity counter when there's significant movement
            self.inactive_frames = max(0, self.inactive_frames - 2)  # Decrease twice as fast
        
        # Make forward movement twice as efficient as backward movement
        if self.speed > 0:  # Forward movement
            movement_cost = distance_moved * ENERGY_DEPLETION_RATE
        else:  # Backward movement
            movement_cost = distance_moved * ENERGY_DEPLETION_RATE * 2
        
        # Calculate total energy cost
        total_energy_cost = movement_cost + rotation_cost
        
        # Double energy cost when in mating mode
        if self.is_mating:
            total_energy_cost *= 2
        
        # Deduct energy
        self.energy -= total_energy_cost
        self.energy = max(0, self.energy)  # Don't go below 0
        
        # Bounce off walls
        if self.x < self.size:
            self.x = self.size
            self.speed *= -0.5
        elif self.x > WIDTH - self.size:
            self.x = WIDTH - self.size
            self.speed *= -0.5
            
        if self.y < self.size:
            self.y = self.size
            self.speed *= -0.5
        elif self.y > HEIGHT - self.size:
            self.y = HEIGHT - self.size
            self.speed *= -0.5
        
        # Check for food collision
        self.check_food_collision(food_pellets)
        
        # Update triangle points
        self.generate_shape()
        
        return None
    
    def calculate_fitness(self):
        """
        Calculate fitness score based on survival, food consumption, and movement
        
        Returns:
            float: Fitness score
        """
        # Weights for different fitness components
        age_weight = 1.0
        food_weight = 10.0
        distance_weight = 0.1
        energy_weight = 0.5
        
        # Calculate fitness components
        age_fitness = self.age * age_weight
        food_fitness = self.food_eaten * food_weight
        distance_fitness = self.distance_traveled * distance_weight
        energy_fitness = self.energy * energy_weight
        
        # Inactivity penalty for fitness calculation
        inactivity_penalty = 0
        if self.inactive_frames > self.inactivity_threshold:
            inactivity_penalty = self.inactive_frames - self.inactivity_threshold
            inactivity_penalty = min(inactivity_penalty * 0.5, age_fitness * 0.5)  # Cap at 50% of age fitness
        
        # Total fitness with inactivity penalty
        fitness = age_fitness + food_fitness + distance_fitness + energy_fitness - inactivity_penalty
        
        return max(0, fitness)  # Ensure fitness is never negative
    
    @classmethod
    def create_offspring(cls, parent1, parent2, mutation_rate=MUTATION_RATE, mutation_scale=MUTATION_SCALE, energy=INITIAL_ENERGY):
        """
        Create a new crater as an offspring of two parents with mutations
        
        Args:
            parent1 (Crater): First parent crater
            parent2 (Crater): Second parent crater
            mutation_rate (float): Probability of mutation for each weight
            mutation_scale (float): Scale of mutations
            energy (float): Initial energy for the offspring
            
        Returns:
            Crater: New crater with combined brain from parents and mutations
        """
        # Create a new brain by combining parents
        if isinstance(parent1.brain, DeepNeuralNetwork):
            # Deep neural network crossover and mutation
            child_brain = copy.deepcopy(parent1.brain)
            
            # Crossover for deep network: Mix weights from both parents layer by layer
            for i in range(len(child_brain.weights)):
                for j in range(child_brain.weights[i].shape[0]):
                    for k in range(child_brain.weights[i].shape[1]):
                        if random.random() < 0.5:
                            child_brain.weights[i][j, k] = parent2.brain.weights[i][j, k]
            
                # Mix biases
                for j in range(child_brain.biases[i].shape[0]):
                    if random.random() < 0.5:
                        child_brain.biases[i][j, 0] = parent2.brain.biases[i][j, 0]
            
            # Apply mutations to weights
            for i in range(len(child_brain.weights)):
                for j in range(child_brain.weights[i].shape[0]):
                    for k in range(child_brain.weights[i].shape[1]):
                        if random.random() < mutation_rate:
                            child_brain.weights[i][j, k] += random.gauss(0, 1) * mutation_scale
                
                # Apply mutations to biases
                for j in range(child_brain.biases[i].shape[0]):
                    if random.random() < mutation_rate:
                        child_brain.biases[i][j, 0] += random.gauss(0, 1) * mutation_scale
        
        else:
            # Simple neural network (original code)
            child_brain = copy.deepcopy(parent1.brain)
            
            # Crossover: Mix weights from both parents (50/50 chance for each weight)
            # Input to hidden weights
            for i in range(child_brain.weights_ih.shape[0]):
                for j in range(child_brain.weights_ih.shape[1]):
                    if random.random() < 0.5:
                        child_brain.weights_ih[i, j] = parent2.brain.weights_ih[i, j]
            
            # Hidden biases
            for i in range(child_brain.bias_h.shape[0]):
                if random.random() < 0.5:
                    child_brain.bias_h[i, 0] = parent2.brain.bias_h[i, 0]
            
            # Hidden to output weights
            for i in range(child_brain.weights_ho.shape[0]):
                for j in range(child_brain.weights_ho.shape[1]):
                    if random.random() < 0.5:
                        child_brain.weights_ho[i, j] = parent2.brain.weights_ho[i, j]
            
            # Output biases
            for i in range(child_brain.bias_o.shape[0]):
                if random.random() < 0.5:
                    child_brain.bias_o[i, 0] = parent2.brain.bias_o[i, 0]
            
            # Apply mutations
            # Input to hidden weights
            for i in range(child_brain.weights_ih.shape[0]):
                for j in range(child_brain.weights_ih.shape[1]):
                    if random.random() < mutation_rate:
                        child_brain.weights_ih[i, j] += random.gauss(0, 1) * mutation_scale
            
            # Hidden biases
            for i in range(child_brain.bias_h.shape[0]):
                if random.random() < mutation_rate:
                    child_brain.bias_h[i, 0] += random.gauss(0, 1) * mutation_scale
            
            # Hidden to output weights
            for i in range(child_brain.weights_ho.shape[0]):
                for j in range(child_brain.weights_ho.shape[1]):
                    if random.random() < mutation_rate:
                        child_brain.weights_ho[i, j] += random.gauss(0, 1) * mutation_scale
            
            # Output biases
            for i in range(child_brain.bias_o.shape[0]):
                if random.random() < mutation_rate:
                    child_brain.bias_o[i, 0] += random.gauss(0, 1) * mutation_scale
        
        # Use position of one of the parents (randomly choose which one)
        parent_pos = parent1 if random.random() < 0.5 else parent2
        
        # Create a new crater with the combined brain at parent's position
        offspring = cls(x=parent_pos.x, y=parent_pos.y, brain=child_brain, font=parent1.font)
        offspring.energy = energy  # Set the offspring's initial energy
        
        return offspring
    
    def get_age_color(self):
        """
        Get color based on crater age
        
        Returns:
            tuple: RGB color tuple
        """
        if self.age < AGE_YOUNG:
            # Young crater: transition from young to adult (blue to green)
            ratio = self.age / AGE_YOUNG
            r = int(YOUNG_COLOR[0] + (ADULT_COLOR[0] - YOUNG_COLOR[0]) * ratio)
            g = int(YOUNG_COLOR[1] + (ADULT_COLOR[1] - YOUNG_COLOR[1]) * ratio)
            b = int(YOUNG_COLOR[2] + (ADULT_COLOR[2] - YOUNG_COLOR[2]) * ratio)
            return (r, g, b)
        
        elif self.age < AGE_ADULT:
            # Adult crater: transition from adult to mature (green to yellow)
            ratio = (self.age - AGE_YOUNG) / (AGE_ADULT - AGE_YOUNG)
            r = int(ADULT_COLOR[0] + (MATURE_COLOR[0] - ADULT_COLOR[0]) * ratio)
            g = int(ADULT_COLOR[1] + (MATURE_COLOR[1] - ADULT_COLOR[1]) * ratio)
            b = int(ADULT_COLOR[2] + (MATURE_COLOR[2] - ADULT_COLOR[2]) * ratio)
            return (r, g, b)
        
        elif self.age < AGE_MATURE:
            # Mature crater: transition from mature to elder (yellow to red)
            ratio = (self.age - AGE_ADULT) / (AGE_MATURE - AGE_ADULT)
            r = int(MATURE_COLOR[0] + (ELDER_COLOR[0] - MATURE_COLOR[0]) * ratio)
            g = int(MATURE_COLOR[1] + (ELDER_COLOR[1] - MATURE_COLOR[1]) * ratio)
            b = int(MATURE_COLOR[2] + (ELDER_COLOR[2] - MATURE_COLOR[2]) * ratio)
            return (r, g, b)
        
        else:
            # Elder crater: red
            return ELDER_COLOR

    def draw(self, surface, draw_sensors=False):
        """
        Draw the crater and optionally its sensors
        
        Args:
            surface: Pygame surface to draw on
            draw_sensors (bool): Whether to draw sensor rays
        """
        # Get color based on age
        crater_color = self.get_age_color()
        
        # Adjust brightness based on energy
        energy_ratio = self.energy / self.max_energy
        brightness_factor = 0.5 + 0.5 * energy_ratio  # 50%-100% brightness
        r = min(255, int(crater_color[0] * brightness_factor))
        g = min(255, int(crater_color[1] * brightness_factor))
        b = min(255, int(crater_color[2] * brightness_factor))
        crater_color = (r, g, b)
        
        # Draw crater triangle
        pygame.draw.polygon(surface, crater_color, self.points)
        
        # If mating, draw a magenta border with pulsing effect to indicate energy drain
        if self.is_mating:
            # Calculate pulsing effect based on mating timer (pulsing increases as timer decreases)
            pulse_intensity = 1.5 + 0.5 * (1 - (self.mating_timer / MATING_DURATION))
            border_width = max(2, int(3 * pulse_intensity))
            pygame.draw.polygon(surface, MATING_COLOR, self.points, width=border_width)
            
            # Add small text indicator showing 2x energy consumption
            if self.font:
                indicator_text = self.font.render("2x", True, MATING_COLOR)
                surface.blit(indicator_text, (self.x + self.size + 5, self.y - 15))
        
        # If inactive and being penalized, draw a dark red border
        elif self.inactive_frames > self.inactivity_threshold:
            # Make border darker red as inactivity increases
            intensity = min(255, 100 + (self.inactive_frames - self.inactivity_threshold) // 2)
            inactive_color = (intensity, 0, 0)  # Dark red
            pygame.draw.polygon(surface, inactive_color, self.points, width=2)
        
        # Draw direction indicator (a small dot at the front)
        front_x, front_y = self.points[0]  # First point is the front
        pygame.draw.circle(surface, DIRECTION_COLOR, (int(front_x), int(front_y)), 3)
        
        # Display energy level in red
        if self.energy > 0 and self.font:
            energy_text = self.font.render(f"{int(self.energy)}", True, ENERGY_TEXT_COLOR)
            surface.blit(energy_text, (self.x - 10, self.y - 5))
        
        # Draw sensors if enabled (and only if the crater is within view)
        if draw_sensors and self.energy > 0:
            # Skip drawing sensors for craters that are likely not visible
            if self.x < -50 or self.x > WIDTH + 50 or self.y < -50 or self.y > HEIGHT + 50:
                return
                
            # Define colors for different sensor types
            WALL_SENSOR_COLOR = (255, 0, 0)       # Red for walls
            CRATER_SENSOR_COLOR = (255, 255, 0)   # Yellow for craters
            MATING_SENSOR_COLOR = (255, 0, 255)   # Magenta for mating craters
            FOOD_SENSOR_COLOR = (0, 255, 0)       # Green for food
            
            # Draw fewer sensor rays if there are many craters
            ray_step = 2 if len(self.sensor_readings) > 40 else 1
            
            for i in range(0, NUM_SENSORS, ray_step):
                angle = self.rotation + (i * (2 * math.pi / NUM_SENSORS))
                
                # Index in sensor_readings array
                base_idx = i * 5
                
                # Wall detection
                wall_dist = self.sensor_readings[base_idx] * SENSOR_RANGE
                crater_dist = self.sensor_readings[base_idx + 1] * SENSOR_RANGE
                mating_dist = self.sensor_readings[base_idx + 2] * SENSOR_RANGE
                crater_energy = self.sensor_readings[base_idx + 3]  # Normalized 0-1
                food_dist = self.sensor_readings[base_idx + 4] * SENSOR_RANGE
                
                # Calculate sensor endpoints
                end_x = self.x + SENSOR_RANGE * math.cos(angle)
                end_y = self.y + SENSOR_RANGE * math.sin(angle)
                
                # Draw a thin background line for the full sensor range
                pygame.draw.line(surface, (50, 50, 50), (self.x, self.y), (end_x, end_y), 1)
                
                # Draw detected objects, starting with the farthest ones first
                sensors_to_draw = []
                
                # Only add sensors that detected something
                if wall_dist < SENSOR_RANGE:
                    sensors_to_draw.append((wall_dist, WALL_SENSOR_COLOR))
                if crater_dist < SENSOR_RANGE:
                    sensors_to_draw.append((crater_dist, CRATER_SENSOR_COLOR))
                if mating_dist < SENSOR_RANGE:
                    sensors_to_draw.append((mating_dist, MATING_SENSOR_COLOR))
                if food_dist < SENSOR_RANGE:
                    sensors_to_draw.append((food_dist, FOOD_SENSOR_COLOR))
                
                # Sort by distance (descending) and only draw if something was detected
                if sensors_to_draw:
                    sensors_to_draw.sort(key=lambda x: x[0], reverse=True)
                    
                    # Draw sensors in order (farthest to closest)
                    for distance, color in sensors_to_draw:
                        detection_end_x = self.x + distance * math.cos(angle)
                        detection_end_y = self.y + distance * math.sin(angle)
                        
                        # Make line thickness based on how close the object is
                        thickness = max(1, int(3 * (1 - distance / SENSOR_RANGE)))
                        
                        # Draw the ray to the detected object
                        pygame.draw.line(surface, color, (self.x, self.y), 
                                      (detection_end_x, detection_end_y), thickness)
                
                # If a crater with energy is detected, draw a small circle showing energy level
                if crater_dist < SENSOR_RANGE and crater_energy > 0:
                    energy_pos_x = self.x + crater_dist * math.cos(angle)
                    energy_pos_y = self.y + crater_dist * math.sin(angle)
                    
                    # Energy indicator color: green (high) to red (low)
                    energy_r = int(255 * (1 - crater_energy))
                    energy_g = int(255 * crater_energy)
                    energy_color = (energy_r, energy_g, 0)
                    
                    # Size based on energy level
                    energy_size = 2 + int(3 * crater_energy)
                    
                    # Draw energy indicator
                    pygame.draw.circle(surface, energy_color, 
                                    (int(energy_pos_x), int(energy_pos_y)), energy_size) 