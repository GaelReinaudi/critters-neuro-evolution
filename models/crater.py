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
    DISTANCE_CUTOFF, PRECOMPUTE_ANGLES
)
from craters.models.neural_network import SimpleNeuralNetwork

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
        # Inputs: sensor readings (distance to objects in different directions) + energy
        # Outputs: forward thrust, reverse thrust, rotation
        if brain is None:
            self.brain = SimpleNeuralNetwork(NUM_SENSORS * 2 + 1, 12, 3)
        else:
            self.brain = brain
        
        # For visualizing sensor rays
        self.sensor_readings = [1.0] * NUM_SENSORS * 2
        
        # Evolution tracking
        self.age = 0  # Age in frames
        self.food_eaten = 0  # Number of food pellets consumed
        self.distance_traveled = 0  # Total distance traveled
        
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
        Cast rays in different directions to detect walls and other craters
        
        Args:
            craters (list): List of all craters in the environment
            food_pellets (list): List of all food pellets
            force_update (bool): Whether to force update even if not time yet
            
        Returns:
            list: Sensor readings (normalized distances)
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
            
            # Wall distance
            wall_distance = self.get_wall_distance(angle)
            self.sensor_readings.append(min(wall_distance / SENSOR_RANGE, 1.0))
            
            # Crater distance
            crater_distance = self.get_crater_distance(angle, craters)
            self.sensor_readings.append(min(crater_distance / SENSOR_RANGE, 1.0))
            
            # Future enhancement: detect food (currently not used in neural input)
        
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
        
        # Limit to sensor range
        return min(wall_dist, SENSOR_RANGE)
    
    def get_crater_distance(self, angle, craters):
        """
        Calculate distance to the nearest crater in the given direction
        
        Args:
            angle (float): Angle of the ray in radians
            craters (list): List of all craters
            
        Returns:
            float: Distance to the nearest crater
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
                
            # Distance to crater center
            distance_to_center = math.sqrt(distance_squared)
            if distance_to_center > SENSOR_RANGE + crater.size:
                continue
                
            # Angle to crater
            angle_to_crater = math.atan2(dy, dx)
            
            # Normalize angle difference to [-pi, pi]
            angle_diff = (angle_to_crater - angle) % (2 * math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
                
            # If not in ray direction (within a small cone), skip
            if abs(angle_diff) > 0.5:  # About 30 degrees
                continue
                
            # Project distance
            distance = distance_to_center * math.cos(angle_diff) - crater.size
            
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
        movement_cost = distance_moved * ENERGY_DEPLETION_RATE
        
        # Deduct energy (movement + rotation)
        self.energy -= (movement_cost + rotation_cost)
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
        
        # Total fitness
        fitness = age_fitness + food_fitness + distance_fitness + energy_fitness
        
        return fitness
    
    @classmethod
    def create_offspring(cls, parent1, parent2, mutation_rate=MUTATION_RATE, mutation_scale=MUTATION_SCALE):
        """
        Create a new crater as an offspring of two parents with mutations
        
        Args:
            parent1 (Crater): First parent crater
            parent2 (Crater): Second parent crater
            mutation_rate (float): Probability of mutation for each weight
            mutation_scale (float): Scale of mutations
            
        Returns:
            Crater: New crater with combined brain from parents and mutations
        """
        # Create a new brain by combining parents
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
        
        # Create a new crater with the combined brain
        return cls(brain=child_brain, font=parent1.font)
    
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
        # Crater color based on mating state or age
        if self.is_mating:
            # Magenta color for mating state
            crater_color = MATING_COLOR
        else:
            # Color based on age
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
        
        # Draw direction indicator (a small dot at the front)
        front_x, front_y = self.points[0]  # First point is the front
        pygame.draw.circle(surface, DIRECTION_COLOR, (int(front_x), int(front_y)), 3)
        
        # Display energy level in red
        if self.energy > 0 and self.font:
            energy_text = self.font.render(f"{int(self.energy)}", True, ENERGY_TEXT_COLOR)
            surface.blit(energy_text, (self.x - 10, self.y - 5))
        
        # Draw sensors if enabled
        if draw_sensors and self.energy > 0:
            for i in range(NUM_SENSORS):
                angle = self.rotation + (i * (2 * math.pi / NUM_SENSORS))
                # Distance from sensor readings (wall and crater)
                wall_dist = self.sensor_readings[i*2] * SENSOR_RANGE
                crater_dist = self.sensor_readings[i*2+1] * SENSOR_RANGE
                
                # Use minimum of wall and crater distance
                ray_length = min(wall_dist, crater_dist)
                
                # Calculate end point
                end_x = self.x + ray_length * math.cos(angle)
                end_y = self.y + ray_length * math.sin(angle)
                
                # Change color based on detection (from red when close to green when far)
                # Normalized distance (0 to 1)
                detection_ratio = ray_length / SENSOR_RANGE
                # Red component decreases with distance
                red = int(255 * (1 - detection_ratio))
                # Green component increases with distance
                green = int(255 * detection_ratio)
                sensor_color = (red, green, 0)
                
                # Draw the ray
                pygame.draw.line(surface, sensor_color, (self.x, self.y), (end_x, end_y), 2) 