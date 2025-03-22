import pygame
import sys
import random
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Configuration
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (0, 0, 0)  # Black
CRATER_COLOR = (150, 150, 150)  # Gray
SENSOR_COLOR = (255, 0, 0)  # Red for ray visualization
DIRECTION_COLOR = (0, 255, 255)  # Cyan for direction indicator
FOOD_COLOR = (0, 255, 0)  # Green for food pellets
NUM_CRATERS = 100
NUM_SENSORS = 8  # Number of sensor rays
SENSOR_RANGE = 100  # How far sensors can detect
NUM_FOOD_PELLETS = 50  # Number of food pellets
FOOD_ENERGY = 100  # Energy gained from food
MAX_ENERGY = 1000  # Maximum energy a crater can have
INITIAL_ENERGY = 5  # Starting energy for craters
ENERGY_DEPLETION_RATE = 0.2  # Energy lost per unit of movement
ENERGY_ROTATION_COST = 0.1  # Energy lost per unit of rotation
FONT_SIZE = 12  # Size for energy display

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Crater Simulation with Neural Networks and Energy")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, FONT_SIZE)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize with random weights
        self.weights_ih = np.random.randn(hidden_size, input_size) * 0.1
        self.bias_h = np.random.randn(hidden_size, 1) * 0.1
        self.weights_ho = np.random.randn(output_size, hidden_size) * 0.1
        self.bias_o = np.random.randn(output_size, 1) * 0.1
        
    def forward(self, inputs):
        # Convert inputs to numpy array
        inputs = np.array(inputs).reshape(-1, 1)
        
        # Hidden layer
        hidden = np.dot(self.weights_ih, inputs) + self.bias_h
        hidden = self.sigmoid(hidden)
        
        # Output layer
        output = np.dot(self.weights_ho, hidden) + self.bias_o
        output = self.sigmoid(output)
        
        return output.flatten()
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class Food:
    def __init__(self, x=None, y=None):
        # Random position if not provided
        self.x = x if x is not None else random.randint(20, WIDTH-20)
        self.y = y if y is not None else random.randint(20, HEIGHT-20)
        self.size = random.randint(5, 10)
        self.energy = FOOD_ENERGY
        self.active = True
    
    def draw(self, surface):
        if self.active:
            pygame.draw.circle(surface, FOOD_COLOR, (int(self.x), int(self.y)), self.size)

class Crater:
    def __init__(self, x=None, y=None, size=None):
        # Initialize crater with random values if not provided
        self.x = x if x is not None else random.randint(50, WIDTH-50)
        self.y = y if y is not None else random.randint(50, HEIGHT-50)
        self.size = size if size is not None else random.randint(10, 30)
        self.rotation = random.uniform(0, 2 * math.pi)
        
        # Movement properties
        self.max_speed = 2.0
        self.speed = 0
        self.angular_velocity = 0
        
        # Energy system
        self.energy = INITIAL_ENERGY
        self.max_energy = MAX_ENERGY
        
        # Create the triangular shape
        self.generate_shape()
        
        # Neural network for crater behavior
        # Inputs: sensor readings (distance to objects in different directions)
        # Outputs: forward thrust, reverse thrust, rotation
        self.brain = SimpleNeuralNetwork(NUM_SENSORS * 2 + 1, 12, 3)  # Added energy input
        
        # For visualizing sensor rays
        self.sensor_readings = [1.0] * NUM_SENSORS * 2  # Initialize with max distance (nothing detected)
    
    def generate_shape(self):
        """Create the triangular shape for the crater"""
        # Define the three points of the triangle relative to center (x, y)
        self.points = []
        # Front point (pointing in direction of rotation)
        front_x = self.x + self.size * math.cos(self.rotation)
        front_y = self.y + self.size * math.sin(self.rotation)
        # Two back points
        left_x = self.x + self.size * math.cos(self.rotation + 2.09)  # ~120 degrees
        left_y = self.y + self.size * math.sin(self.rotation + 2.09)
        right_x = self.x + self.size * math.cos(self.rotation - 2.09)
        right_y = self.y + self.size * math.sin(self.rotation - 2.09)
        
        self.points = [(front_x, front_y), (left_x, left_y), (right_x, right_y)]
    
    def sense_environment(self, craters, food_pellets):
        """Cast rays in different directions to detect walls and other craters"""
        self.sensor_readings = []
        
        # Cast rays in multiple directions
        for i in range(NUM_SENSORS):
            angle = self.rotation + (i * (2 * math.pi / NUM_SENSORS))
            
            # Wall distance
            wall_distance = self.get_wall_distance(angle)
            self.sensor_readings.append(min(wall_distance / SENSOR_RANGE, 1.0))
            
            # Crater distance
            crater_distance = self.get_crater_distance(angle, craters)
            self.sensor_readings.append(min(crater_distance / SENSOR_RANGE, 1.0))
            
            # Future enhancement: detect food (currently not used in neural input)
        
        return self.sensor_readings
    
    def get_wall_distance(self, angle):
        """Calculate distance to the nearest wall in the given direction"""
        # Ray starting point
        ray_x = self.x
        ray_y = self.y
        
        # Ray direction
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
        """Calculate distance to the nearest crater in the given direction"""
        # Ray starting point
        ray_x = self.x
        ray_y = self.y
        
        min_distance = SENSOR_RANGE
        
        for crater in craters:
            if crater is self:
                continue
                
            # Vector to crater
            dx = crater.x - ray_x
            dy = crater.y - ray_y
            
            # Distance to crater center
            distance_to_center = math.sqrt(dx*dx + dy*dy)
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
        """Check for collision with food pellets and absorb energy"""
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
                return True
        
        return False
    
    def update(self, craters, food_pellets):
        """Update crater based on neural network output and energy level"""
        # Skip updates if too low on energy
        if self.energy <= 0:
            return
            
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
        old_speed = self.speed
        acceleration = forward_thrust * 0.1 - reverse_thrust * 0.05
        self.speed += acceleration
        self.speed *= 0.95  # Friction
        self.speed = max(-self.max_speed, min(self.max_speed, self.speed))
        
        # Calculate movement energy cost
        speed_cost = abs(self.speed) * ENERGY_DEPLETION_RATE
        
        # Update position
        old_x, old_y = self.x, self.y
        self.x += self.speed * math.cos(self.rotation)
        self.y += self.speed * math.sin(self.rotation)
        
        # Calculate distance moved
        distance_moved = math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2)
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
    
    def draw(self, surface, draw_sensors=False):
        """Draw the crater and optionally its sensors"""
        # Crater color based on energy (from gray to white as energy increases)
        energy_ratio = self.energy / self.max_energy
        color_value = min(255, int(150 + 105 * energy_ratio))
        crater_color = (color_value, color_value, color_value)
        
        # Draw crater triangle
        pygame.draw.polygon(surface, crater_color, self.points)
        
        # Draw direction indicator (a small dot at the front)
        front_x, front_y = self.points[0]  # First point is the front
        pygame.draw.circle(surface, DIRECTION_COLOR, (int(front_x), int(front_y)), 3)
        
        # Display energy level in red
        if self.energy > 0:
            energy_text = font.render(f"{int(self.energy)}", True, (255, 0, 0))
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

class CraterSimulation:
    def __init__(self, num_craters=NUM_CRATERS, num_food=NUM_FOOD_PELLETS):
        self.craters = [Crater() for _ in range(num_craters)]
        self.food_pellets = [Food() for _ in range(num_food)]
        self.show_sensors = True  # Toggle for sensor visualization (now enabled by default)
        self.food_spawn_timer = 0
        self.food_spawn_interval = 60  # Frames between spawning food
    
    def update(self):
        """Update all craters and food"""
        # Update craters and handle energy depletion
        craters_to_remove = []
        
        for i, crater in enumerate(self.craters):
            # Update crater
            crater.update(self.craters, self.food_pellets)
            
            # Check if crater ran out of energy
            if crater.energy <= 0:
                # Mark for removal and create food pellet at its position
                craters_to_remove.append(i)
                # Create new food pellet at crater's position
                new_food = Food(crater.x, crater.y)
                self.food_pellets.append(new_food)
        
        # Remove dead craters (in reverse order to avoid index issues)
        for i in sorted(craters_to_remove, reverse=True):
            del self.craters[i]
        
        # Check and replace consumed food
        active_food = 0
        for food in self.food_pellets:
            if food.active:
                active_food += 1
        
        # Spawn new food periodically
        self.food_spawn_timer += 1
        if self.food_spawn_timer >= self.food_spawn_interval:
            if active_food < NUM_FOOD_PELLETS:
                # Replace consumed food
                for food in self.food_pellets:
                    if not food.active:
                        food.x = random.randint(20, WIDTH-20)
                        food.y = random.randint(20, HEIGHT-20)
                        food.active = True
                        break
            self.food_spawn_timer = 0
    
    def draw(self, surface):
        """Draw all craters and food"""
        # Draw food
        for food in self.food_pellets:
            food.draw(surface)
        
        # Draw craters
        for crater in self.craters:
            crater.draw(surface, self.show_sensors)
        
        # Display information
        active_food = sum(1 for food in self.food_pellets if food.active)
        active_craters = sum(1 for crater in self.craters if crater.energy > 0)
        info_text = f"Food: {active_food} | Craters: {active_craters}/{len(self.craters)}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        surface.blit(text_surface, (10, 10))

def main():
    simulation = CraterSimulation()
    
    # Print information about controls
    print("Simulation started. Press 'S' to toggle sensor visibility.")
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    # Toggle sensor visualization with 's' key
                    simulation.show_sensors = not simulation.show_sensors
        
        # Clear the screen
        screen.fill(BACKGROUND_COLOR)
        
        # Update and draw simulation
        simulation.update()
        simulation.draw(screen)
        
        # Update display
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 