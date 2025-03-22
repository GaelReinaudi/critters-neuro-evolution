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
NUM_CRATERS = 100
NUM_SENSORS = 8  # Number of sensor rays
SENSOR_RANGE = 100  # How far sensors can detect

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Crater Simulation with Neural Networks")
clock = pygame.time.Clock()

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
        
        # Create the triangular shape
        self.generate_shape()
        
        # Neural network for crater behavior
        # Inputs: sensor readings (distance to objects in different directions)
        # Outputs: forward thrust, reverse thrust, rotation
        self.brain = SimpleNeuralNetwork(NUM_SENSORS * 2, 12, 3)
        
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
    
    def sense_environment(self, craters):
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
    
    def update(self, craters):
        """Update crater based on neural network output"""
        # Get sensor readings
        sensor_data = self.sense_environment(craters)
        
        # Feed sensor data to neural network
        output = self.brain.forward(sensor_data)
        
        # Interpret neural network output
        forward_thrust = (output[0] * 2) - 1  # Range: -1 to 1
        reverse_thrust = (output[1] * 2) - 1  # Range: -1 to 1
        rotation_force = (output[2] * 2) - 1  # Range: -1 to 1
        
        # Apply rotation
        self.angular_velocity += rotation_force * 0.05
        self.angular_velocity *= 0.9  # Damping
        self.rotation += self.angular_velocity
        
        # Apply thrust
        acceleration = forward_thrust * 0.1 - reverse_thrust * 0.05
        self.speed += acceleration
        self.speed *= 0.95  # Friction
        self.speed = max(-self.max_speed, min(self.max_speed, self.speed))
        
        # Update position
        self.x += self.speed * math.cos(self.rotation)
        self.y += self.speed * math.sin(self.rotation)
        
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
        
        # Update triangle points
        self.generate_shape()
    
    def draw(self, surface, draw_sensors=False):
        """Draw the crater and optionally its sensors"""
        # Draw crater triangle
        pygame.draw.polygon(surface, CRATER_COLOR, self.points)
        
        # Draw direction indicator (a small dot at the front)
        front_x, front_y = self.points[0]  # First point is the front
        pygame.draw.circle(surface, DIRECTION_COLOR, (int(front_x), int(front_y)), 3)
        
        # Draw sensors if enabled
        if draw_sensors:
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
    def __init__(self, num_craters=NUM_CRATERS):
        self.craters = [Crater() for _ in range(num_craters)]
        self.show_sensors = True  # Toggle for sensor visualization (now enabled by default)
    
    def update(self):
        """Update all craters"""
        for crater in self.craters:
            crater.update(self.craters)
    
    def draw(self, surface):
        """Draw all craters"""
        for crater in self.craters:
            crater.draw(surface, self.show_sensors)

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