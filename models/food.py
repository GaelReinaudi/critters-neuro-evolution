"""
Food model for the crater simulation
"""
import random
import pygame
from craters.config import WIDTH, HEIGHT, FOOD_COLOR, FOOD_ENERGY

class Food:
    """
    Represents a food pellet that can be consumed by craters
    """
    def __init__(self, x=None, y=None):
        """
        Initialize a food pellet with random or specified position
        
        Args:
            x (float, optional): X coordinate. If None, a random position is used.
            y (float, optional): Y coordinate. If None, a random position is used.
        """
        # Set coordinates (random if not provided)
        self.set_position(x, y)
        self.size = random.randint(5, 10)
        self.energy = FOOD_ENERGY
        self.active = True
    
    def set_position(self, x=None, y=None):
        """
        Set the position of the food, using random coordinates if not specified
        
        Args:
            x (float, optional): X coordinate. If None, a random position is used.
            y (float, optional): Y coordinate. If None, a random position is used.
        """
        self.x = x if x is not None else random.randint(20, WIDTH-20)
        self.y = y if y is not None else random.randint(20, HEIGHT-20)
    
    def draw(self, surface):
        """
        Draw the food pellet on the given surface
        
        Args:
            surface: Pygame surface to draw on
        """
        if self.active:
            pygame.draw.circle(surface, FOOD_COLOR, (int(self.x), int(self.y)), self.size) 