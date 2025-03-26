"""
Food model for the crater simulation
"""
import random
import pygame
from config import (
    WIDTH, HEIGHT, FOOD_COLOR, FOOD_ENERGY
)

class Food:
    """
    Represents a food pellet that craters can consume for energy
    """
    def __init__(self, x=None, y=None, size=None, color=None):
        """
        Initialize food with random or specified attributes
        
        Args:
            x (float, optional): X coordinate. If None, a random position is used.
            y (float, optional): Y coordinate. If None, a random position is used.
            size (int, optional): Food size. If None, a random size is used.
            color (tuple, optional): RGB color tuple. If None, default food color is used.
        """
        self.set_position(x, y)
        self.size = size if size is not None else random.randint(5, 8)
        self.energy = FOOD_ENERGY
        self.active = True
        self.color = color if color is not None else FOOD_COLOR
    
    def set_position(self, x=None, y=None):
        """
        Set the position of the food pellet
        
        Args:
            x (float, optional): X coordinate. If None, a random position is used.
            y (float, optional): Y coordinate. If None, a random position is used.
        """
        # Use the provided coordinates or generate random ones
        self.x = x if x is not None else random.randint(50, WIDTH-50)
        self.y = y if y is not None else random.randint(50, HEIGHT-50)
    
    def draw(self, surface):
        """
        Draw the food pellet on the given surface
        
        Args:
            surface: Pygame surface to draw on
        """
        if self.active:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.size) 