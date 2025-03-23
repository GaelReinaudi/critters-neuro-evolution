"""
Food model for the crater simulation
"""
import random
import pygame
from craters.config import (
    WIDTH, HEIGHT, FOOD_COLOR, FOOD_ENERGY, FOOD_SPAWN_INTERVAL
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
        self.spawn_timer = 0
        self.lifespan = 1000  # Lifespan in frames before disappearing
        self.age = 0
    
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
    
    def update(self):
        """Update food state"""
        if not self.active:
            # Increment spawn timer for inactive food
            self.spawn_timer += 1
            if self.spawn_timer >= FOOD_SPAWN_INTERVAL * 10:  # Longer interval for respawning
                # Reset to active with new position
                self.set_position()
                self.active = True
                self.spawn_timer = 0
        else:
            # Increment age
            self.age += 1
            if self.age > self.lifespan:
                self.active = False
    
    @property
    def exists(self):
        """
        Check if the food still exists in the simulation
        
        Returns:
            bool: True if the food is still part of the simulation
        """
        # Food exists if it's active or has a chance to become active again
        return self.active or self.spawn_timer < FOOD_SPAWN_INTERVAL * 10 