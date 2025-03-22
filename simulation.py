"""
Main simulation logic for crater simulation
"""
import pygame
from craters.config import (
    NUM_CRATERS, NUM_FOOD_PELLETS, TEXT_COLOR,
    FOOD_SPAWN_INTERVAL
)
from craters.models.crater import Crater
from craters.models.food import Food

class CraterSimulation:
    """
    Main simulation class that manages all entities and updates
    """
    def __init__(self, num_craters=NUM_CRATERS, num_food=NUM_FOOD_PELLETS, font=None):
        """
        Initialize the simulation with craters and food
        
        Args:
            num_craters (int): Number of craters to create
            num_food (int): Number of food pellets to create
            font: Pygame font for text display
        """
        self.font = font
        self.craters = [Crater(font=font) for _ in range(num_craters)]
        self.food_pellets = [Food() for _ in range(num_food)]
        self.show_sensors = True  # Toggle for sensor visualization
        self.food_spawn_timer = 0
        self.food_spawn_interval = FOOD_SPAWN_INTERVAL
    
    def update(self):
        """
        Update all entities in the simulation for one frame
        """
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
        
        # Count active food
        active_food = sum(1 for food in self.food_pellets if food.active)
        
        # Spawn new food periodically
        self.food_spawn_timer += 1
        if self.food_spawn_timer >= self.food_spawn_interval:
            if active_food < NUM_FOOD_PELLETS:
                # Replace consumed food
                for food in self.food_pellets:
                    if not food.active:
                        food.set_position()  # Generate new random position
                        food.active = True
                        break
            self.food_spawn_timer = 0
    
    def draw(self, surface):
        """
        Draw all entities to the given surface
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw food
        for food in self.food_pellets:
            food.draw(surface)
        
        # Draw craters
        for crater in self.craters:
            crater.draw(surface, self.show_sensors)
        
        # Display information
        if self.font:
            active_food = sum(1 for food in self.food_pellets if food.active)
            active_craters = len(self.craters)
            info_text = f"Food: {active_food} | Craters: {active_craters}/{NUM_CRATERS}"
            text_surface = self.font.render(info_text, True, TEXT_COLOR)
            surface.blit(text_surface, (10, 10))
    
    def toggle_sensors(self):
        """Toggle the visibility of sensor rays"""
        self.show_sensors = not self.show_sensors 