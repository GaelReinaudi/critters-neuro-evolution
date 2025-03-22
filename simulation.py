"""
Main simulation logic for crater simulation
"""
import random
import pygame
from craters.config import (
    NUM_CRATERS, NUM_FOOD_PELLETS, TEXT_COLOR,
    FOOD_SPAWN_INTERVAL, ORANGE_FOOD_COLOR
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
        
        # Track statistics
        self.generation = 1
        self.mating_events = 0
        self.births = 0
    
    def update(self):
        """
        Update all entities in the simulation for one frame
        """
        # Update craters and handle energy depletion
        craters_to_remove = []
        mating_pairs = []  # Track craters that are mating this frame
        new_craters = []  # Track new craters born this frame
        
        # First pass: update all craters and check for mating
        for i, crater in enumerate(self.craters):
            # Update crater (returns other crater if mating occurred)
            other_crater = crater.update(self.craters, self.food_pellets)
            
            if other_crater and other_crater not in mating_pairs and crater not in mating_pairs:
                mating_pairs.append(crater)
                mating_pairs.append(other_crater)
            
            # Check if crater ran out of energy
            if crater.energy <= 0:
                # Mark for removal and create food pellet at its position
                craters_to_remove.append(i)
                # Create new orange food pellet at crater's position
                new_food = Food(crater.x, crater.y, color=ORANGE_FOOD_COLOR)
                self.food_pellets.append(new_food)
        
        # Handle mating pairs
        for i in range(0, len(mating_pairs), 2):
            if i+1 < len(mating_pairs):  # Ensure we have a pair
                parent1 = mating_pairs[i]
                parent2 = mating_pairs[i+1]
                
                # Each parent loses half their energy
                parent1.energy /= 2
                parent2.energy /= 2
                
                # Reset mating state
                parent1.is_mating = False
                parent2.is_mating = False
                
                # Create two offspring
                for _ in range(2):
                    offspring = Crater.create_offspring(parent1, parent2)
                    offspring.font = self.font
                    new_craters.append(offspring)
                
                # Track mating statistics
                self.mating_events += 1
                self.births += 2
        
        # Add new craters
        self.craters.extend(new_craters)
        
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
            mating_craters = sum(1 for crater in self.craters if crater.is_mating)
            
            # Basic info
            info_text = f"Food: {active_food} | Craters: {active_craters}"
            text_surface = self.font.render(info_text, True, TEXT_COLOR)
            surface.blit(text_surface, (10, 10))
            
            # Mating info
            mating_info = f"Mating Craters: {mating_craters} | Mating Events: {self.mating_events} | Births: {self.births}"
            mating_surface = self.font.render(mating_info, True, TEXT_COLOR)
            surface.blit(mating_surface, (10, 30))
    
    def toggle_sensors(self):
        """Toggle the visibility of sensor rays"""
        self.show_sensors = not self.show_sensors 