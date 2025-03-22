"""
Main simulation logic for crater simulation
"""
import random
import pygame
from craters.config import (
    NUM_CRATERS, NUM_FOOD_PELLETS, TEXT_COLOR,
    FOOD_SPAWN_INTERVAL, ENABLE_EVOLUTION, EVOLUTION_INTERVAL,
    MIN_POPULATION, SELECTION_PERCENTAGE
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
        
        # Evolution tracking
        self.generation = 1
        self.frames_since_evolution = 0
        self.evolution_history = []  # Track fitness across generations
    
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
        
        # Check if we should evolve the population
        if ENABLE_EVOLUTION:
            self.frames_since_evolution += 1
            
            # Evolve if it's been long enough or population is too small
            if (self.frames_since_evolution >= EVOLUTION_INTERVAL or 
                len(self.craters) <= MIN_POPULATION):
                self.evolve_population()
                self.frames_since_evolution = 0
    
    def evolve_population(self):
        """
        Apply evolutionary algorithm to the crater population:
        1. Select top-performing craters as parents
        2. Create offspring with mutations
        3. Replace population with offspring
        """
        if not self.craters:  # No craters to evolve
            self.craters = [Crater(font=self.font) for _ in range(NUM_CRATERS)]
            return
        
        # Calculate fitness for all craters
        for crater in self.craters:
            crater.calculate_fitness()
        
        # Sort craters by fitness (descending)
        self.craters.sort(key=lambda c: c.calculate_fitness(), reverse=True)
        
        # Record best fitness for history
        if self.craters:
            best_fitness = self.craters[0].calculate_fitness()
            avg_fitness = sum(c.calculate_fitness() for c in self.craters) / len(self.craters)
            self.evolution_history.append((self.generation, best_fitness, avg_fitness))
        
        # Select top percentage as parents
        num_parents = max(2, int(len(self.craters) * SELECTION_PERCENTAGE))
        parents = self.craters[:num_parents]
        
        # Create new population from parents
        new_population = []
        
        # First, keep the best parent unchanged (elitism)
        if parents:
            new_population.append(Crater(
                brain=parents[0].brain,  # Keep brain unchanged
                font=self.font
            ))
        
        # Fill the rest with offspring
        while len(new_population) < NUM_CRATERS:
            # Select a random parent (weighted by fitness)
            parent = random.choices(
                parents,
                weights=[c.calculate_fitness() for c in parents],
                k=1
            )[0]
            
            # Create an offspring with mutations
            offspring = Crater.create_offspring(parent)
            offspring.font = self.font
            new_population.append(offspring)
        
        # Replace the old population
        self.craters = new_population
        
        # Increment generation counter
        self.generation += 1
        
        print(f"Evolution completed. Generation {self.generation}, Population: {len(self.craters)}")
    
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
            
            # Basic info
            info_text = f"Food: {active_food} | Craters: {active_craters}/{NUM_CRATERS} | Generation: {self.generation}"
            text_surface = self.font.render(info_text, True, TEXT_COLOR)
            surface.blit(text_surface, (10, 10))
            
            # Evolution info
            if self.evolution_history and self.font:
                gen_info = f"Evolution in: {(EVOLUTION_INTERVAL - self.frames_since_evolution) // 60}s"
                evolution_surface = self.font.render(gen_info, True, TEXT_COLOR)
                surface.blit(evolution_surface, (10, 30))
    
    def toggle_sensors(self):
        """Toggle the visibility of sensor rays"""
        self.show_sensors = not self.show_sensors 