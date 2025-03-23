"""
Main simulation logic for crater simulation
"""
import random
import time
import pygame
from craters.config import (
    NUM_CRATERS, NUM_FOOD_PELLETS, TEXT_COLOR,
    FOOD_SPAWN_INTERVAL, ORANGE_FOOD_COLOR,
    AGE_YOUNG, AGE_ADULT, AGE_MATURE,
    DISTANCE_CUTOFF, USE_SPATIAL_HASH,
    BATCH_PROCESSING, SKIP_FRAMES_WHEN_LAGGING,
    FPS, MATING_DURATION, CELL_SIZE,
    FOOD_DETECTION_RANGE, WIDTH, HEIGHT
)
from craters.models.crater import Crater
from craters.models.food import Food
from craters.spatial_hash import SpatialHash

class CraterSimulation:
    """
    Main simulation class that manages all entities and updates
    """
    def __init__(self, font=None):
        """
        Initialize simulation with craters, food, and other settings
        
        Args:
            font: Pygame font for text display
        """
        self.font = font
        self.craters = []
        self.food_pellets = []
        self.spatial_hash = SpatialHash(CELL_SIZE) if USE_SPATIAL_HASH else None
        
        # Performance tracking
        self.frame_times = []
        self.avg_frame_time = 0.0
        self.last_frame_time = time.time()
        
        # Evolution tracking
        self.generation = 0
        self.mating_events = 0
        self.births = 0
        self.show_sensors = True  # Sensor visualization option
        
        # Initialize the PyGAD genetic algorithm manager
        from craters.models.pygad_evolution import GeneticAlgorithmManager
        self.ga_manager = GeneticAlgorithmManager()
        self.ga_manager.setup_genetic_algorithm(NUM_CRATERS)
        
        # Initialize craters
        for _ in range(NUM_CRATERS):
            self.craters.append(Crater(font=self.font))
            
        # Initialize food pellets
        for _ in range(NUM_FOOD_PELLETS):
            self.food_pellets.append(Food(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
        
        self.food_spawn_timer = 0
        self.food_spawn_interval = FOOD_SPAWN_INTERVAL
    
    def _update_spatial_hash(self):
        """Update the spatial hash with current entity positions"""
        if not USE_SPATIAL_HASH:
            return
            
        self.spatial_hash.clear()
        for crater in self.craters:
            self.spatial_hash.insert(crater)
        
        for food in self.food_pellets:
            if food.active:
                self.spatial_hash.insert(food)
    
    def get_nearby_craters(self, crater):
        """Get craters near the given crater efficiently"""
        if USE_SPATIAL_HASH:
            nearby = self.spatial_hash.get_nearby_entities(crater, DISTANCE_CUTOFF)
            # Ensure we only return Crater instances and not the crater itself
            return [c for c in nearby if isinstance(c, Crater) and c is not crater]
        else:
            # Traditional O(n) approach
            return [c for c in self.craters if c is not crater]
            
    def get_nearby_food(self, crater):
        """Get active food pellets near the given crater efficiently"""
        if USE_SPATIAL_HASH:
            # Use the longer FOOD_DETECTION_RANGE for food detection
            nearby = self.spatial_hash.get_nearby_entities(crater, FOOD_DETECTION_RANGE)
            return [f for f in nearby if isinstance(f, Food) and f.active]
        else:
            # Traditional approach
            return [f for f in self.food_pellets if f.active]
    
    def update(self):
        """Update all entities and handle mating and evolution"""
        # Skip frames if running too slow
        if SKIP_FRAMES_WHEN_LAGGING and self.avg_frame_time > 1.0 / 30:
            # Skip update if frame time is over 30 FPS (prioritize rendering)
            if self.skip_frame:
                self.skip_frame = False
                return
            else:
                self.skip_frame = True
        
        # Start timer
        start_time = time.time()
        
        # Update food spawn timer
        self.food_spawn_timer += 1
        if self.food_spawn_timer >= self.food_spawn_interval:
            self.food_spawn_timer = 0
            
            # Only spawn new food if there are less than max food pellets
            active_pellets = sum(1 for f in self.food_pellets if f.active)
            if active_pellets < NUM_FOOD_PELLETS:
                # Add new food in random location
                self.food_pellets.append(Food(random.randint(50, WIDTH-50), random.randint(50, HEIGHT-50)))
        
        # Update spatial hash if used
        if USE_SPATIAL_HASH:
            self._update_spatial_hash()
        
        # Process craters in batches for better cache locality if enabled
        if BATCH_PROCESSING:
            # Update all craters and track those that need to mate
            mating_pairs = []
            for i, crater in enumerate(self.craters):
                # Skip if crater has no energy
                if crater.energy <= 0:
                    continue
                    
                # Update the crater and get mating partner if any
                other_crater = crater.update(self.craters, self.food_pellets)
                
                # If there's a mating partner, create offspring
                if other_crater and (crater, other_crater) not in mating_pairs and (other_crater, crater) not in mating_pairs:
                    mating_pairs.append((crater, other_crater))
            
            # Process all mating after updates
            for crater1, crater2 in mating_pairs:
                self.mate_craters(crater1, crater2)
        else:
            # Original update logic without batching
            # Update craters
            for crater in self.craters:
                if crater.energy <= 0:
                    continue
                    
                # Check for mating
                other_crater = crater.update(self.craters, self.food_pellets)
                if other_crater:
                    self.mate_craters(crater, other_crater)
        
        # Remove dead craters (energy <= 0) and create food in their place
        for crater in list(self.craters):
            if crater.energy <= 0:
                # Create a food pellet at the crater's position
                orange_food = Food(crater.x, crater.y, color=ORANGE_FOOD_COLOR, energy=crater.energy/2)
                self.food_pellets.append(orange_food)
                self.craters.remove(crater)
        
        # Update food pellets
        for food in self.food_pellets:
            food.update()
        
        # Remove inactive food pellets
        self.food_pellets = [food for food in self.food_pellets if food.exists]
        
        # Calculate frame time and update average
        end_time = time.time()
        frame_time = end_time - start_time
        
        # Track frame times for rolling average (last 60 frames)
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
            
        # Calculate average frame time
        self.avg_frame_time = sum(self.frame_times) / len(self.frame_times)
    
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
            if crater.energy > 0:
                crater.draw(surface, self.show_sensors, self.craters)
        
        # Display information
        if self.font:
            active_food = sum(1 for food in self.food_pellets if food.active)
            active_craters = len(self.craters)
            mating_craters = sum(1 for crater in self.craters if crater.is_mating)
            
            # Calculate age statistics
            total_age = sum(crater.age for crater in self.craters) if self.craters else 0
            avg_age = total_age / len(self.craters) if self.craters else 0
            max_age = max((crater.age for crater in self.craters), default=0)
            
            young_craters = sum(1 for crater in self.craters if crater.age < AGE_YOUNG)
            adult_craters = sum(1 for crater in self.craters if AGE_YOUNG <= crater.age < AGE_ADULT)
            mature_craters = sum(1 for crater in self.craters if AGE_ADULT <= crater.age < AGE_MATURE)
            elder_craters = sum(1 for crater in self.craters if crater.age >= AGE_MATURE)
            
            # Calculate generation depth statistics
            if self.craters:
                generation_depths = [crater.generation_depth for crater in self.craters]
                avg_generation = sum(generation_depths) / len(generation_depths)
                max_generation = max(generation_depths)
                min_generation = min(generation_depths)
            else:
                avg_generation = max_generation = min_generation = 0
            
            # Count inactive craters
            inactive_craters = sum(1 for crater in self.craters if crater.inactive_frames > crater.inactivity_threshold)
            avg_inactive_frames = sum(crater.inactive_frames for crater in self.craters) / len(self.craters) if self.craters else 0
            
            # Basic info
            info_text = f"Food: {active_food} | Craters: {active_craters}"
            text_surface = self.font.render(info_text, True, TEXT_COLOR)
            surface.blit(text_surface, (10, 10))
            
            # Generation stats info
            generation_info = f"Generation Stats - Avg: {avg_generation:.1f} | Max: {max_generation} | Min: {min_generation}"
            generation_surface = self.font.render(generation_info, True, TEXT_COLOR)
            surface.blit(generation_surface, (10, 30))
            
            # Mating info
            mating_info = f"Mating Craters: {mating_craters} | Mating Events: {self.mating_events} | Births: {self.births}"
            mating_surface = self.font.render(mating_info, True, TEXT_COLOR)
            surface.blit(mating_surface, (10, 50))
            
            # Age info
            age_info = f"Avg Age: {int(avg_age)} | Max Age: {max_age} | Y: {young_craters} | A: {adult_craters} | M: {mature_craters} | E: {elder_craters}"
            age_surface = self.font.render(age_info, True, TEXT_COLOR)
            surface.blit(age_surface, (10, 70))
            
            # Inactivity info
            inactive_info = f"Inactive Craters: {inactive_craters} | Avg Inactive Frames: {int(avg_inactive_frames)}"
            inactive_surface = self.font.render(inactive_info, True, TEXT_COLOR)
            surface.blit(inactive_surface, (10, 90))
            
            # Performance info
            perf_info = f"Frame Time: {self.avg_frame_time*1000:.1f}ms | FPS: {1.0/max(self.avg_frame_time, 0.001):.1f}"
            perf_surface = self.font.render(perf_info, True, TEXT_COLOR)
            surface.blit(perf_surface, (10, 110))
    
    def toggle_sensors(self):
        """Toggle the visibility of sensor rays"""
        self.show_sensors = not self.show_sensors
        
    def force_mating(self, percentage=0.5):
        """
        Force the top percentage of highest-energy craters to mate
        
        Args:
            percentage (float): Percentage of population to mate (0.0-1.0)
        """
        # Get active craters and sort by energy
        active_craters = [crater for crater in self.craters if crater.energy > 0]
        if len(active_craters) < 2:
            return
            
        # Sort by energy (highest first)
        top_craters = sorted(active_craters, key=lambda c: c.energy, reverse=True)
        
        # Select top percentage
        num_to_mate = max(2, int(len(top_craters) * percentage))
        num_to_mate = num_to_mate if num_to_mate % 2 == 0 else num_to_mate - 1  # Ensure even number
        top_craters = top_craters[:num_to_mate]
        
        # Randomize pairings to avoid inbreeding
        random.shuffle(top_craters)
        
        # Create offspring
        new_craters = []
        
        # Increment generation when evolution occurs
        self.generation += 1
        
        # Create pairs and produce offspring
        for i in range(0, num_to_mate, 2):
            if i+1 >= len(top_craters):
                break  # Skip last unpaired crater if somehow we have an odd number
                
            parent1 = top_craters[i]
            parent2 = top_craters[i+1]
            
            # Calculate total energy to distribute to offspring
            total_offspring_energy = (parent1.energy / 2) + (parent2.energy / 2)
            energy_per_offspring = total_offspring_energy / 2
            
            # Each parent loses half their energy
            parent1.energy /= 2
            parent2.energy /= 2
            
            # Reset mating state if they were in it
            parent1.is_mating = False
            parent2.is_mating = False
            
            # Create two offspring
            for _ in range(2):
                offspring = Crater.create_offspring(parent1, parent2, energy=energy_per_offspring)
                offspring.font = self.font
                new_craters.append(offspring)
            
            # Flash the pair with mating color briefly for visual feedback
            parent1.is_mating = True
            parent2.is_mating = True
            parent1.mating_timer = 10  # Very short duration just for visual feedback
            parent2.mating_timer = 10
            
            # Track mating statistics
            self.mating_events += 1
            self.births += 2
            
        # Add new craters
        self.craters.extend(new_craters)
            
        # Print information about forced mating
        print(f"Forced {num_to_mate//2} pairs of craters to mate, creating {len(new_craters)} offspring")

    def mate_craters(self, crater1, crater2):
        """
        Create offspring from two parent craters and handle mating process
        
        Args:
            crater1 (Crater): First parent crater
            crater2 (Crater): Second parent crater
        """
        # Create offspring using PyGAD-based genetic evolution
        offspring = Crater.create_offspring(crater1, crater2)
        
        # Add to craters list
        self.craters.append(offspring)
        
        # Reduce energy of parents - reproduction costs energy
        crater1.energy /= 2
        crater2.energy /= 2
        
        # Reset mating state
        crater1.is_mating = False
        crater2.is_mating = False
        
        # Update stats
        self.mating_events += 1
        self.births += 1
        
        return offspring 