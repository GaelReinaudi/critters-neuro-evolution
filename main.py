"""
Main entry point for crater simulation
"""
import sys
import pygame
from craters.config import (
    WIDTH, HEIGHT, BACKGROUND_COLOR, FPS, FONT_SIZE,
    USE_SPATIAL_HASH, BATCH_PROCESSING, PRECOMPUTE_ANGLES
)
from craters.simulation import CraterSimulation

def main():
    """Main function to run the crater simulation"""
    # Initialize Pygame
    pygame.init()
    
    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Crater Simulation with Neural Networks and Evolution")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, FONT_SIZE)
    
    # Create the simulation
    simulation = CraterSimulation(font=font)
    
    # Print information about controls and optimizations
    print("Simulation started.")
    print("Controls:")
    print("  S - Toggle sensor visualization")
    print("  E - Manually trigger evolution")
    print("  ESC - Quit simulation")
    
    print("\nEnhanced Sensor System:")
    print("  Red rays - Wall detection")
    print("  Yellow rays - Crater detection")
    print("  Magenta rays - Mating crater detection")
    print("  Green rays - Food detection")
    print("  Colored dots - Energy level of detected craters (red = low, green = high)")
    
    print("\nOptimization settings:")
    print(f"  Spatial hashing: {'ENABLED' if USE_SPATIAL_HASH else 'DISABLED'}")
    print(f"  Batch processing: {'ENABLED' if BATCH_PROCESSING else 'DISABLED'}")
    print(f"  Precomputed angles: {'ENABLED' if PRECOMPUTE_ANGLES else 'DISABLED'}")
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    # Toggle sensor visualization
                    simulation.toggle_sensors()
                elif event.key == pygame.K_e:
                    # Manually trigger evolution
                    print("Evolution manually triggered")
                    # Uncomment when evolution feature is implemented
                    # simulation.evolve_population()
                elif event.key == pygame.K_ESCAPE:
                    # Quit simulation
                    running = False
        
        # Clear the screen
        screen.fill(BACKGROUND_COLOR)
        
        # Update and draw simulation
        simulation.update()
        simulation.draw(screen)
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 