"""
Main entry point for crater simulation
"""
import sys
import pygame
from craters.config import (
    WIDTH, HEIGHT, BACKGROUND_COLOR, FPS, FONT_SIZE
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
    
    # Print information about controls
    print("Simulation started.")
    print("Controls:")
    print("  S - Toggle sensor visibility")
    print("  E - Manually trigger evolution")
    print("  ESC - Quit simulation")
    
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
                    simulation.evolve_population()
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