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
    pygame.display.set_caption("Crater Simulation with Neural Networks and Energy")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, FONT_SIZE)
    
    # Create the simulation
    simulation = CraterSimulation(font=font)
    
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
                    simulation.toggle_sensors()
        
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