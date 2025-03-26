"""
Main entry point for crater simulation
"""
import sys
import pygame
from config import (
    WIDTH, HEIGHT, BACKGROUND_COLOR, FPS, FONT_SIZE,
    USE_SPATIAL_HASH, BATCH_PROCESSING, PRECOMPUTE_ANGLES,
    USE_GPU_ACCELERATION, USE_DOUBLE_BUFFER, VSYNC_ENABLED
)
from simulation import CraterSimulation

def main():
    """Main function to run the crater simulation"""
    # Initialize Pygame
    pygame.init()
    
    # Set up the display with hardware acceleration based on config
    display_flags = 0
    if USE_GPU_ACCELERATION:
        display_flags |= pygame.HWSURFACE
    if USE_DOUBLE_BUFFER:
        display_flags |= pygame.DOUBLEBUF
    if VSYNC_ENABLED:
        display_flags |= pygame.SCALED  # Required for vsync in pygame 2.0+
        screen = pygame.display.set_mode((WIDTH, HEIGHT), display_flags, vsync=1)
    else:
        screen = pygame.display.set_mode((WIDTH, HEIGHT), display_flags)
        
    pygame.display.set_caption("Crater Simulation with Neural Networks and Evolution")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, FONT_SIZE)
    
    # Create the simulation
    simulation = CraterSimulation(font=font)
    
    # Create hardware-accelerated surface for rendering
    hw_surface = pygame.Surface((WIDTH, HEIGHT), pygame.HWSURFACE)
    
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
    print(f"  Hardware acceleration: {'ENABLED' if USE_GPU_ACCELERATION else 'DISABLED'}")
    print(f"  Double buffering: {'ENABLED' if USE_DOUBLE_BUFFER else 'DISABLED'}")
    print(f"  VSync: {'ENABLED' if VSYNC_ENABLED else 'DISABLED'}")
    
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
                    # Force top energy craters to mate
                    simulation.force_mating(percentage=0.2)
                elif event.key == pygame.K_ESCAPE:
                    # Quit simulation
                    running = False
        
        # Clear the hardware surface
        hw_surface.fill(BACKGROUND_COLOR)
        
        # Update and draw simulation to hardware surface
        simulation.update()
        simulation.draw(hw_surface)
        
        # Blit hardware surface to screen
        screen.blit(hw_surface, (0, 0))
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 