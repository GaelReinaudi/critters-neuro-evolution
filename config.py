# Configuration parameters for the crater simulation

# Window settings
WIDTH = 1800
HEIGHT = 1600

# Colors
BACKGROUND_COLOR = (0, 0, 0)  # Black
CRATER_COLOR = (150, 150, 150)  # Gray
SENSOR_COLOR = (255, 0, 0)  # Red for ray visualization
DIRECTION_COLOR = (0, 255, 255)  # Cyan for direction indicator
FOOD_COLOR = (0, 255, 0)  # Green for food pellets
TEXT_COLOR = (255, 255, 255)  # White for general text
ENERGY_TEXT_COLOR = (255, 0, 0)  # Red for energy display

# Simulation parameters
NUM_CRATERS = 100
NUM_SENSORS = 8  # Number of sensor rays
SENSOR_RANGE = 100  # How far sensors can detect
NUM_FOOD_PELLETS = 50  # Number of food pellets
FOOD_SPAWN_INTERVAL = 60  # Frames between food spawning attempts

# Energy settings
FOOD_ENERGY = 200  # Energy gained from food (increased)
MAX_ENERGY = 1000  # Maximum energy a crater can have
INITIAL_ENERGY = 300  # Starting energy for craters (increased)
ENERGY_DEPLETION_RATE = 0.15  # Energy lost per unit of movement (decreased)
ENERGY_ROTATION_COST = 0.05  # Energy lost per unit of rotation (decreased)

# Movement settings
MAX_SPEED = 4.0  # Maximum speed (increased)
ACCELERATION_FACTOR = 0.25  # Acceleration multiplier (increased)
FRICTION = 0.98  # Friction coefficient (reduced friction)

# Display settings
FONT_SIZE = 12  # Size for energy display
FPS = 60  # Frames per second 