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
ORANGE_FOOD_COLOR = (255, 165, 0)  # Orange for food from dead craters
TEXT_COLOR = (255, 255, 255)  # White for general text
ENERGY_TEXT_COLOR = (0, 0, 0)  # Red for energy display
MATING_COLOR = (255, 0, 255)  # Magenta for mating state

# Age color thresholds
YOUNG_COLOR = (100, 100, 255)  # Blue for young craters
ADULT_COLOR = (100, 255, 100)  # Green for adult craters
MATURE_COLOR = (255, 255, 100)  # Yellow for mature craters
ELDER_COLOR = (255, 100, 100)  # Red for elder craters
AGE_YOUNG = 500      # Frames until considered adult
AGE_ADULT = 2000     # Frames until considered mature
AGE_MATURE = 5000    # Frames until considered elder

# Simulation parameters
NUM_CRATERS = 200
NUM_SENSORS = 8  # Number of sensor rays
SENSOR_RANGE = 100  # How far sensors can detect
NUM_FOOD_PELLETS = 1000  # Number of food pellets
FOOD_SPAWN_INTERVAL = 30  # Frames between food spawning attempts

# Energy settings
FOOD_ENERGY = 200  # Energy gained from food (increased)
MAX_ENERGY = 1000  # Maximum energy a crater can have
INITIAL_ENERGY = 100  # Starting energy for craters (increased)
ENERGY_DEPLETION_RATE = 0.15  # Energy lost per unit of movement (decreased)
ENERGY_ROTATION_COST = 0.05  # Energy lost per unit of rotation (decreased)

# Movement settings
MAX_SPEED = 4.0  # Maximum speed (increased)
ACCELERATION_FACTOR = 0.25  # Acceleration multiplier (increased)
FRICTION = 0.98  # Friction coefficient (reduced friction)

# Mating parameters
MATING_ENERGY_THRESHOLD = 400  # Energy required to enter mating state
MATING_RADIUS = 40  # Distance for detecting other mating craters
MATING_PROBABILITY = 0.005  # Chance per frame to enter mating state when above threshold
MATING_DURATION = 300  # How long a crater stays in mating state (in frames)
MUTATION_RATE = 0.1  # Probability of mutation for each weight
MUTATION_SCALE = 0.2  # Scale of mutations

# Optimization settings
USE_SPATIAL_HASH = True           # Use spatial partitioning for faster collision detection
SENSOR_UPDATE_FRAMES = 3          # Update sensors every N frames
DISTANCE_CUTOFF = 150             # Ignore interactions beyond this distance
BATCH_PROCESSING = True           # Process entities in batches
USE_NUMBA = False                 # Use Numba JIT compilation if available
PRECOMPUTE_ANGLES = True          # Precompute trigonometric functions
SKIP_FRAMES_WHEN_LAGGING = True   # Skip frames if FPS drops too low

# Display settings
FONT_SIZE = 24  # Size for energy display
FPS = 60  # Frames per second 