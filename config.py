# Configuration parameters for the crater simulation

# Window settings
WIDTH = 1900
HEIGHT = 1000

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
NUM_FOOD_PELLETS = 500  # Number of food pellets
FOOD_SPAWN_INTERVAL = 60  # Frames between food spawning attempts

# Energy settings
FOOD_ENERGY = 200  # Energy gained from food (increased)
MAX_ENERGY = 1000  # Maximum energy a crater can have
INITIAL_ENERGY = 20  # Starting energy for craters (increased)
ENERGY_DEPLETION_RATE = 0.15  # Energy lost per unit of movement (decreased)
ENERGY_ROTATION_COST = 0.05  # Energy lost per unit of rotation (decreased)

# Movement settings
MAX_SPEED = 4.0  # Maximum speed (increased)
ACCELERATION_FACTOR = 0.25  # Acceleration multiplier (increased)
FRICTION = 0.98  # Friction coefficient (reduced friction)

# Mating parameters
MATING_ENERGY_THRESHOLD = 300  # Energy required to enter mating state
MATING_RADIUS = 40  # Distance for detecting other mating craters
MATING_PROBABILITY = 0.005  # Chance per frame to enter mating state when above threshold
MATING_DURATION = 300  # How long a crater stays in mating state (in frames)
MUTATION_RATE = 0.1  # Probability of mutation for each weight
MUTATION_SCALE = 0.2  # Scale of mutations

# Neural network settings
USE_DEEP_NETWORK = True  # Use the advanced multi-layer network instead of simple one
NETWORK_HIDDEN_LAYERS = [24, 16, 8]  # Sizes of hidden layers for deep network
NETWORK_ACTIVATION = 'relu'  # Activation function: 'relu', 'leaky_relu', 'tanh', or 'sigmoid'

# Optimization settings
USE_SPATIAL_HASH = True  # Use spatial partitioning for faster collision detection
CELL_SIZE = 120  # Size of each spatial hash cell (should match DISTANCE_CUTOFF)
BATCH_PROCESSING = True  # Process entities in batches for better cache locality
SENSOR_UPDATE_FRAMES = 5  # Update sensors every N frames (higher = more performance, less accuracy)
DISTANCE_CUTOFF = 120  # Maximum distance to check for interactions (lower = better performance)
PRECOMPUTE_ANGLES = True  # Precompute sin/cos values for common angles
SKIP_FRAMES_WHEN_LAGGING = True  # Skip update frames if framerate is too low
USE_SENSOR_CACHING = True  # Cache sensor readings between updates
USE_FAST_MATH = True  # Use faster math approximations
REDUCE_DRAW_DETAIL = True  # Reduce drawing detail when many entities are visible

# Display settings
FONT_SIZE = 18  # Size for energy display
FPS = 60  # Frames per second 