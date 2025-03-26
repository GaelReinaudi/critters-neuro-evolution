# Critter Simulation

A 2D simulation of triangular critters moving on a plane with neural network-controlled behavior, energy management, and natural mating-based evolution.

## Demo Video
[![Evolution in Action: Neural Network Critters Learn to Survive](https://img.youtube.com/vi/88SG8uD1GIw/maxresdefault.jpg)](https://www.youtube.com/watch?v=88SG8uD1GIw)

Watch as triangular critters evolve from random movement to purposeful behavior in this fascinating simulation! The video shows a comparison between newly spawned critters moving randomly (top) and evolved critters after 50 generations showing intelligent behavior (bottom).

## Requirements
- Python 3.6 or higher
- Required Python packages (can be installed via pip):
  - pygame
  - numpy
  - neat-python

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd critters-neuro-evolution
   ```

2. Install the required dependencies using one of these methods:

   **Method 1: Using requirements.txt**
   ```
   pip install -r requirements.txt
   ```

   **Method 2: Using setup.py**
   ```
   pip install -e .
   ```

## Running the Simulation

You can run the simulation directly from the root directory using:

```
python run.py
```

or:

```
python main.py
```

or (if you've made run.py executable):

```
./run.py
```

## Troubleshooting

- **Display Issues**: If you encounter display errors, try adjusting the display settings in `config.py`:
  - Set `USE_GPU_ACCELERATION = False` to disable hardware acceleration
  - Set `USE_DOUBLE_BUFFER = False` if double buffering causes issues
  - Reduce `WIDTH` and `HEIGHT` for lower-end machines

- **Performance Issues**: For improved performance on slower machines:
  - Reduce `NUM_CRATERS` and `NUM_FOOD_PELLETS` in `config.py`
  - Set `REDUCE_DRAW_DETAIL = True` to simplify rendering

- **ModuleNotFoundError**: If you get import errors, ensure you're running from the project root directory

## Controls
- **S key**: Toggle sensor visualization (on by default)
- **E key**: Force top 20% of highest-energy critters to enter mating mode
- **ESC key**: Quit the simulation

## Features
- 100 triangular critters with neural network-controlled movement
- Energy system:
  - Critters consume energy when moving and rotating
  - Critters must collect green food pellets to replenish energy
  - Energy level displayed as a red number on each critter
  - Critters get brighter as they gain more energy
  - Critters that run out of energy transform into orange food pellets
- Natural mating-based evolution:
  - Critters automatically enter mating state (magenta color) when energy exceeds threshold
  - Mating critters consume energy at twice the normal rate (indicated by "2x" and pulsing border)
  - When two mating critters meet, they produce offspring
  - Each parent loses half its energy during reproduction
  - Offspring inherit neural networks from both parents with some mutations
  - Population gradually evolves more effective behaviors over time
- Age-based coloring:
  - Young critters (blue): First stage of life
  - Adult critters (green): Second stage of life
  - Mature critters (yellow): Third stage of life 
  - Elder critters (red): Final stage of life
  - Colors transition smoothly between stages
  - Brightness still affected by energy level
- Visual enhancements:
  - Cyan dot indicating direction of movement
  - Multi-colored sensors showing what is being detected:
    - Red rays - Wall detection
    - Yellow rays - Critter detection
    - Magenta rays - Mating critter detection
    - Green rays - Food detection
    - Colored dots - Energy level of detected critters
  - Green food pellets appear randomly on the screen
  - Orange food pellets come from dead critters
  - Magenta pulsing border indicates critters in mating state
- Each critter has 8 sensors detecting:
  - Distance to walls in different directions
  - Distance to other critters in different directions
- Simple neural network with:
  - 17 inputs (8 directions x 2 readings per direction + energy level)
  - 12 hidden neurons
  - 3 outputs (forward thrust, reverse thrust, rotation)
- Realistic physics with:
  - Acceleration and deceleration
  - Angular momentum and rotation
  - Wall collision handling

## How It Works
Each critter uses ray-casting to detect nearby objects and feeds this data to a neural network that decides how to move. The neural network outputs control thrust (forward/backward) and rotation. Critters must balance exploring to find food with conserving energy for survival. When a critter runs out of energy, it disappears and becomes an orange food pellet, creating a natural energy cycle in the ecosystem.

When a critter's energy exceeds a threshold, it automatically enters a mating state (turning magenta). If two critters in mating state collide, they produce offspring with neural networks that combine features from both parents with some random mutations. This natural selection process favors critters with strategies that effectively find food and manage energy. Over generations, the population evolves more sophisticated behaviors without any external intervention.

As critters age, their color changes to reflect their life stage, progressing from blue (young) to green (adult) to yellow (mature) to red (elder). This visual cue allows you to identify which critters have survived the longest and are potentially carrying the most successful genetic information.

## Code Structure
The project follows a modular architecture:
- `config.py` - Central configuration parameters
- `main.py` - Entry point to run the simulation
- `simulation.py` - Main simulation logic class
- `spatial_hash.py` - Spatial partitioning for efficient collision detection
- `models/`
  - `crater.py` - Critter entity with neural network brain
  - `food.py` - Food pellet entity
  - `neural_network.py` - Simple neural network implementation

## Customization
Edit `config.py` to modify:
- Number of critters and food pellets
- Energy consumption and replenishment rates
- Neural network architecture
- Sensor range and sensitivity
- Movement parameters (speed, acceleration, friction)
- Mating parameters (energy threshold, duration)
- Age thresholds and colors
- Mutation rate and scale
