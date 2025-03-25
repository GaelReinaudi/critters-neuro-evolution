# Crater Simulation

A 2D simulation of triangular craters moving on a plane with neural network-controlled behavior, energy management, and natural mating-based evolution.

## Demo Video
[![Evolution in Action: Neural Network Craters Learn to Survive](https://img.youtube.com/vi/88SG8uD1GIw/maxresdefault.jpg)](https://www.youtube.com/watch?v=88SG8uD1GIw)

Watch as triangular craters evolve from random movement to purposeful behavior in this fascinating simulation! The video shows a comparison between newly spawned craters moving randomly (top) and evolved craters after 50 generations showing intelligent behavior (bottom).

## Requirements
- Python 3.x
- Pygame (`pip install pygame`)
- NumPy (`pip install numpy`)

## Running the Simulation
```
python -m craters.main
```

## Controls
- **S key**: Toggle sensor visualization (on by default)
- **E key**: Force top 20% of highest-energy craters to enter mating mode
- **ESC key**: Quit the simulation

## Features
- 100 triangular craters with neural network-controlled movement
- Energy system:
  - Craters consume energy when moving and rotating
  - Craters must collect green food pellets to replenish energy
  - Energy level displayed as a red number on each crater
  - Craters get brighter as they gain more energy
  - Craters that run out of energy transform into orange food pellets
- Natural mating-based evolution:
  - Craters automatically enter mating state (magenta color) when energy exceeds threshold
  - Mating craters consume energy at twice the normal rate (indicated by "2x" and pulsing border)
  - When two mating craters meet, they produce offspring
  - Each parent loses half its energy during reproduction
  - Offspring inherit neural networks from both parents with some mutations
  - Population gradually evolves more effective behaviors over time
- Age-based coloring:
  - Young craters (blue): First stage of life
  - Adult craters (green): Second stage of life
  - Mature craters (yellow): Third stage of life 
  - Elder craters (red): Final stage of life
  - Colors transition smoothly between stages
  - Brightness still affected by energy level
- Visual enhancements:
  - Cyan dot indicating direction of movement
  - Multi-colored sensors showing what is being detected:
    - Red rays - Wall detection
    - Yellow rays - Crater detection
    - Magenta rays - Mating crater detection
    - Green rays - Food detection
    - Colored dots - Energy level of detected craters
  - Green food pellets appear randomly on the screen
  - Orange food pellets come from dead craters
  - Magenta pulsing border indicates craters in mating state
- Each crater has 8 sensors detecting:
  - Distance to walls in different directions
  - Distance to other craters in different directions
- Simple neural network with:
  - 17 inputs (8 directions x 2 readings per direction + energy level)
  - 12 hidden neurons
  - 3 outputs (forward thrust, reverse thrust, rotation)
- Realistic physics with:
  - Acceleration and deceleration
  - Angular momentum and rotation
  - Wall collision handling

## How It Works
Each crater uses ray-casting to detect nearby objects and feeds this data to a neural network that decides how to move. The neural network outputs control thrust (forward/backward) and rotation. Craters must balance exploring to find food with conserving energy for survival. When a crater runs out of energy, it disappears and becomes an orange food pellet, creating a natural energy cycle in the ecosystem.

When a crater's energy exceeds a threshold, it automatically enters a mating state (turning magenta). If two craters in mating state collide, they produce offspring with neural networks that combine features from both parents with some random mutations. This natural selection process favors craters with strategies that effectively find food and manage energy. Over generations, the population evolves more sophisticated behaviors without any external intervention.

As craters age, their color changes to reflect their life stage, progressing from blue (young) to green (adult) to yellow (mature) to red (elder). This visual cue allows you to identify which craters have survived the longest and are potentially carrying the most successful genetic information.

## Code Structure
The project follows a modular architecture:
- `config.py` - Central configuration parameters
- `main.py` - Entry point to run the simulation
- `simulation.py` - Main simulation logic class
- `models/`
  - `crater.py` - Crater entity with neural network brain
  - `food.py` - Food pellet entity
  - `neural_network.py` - Simple neural network implementation

## Customization
Edit `config.py` to modify:
- Number of craters and food pellets
- Energy consumption and replenishment rates
- Neural network architecture
- Sensor range and sensitivity
- Movement parameters (speed, acceleration, friction)
- Mating parameters (energy threshold, duration)
- Age thresholds and colors
- Mutation rate and scale
