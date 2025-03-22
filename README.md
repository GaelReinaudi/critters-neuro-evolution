# Crater Simulation

A 2D simulation of triangular craters moving on a plane with neural network-controlled behavior, energy management, and evolutionary optimization.

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
- **E key**: Manually trigger evolution
- **ESC key**: Quit the simulation

## Features
- 100 triangular craters with neural network-controlled movement
- Energy system:
  - Craters consume energy when moving and rotating
  - Craters must collect green food pellets to replenish energy
  - Energy level displayed as a red number on each crater
  - Craters get brighter as they gain more energy
  - Craters that run out of energy transform into new food pellets
- Evolutionary algorithm:
  - Automatic evolution every 30 seconds or when population drops too low
  - Selection of top-performing craters based on fitness metrics
  - Fitness calculated from age, energy, food consumption, and distance traveled
  - Offspring inherit neural networks with random mutations
  - Population improves behavior over generations
- Visual enhancements:
  - Cyan dot indicating direction of movement
  - Color-changing sensors (red when detecting close objects, green when clear)
  - Green food pellets appear randomly on the screen
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
Each crater uses ray-casting to detect nearby objects and feeds this data to a neural network that decides how to move. The neural network outputs control thrust (forward/backward) and rotation. Craters must balance exploring to find food with conserving energy for survival. When a crater runs out of energy, it disappears and becomes a food pellet, creating a natural energy cycle in the ecosystem.

Over time, the evolutionary algorithm selects the most successful craters (those that survive longest, collect the most food, and efficiently explore the environment) and uses them as parents for the next generation. Each new generation inherits the "knowledge" encoded in the neural networks of successful parents, with small random mutations that allow for ongoing adaptation and improvement.

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
- Evolution parameters (interval, selection percentage, mutation rate)
