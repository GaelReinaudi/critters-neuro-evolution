# Crater Simulation

A 2D simulation of triangular craters moving on a plane with neural network-controlled behavior.

## Requirements
- Python 3.x
- Pygame (`pip install pygame`)
- NumPy (`pip install numpy`)

## Running the Simulation
```
python craters_sim.py
```

## Controls
- **S key**: Toggle sensor visualization (on by default)
- **Close window**: Quit the simulation

## Features
- 100 triangular craters with neural network-controlled movement
- Visual enhancements:
  - Cyan dot indicating direction of movement
  - Color-changing sensors (red when detecting close objects, green when clear)
- Each crater has 8 sensors detecting:
  - Distance to walls in different directions
  - Distance to other craters in different directions
- Simple neural network with:
  - 16 inputs (8 directions x 2 readings per direction)
  - 12 hidden neurons
  - 3 outputs (forward thrust, reverse thrust, rotation)
- Realistic physics with:
  - Acceleration and deceleration
  - Angular momentum and rotation
  - Wall collision handling

## How It Works
Each crater uses ray-casting to detect nearby objects and feeds this data to a neural network that decides how to move. The neural network outputs control thrust (forward/backward) and rotation, creating emergent behavior patterns.

## Customization
Edit `craters_sim.py` to modify:
- Number of craters
- Simulation parameters
- Neural network architecture
- Sensor range and sensitivity
