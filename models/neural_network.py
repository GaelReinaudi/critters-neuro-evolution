"""
Neural network implementations for craters simulation
"""
import numpy as np
import pygad.nn as pygad_nn

class SimpleNeuralNetwork:
    """
    A basic feedforward neural network with one hidden layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize neural network with random weights
        
        Args:
            input_size (int): Number of input neurons
            hidden_size (int): Number of hidden neurons
            output_size (int): Number of output neurons
        """
        # Initialize with random weights, scaled to prevent saturation
        self.weights_ih = np.random.randn(hidden_size, input_size) * 0.1
        self.bias_h = np.random.randn(hidden_size, 1) * 0.1
        self.weights_ho = np.random.randn(output_size, hidden_size) * 0.1
        self.bias_o = np.random.randn(output_size, 1) * 0.1
        
        # Cache for forward pass
        self.input_cache = None
        self.hidden_cache = None
        self.last_input = None
        
    def forward(self, inputs):
        """
        Forward pass through the network
        
        Args:
            inputs (list or array): Input values
            
        Returns:
            array: Output values
        """
        # Check if we can reuse cached results
        inputs_array = np.array(inputs).reshape(-1, 1)
        if self.input_cache is not None and np.array_equal(inputs_array, self.last_input):
            return self.output_cache.flatten()
        
        # Save the current input
        self.last_input = inputs_array.copy()
        
        # Hidden layer
        hidden = np.dot(self.weights_ih, inputs_array) + self.bias_h
        hidden = self.fast_sigmoid(hidden)
        self.hidden_cache = hidden
        
        # Output layer
        output = np.dot(self.weights_ho, hidden) + self.bias_o
        output = self.fast_sigmoid(output)
        
        # Cache the result
        self.output_cache = output
        
        return output.flatten()
    
    @staticmethod
    def sigmoid(x):
        """
        Standard sigmoid activation function
        
        Args:
            x (array): Input values
            
        Returns:
            array: Activated values
        """
        return 1 / (1 + np.exp(-x))
        
    @staticmethod
    def fast_sigmoid(x):
        """
        Fast approximate sigmoid function
        
        Args:
            x (array): Input values
            
        Returns:
            array: Activated values
        """
        # Using a faster approximation of sigmoid
        # For values in a reasonable range (-5 to 5), this is quite accurate
        # and much faster than exp
        return 0.5 * (1 + np.tanh(0.5 * x))


class DeepNeuralNetwork:
    """
    An advanced feedforward neural network with multiple hidden layers and various activation functions
    """
    def __init__(self, input_size, hidden_layers, output_size, activation='relu'):
        """
        Initialize deep neural network with random weights
        
        Args:
            input_size (int): Number of input neurons
            hidden_layers (list): List of integers specifying size of each hidden layer
            output_size (int): Number of output neurons
            activation (str): Activation function to use ('relu', 'leaky_relu', 'tanh', 'sigmoid')
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_name = activation
        
        # Select activation function
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'leaky_relu':
            self.activation = self.leaky_relu
        elif activation == 'tanh':
            self.activation = np.tanh
        else:
            self.activation = self.fast_sigmoid
        
        # Initialize weights and biases for all layers
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # He initialization for ReLU and variants
        if activation in ['relu', 'leaky_relu']:
            scale_method = lambda fan_in: np.sqrt(2.0 / fan_in)
        else:
            # Xavier initialization for sigmoid and tanh
            scale_method = lambda fan_in: np.sqrt(1.0 / fan_in)
            
        # Initialize all layers
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = scale_method(fan_in)
            
            # Initialize weights and biases with appropriate scaling
            self.weights.append(np.random.randn(fan_out, fan_in) * scale)
            self.biases.append(np.random.randn(fan_out, 1) * 0.01)  # Small bias initialization
        
        # Cache for forward pass to improve performance
        self.activations = [None] * (len(layer_sizes))
        self.last_input = None
        self.output_cache = None
    
    def forward(self, inputs):
        """
        Forward pass through the deep network
        
        Args:
            inputs (list or array): Input values
            
        Returns:
            array: Output values
        """
        # Check if we can reuse cached results
        inputs_array = np.array(inputs).reshape(-1, 1)
        if self.last_input is not None and np.array_equal(inputs_array, self.last_input):
            return self.output_cache.flatten()
        
        # Save the current input
        self.last_input = inputs_array.copy()
        self.activations[0] = inputs_array
        
        # Forward propagation through all layers except output
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.activations[i + 1] = self.activation(z)
        
        # Output layer with sigmoid activation for [0,1] output
        z_out = np.dot(self.weights[-1], self.activations[-2]) + self.biases[-1]
        self.activations[-1] = self.fast_sigmoid(z_out)
        
        # Cache the result
        self.output_cache = self.activations[-1]
        
        return self.output_cache.flatten()
    
    @staticmethod
    def relu(x):
        """
        ReLU activation function
        
        Args:
            x (array): Input values
            
        Returns:
            array: Activated values
        """
        return np.maximum(0, x)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        Leaky ReLU activation function
        
        Args:
            x (array): Input values
            alpha (float): Leak coefficient
            
        Returns:
            array: Activated values
        """
        return np.maximum(alpha * x, x)
    
    @staticmethod
    def fast_sigmoid(x):
        """
        Fast approximate sigmoid function
        
        Args:
            x (array): Input values
            
        Returns:
            array: Activated values
        """
        # Using a faster approximation of sigmoid
        # For values in a reasonable range (-5 to 5), this is quite accurate
        # and much faster than exp
        return 0.5 * (1 + np.tanh(0.5 * x))


class PyGADNeuralNetwork:
    """
    Neural network implementation using pyGAD
    """
    def __init__(self, input_size, hidden_layers, output_size, weights=None):
        """
        Initialize neural network with random weights or provided weights
        
        Args:
            input_size (int): Number of input neurons
            hidden_layers (list): List of hidden layer sizes
            output_size (int): Number of output neurons
            weights (list, optional): Flattened list of weights if already available
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        
        # Define network architecture
        self.network_architecture = [input_size] + hidden_layers + [output_size]
        
        # Calculate the number of weights needed
        self.num_weights = 0
        for i in range(len(self.network_architecture) - 1):
            # Weights + biases for this layer
            layer_weights = (self.network_architecture[i] * self.network_architecture[i+1]) + self.network_architecture[i+1]
            self.num_weights += layer_weights
        
        # Initialize weights manually if not provided
        if weights is None:
            # Create random weights
            self.weights = np.random.uniform(-0.1, 0.1, size=self.num_weights)
        else:
            # Use provided weights
            self.weights = np.array(weights)
        
        # Create the layer weights and biases
        self._create_layers_from_weights()
            
        # Cache for forward pass
        self.last_input = None
        self.output_cache = None
    
    def _create_layers_from_weights(self):
        """Convert flat weights array to weight matrices and bias vectors for each layer"""
        self.layer_weights = []
        self.layer_biases = []
        
        # Index to keep track of position in flattened weights array
        idx = 0
        
        # For each layer
        for i in range(len(self.network_architecture) - 1):
            # Get dimensions
            input_size = self.network_architecture[i]
            output_size = self.network_architecture[i+1]
            
            # Extract weights for this layer
            w_size = input_size * output_size
            w = self.weights[idx:idx+w_size].reshape(output_size, input_size)
            idx += w_size
            
            # Extract biases for this layer
            b = self.weights[idx:idx+output_size].reshape(output_size, 1)
            idx += output_size
            
            # Save weights and biases
            self.layer_weights.append(w)
            self.layer_biases.append(b)
    
    def forward(self, inputs):
        """
        Forward pass through the network
        
        Args:
            inputs (list or array): Input values
            
        Returns:
            array: Output values (0-1 range)
        """
        # Check if we can reuse cached results
        inputs_array = np.array(inputs)
        if self.last_input is not None and np.array_equal(inputs_array, self.last_input):
            return self.output_cache
        
        # Save the current input
        self.last_input = inputs_array.copy()
        
        # Convert to column vector
        x = inputs_array.reshape(-1, 1)
        
        # Forward pass through each layer
        for i in range(len(self.layer_weights) - 1):
            # Linear transformation
            z = np.dot(self.layer_weights[i], x) + self.layer_biases[i]
            # ReLU activation for hidden layers
            x = np.maximum(0, z)
        
        # Output layer with sigmoid activation
        z_out = np.dot(self.layer_weights[-1], x) + self.layer_biases[-1]
        output = 1.0 / (1.0 + np.exp(-z_out))
        
        # Cache the result
        self.output_cache = output.flatten()
        
        return self.output_cache
    
    def get_weights(self):
        """Get flattened weights for genetic algorithm
        
        Returns:
            list: Flattened weights and biases
        """
        return self.weights.tolist() 