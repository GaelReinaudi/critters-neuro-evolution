"""
Simple neural network implementation for craters simulation
"""
import numpy as np

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