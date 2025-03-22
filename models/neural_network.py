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
        
    def forward(self, inputs):
        """
        Forward pass through the network
        
        Args:
            inputs (list or array): Input values
            
        Returns:
            array: Output values
        """
        # Convert inputs to numpy array
        inputs = np.array(inputs).reshape(-1, 1)
        
        # Hidden layer
        hidden = np.dot(self.weights_ih, inputs) + self.bias_h
        hidden = self.sigmoid(hidden)
        
        # Output layer
        output = np.dot(self.weights_ho, hidden) + self.bias_o
        output = self.sigmoid(output)
        
        return output.flatten()
    
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        
        Args:
            x (array): Input values
            
        Returns:
            array: Activated values
        """
        return 1 / (1 + np.exp(-x)) 