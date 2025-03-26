"""
Neural network implementations for craters simulation
"""
import numpy as np
import pygad.nn as pygad_nn
import neat
import math
import random
import os
import itertools

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


class NEATNeuralNetwork:
    """
    Neural network implementation using NEAT-Python
    """
    
    @staticmethod
    def create_default_config():
        """
        Create a default NEAT configuration
        
        Returns:
            neat.config.Config: Default NEAT configuration
        """
        # Create a directory for NEAT config if it doesn't exist
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        # Create config file path
        config_path = os.path.join(config_dir, 'neat-config.txt')
        
        # Create default config file if it doesn't exist
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                f.write("""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1000.0
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_direct

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = 0
num_outputs             = 0

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
                """)
        
        # Load the config
        config = neat.config.Config(
            neat.genome.DefaultGenome,
            neat.reproduction.DefaultReproduction,
            neat.species.DefaultSpeciesSet,
            neat.stagnation.DefaultStagnation,
            config_path
        )
        
        return config
    
    def __init__(self, input_size=None, output_size=None, genome=None):
        """
        Initialize a NEAT neural network with either an existing genome or by creating a new one
        
        Args:
            input_size (int): Number of inputs for the neural network
            output_size (int): Number of outputs for the neural network
            genome (neat.genome.DefaultGenome): Optional existing genome to use
        """
        self.input_size = input_size or 0
        self.output_size = output_size or 0
        self.config = self.create_default_config()
        
        # Update config with actual input/output sizes
        if input_size is not None and output_size is not None:
            self.config.genome_config.num_inputs = input_size
            self.config.genome_config.num_outputs = output_size
            
            # Initialize node indexer to ensure proper ID assignment
            self.config.genome_config.node_indexer = itertools.count(input_size + output_size)
        
        # If genome provided, use it
        if genome:
            self.genome = genome
        else:
            # Create a new genome with random initial weights
            genome_type = neat.genome.DefaultGenome
            self.genome = genome_type(0)
            
            # Manually create the required nodes since we're not using a NEAT population
            # First, create input nodes (keys 0 to input_size-1)
            for i in range(self.input_size):
                self.genome.nodes[i] = neat.genome.DefaultNodeGene(i)
                self.genome.nodes[i].activation = 'identity'
                self.genome.nodes[i].aggregation = 'sum'
            
            # Create output nodes (keys input_size to input_size+output_size-1)
            for i in range(self.input_size, self.input_size + self.output_size):
                self.genome.nodes[i] = neat.genome.DefaultNodeGene(i)
                self.genome.nodes[i].activation = 'sigmoid'
                self.genome.nodes[i].aggregation = 'sum'
            
            # Create connections between input and output nodes
            for i in range(self.input_size):
                for j in range(self.input_size, self.input_size + self.output_size):
                    key = (i, j)
                    self.genome.connections[key] = neat.genome.DefaultConnectionGene(key)
                    self.genome.connections[key].weight = random.uniform(-1, 1)
                    self.genome.connections[key].enabled = True
        
        # Create a custom FeedForwardNetwork instead of using neat.nn.FeedForwardNetwork
        self.network = self._create_network()
        self.last_outputs = None
    
    def _create_network(self):
        """
        Create a custom feed-forward network from the genome
        
        Returns:
            object: Custom feed-forward neural network
        """
        # Define input and output nodes
        input_nodes = list(range(self.input_size))
        output_nodes = list(range(self.input_size, self.input_size + self.output_size))
        
        # Create a set of node evaluations (for computation)
        node_evals = []
        
        # For each output node, gather all the connections
        for i in range(self.input_size, self.input_size + self.output_size):
            inputs = []
            for j in range(self.input_size):
                key = (j, i)
                if key in self.genome.connections and self.genome.connections[key].enabled:
                    inputs.append((j, self.genome.connections[key].weight))
            
            # Add the node evaluation
            node_evals.append((i, 'sigmoid', sum, 0.0, 1.0, inputs))
        
        # Create a custom network object (not using NEAT's FeedForwardNetwork directly)
        class CustomNetwork:
            def __init__(self, inputs, outputs, node_evals):
                self.input_nodes = inputs
                self.output_nodes = outputs
                self.node_evals = node_evals
                
        return CustomNetwork(input_nodes, output_nodes, node_evals)
    
    def forward(self, inputs):
        """
        Forward pass through the network
        
        Args:
            inputs (list): Input values for the neural network
            
        Returns:
            list: Output values from the neural network
        """
        # Ensure inputs match the expected size
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        # Create a dictionary of inputs
        input_dict = {i: inputs[i] for i in range(self.input_size)}
        
        # Calculate outputs
        outputs = {}
        for node_key, activation, agg_func, bias, response, links in self.network.node_evals:
            node_inputs = [input_dict[i] * w for i, w in links]
            s = agg_func(node_inputs)
            outputs[node_key] = self._sigmoid(bias + response * s)
        
        # Convert to list in the order of output nodes
        self.last_outputs = [outputs[i] for i in self.network.output_nodes]
        
        return self.last_outputs
    
    def _sigmoid(self, x):
        """
        Sigmoid activation function
        
        Args:
            x (float): Input value
            
        Returns:
            float: Output value (0-1 range)
        """
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    def get_genome(self):
        """
        Get the NEAT genome that defines this network
        
        Returns:
            neat.genome.DefaultGenome: The NEAT genome
        """
        return self.genome
    
    @classmethod
    def crossover(cls, parent1, parent2):
        """
        Create a new neural network by performing crossover between two parent networks
        
        Args:
            parent1 (NEATNeuralNetwork): First parent network
            parent2 (NEATNeuralNetwork): Second parent network
            
        Returns:
            NEATNeuralNetwork: Child network with genome created from parents
        """
        # Create new network with same structure
        new_network = cls(
            input_size=parent1.input_size,
            output_size=parent1.output_size
        )
        
        # Perform crossover by directly copying and combining connections
        # We'll use a simpler approach since we're not using the full NEAT genome evolution
        
        # For each possible connection, randomly inherit from either parent
        for i in range(parent1.input_size):
            for j in range(parent1.input_size, parent1.input_size + parent1.output_size):
                connection_key = (i, j)
                
                # Check if connection exists in parent networks
                p1_has_conn = False
                p2_has_conn = False
                p1_weight = 0.0
                p2_weight = 0.0
                
                # Check connections in parent1
                for node_key, activation, agg_func, bias, response, links in parent1.network.node_evals:
                    if node_key == j:
                        for input_idx, weight in links:
                            if input_idx == i:
                                p1_has_conn = True
                                p1_weight = weight
                                break
                
                # Check connections in parent2
                for node_key, activation, agg_func, bias, response, links in parent2.network.node_evals:
                    if node_key == j:
                        for input_idx, weight in links:
                            if input_idx == i:
                                p2_has_conn = True
                                p2_weight = weight
                                break
                
                # If both parents have the connection, inherit randomly from one
                if p1_has_conn and p2_has_conn:
                    if random.random() < 0.5:
                        weight = p1_weight
                    else:
                        weight = p2_weight
                    
                    # Apply slight mutation
                    if random.random() < 0.1:  # 10% chance of mutation
                        weight += random.gauss(0, 0.5)
                    
                    # Find the connection in the new network and update its weight
                    for idx, (node_key, activation, agg_func, bias, response, links) in enumerate(new_network.network.node_evals):
                        if node_key == j:
                            for link_idx, (input_idx, _) in enumerate(links):
                                if input_idx == i:
                                    # Update the weight
                                    new_links = list(links)
                                    new_links[link_idx] = (input_idx, weight)
                                    new_network.network.node_evals[idx] = (node_key, activation, agg_func, bias, response, new_links)
                                    break
                
                # If only one parent has the connection, inherit with 50% probability
                elif p1_has_conn and random.random() < 0.5:
                    # Find the output node in the new network
                    for idx, (node_key, activation, agg_func, bias, response, links) in enumerate(new_network.network.node_evals):
                        if node_key == j:
                            # Add the connection
                            new_links = list(links)
                            new_links.append((i, p1_weight))
                            new_network.network.node_evals[idx] = (node_key, activation, agg_func, bias, response, new_links)
                            break
                
                elif p2_has_conn and random.random() < 0.5:
                    # Find the output node in the new network
                    for idx, (node_key, activation, agg_func, bias, response, links) in enumerate(new_network.network.node_evals):
                        if node_key == j:
                            # Add the connection
                            new_links = list(links)
                            new_links.append((i, p2_weight))
                            new_network.network.node_evals[idx] = (node_key, activation, agg_func, bias, response, new_links)
                            break
                            
        # Apply some mutations (adding new connections)
        if random.random() < 0.05:  # 5% chance of adding a new connection
            # Select random input and output nodes
            input_node = random.randrange(parent1.input_size)
            output_node = random.randrange(parent1.input_size, parent1.input_size + parent1.output_size)
            
            # Check if this connection already exists
            connection_exists = False
            for node_key, activation, agg_func, bias, response, links in new_network.network.node_evals:
                if node_key == output_node:
                    for input_idx, _ in links:
                        if input_idx == input_node:
                            connection_exists = True
                            break
            
            # If connection doesn't exist, add it
            if not connection_exists:
                for idx, (node_key, activation, agg_func, bias, response, links) in enumerate(new_network.network.node_evals):
                    if node_key == output_node:
                        # Add the new connection with a random weight
                        new_links = list(links)
                        new_links.append((input_node, random.uniform(-1, 1)))
                        new_network.network.node_evals[idx] = (node_key, activation, agg_func, bias, response, new_links)
                        break
        
        return new_network 