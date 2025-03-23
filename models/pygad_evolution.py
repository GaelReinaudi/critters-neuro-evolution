"""
PyGAD integration for crater evolution
"""
import random
import numpy as np
import pygad
from craters.config import (
    MUTATION_RATE, MUTATION_SCALE, 
    USE_DEEP_NETWORK, NETWORK_HIDDEN_LAYERS
)
from craters.models.neural_network import SimpleNeuralNetwork, DeepNeuralNetwork

class GeneticAlgorithmManager:
    """
    Manages genetic evolution using PyGAD for crater populations
    """
    def __init__(self):
        """Initialize the genetic algorithm manager"""
        pass
        
    def setup_genetic_algorithm(self, population_size, num_generations=1):
        """
        Setup the genetic algorithm parameters
        
        Args:
            population_size (int): Size of the population
            num_generations (int): Number of generations to evolve
        """
        # No setup needed when using only PyGAD's crossover and mutation functions directly
        pass

    def create_offspring(self, parent1, parent2, energy=None):
        """
        Create offspring by mating two parents using PyGAD
        
        Args:
            parent1: First parent crater
            parent2: Second parent crater
            energy (float, optional): Initial energy for offspring
            
        Returns:
            tuple: (neural_network, generation_depth)
        """
        # Convert parent networks to chromosome arrays
        parent1_chromosome = self._network_to_chromosome(parent1.brain)
        parent2_chromosome = self._network_to_chromosome(parent2.brain)
        
        # Create parents array for PyGAD
        parents = np.array([parent1_chromosome, parent2_chromosome])
        
        # Use PyGAD for crossover and mutation
        if USE_DEEP_NETWORK:
            # For deep networks, use single point crossover
            # The direct functions are not available in version 3.x, so we'll create a GA instance temporarily
            ga_instance = pygad.GA(
                num_generations=1,
                num_parents_mating=2,
                sol_per_pop=2,
                num_genes=len(parent1_chromosome),
                fitness_func=lambda x, y: 0,  # Dummy fitness function
                crossover_type="single_point",
                mutation_type="random",
                mutation_percent_genes=MUTATION_RATE * 100,
                mutation_by_replacement=False,
                random_mutation_min_val=-MUTATION_SCALE,
                random_mutation_max_val=MUTATION_SCALE,
                keep_parents=0
            )
            
            # Perform crossover
            offspring = ga_instance.crossover(parents, (1, len(parent1_chromosome)))[0]
            
            # Perform mutation
            offspring = ga_instance.mutation(offspring)
            
            # Convert chromosome back to network
            child_brain = self._chromosome_to_network(offspring, parent1.brain)
        else:
            # For simple networks, same approach
            ga_instance = pygad.GA(
                num_generations=1,
                num_parents_mating=2,
                sol_per_pop=2,
                num_genes=len(parent1_chromosome),
                fitness_func=lambda x, y: 0,  # Dummy fitness function
                crossover_type="single_point",
                mutation_type="random",
                mutation_percent_genes=MUTATION_RATE * 100,
                mutation_by_replacement=False,
                random_mutation_min_val=-MUTATION_SCALE,
                random_mutation_max_val=MUTATION_SCALE,
                keep_parents=0
            )
            
            # Perform crossover
            offspring = ga_instance.crossover(parents, (1, len(parent1_chromosome)))[0]
            
            # Perform mutation
            offspring = ga_instance.mutation(offspring)
            
            # Convert chromosome back to network
            child_brain = self._chromosome_to_network(offspring, parent1.brain)
            
        # Calculate new generation depth
        generation_depth = max(parent1.generation_depth, parent2.generation_depth) + 1
        
        return child_brain, generation_depth
    
    def _network_to_chromosome(self, network):
        """
        Convert neural network to a flat chromosome array
        
        Args:
            network: Neural network (Simple or Deep)
            
        Returns:
            array: Flattened weights and biases
        """
        if isinstance(network, DeepNeuralNetwork):
            # For deep network
            chromosome = []
            for i in range(len(network.weights)):
                # Add flattened weights
                chromosome.extend(network.weights[i].flatten())
                # Add flattened biases
                chromosome.extend(network.biases[i].flatten())
            return np.array(chromosome)
        else:
            # For simple network
            chromosome = []
            chromosome.extend(network.weights_ih.flatten())
            chromosome.extend(network.bias_h.flatten())
            chromosome.extend(network.weights_ho.flatten())
            chromosome.extend(network.bias_o.flatten())
            return np.array(chromosome)
    
    def _chromosome_to_network(self, chromosome, template_network):
        """
        Convert chromosome back to neural network
        
        Args:
            chromosome: Flattened array of weights and biases
            template_network: Network to use as template for structure
            
        Returns:
            Neural network with weights from chromosome
        """
        if isinstance(template_network, DeepNeuralNetwork):
            # For deep network
            new_network = DeepNeuralNetwork(
                input_size=template_network.input_size, 
                hidden_layers=template_network.hidden_layers,
                output_size=template_network.output_size,
                activation=template_network.activation_name
            )
            
            # Track position in chromosome
            pos = 0
            
            # Fill weights and biases
            for i in range(len(template_network.weights)):
                weight_shape = template_network.weights[i].shape
                bias_shape = template_network.biases[i].shape
                
                # Get weights
                weights_size = weight_shape[0] * weight_shape[1]
                weights = chromosome[pos:pos+weights_size].reshape(weight_shape)
                new_network.weights[i] = weights
                pos += weights_size
                
                # Get biases
                biases_size = bias_shape[0] * bias_shape[1]
                biases = chromosome[pos:pos+biases_size].reshape(bias_shape)
                new_network.biases[i] = biases
                pos += biases_size
                
            return new_network
        else:
            # For simple network
            new_network = SimpleNeuralNetwork(
                template_network.weights_ih.shape[1],
                template_network.weights_ih.shape[0],
                template_network.weights_ho.shape[0]
            )
            
            # Track position in chromosome
            pos = 0
            
            # Get weights_ih
            weights_ih_size = template_network.weights_ih.shape[0] * template_network.weights_ih.shape[1]
            new_network.weights_ih = chromosome[pos:pos+weights_ih_size].reshape(template_network.weights_ih.shape)
            pos += weights_ih_size
            
            # Get bias_h
            bias_h_size = template_network.bias_h.shape[0] * template_network.bias_h.shape[1]
            new_network.bias_h = chromosome[pos:pos+bias_h_size].reshape(template_network.bias_h.shape)
            pos += bias_h_size
            
            # Get weights_ho
            weights_ho_size = template_network.weights_ho.shape[0] * template_network.weights_ho.shape[1]
            new_network.weights_ho = chromosome[pos:pos+weights_ho_size].reshape(template_network.weights_ho.shape)
            pos += weights_ho_size
            
            # Get bias_o
            bias_o_size = template_network.bias_o.shape[0] * template_network.bias_o.shape[1]
            new_network.bias_o = chromosome[pos:pos+bias_o_size].reshape(template_network.bias_o.shape)
            
            return new_network 