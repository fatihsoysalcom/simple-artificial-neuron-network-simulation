import math
import random

# Sigmoid activation function - simulates a neuron's firing
# This function determines if a 'neuron' activates based on its input sum.
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid - used in backpropagation for learning
# This helps the network understand how much to adjust its 'connections' (weights).
def sigmoid_derivative(x):
    # Note: x here is assumed to be the output of the sigmoid function
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        # Initialize the number of nodes in each layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights with random values between -1 and 1
        # These weights represent the strength of connections between 'neurons'.
        # They are crucial for information transmission and processing.
        self.weights_ih = [[random.uniform(-1, 1) for _ in range(self.hidden_nodes)] for _ in range(self.input_nodes)]
        self.weights_ho = [[random.uniform(-1, 1) for _ in range(self.output_nodes)] for _ in range(self.hidden_nodes)]

        # Learning rate - how quickly the network adjusts its weights during training
        self.learning_rate = 0.1

    # Forward propagation: inputs through the network to produce outputs
    # This simulates how information flows through a network of neurons.
    def feedforward(self, input_array):
        # Calculate hidden layer inputs and apply activation
        hidden_inputs = [0.0] * self.hidden_nodes
        for j in range(self.hidden_nodes):
            for i in range(self.input_nodes):
                hidden_inputs[j] += input_array[i] * self.weights_ih[i][j]
        
        hidden_outputs = [sigmoid(x) for x in hidden_inputs] # Activation function applied to hidden 'neurons'

        # Calculate output layer inputs and apply activation
        output_inputs = [0.0] * self.output_nodes
        for k in range(self.output_nodes):
            for j in range(self.hidden_nodes):
                output_inputs[k] += hidden_outputs[j] * self.weights_ho[j][k]
        
        final_outputs = [sigmoid(x) for x in output_inputs] # Activation function applied to output 'neurons'
        
        return hidden_outputs, final_outputs

    # Backpropagation: adjust weights based on error
    # This is the 'learning' phase, where the network refines its connections.
    def train(self, input_array, target_array):
        hidden_outputs, final_outputs = self.feedforward(input_array)

        # Calculate output errors (difference between target and predicted)
        output_errors = [target_array[i] - final_outputs[i] for i in range(self.output_nodes)]

        # Calculate output gradients (error * derivative of activation)
        # This tells us how much each output 'neuron' contributed to the error.
        output_gradients = [output_errors[i] * sigmoid_derivative(final_outputs[i]) for i in range(self.output_nodes)]

        # Calculate hidden errors (propagated from output errors)
        # Errors are propagated backward through the network.
        hidden_errors = [0.0] * self.hidden_nodes
        for j in range(self.hidden_nodes):
            for k in range(self.output_nodes):
                hidden_errors[j] += output_gradients[k] * self.weights_ho[j][k]

        # Calculate hidden gradients
        hidden_gradients = [hidden_errors[j] * sigmoid_derivative(hidden_outputs[j]) for j in range(self.hidden_nodes)]

        # Update weights_ho (hidden to output)
        # Adjusting the strength of connections based on learning.
        for j in range(self.hidden_nodes):
            for k in range(self.output_nodes):
                delta_weight = output_gradients[k] * hidden_outputs[j] * self.learning_rate
                self.weights_ho[j][k] += delta_weight

        # Update weights_ih (input to hidden)
        # Further adjustment of connections.
        for i in range(self.input_nodes):
            for j in range(self.hidden_nodes):
                delta_weight = hidden_gradients[j] * input_array[i] * self.learning_rate
                self.weights_ih[i][j] += delta_weight

# --- Main execution --- 
if __name__ == "__main__":
    # Define the structure of the artificial neural network
    nn = NeuralNetwork(input_nodes=2, hidden_nodes=4, output_nodes=1) # 2 inputs, 4 hidden 'neurons', 1 output

    # Training data for a simple XOR gate
    # This simulates learning patterns, much like a brain learns from experience.
    training_data = [
        {"inputs": [0, 0], "targets": [0]},
        {"inputs": [0, 1], "targets": [1]},
        {"inputs": [1, 0], "targets": [1]},
        {"inputs": [1, 1], "targets": [0]}
    ]

    print("Training the neural network...")
    # Train the network for a number of epochs (iterations)
    for epoch in range(10000):
        for data in training_data:
            nn.train(data["inputs"], data["targets"])
    print("Training complete.\n")

    # Test the trained network
    print("Testing the trained network:")
    for data in training_data:
        _, output = nn.feedforward(data["inputs"])
        # The output is a probability, so we round it to get a binary classification
        print(f"Input: {data['inputs']}, Target: {data['targets'][0]}, Predicted: {output[0]:.4f} (Rounded: {round(output[0])})")
