import numpy as np
class Neuron:
    def __init__(self, n_inputs, bias=0., weights=None):
        self.b = bias
        if weights:
            self.ws = np.array(weights)
        else:
            self.ws = np.random.rand(n_inputs)

        self.last_input = None
        self.gradient = None

    def _f(self, x):
        return max(0.1 * x, x)

    def _df(self, x):
        return 1.0 if x >= 0 else 0.1

    def __call__(self, xs):
        self.last_input = xs
        return self._f(xs @ self.ws + self.b)

    def compute_gradient(self, delta):
        self.gradient = delta * self._df(self.last_input @ self.ws + self.b)
        return self.gradient

    def update_weights(self, learning_rate):
        self.ws -= learning_rate * self.gradient * self.last_input

    def reset_gradient(self):
        self.gradient = None

class Layer:
    def __init__(self, n_neurons, n_inputs_per_neuron):
        self.neurons = [Neuron(n_inputs_per_neuron) for _ in range(n_neurons)]

    def __call__(self, inputs):
        return np.array([neuron(inputs) for neuron in self.neurons])

    def compute_gradients(self, deltas):
        return np.array([neuron.compute_gradient(delta) for neuron, delta in zip(self.neurons, deltas)])

    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)

    def reset_gradients(self):
        for neuron in self.neurons:
            neuron.reset_gradient()

class ANN:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def backpropagate(self, inputs, targets, learning_rate):
        outputs = self(inputs)
        delta = (outputs - targets).reshape(-1, 1)

        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                deltas = delta
            else:
                deltas = np.dot(self.layers[i+1].neurons[0].ws.T, np.squeeze(delta)) * np.array([neuron._df(neuron.last_input @ neuron.ws + neuron.b) for neuron in self.layers[i].neurons])

            self.layers[i].compute_gradients(deltas)

        for layer in self.layers:
            layer.update_weights(learning_rate)

        for layer in self.layers:
            layer.reset_gradients()

input_layer_size = 3
hidden_layer1_size = 4
hidden_layer2_size = 4
output_layer_size = 1

input_layer = Layer(input_layer_size, input_layer_size)
hidden_layer1 = Layer(hidden_layer1_size, input_layer_size)
hidden_layer2 = Layer(hidden_layer2_size, hidden_layer1_size)
output_layer = Layer(output_layer_size, hidden_layer2_size)

ann = ANN([hidden_layer1, hidden_layer2, output_layer])

training_inputs = np.array([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6],
                            [0.7, 0.8, 0.9]])
training_targets = np.array([[0.4],
                             [0.7],
                             [1.0]])

epochs = 1000
print_every = 100
learning_rate = 0.1

for epoch in range(epochs):
    for inputs, targets in zip(training_inputs, training_targets):
        ann.backpropagate(inputs, targets, learning_rate)

    if epoch % print_every == 0:
        input_data = np.random.rand(input_layer_size)
        output = ann(input_data)
        print(f"Epoch {epoch}: Input: {input_data}, Output: {output}")



validation_data = [
    (np.array([0.3, 0.3, 0.3]), 0.3),
    (np.array([0.5, 0.5, 0.5]), 0.5),
    (np.array([0.8, 0.8, 0.8]), 0.9)
]

print("\nValidation after training:")
for input_data, target in validation_data:
    output = ann(input_data)
    print(f"Input: {input_data}, Target: {target}, Output: {output}")
