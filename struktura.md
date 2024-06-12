```python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class Neuron:
    def __init__(self, n_inputs, bias=0., weights=None):
        self.b = bias
        if weights:
            self.ws = np.array(weights)
        else:
            self.ws = np.random.rand(n_inputs)

    def _f(self, x):
        return max(0.1 * x, x)

    def __call__(self, xs):
        return self._f(xs @ self.ws + self.b)

class Layer:
    def __init__(self, n_neurons, n_inputs_per_neuron):
        self.neurons = [Neuron(n_inputs_per_neuron) for _ in range(n_neurons)]

    def __call__(self, inputs):
        return np.array([neuron(inputs) for neuron in self.neurons])

class ANN:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


input_layer_size = 3
hidden_layer1_size = 4
hidden_layer2_size = 4
output_layer_size = 1


input_layer = Layer(input_layer_size, input_layer_size)
hidden_layer1 = Layer(hidden_layer1_size, input_layer_size)
hidden_layer2 = Layer(hidden_layer2_size, hidden_layer1_size)
output_layer = Layer(output_layer_size, hidden_layer2_size)

ann = ANN([hidden_layer1, hidden_layer2, output_layer])


def draw_network(layers):
    layer_sizes = [input_layer_size] + [len(layer.neurons) for layer in layers]
    G = nx.DiGraph()

    pos = {}
    node_id = 0
    v_spacing = 1.0 / float(max(layer_sizes))
    h_spacing = 1.0 / float(len(layer_sizes) - 1)

    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2
        for j in range(layer_size):
            pos[node_id] = (i * h_spacing, layer_top - j * v_spacing)
            node_id += 1

    node_id = 0
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                G.add_edge(node_id + j, node_id + layer_size_a + k)
        node_id += layer_size_a

    plt.figure(figsize=(8, 7))
    nx.draw(G, pos, with_labels=False, node_size=3000, node_color='skyblue', edge_color='k', width=1.5, arrowsize=20)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.axis('off')
    plt.show()

draw_network([hidden_layer1, hidden_layer2, output_layer])
```
