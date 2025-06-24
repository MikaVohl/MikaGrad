import numpy as np
from engine import Value

class Module:
    def zero_grad(self):
        for param in self.parameters():
            param.grad.fill(0.0)

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, num_in, nonlin=True):
        self.weights = Value(np.random.uniform(-1, 1), size=(num_in,))
        self.bias = Value(0.0)
        self.nonlin = nonlin
    
    def __call__(self, x):
        x = x if isinstance(x, Value) else Value(np.asarray(x, dtype=float))
        act = self.weights.dot(x) + self.bias
        return act.relu() if self.nonlin else act
    
    def parameters(self):
        return [self.weights, self.bias]
    
    def __repr__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias}, nonlin={self.nonlin})"

class Layer(Module):
    def __init__(self, num_in, num_out, **kwargs):
        self.neurons = [Neuron(num_in, **kwargs) for _ in range(num_out)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]
    
    def __repr__(self):
        return f"Layer(neurons={self.neurons})"
    
class MLP(Module):
    def __init__(self, num_in, num_outs):
        sz = [num_in] + num_outs
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=i!=len(num_outs)-1) for i in range(len(num_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
    def __repr__(self):
        return f"MLP(layers={self.layers})"