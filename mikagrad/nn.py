import numpy as np
from .engine import Value

class Module:
    def zero_grad(self):
        for param in self.parameters():
            param.grad.fill(0.0)

    def parameters(self):
        return []

class Layer(Module):
    def __init__(self, num_in, num_out, nonlin=True):
        self.weights = Value(np.random.uniform(-1, 1, size=(num_out, num_in)))
        self.bias = Value(np.zeros(num_out))
        self.nonlin = nonlin
    
    def __call__(self, x):
        x = x if isinstance(x, Value) else Value(np.asarray(x, dtype=float))
        out = x @ self.weights.T + self.bias
        return out.relu() if self.nonlin else out
    
    def parameters(self):
        return [self.weights, self.bias]
    
    def __repr__(self):
        return f"Layer(weights={self.weights}, bias={self.bias}, nonlin={self.nonlin})"
    
class MLP(Module):
    def __init__(self, num_in, num_outs):
        sz = [num_in] + num_outs
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=(i != len(num_outs)-1)) for i in range(len(num_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
    def __repr__(self):
        return f"MLP(layers={self.layers})"