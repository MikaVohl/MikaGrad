import numpy as np
from numbers import Real

def match_shape(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    # right-align shapes
    gs = grad.shape
    ts = (1,)*(grad.ndim - len(target_shape)) + target_shape

    # build list of axes that must be summed away
    axes = [i for i,(g,t) in enumerate(zip(gs, ts)) if g!=t]
    if axes:
        grad = grad.sum(axis=tuple(axes), keepdims=True)

    return grad.reshape(target_shape)


class Value():
    def __init__(self, data, operation="", children=()):
        if isinstance(data, (list, tuple, Real)):
            data = np.array(data, dtype=float)
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        self.data = data
        self.operation = operation
        self.children = children
        self.grad = np.zeros_like(self.data, dtype=float)
        self._backward = lambda: None

    def __repr__(self):
        return str(self.data)
    
    def dot(self, other):
        other = other if isinstance(other, Value) else Value(other)
        assert self.data.ndim == 1 and other.data.ndim == 1, "Dot product is only defined for vectors"
        result = Value(np.dot(self.data, other.data), "•", (self, other))
        def _backward():
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data
        result._backward = _backward
        return result

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, "+", (self, other))
        def _backward():
            self.grad += match_shape(result.grad, self.data.shape)
            other.grad += match_shape(result.grad, other.data.shape)
        result._backward = _backward
        return result

    def __mul__(self, other): # multiplication is element-wise
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, "*", (self, other))

        def _backward():
            self.grad += match_shape(result.grad * other.data, self.data.shape)
            other.grad += match_shape(result.grad * self.data, other.data.shape)
        result._backward = _backward
        return result
    
    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(np.matmul(self.data, other.data), "@", (self, other))
        
        def _backward():
            self_grad = np.matmul(result.grad, other.data.swapaxes(-1, -2))
            other_grad = np.matmul(self.data.swapaxes(-1, -2), result.grad)
            self.grad  += match_shape(self_grad, self.data.shape)
            other.grad += match_shape(other_grad, other.data.shape)
        result._backward = _backward
        return result
    
    def __pow__(self, other):
        assert isinstance(other, (float, int)), "power must be float or integer" # must force "other" to be primitive float or int
        result = Value(self.data ** other, "^", (self,))
        def _backward():
            self.grad += match_shape(other * self.data ** (other - 1) * result.grad, self.data.shape)
        result._backward = _backward
        return result
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + -other
    
    def __truediv__(self, other):
        return self * other ** -1
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return -self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return Value(other) / self
    
    def sum(self, axis=None, keepdims=False):
        result = Value(np.sum(self.data, axis=axis, keepdims=keepdims), "Σ", (self,))
        def _backward():
            factor = result.grad
            if axis is not None and not keepdims:
                factor = np.expand_dims(factor, axis)
            self.grad += match_shape(factor * np.ones_like(self.data), self.data.shape)
        result._backward = _backward
        return result
    
    def backward(self):
        # A topological sort is a graph traversal in which each node v is visited only after all its dependencies are visited.
        # We need this, since we must call backward() on a child strictly after it has been called on its parent
        nodes = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                nodes.append(v)
        build_topo(self)
        self.grad = np.ones(self.data.shape)
        for node in reversed(nodes):
            node._backward()

    def zero_grad(self):
        visited = set()
        def _zero(v):
            if v not in visited:
                visited.add(v)
                v.grad = np.zeros_like(v.data, dtype=float)
                for child in v.children:
                    assert isinstance(child, Value), "Child must be an instance of Value"
                    _zero(child)
        _zero(self)

    def relu(self):
        result = Value(np.maximum(0, self.data), "ReLU", (self,))
        def _backward():
            self.grad += match_shape((result.data > 0) * result.grad, self.data.shape)
        result._backward = _backward
        return result