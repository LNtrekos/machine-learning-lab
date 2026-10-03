import math 
import numpy as np
import matplotlib.pyplot as plt

from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) 
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


class Value: 

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None 
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return f"Value(data={self.data}, op={self._op}, label={self.label})"

    def __gt__(self, other):
        other_data = other.data if isinstance(other, Value) else other
        return self.data > other_data

    def __lt__(self, other):
        other_data = other.data if isinstance(other, Value) else other
        return self.data < other_data
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')
        
        def _backward(): 
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, _children=(self,), _op=f'**{other}')
        
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), _children=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
    
    
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        
    
class Neuron:

    def __init__(self, nin):
        self.w = [Value(np.random.randn(), label=f'w{i}') for i in range(nin)]
        self.b = Value(0.0, label='b')

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]


class Layer: 
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.881373587019543, label='b')

x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

#e = (2*n).exp(); e.label = 'e'
#o = (e - 1) / (e + 1); o.label = 'o'

#o.backward()
#dot = draw_dot(o)
#dot.render('graph', format='png', view=True)
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]
learning_rate = 0.05
threshold = 0.01
max_iterations = 10_000

for k in range(max_iterations):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypred, ys))

    # Stop once the target loss is reached
    if loss.data <= threshold:
        print(f"Converged after {k} iterations")
        break

    # Reset gradients from the previous iteration
    for p in n.parameters():
        p.grad = 0.0

    # Backward pass
    loss.backward()

    # Gradient-descent update
    for p in n.parameters():
        p.data -= learning_rate * p.grad

    if k % 10 == 0:
        print(f"iteration {k}: loss = {loss.data:.6f}")

else:
    print("Maximum number of iterations reached")

ypred = [n(x) for x in xs]
loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypred, ys))

print(f"\nFinal loss: {loss.data:.6f}")

for prediction, target in zip(ypred, ys):
    print(
        f"prediction: {prediction.data:.4f}, "
        f"target: {target:.1f}"
    )


plot = draw_dot(loss)
plot.render('mlp_graph', format='png', view=True)