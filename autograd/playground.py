from autograd.neural_network import Value


a = Value(5.0, label="a", _op = '')
b = Value(21.0, label="b", _op = '')
print(a)
print(repr(a))
print(b)
print(a + b)
print(repr(a + b))
print(a * b)
print(repr(a * b))

print(a + 1)
print(b + 2)
