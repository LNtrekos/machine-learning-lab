from autograd.classes import Value


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
c = a+b
d = a*b + c
print(a)
print(b)
print(c)
print(d)

