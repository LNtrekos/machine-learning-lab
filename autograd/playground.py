from autograd.classes import Value
from autograd.ultils import trace, draw_dot

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a*b + c
print(c)
dot = draw_dot(d)

dot.render(directory = "autograd/graphviz_output", view = True)



