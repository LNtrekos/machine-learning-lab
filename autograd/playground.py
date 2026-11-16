from classes import MLP
from ultils import trace, draw_dot


xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(2, [4, 1])
ypred = [n(x) for x in xs]

loss = sum((y_i - y_hat)**2 for y_i, y_hat in zip(ys, ypred))
print(loss)

loss_dot = draw_dot(loss)
loss_dot.render(directory = 'graphviz_outputs', view=True)

