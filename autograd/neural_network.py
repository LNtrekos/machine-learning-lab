

class Value:

    def __init__(self, data, label='', _op=''):
        self.data = data
        self.grad = 0.0
        self.label = label 
        self._op = _op


    def __repr__(self):
        return f"Value(data={self.data}, label={self.label}, _op={self._op})"

    def __str__(self):
        return str(self.data)


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, label=f"{self.label}+{other.label}", _op = "+")

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, label=f"{self.label}*{other.label}", _op = "*")
        
        return out 
