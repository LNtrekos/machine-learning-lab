class Value:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):

        result = Value(self.data + other.data)

        return result

    def __mul__(self, other):

        result = Value(self.data * other.data)

        return result

