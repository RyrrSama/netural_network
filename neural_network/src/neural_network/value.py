# Default import
import math

# Since the neural network is the big complex mathematically expression so we need a data structure class to track the mathmatically exopresssion how a vavrible is derivated


# Data struture  class
class Value:
    def __init__(self, data, children=None, opt="", label=""):
        self.data = data
        if not children:
            children = {}
        self.prev = set(children)
        self.opt = opt
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value data = {self.data}"

    def __add__(self, other):
        # Returning the same data type instead of number
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data, {self, other}, "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data, {self, other}, "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            # The derivate of exp is the fuction itself that is f(x) = x^x then f`(x) = x^x
            #  Thats is the value of exp to the function is same as slope f(2) = 2
            self.grad = out.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        #  Invoked when negtive of the number required
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        # return self.data - other
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def get_parent(self):
        return self.prev

    def get_operator(self):
        return self.opt

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only support int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        #  Tanh formula (e.pow(2x) - 1 / e.pow(2x) +1)
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(root):
            if root not in visited:
                visited.add(root)
                for child in root.prev:
                    build_topo(child)
                topo.append(root)

        build_topo(self)

        self.grad = 1.0
        for each in reversed(topo):
            each._backward()
