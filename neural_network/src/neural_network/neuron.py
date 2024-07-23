from .value import Value
import random

#  Create Neuron class
class Neuron:

  def __init__(self, no_of_input):
    # Here we are creating a neuron which have 'no_of_input' input and we are creating a random weights range from (-1, 1) for each input and multipling it with the respected weights
    #  Finally we are creating  the bias for our neuron from range (-1, 1)
    self.weights = [Value(random.uniform(-1,1)) for _ in range(no_of_input)]
    self.bias = Value(random.uniform(-1, 1))

  def __call__(self, input_list):
    # w* x + b
    # Once the neuron is called with a list_inputs we are zipping the neuron with represented weights and multipling it. And finally we adding the bias with the finally output.
    activation = sum( ( weight_i * input_i for weight_i, input_i in zip(self.weights, input_list) ) , self.bias)
    #  Finally we creating the tanh of the neuron output. this will return a neuron of tanh as a activation layer
    return activation.tanh()

  def parameters(self):
    return self.weights + [self.bias]