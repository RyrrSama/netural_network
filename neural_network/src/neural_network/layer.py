from .neuron import Neuron

class Layer:

  def __init__(self, no_of_input_per_neuron, no_of_neuron):
    # Here the the Layer call is used to create the sequence of neuron  with 'no_of_input_per_neuron'  and how many neroun we want in the sequence layer
    # The self.neurons contain the sequence of neuron with no_of_input_per_neuron
    self.neurons = [Neuron(no_of_input_per_neuron) for _ in range(no_of_neuron)]

  def __call__(self, input_list):
    # This call function will pass the list of given input to the neuron for each neuron in the sequence
    # Same input is passed to all neurons in the n sequence
    output_list = [each_neuron(input_list) for each_neuron in self.neurons]
    #  Here we are checking if the sequence contain a single neuron then we gonna return the single neuron and consisder as a input if not return the array of neurons
    return output_list[0] if len(output_list) == 1 else output_list

  def parameters(self):
    return [params for each_neuron in self.neurons for params in each_neuron.parameters()]