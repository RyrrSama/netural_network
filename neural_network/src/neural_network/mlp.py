from .layer import Layer

class MLP:
  def __init__(self, no_of_input_neuron, list_of_neuron_layer):
    # This create a list of input, seq_of_neuron,..., output_neurons]
    list_neuron_sequence = [no_of_input_neuron] + list_of_neuron_layer
    # We are creating a seqence layer of neurons with the current layer number of neuron as a input count and next count as number of neuron
    # Which is given  [3,4,4,1] = > 3 input neuron, 4 sequence neuron with 3 input each , 4 sequence neuron with 4 input each, finally a single output neuron with single output neuron
    # The output will be a list of sequence of neurons.
    self.layers = [Layer(list_neuron_sequence[i], list_neuron_sequence[i + 1]) for i in range(len(list_of_neuron_layer))]

  def __call__(self, list_input):
    #  Here we are itrators into all layers and calling each layer of neurons with the output of the prev neurons and finally return the final neuron layer
    for each_layer in self.layers:
      # Here we are calling the each neuron layer with the output of previous neuron layer
      #  list_input is updating in each itrator with the output of the prev neuron layer
      list_input = each_layer(list_input)
    # returning the final output layer neurons
    return list_input

  def parameters(self):

    return [params for layer in self.layers for params in layer.parameters()]