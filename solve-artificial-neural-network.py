
# @author Nihal Afsal
# Code written in python to solve artificial neural network examples

import math
import numpy as np

# A two-input neuron with a hard limit transfer function that has an input vector of [-5 6], a weight of [3 2], and a bias of 1.2.
def limit_transfer_function(ValInputs, ValWeights, ValBias):
	sum = np.dot(ValInputs, ValWeights) + ValBias
	output = np.where(sum >= 0, 1, 0)
	print(output)


# A single-input neuron with a log-sigmoid transfer function that has an input of 2.0, a weight of 2.3, and a bias of -3.0.
def log_sigmoid_function(valInput, valweight, valBias):
	return 1 / (1 + math.exp(-valweight * valInput + valBias))

# A two-input neuron with a log-sigmoid transfer function that has an input vector of [-5 6], a weight of [3 2], and a bias of 1.2.
def log_sigmoid_function_two(ValInput, ValWeights, ValBias):
	val = np.dot(ValInput, ValWeights) + ValBias
	output = 1 / (1 + np.exp(-val))
	print(output)

# A single-input neuron with a hard limit transfer function that has an input of 2.0, a weight of 2.3, and a bias of -3.0.
def hard_limit_function(valInput, valweight, valBias):
	if valweight * valInput + valBias >= 0:
		return 1
	else:
		return 0

# Limit transfer function
ValInputs = np.array([-5, 6])
ValWeights = np.array([3, 2])
ValBias = 1.2
limit_transfer_function(ValInputs, ValWeights, ValBias)

# Log sigmoid function
valInput = 2.0
valweight = 2.3
valBias = -3.0
output = log_sigmoid_function(valInput, valweight, valBias)
print(output)

# Log sigmoid function two
ValInput = np.array([-5, 6])
ValWeights = np.array([3, 2])
ValBias = 1.2
log_sigmoid_function_two(ValInput, ValWeights, ValBias)

# Hard limit function
valInput = 2.0
valweight = 2.3
valBias = -3.0
output = hard_limit_function(valInput, valweight, valBias)
print(output)
