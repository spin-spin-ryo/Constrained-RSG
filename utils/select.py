import jax.nn as nn
from utils.jax_layers import cross_entropy_loss
from utils.calculate import constant,inverse

def get_activation(activation_name):
  if activation_name == "sigmoid":
    activation = nn.sigmoid
  elif activation_name == "relu":
    activation = nn.relu
  else:
    raise ValueError("No activation")
  return activation

def get_criterion(criterion_name):
  if criterion_name == "CrossEntropy":
    criterion = cross_entropy_loss
  else:
    raise ValueError("No criterion")
  return criterion
    
def get_step_scheduler_func(scheduler):
  if scheduler == "constant":
    return constant
  elif scheduler == "inverse":
    return inverse
  else:
    raise ValueError(f"{scheduler} is not implemented.")