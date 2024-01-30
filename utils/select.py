import torch
import torch.nn.functional as F
import torch.nn as nn


def get_activation(activation_name):
  if activation_name == "sigmoid":
    activation = torch.sigmoid
  elif activation_name == "relu":
    activation = F.relu
  elif activation_name == "mish":
    activation = F.mish
  else:
    raise ValueError("No activation")
  return activation

def get_criterion(criterion_name):
  if criterion_name == "CrossEntropy":
    criterion = nn.CrossEntropyLoss()
  else:
    raise ValueError("No criterion")
  return criterion
    