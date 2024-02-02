import torch
import numpy as np
from environments import DATAPATH
import os
import pickle
import jax.numpy as jnp


data_path = os.path.join(DATAPATH,"mnist","cnn","labels.pth")
U = torch.load(data_path,map_location="cpu")
U = jnp.array(U.detach().numpy())

data_path = ".".join(data_path.split(".")[:-1]) + ".npy"
jnp.save(data_path,U)