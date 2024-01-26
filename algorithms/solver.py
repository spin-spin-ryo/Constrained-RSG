from algorithms.descent_method import *
from algorithms.constrained_descent_method import *
from algorithms.proposed_method import *
from environments import *

def get_solver(solver_name,backward_mode):
    if solver_name == GRADIENT_DESCENT:
        solver = GradientDescent(backward_mode)
    elif solver_name == SUBSPACE_GRADIENT_DESCENT:
        solver = SubspaceGD(backward_mode)
    elif solver_name == ACCELERATED_GRADIENT_DESCENT:
        solver = AcceleratedGD(backward_mode)
    elif solver_name == NEWTON:
        solver = NewtonMethod(backward_mode)
    elif solver_name == LIMITED_MEMORY_NEWTON:
        solver = LimitedMemoryNewton(backward_mode)
    elif solver_name == PROXIMAL_GRADIENT_DESCENT:
        solver = BacktrackingProximalGD(backward_mode)
    elif solver_name == ACCELERATED_PROXIMAL_GRADIENT_DESCENT:
        solver = BacktrackingAcceleratedProximalGD(backward_mode)
    elif solver_name == GRADIENT_PROJECTION:
        solver = GradientProjectionMethod(backward_mode)
    elif solver_name == DYNAMIC_BARRIER:
        solver = DynamicBarrierGD(backward_mode)
    elif solver_name == PRIMALDUAL:
        solver = PrimalDualInteriorPointMethod(backward_mode)
    elif solver_name == RSG_LC:
        solver = RSGLC(backward_mode)
    elif solver_name == RSG_NC:
        solver = RSGNC(backward_mode)
    else:
        raise ValueError(f"{solver_name} is not implemetend.")
    return solver

def get_params_from_config(config):
    