from algorithms.descent_method import *
from algorithms.constrained_descent_method import *
from algorithms.proposed_method import *
from environments import *

def get_solver(solver_name,device,dtype):
    if solver_name == GRADIENT_DESCENT:
        solver = GradientDescent(device=device,dtype=dtype)
    elif solver_name == SUBSPACE_GRADIENT_DESCENT:
        solver = SubspaceGD(device=device,dtype=dtype)
    elif solver_name == ACCELERATED_GRADIENT_DESCENT:
        solver = AcceleratedGD(device=device,dtype=dtype)
    elif solver_name == NEWTON:
        solver = NewtonMethod(device=device,dtype=dtype)
    elif solver_name == LIMITED_MEMORY_NEWTON:
        solver = LimitedMemoryNewton(device=device,dtype=dtype)
    elif solver_name == PROXIMAL_GRADIENT_DESCENT:
        solver = BacktrackingProximalGD(device=device,dtype=dtype)
    elif solver_name == ACCELERATED_PROXIMAL_GRADIENT_DESCENT:
        solver = BacktrackingAcceleratedProximalGD(device=device,dtype=dtype)
    elif solver_name == GRADIENT_PROJECTION:
        solver = GradientProjectionMethod(device=device,dtype=dtype)
    elif solver_name == DYNAMIC_BARRIER:
        solver = DynamicBarrierGD(device=device,dtype=dtype)
    elif solver_name == PRIMALDUAL:
        solver = PrimalDualInteriorPointMethod(device=device,dtype=dtype)
    elif solver_name == RSG_LC:
        solver = RSGLC(device=device,dtype=dtype)
    elif solver_name == RSG_NC:
        solver = RSGNC(device=device,dtype=dtype)
    else:
        raise ValueError(f"{solver_name} is not implemented.")
    return solver
    