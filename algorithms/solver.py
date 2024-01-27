from algorithms.descent_method import *
from algorithms.constrained_descent_method import *
from algorithms.proposed_method import *
from environments import *

def get_solver(solver_name):
    if solver_name == GRADIENT_DESCENT:
        solver = GradientDescent()
    elif solver_name == SUBSPACE_GRADIENT_DESCENT:
        solver = SubspaceGD()
    elif solver_name == ACCELERATED_GRADIENT_DESCENT:
        solver = AcceleratedGD()
    elif solver_name == NEWTON:
        solver = NewtonMethod()
    elif solver_name == LIMITED_MEMORY_NEWTON:
        solver = LimitedMemoryNewton()
    elif solver_name == PROXIMAL_GRADIENT_DESCENT:
        solver = BacktrackingProximalGD()
    elif solver_name == ACCELERATED_PROXIMAL_GRADIENT_DESCENT:
        solver = BacktrackingAcceleratedProximalGD()
    elif solver_name == GRADIENT_PROJECTION:
        solver = GradientProjectionMethod()
    elif solver_name == DYNAMIC_BARRIER:
        solver = DynamicBarrierGD()
    elif solver_name == PRIMALDUAL:
        solver = PrimalDualInteriorPointMethod()
    elif solver_name == RSG_LC:
        solver = RSGLC()
    elif solver_name == RSG_NC:
        solver = RSGNC()
    else:
        raise ValueError(f"{solver_name} is not implemented.")
    return solver
    