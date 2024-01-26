from problems.objectives import *
from problems.constraints import *

# 目的関数一覧
QUADRATIC = "Quadratic"
SPARSEQUADRATIC = "SparseQuadratic"
MATRIXFACTORIZATION = "MatrixFactorization"
MATRIXFACTORIZATION_COMPLETION = "MatrixFactorization_Completions"
LEASTSQUARE = "LeastSquare"
MLPNET = "MLPNET"
CNN = "CNN"

# 制約一覧
POLYTOPE = "Polytope"
NONNEGATIVE = "NonNegative"
FUSEDLASSO = "FusedLasso"
BALL = "Ball"
HUBER = "Huber"


def generate_objective(function_name,function_properties):
    if function_name == QUADRATIC:
        f = generate_quadratic(function_properties[function_name])
    elif function_name == SPARSEQUADRATIC:
        f = generate_sparse_quadratic(function_properties[function_name])
    elif function_name == MATRIXFACTORIZATION:
        f = generate_matrix_factorization(function_properties[function_name])
    elif function_name == MATRIXFACTORIZATION_COMPLETION:
        f = generate_matrix_factorization_completion(function_properties[function_name])
    elif function_name == LEASTSQUARE:
        f = generate_least_square(function_properties[function_name])
    elif function_name == MLPNET:
        f = generate_mlpnet(function_properties[function_name])
    elif function_name == CNN:
        f = generate_cnn(function_properties[function_name])
    else:
        raise ValueError(f"{function_name} is not implemented.")
    return f

def generate_constraints(constraints_name, constraints_properties):
    if constraints_name == POLYTOPE:
        constraints = generate_polytope(constraints_properties[constraints_name])
    elif constraints_name == NONNEGATIVE:
        constraints = generate_nonnegative(constraints_properties[constraints_name])
    elif constraints_name == QUADRATIC:
        constraints = generate_quadratic_constraints(constraints_properties[constraints_name])
    elif constraints_name == FUSEDLASSO:
        constraints = generate_fusedlasso(constraints_properties[constraints_name])
    elif constraints_name == BALL:
        constraints = generate_ball(constraints_properties[constraints_name])
    elif constraints_name == HUBER:
        constraints = generate_huber(constraints_properties[constraints_name])
    else:
        raise ValueError(f"{constraints_name} is not implemented.")
    return constraints    

def generate_initial_points(func,constraints_name):
    return

def generate_quadratic(properties):
    dim = int(properties["dim"])
    convex = properties["convex"]
    rank = int(properties["rank"])
    return