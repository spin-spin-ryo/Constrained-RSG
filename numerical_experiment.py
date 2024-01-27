from algorithms.solver import get_solver
from problems.generate_problem import generate_objective,generate_constraints,generate_initial_points,objective_properties_key,constraints_properties_key
from environments import NOCONSTRAINTS


def get_objects_from_config(config):
    algorithms_config = config["algorithms"]
    objectives_config = config["objectives"]
    constraints_config = config["constraints"]

    # solverを取得
    solver_name = algorithms_config["solver_name"]
    backward_mode = algorithms_config["backward"]
    solver = get_solver(solver_name=solver_name,backward_mode=backward_mode)
    solver_params = {}
    for param in solver.params_key:
      solver_params[param] = algorithms_config[param]
    
    # objectiveを取得
    objective_name = objectives_config["objective_name"]
    function_properties = {}
    for param in objective_properties_key[objective_name]:
      function_properties[param] = objectives_config[param]
    f = generate_objective(function_name=objective_name,function_properties=function_properties)

    # constraintsを取得
    constraints_name = constraints_config["constraints_name"]
    constraints_properties = {}
    if constraints_name != NOCONSTRAINTS:
      for param in constraints_properties_key[constraints_name]:
        constraints_properties[param] = constraints_config[param]
      con = generate_constraints(constraints_name=constraints_name,constraints_properties=constraints_properties)
    else:
      con = None
      
    x0 = generate_initial_points(func=f,
                                 function_name=objective_name,
                                 constraints_name=constraints_name,
                                 function_properties=function_properties)
    
    return solver,solver_params,f,function_properties,con,constraints_properties,x0


        