from environments import *

def get_algorithm_parameters(solver_name,allparams_config=None):
  config = {}
  config["solver_name"] = solver_name
  if allparams_config is not None:
    for key in algorithm_parameters_key[solver_name]:
      config[key] = allparams_config[key]
    return config
  if solver_name == GRADIENT_DESCENT:
    config["lr"] = 1
    config["eps"] = 1e-4
    config["backward"] = True
    config["linesearch"] = True
  elif solver_name == SUBSPACE_GRADIENT_DESCENT:
    config["lr"] = 1
    config["reduced_dim"] = 100
    config["dim"] = 1000
    config["mode"] = "random"
    config["eps"] = 1e-4
    config["backward"] = True
    config["linesearch"] = True
  elif solver_name == ACCELERATED_GRADIENT_DESCENT:
    config["lr"] = 1
    config["eps"] = 1e-4
    config["backward"] = True
    config["restart"] = True
  elif solver_name == BFGS_QUASI_NEWTON:
    config["alpha"] = 0.3
    config["beta"] = 0.8
    config["backward"] = True
    config["eps"] = 1e-4
  elif solver_name == LIMITED_MEMORY_BFGS:
    config["alpha"] = 0.3
    config["beta"] = 0.8
    config["backward"] = True
    config["eps"] = 1e-4
    config["memory_size"] = 10
  elif solver_name == PROXIMAL_GRADIENT_DESCENT:
    config["eps"] = 1e-4
    config["beta"] = 0.8
    config["alpha"] = 100
    config["backward"] = DIRECTIONALDERIVATIVE
  elif solver_name == ACCELERATED_PROXIMAL_GRADIENT_DESCENT:
    config["restart"] = True
    config["eps"] = 1e-4
    config["beta"] = 0.8
    config["alpha"] = 1
    config["backward"] = True
  elif solver_name == NEWTON:
    config["alpha"] = 0.3
    config["beta"] = 0.8
    config["backward"] = True
    config["eps"] = 1e-4
  elif solver_name == SUBSPACE_NEWTON:
    config["alpha"] = 0.3
    config["beta"] = 0.8
    config["backward"] = True
    config["eps"] = 1e-4
    config["dim"] = 1000
    config["reduced_dim"] = 10
    config["mode"] = "random"
  elif solver_name == LIMITED_MEMORY_NEWTON:
    config["reduced_dim"] = 100
    config["threshold_eigenvalue"] = 1e-2
    config["alpha"] = 0.3
    config["beta"] = 0.8
    config["backward"] = True
    config["eps"] = 1e-4
    config["mode"] = LEESELECTION
  elif solver_name == RANDOM_BFGS:
    config["backward"] = True
    config["eps"] = 1e-4
    config["dim"] = 1000
    config["reduced_dim"] = 10
  elif solver_name == SUBSPACE_REGULARIZED_NEWTON:
    config["backward"] = True
    config["eps"] = 1e-4
    config["dim"] = 1000
    config["reduced_dim"] = 10
    config["alpha"] = 0.3
    config["beta"] = 0.8
    config["gamma"] = 1
    config["c1"] = 1
    config["c2"] = 2
  elif solver_name == GRADIENT_PROJECTION:
    config["eps"] = 1e-4
    config["delta"] = 1e-4
    config["lr"] = 1
    config["alpha"] = 0.3
    config["beta"] = 0.8
    config["backward"] = True
  elif solver_name == DYNAMIC_BARRIER:
    config["lr"] = 0.1
    config["alpha"] = 1
    config["beta"] = 1
    config["backward"] = DIRECTIONALDERIVATIVE
    config["inner_iteration"] = 10000
    config["sub_problem_eps"] = 1e-4
    config["barrier_func_type"] = "values"
  elif solver_name == PRIMALDUAL:
    config["mu"] = 1.5
    config["eps"] = 1e-4
    config["eps_feas"] = 1e-4
    config["beta"] = 0.8
    config["alpha"] = 0.3
    config["backward"] = True
  elif solver_name == RSG_LC:
    config["eps0"] = 1e-4
    config["delta1"] = 1e-5
    config["eps2"] = 1e-6
    config["dim"] = 13125
    config["reduced_dim"] = 600
    config["alpha"] = 100
    config["beta"] = 0.8
    config["mode"] = RANDOM
    config["backward"] = True
  elif solver_name == G_LC:
    config["eps0"] = 1e-4
    config["delta1"] = 1e-5
    config["eps2"] = 1e-6
    config["dim"] = 13125
    config["alpha"] = 0.01
    config["beta"] = 0.8
    config["backward"] = True
  elif solver_name == RSG_NC:
    config["eps0"] = 0.1
    config["delta1"] = 1e-8
    config["eps2"] = 1e-6
    config["dim"] = 33738
    config["reduced_dim"] = 10
    config["alpha"] =1000000.0
    config["beta"] = 0.8
    config["mode"] = RANDOM
    config["r"] = 0.5
    config["backward"] = DIRECTIONALDERIVATIVE
  elif solver_name == G_NC:
    config["eps0"] = 0.1
    config["delta1"] = 1e-8
    config["eps2"] = 1e-6
    config["dim"] = 33738
    config["alpha"] = 10
    config["beta"] = 0.8
    config["r"] = 0.5
    config["backward"] = DIRECTIONALDERIVATIVE
  return config