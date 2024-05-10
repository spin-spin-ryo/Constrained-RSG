from utils.save_func import save_config
from environments import *
from template.objective_config import get_objective_config
from template.constraints_config import get_constraints_config
from template.algorithm_config import get_algorithm_parameters

save_path = "configs/config.json"

# Please edit the template/objective_config.py, template/constraints_config.py, template/algorithm_config.py and set parameters.

objective_name = MATRIXFACTORIZATION_COMPLETION
constraints_name = NONNEGATIVE
solver_name = G_LC

iteration = 10000
log_interval = 100


config = {
  "objective": get_objective_config(objective_name),
  "constraints":get_constraints_config(constraints_name),
  "algorithms":get_algorithm_parameters(solver_name),
  "iteration":iteration,
  "log_interval":log_interval
}

save_config(config=config,save_path=save_path)
