from environments import * 
from numerical_experiment import get_objects_from_config
from utils.save_func import get_path_form_params,save_result_json,load_config
import os
import torch
import sys


def run_numerical_experiment(config):
  iteration = config["iteration"]
  log_interval = config["log_interval"]
  algorithms_config = config["algorithms"]
  objectives_config = config["objectives"]
  constraints_config = config["constraints"]
  solver_name = algorithms_config["solver_name"]
  objective_name = objectives_config["objective_name"]
  constraints_name = constraints_config["constraints_name"]

  solver,solver_params,f,function_properties,con,constraints_properties,x0 = get_objects_from_config(config)
  solver_dir = get_path_form_params(solver_params)
  func_dir = get_path_form_params(function_properties)
  if constraints_name != NOCONSTRAINTS:
    con_dir = get_path_form_params(constraints_properties)
    save_path = os.path.join(RESULTPATH,
                            objective_name,func_dir,
                            constraints_name,con_dir,
                            solver_name,solver_dir)

  else:
    save_path = os.path.join(RESULTPATH,
                            objective_name,func_dir,
                            constraints_name,
                            solver_name,solver_dir)


  os.makedirs(save_path,exist_ok=True)

  # 実験開始
  if constraints_name != NOCONSTRAINTS:
    solver.run(f=f,
              con=con,
              x0=x0,
              iteration=iteration,
              params=solver_params,
              save_path=save_path,
              log_interval=log_interval
              )
  else:
    solver.run(f=f,
              x0=x0,
              iteration=iteration,
              params=solver_params,
              save_path=save_path,
              log_interval=log_interval
              )

  min_f_value = torch.min(solver.save_values["func_values"]).item()
  execution_time = solver.save_values["time"][-1]
  values_dict = {
    "min_value":min_f_value,
    "time":execution_time
  }

  save_result_json(save_path=os.path.join(save_path,"result.json"),
                  values_dict=values_dict,
                  iteration=iteration)

if __name__ == "__main__":
  args = sys.argv
  config_path = args[1]
  config = load_config(config_path)
  run_numerical_experiment(config)