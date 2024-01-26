from environments import * 
from numerical_experiment import get_objects_from_config
from utils.save_func import get_path_form_params
import os

# 以下設定
objective_name = QUADRATIC
constraints_name = POLYTOPE
solver_name = GRADIENT_DESCENT



# 問題関連のパラメータ
dim = 1000
constraints_num = 100


# アルゴリズム関連のパラメータ
backward_mode = True
iteration = 100
log_interval = 10

config = {
	"objective":{
		"objective_name":objective_name
	},

	"constraints":{
    "constraints_name":constraints_name
	},
	"algorithms":{
		"solver_name":solver_name,
    "backward_mode":backward_mode
	}
}


solver,solver_params,f,function_properties,con,constraints_properties,x0 = get_objects_from_config(config)

solver_dir = get_path_form_params(solver_params)
func_dir = get_path_form_params(function_properties)
con_dir = get_path_form_params(constraints_properties)

save_path = os.path.join(RESULTPATH,objective_name,func_dir,constraints_name,con_dir,solver_name,solver_dir)
os.makedirs(save_path,exist_ok=True)

# 実験開始
solver.run(f=f,
           x0=x0,
           iteration=iteration,
           params=solver_params,
           save_path=save_path,
           log_interval=log_interval
          )