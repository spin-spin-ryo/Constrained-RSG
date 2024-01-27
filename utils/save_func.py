import os
from environments import DISTINCT_PARAM_VALUE,DISTINCT_PARAMS
import json
import numpy as np

def get_path_form_params(params:dict):
    save_path = ""
    for k,v in params.items():
        save_path += str(k) + DISTINCT_PARAM_VALUE + str(v) + DISTINCT_PARAMS

    save_path = save_path[:-1]
    return save_path

def get_params_from_path(save_path):
  return

def save_config(config,save_path):
	with open(save_path,"w") as f:
		json.dump(config,f,indent=2)
	return

def load_config(config_path):
	with open(config_path,"r") as f:
		config = json.load(f)
	return config

def save_result_json(save_path,values_dict:dict,iteration):
	if os.path.exists(save_path):
		with open(save_path,"r") as f:
			result_json = json.load(f)
		for index in range(len(result_json["result"])):
			if result_json["result"][index]["iteration"] == iteration:
				result_json["result"][index]["save_values"].append(values_dict)
				with open(save_path,"w") as f2:
					json.dump(result_json,f2,indent=2)
				return

	else:
		result_json ={"result":[]} 
		each_json = {
									"iteration":iteration,
									"save_values":[values_dict],
								}
		result_json["result"].append(each_json)
		with open(save_path,"w") as f2:
			json.dump(result_json,f2,indent=2)
		