import os
from environments import DISTINCT_PARAM_VALUE,DISTINCT_PARAMS


def get_path_form_params(params:dict):
    save_path = ""
    for k,v in params.items():
        save_path += str(k) + DISTINCT_PARAM_VALUE + str(v) + DISTINCT_PARAMS

    save_path = save_path[:-1]
    return save_path

def get_params_from_path(save_path):
    return