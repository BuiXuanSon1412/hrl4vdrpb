import json
import os
from moo_algorithm.metric import cal_hv
import numpy as np


def cal_HV_algorithm(data_name, number_customers, algorithm):
    data_path = os.path.join(
        "result", algorithm, f"N{number_customers}", f"{data_name}.json"
    )

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    objectives_list = data["history"]["99"]
    hv_value = cal_hv(np.array(objectives_list), ref_point=np.array([100, 500000]))
    return hv_value


algorithms = ["NSGA_II", "NSGA_III", "MOEAD", "PFG_MOEA", "AGEA", "IAGEA"]


if __name__ == "__main__":
    for algor in algorithms:
        hv = cal_HV_algorithm("S043_N50_RC_R50", 50, algor)
        print(f"{algor}: HV = {hv}")
