import os 
import json
from vrp_reader import read_vrp_file
from TabuSearch import *
from Genetic import memetic_solve_cvrp
from AntColony import aco_solve_cvrp

import os 
current_file = os.path.abspath(__file__)
project_path = os.path.abspath(os.path.join(current_file, "..", ))

"""
V1: 无初始化
V2: 添加了初始化
V3: 添加了初始化，但没有局部搜索LS
"""

def main():
    dataset_path = os.path.join(project_path, 'dataset', 'Vrp-Set-Golden', 'Golden')
    for i in range(1, 21):
        cvrp_file = os.path.join(dataset_path, f'Golden_{i}.vrp')
        sol_file = os.path.join(dataset_path, f'Golden_{i}.sol')
        # TS算法
        solve_cvrp(dataset_filename=cvrp_file)
        # 蚁群算法
        aco_solve_cvrp(vrp_filepath=cvrp_file)
        # 遗传算法
        memetic_solve_cvrp(vrp_file_path=cvrp_file, sol_file_path=sol_file)
        
        # break

if __name__ == '__main__':
    main()
    print("CVRP 求解完成。请检查 'results' 目录中的结果文件。")