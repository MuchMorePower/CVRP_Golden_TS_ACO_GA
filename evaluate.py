import json
import math
import os
import re
from vrp_reader import read_vrp_file, read_sol_file
from utils import *




def evaluate(solution_filepath, vrp_filepath, bks_cost=None):
    """
    评估单个 CVRP 解决方案文件，并打印详细指标。

    参数:
        solution_filepath (str): 保存的解决方案 JSON 文件路径。
        vrp_filepath (str): 原始 VRP 文件路径。
        bks_cost (float, optional): 该问题的最佳已知解成本。

    返回:
        dict: 包含评估指标的字典，如果评估失败则返回 None。
    """
    print(f"\n--- 开始评估: {os.path.basename(solution_filepath)} ---")

    # 1. 读取 VRP 文件
    vrp_raw_data = read_vrp_file(vrp_filepath)
    if not vrp_raw_data:
        print("  评估失败：无法读取 VRP 文件。")
        return None

    capacity = vrp_raw_data['capacity']
    distance_matrix, node_map = calculate_distance_matrix(vrp_raw_data['coords'])
    demands = [vrp_raw_data['demands'][node_id] for node_id in sorted(vrp_raw_data['coords'].keys())]
    depot_index = node_map[vrp_raw_data['depot']]

    # 2. 读取解决方案文件
    try:
        with open(solution_filepath, 'r') as f:
            sol_data = json.load(f)
    except Exception as e:
        print(f"  评估失败：读取解决方案文件时出错 - {e}")
        return None

    # 3. 提取并验证数据
    solution = sol_data.get('routes', [])
    saved_cost = sol_data.get('best_cost', float('inf'))
    exec_time = sol_data.get('execution_time_seconds', 0)
    
    if not solution:
        print("  评估失败：解决方案文件中没有路径数据。")
        return None

    recalculated_cost = calculate_solution_cost(solution, distance_matrix, depot_index)
    is_feasible, _ = check_solution_feasibility(solution, demands, capacity, depot_index)
    num_vehicles = len(solution)
    
    route_costs = [calculate_route_cost(r, distance_matrix, depot_index) for r in solution]
    route_loads = [sum(demands[n] for n in r) for r in solution]

    metrics = {
        "problem_name": sol_data.get('problem_name', 'N/A'),
        "solution_file": os.path.basename(solution_filepath),
        "is_feasible": is_feasible,
        "saved_cost": saved_cost,
        "recalculated_cost": recalculated_cost,
        "vehicles_used": num_vehicles,
        "execution_time": exec_time,
        "bks_cost": bks_cost,
        "gap_to_bks": ((recalculated_cost - bks_cost) / bks_cost) * 100 if bks_cost and is_feasible else None,
        "min_route_cost": min(route_costs) if route_costs else 0,
        "max_route_cost": max(route_costs) if route_costs else 0,
        "avg_route_cost": sum(route_costs) / num_vehicles if num_vehicles > 0 else 0,
        "min_route_load": min(route_loads) if route_loads else 0,
        "max_route_load": max(route_loads) if route_loads else 0,
        "avg_route_load": sum(route_loads) / num_vehicles if num_vehicles > 0 else 0,
        "avg_capacity_utilization": (sum(route_loads) / (num_vehicles * capacity)) * 100 if num_vehicles > 0 and capacity > 0 else 0,
    }

    # 4. 打印指标
    print("--- 评估指标 ---")
    print(f"  问题名称:         {metrics['problem_name']}")
    print(f"  解决方案文件:     {metrics['solution_file']}")
    print(f"  是否可行:         {'是' if metrics['is_feasible'] else '否'}")
    print(f"  总成本 (重算):    {metrics['recalculated_cost']:.2f} (保存值: {metrics['saved_cost']:.2f})")
    print(f"  使用车辆数:       {metrics['vehicles_used']}")
    print(f"  执行时间 (秒):    {metrics['execution_time']:.2f}")
    if metrics['bks_cost']:
        print(f"  BKS 成本:         {metrics['bks_cost']:.2f}")
        print(f"  与 BKS 差距:    {metrics['gap_to_bks']:.2f}%" if metrics['gap_to_bks'] is not None else "N/A (不可行或无BKS)")
    print(f"  路径成本 (Min/Max/Avg): {metrics['min_route_cost']:.2f} / {metrics['max_route_cost']:.2f} / {metrics['avg_route_cost']:.2f}")
    print(f"  路径载重 (Min/Max/Avg): {metrics['min_route_load']:.0f} / {metrics['max_route_load']:.0f} / {metrics['avg_route_load']:.0f}")
    print(f"  平均容量利用率:   {metrics['avg_capacity_utilization']:.2f}%")
    print("--- 评估完成 ---")

    return metrics


if __name__ == '__main__':

    version = '3'

    
    project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), ".."))
    vrp_file_path = f'{project_path}/dataset/Vrp-Set-Golden/Golden/Golden_{version}.vrp'
    results_dir = os.path.join(project_path, 'results')
    
    
    solution_to_evaluate = os.path.join(results_dir, f'Golden_{version}_ts_run1.json') 

    sol_file_path = f'{project_path}/dataset/Vrp-Set-Golden/Golden/Golden_{version}.sol'
    sol_data = read_sol_file(sol_file_path)
    bks_cost_golden_1 = sol_data['cost']

    
    if os.path.exists(solution_to_evaluate):
        evaluate(solution_to_evaluate, vrp_file_path, bks_cost=bks_cost_golden_1)
    else:
        print(f"错误: 找不到要评估的文件 {solution_to_evaluate}")
        print("请先运行求解器生成解决方案文件。")

    