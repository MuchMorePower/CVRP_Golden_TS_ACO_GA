import os
import math
import json

def calculate_distance_matrix(coords):
    """根据坐标计算欧几里得距离矩阵，并返回 0-based 映射。"""
    node_ids = sorted(coords.keys())
    num_nodes = len(node_ids)
    distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    node_map = {node_id: i for i, node_id in enumerate(node_ids)}

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            coord_i = coords[node_ids[i]]
            coord_j = coords[node_ids[j]]
            dist = math.hypot(coord_i[0] - coord_j[0], coord_i[1] - coord_j[1])
            distance_matrix[i][j] = dist 

    return distance_matrix, node_map

def calculate_route_cost(route, distance_matrix, depot_index):
    """计算单条路径的总距离。"""
    cost = 0.0
    current_node = depot_index
    for node in route:
        cost += distance_matrix[current_node][node]
        current_node = node
    cost += distance_matrix[current_node][depot_index]
    return cost

def calculate_solution_cost(solution, distance_matrix, depot_index):
    """计算整个解决方案（多条路径）的总距离。"""
    total_cost = 0.0
    for route in solution:
        if route: # 确保路径不为空
            total_cost += calculate_route_cost(route, distance_matrix, depot_index)
    return total_cost

def check_solution_feasibility(solution, demands, capacity, depot_index):
    """
    检查解决方案是否满足容量约束。

    返回:
        tuple: (bool, float) - (是否可行, 如果不可行则为超载量，否则为 0)
    """
    for route in solution:
        
        load = sum(demands[node] for node in route if node != depot_index)
        if load > capacity:
            return False, load # <--- 返回 False 和 超载量
    return True, 0 # <--- 返回 True 和 0

def calculate_insertion_cost(route, new_customer, pos, distance_matrix, depot_index):
    """计算在指定位置插入客户的成本增量。"""
    node_before = route[pos-1] if pos > 0 else depot_index
    node_after = route[pos] if pos < len(route) else depot_index
    cost_increase = (distance_matrix[node_before][new_customer] + 
                     distance_matrix[new_customer][node_after] - 
                     distance_matrix[node_before][node_after])
    return cost_increase

def save_solution(solution, cost, demands, distance_matrix, depot_index,
                  vrp_data, execution_time, filepath="solution.json"):
    """
    将解决方案保存为 JSON 文件。

    参数:
        solution (list): 解决方案的路径列表。
        cost (float): 解决方案的总成本。
        demands (list): 需求列表 (0-based)。
        distance_matrix (list[list]): 距离矩阵。
        depot_index (int): 仓库索引 (0-based)。
        vrp_data (dict): 原始 VRP 数据 (用于元数据)。
        execution_time (float): 算法执行时间 (秒)。
        filepath (str): 保存文件的路径。
    """
    num_vehicles = len(solution)
    total_load = sum(demands[node] for route in solution for node in route)

    solution_data = {
        'problem_name': vrp_data.get('name', 'N/A'),
        'dimension': vrp_data.get('dimension', 'N/A'),
        'capacity': vrp_data.get('capacity', 'N/A'),
        'best_cost': cost,
        'vehicles_used': num_vehicles,
        'total_load': total_load,
        'execution_time_seconds': execution_time,
        'depot_index': depot_index,
        'routes': solution # 0-based 索引
    }

    try:
        with open(filepath, 'w') as f:
            json.dump(solution_data, f, indent=4)
        print(f"\n解决方案已成功保存到: {filepath}")
    except Exception as e:
        print(f"\n错误: 保存解决方案失败 - {e}")

        