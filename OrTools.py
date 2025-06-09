import re
import os
import math
from functools import partial
from vrp_reader import read_vrp_file
# from utils import calculate_distance_matrix, check_solution_feasibility, calculate_solution_cost, calculate_route_cost
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp




def calculate_distance_matrix_and_map(coords):
    """计算距离矩阵并返回 0-based 映射和反向映射。"""
    node_ids = sorted(coords.keys())
    num_nodes = len(node_ids)
    distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    node_map_to_zero = {node_id: i for i, node_id in enumerate(node_ids)}
    node_map_from_zero = {i: node_id for i, node_id in enumerate(node_ids)}

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            coord_i = coords[node_ids[i]]
            coord_j = coords[node_ids[j]]
            dist = math.hypot(coord_i[0] - coord_j[0], coord_i[1] - coord_j[1])
            
            int_dist = int(dist + 0.5)
            distance_matrix[i][j] = int_dist
            distance_matrix[j][i] = int_dist

    return distance_matrix, node_map_to_zero, node_map_from_zero


def create_data_model(vrp_data, distance_matrix, node_map_to_zero):
    """为 OR-Tools 创建数据模型。"""
    data = {}
    data['distance_matrix'] = distance_matrix
    # 将 1-based demands 转换为 0-based
    data['demands'] = [vrp_data['demands'][node_id] for node_id in sorted(vrp_data['demands'].keys())]
    
    data['vehicle_capacities'] = [vrp_data['capacity']] * 30 
    data['num_vehicles'] = len(data['vehicle_capacities'])
    data['depot'] = node_map_to_zero[vrp_data['depot']]
    return data


def print_solution(data, manager, routing, solution):
    """打印 OR-Tools 解决方案。"""
    print(f"目标函数值 (总距离): {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'路径 {vehicle_id}:'
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += f' {node_index} -> '
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        node_index = manager.IndexToNode(index) # 终点 (仓库)
        plan_output += f'{node_index}'
        # 仅打印非空路径
        if route_distance > 0:
            print(f"{plan_output}")
            print(f'  距离: {route_distance}m')
            print(f'  载重: {route_load}')
            total_distance += route_distance
            total_load += route_load
    print(f'\n总距离: {total_distance}m')
    print(f'总载重: {total_load}')


def solve_cvrp_with_or_tools(filepath):
    """使用 OR-Tools 读取并求解 CVRP 问题。"""
    # 1. 读取 VRP 文件
    vrp_data = read_vrp_file(filepath)
    if not vrp_data:
        return

    # 2. 计算距离矩阵和映射
    distance_matrix, node_map_to_zero, node_map_from_zero = calculate_distance_matrix_and_map(vrp_data['coords'])

    # 3. 创建 OR-Tools 数据模型
    data = create_data_model(vrp_data, distance_matrix, node_map_to_zero)

    # 4. 创建 Routing Index Manager 和 Routing Model
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # 5. 创建距离回调函数 (Transit Callback)
    def distance_callback(from_index, to_index):
        """返回两点之间的距离。"""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 6. 创建需求回调函数并添加容量维度 (Capacity Dimension)
    def demand_callback(from_index):
        """返回节点的需求。"""
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  
        data['vehicle_capacities'],  
        True,  
        'Capacity')

    # 7. 设置搜索参数
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(30) # 设置 30 秒搜索时间限制

    # 8. 求解问题
    print("正在使用 OR-Tools 求解...")
    solution = routing.SolveWithParameters(search_parameters)

    # 9. 打印结果
    if solution:
        print("求解成功!")
        print_solution(data, manager, routing, solution)
    else:
        print("未找到解决方案！")



if __name__ == "__main__":
    
    current_file = os.path.abspath(__file__) 
    project_path = os.path.abspath(os.path.join(current_file, "..", ))
    file_to_read = f'{project_path}/dataset/Vrp-Set-Golden/Golden/Golden_1.vrp'
    
    

    if os.path.exists(file_to_read):
         solve_cvrp_with_or_tools(file_to_read)
    else:
        print(f"错误: 文件 '{file_to_read}' 不存在。请检查路径。")
        print("你需要将 'path/to/your/Golden_1.vrp' 替换为实际的文件路径。")