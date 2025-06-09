import os
import re
import math
import random
import copy
import json
import time
# import numpy as np
from tqdm import tqdm
from vrp_reader import read_vrp_file
# from utils import *
from utils import calculate_distance_matrix, calculate_solution_cost, check_solution_feasibility, save_solution
current_file = os.path.abspath(__file__)
project_path = os.path.abspath(os.path.join(current_file, "..", ))



def generate_initial_solution(demands, capacity, distance_matrix, depot_index):
    """
    使用贪心算法生成一个初始可行解。
    从仓库出发，为每辆车选择最近的未访问客户，直到容量不允许。
    """
    solution = []
    num_nodes = len(demands)
    unvisited = set(range(num_nodes))
    unvisited.remove(depot_index)

    while unvisited:
        current_route = []
        current_load = 0
        current_node = depot_index

        while True:
            best_candidate = None
            min_dist = float('inf')

            # 寻找最近的、可行的、未访问的客户
            for node in unvisited:
                if current_load + demands[node] <= capacity:
                    dist = distance_matrix[current_node][node]
                    if dist < min_dist:
                        min_dist = dist
                        best_candidate = node

            if best_candidate:
                current_route.append(best_candidate)
                current_load += demands[best_candidate]
                current_node = best_candidate
                unvisited.remove(best_candidate)
            else:
                # 没有可行的客户了，结束当前路径
                break

        if current_route:
            solution.append(current_route)

    return solution


def get_relocate_neighbors(solution, demands, capacity, distance_matrix, depot_index):
    """
    生成通过 'Relocate' 操作获得的邻居解。
    尝试将每个客户移动到其他路径的不同位置。
    """
    neighbors = []
    num_routes = len(solution)

    for r1_idx in range(num_routes):
        for i in range(len(solution[r1_idx])):
            customer_to_move = solution[r1_idx][i]

           
            for r2_idx in range(num_routes + 1): 
                
                target_route_len = len(solution[r2_idx]) if r2_idx < num_routes else 0
                for j in range(target_route_len + 1):
                    new_solution = copy.deepcopy(solution)

                    
                    moved_customer = new_solution[r1_idx].pop(i)

                    
                    if r2_idx < num_routes:
                         
                        current_load_r2 = sum(demands[node] for node in new_solution[r2_idx])
                        if current_load_r2 + demands[moved_customer] <= capacity:
                            new_solution[r2_idx].insert(j, moved_customer)
                        else:
                            continue 
                    else: 
                        if demands[moved_customer] <= capacity:
                            new_solution.append([moved_customer])
                        else:
                            continue 

                    
                    new_solution = [route for route in new_solution if route]

                    
                    is_valid_move = True
                    if r1_idx == r2_idx and i == j:
                        is_valid_move = False 

                    if is_valid_move:
                         cost = calculate_solution_cost(new_solution, distance_matrix, depot_index)
                         
                         move_info = (moved_customer, r1_idx, r2_idx)
                         neighbors.append((new_solution, cost, move_info))

    return neighbors


def tabu_search_solver(initial_solution, demands, capacity, distance_matrix, depot_index,
                       max_iterations=1000, max_evaluations=10000, tabu_tenure=10):
    """
    实现禁忌搜索算法 (带随机化领带打破)。

    参数:
        initial_solution: 初始解 (路径列表)。
        ... (其他 VRP 数据) ...
        max_iterations: 最大迭代次数。
        max_evaluations: 最大邻居评估次数。
        tabu_tenure: 禁忌期限。

    返回:
        (best_solution, best_cost)
    """
    current_solution = copy.deepcopy(initial_solution)
    current_cost = calculate_solution_cost(current_solution, distance_matrix, depot_index)
    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost

    tabu_list = {}  
    eval_count = 0
    
    print(f"初始解成本: {best_cost:.2f}, 路径数: {len(best_solution)}")

    
    for iteration in tqdm(range(max_iterations), desc="禁忌搜索进度"):
        if eval_count >= max_evaluations:
            print(f"\n达到最大评估次数 {max_evaluations}。")
            break

        neighbors = get_relocate_neighbors(current_solution, demands, capacity, distance_matrix, depot_index)
        eval_count += len(neighbors)

        # 如果没有邻居，则可能陷入困境，可以考虑停止或执行多样化
        if not neighbors:
             print(f"\n迭代 {iteration+1}: 未找到任何邻居。")
             break 

        best_neighbor_cost = float('inf')
        candidate_neighbors = [] 

        
        for neighbor, cost, move in neighbors:
            is_tabu = move in tabu_list and tabu_list[move] > iteration
            aspiration_met = cost < best_cost

            if not is_tabu or aspiration_met:
                
                if cost < best_neighbor_cost:
                    best_neighbor_cost = cost
                    candidate_neighbors = [(neighbor, move)] # 发现更好的，重置列表
                elif cost == best_neighbor_cost:
                    candidate_neighbors.append((neighbor, move)) # 成本相同，添加到列表

        
        if candidate_neighbors:
           
            best_neighbor, best_neighbor_move = random.choice(candidate_neighbors)

            current_solution = best_neighbor
            current_cost = best_neighbor_cost

            # 更新禁忌列表
            if best_neighbor_move:
                
                tabu_list[best_neighbor_move] = iteration + tabu_tenure

            # 更新全局最优解
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
                
                tqdm.write(f"\n迭代 {iteration+1}: 找到新最优解! 成本: {best_cost:.2f}, 评估次数: {eval_count}")

            # 清理过期的禁忌项
            keys_to_remove = [k for k, v in tabu_list.items() if v <= iteration]
            for k in keys_to_remove:
                del tabu_list[k]

        else:
            
            tqdm.write(f"\n迭代 {iteration+1}: 未找到合适的非禁忌邻居。")
            


    print(f"\n禁忌搜索完成。最优成本: {best_cost:.2f}, 路径数: {len(best_solution)}")
    return best_solution, best_cost



def solve_cvrp(dataset_filename):

    # 设置 VRP 文件路径 
    vrp_file_path = dataset_filename

    NUM_RUNS = 1 # 运行次数
    # 禁忌搜索参数
    MAX_ITERATIONS = 500       # 最大迭代次数
    MAX_EVALUATIONS = 600000     # 最大评估次数 
    TABU_TENURE = 20            # 禁忌期限 (需要根据问题规模调整)

    # *** 执行流程 ***
    print(f"--- 开始处理 CVRP 问题 (自定义禁忌搜索): {vrp_file_path} ---")
    print(f"--- 将执行 {NUM_RUNS} 次 ---")

    # 读取 VRP 文件
    print("\n1. 正在读取 VRP 文件...")
    vrp_raw_data = read_vrp_file(vrp_file_path)

    if vrp_raw_data:
        print(f"   文件读取成功: {vrp_raw_data.get('name', 'N/A')}, 维度: {vrp_raw_data.get('dimension', 'N/A')}")
        capacity = vrp_raw_data['capacity']

        # 计算距离矩阵和准备数据
        print("\n2. 正在准备数据...")
        distance_matrix, node_map = calculate_distance_matrix(vrp_raw_data['coords'])
        # 按照 0-based 索引重新排列需求
        demands = [vrp_raw_data['demands'][node_id] for node_id in sorted(vrp_raw_data['coords'].keys())]
        depot_index = node_map[vrp_raw_data['depot']]
        print(f"   数据准备完成. 节点数: {len(demands)}, 仓库索引: {depot_index}")

        solution_filepaths_generated = [] # 存储本次运行生成的文件路径

        
        for run_id in range(1, NUM_RUNS + 1):
            print(f"\n--- 第 {run_id}/{NUM_RUNS} 次运行 ---")

            # 生成初始解
            print("   正在生成初始解...")
            initial_solution = generate_initial_solution(demands, capacity, distance_matrix, depot_index)
            is_feasible, _ = check_solution_feasibility(initial_solution, demands, capacity, depot_index)

            if not is_feasible:
                print("   错误：生成的初始解不可行！跳过此次运行。")
                continue 
            print(f"   初始解生成。路径数: {len(initial_solution)}")

            # 运行禁忌搜索
            print("   正在运行禁忌搜索...")
            start_ts_time = time.time()
            best_sol, best_c = tabu_search_solver(
                initial_solution, demands, capacity, distance_matrix, depot_index,
                max_iterations=MAX_ITERATIONS,
                max_evaluations=MAX_EVALUATIONS,
                tabu_tenure=TABU_TENURE
            )
            end_ts_time = time.time()
            execution_time = end_ts_time - start_ts_time
            print(f"   运行 {run_id} 完成. 成本: {best_c:.2f}, 时间: {execution_time:.2f}s")

            # 保存结果
            results_dir = os.path.join(project_path, 'results')
            os.makedirs(results_dir, exist_ok=True)
            problem_name = vrp_raw_data.get('name', 'unknown_solution')
            solution_filename = f"{problem_name}_ts_run{run_id}.json"
            solution_filepath = os.path.join(results_dir, solution_filename)

            save_solution(best_sol, best_c, demands, distance_matrix, depot_index,
                          vrp_raw_data, execution_time, solution_filepath)
            solution_filepaths_generated.append(solution_filepath) 

        print(f"\n--- {len(solution_filepaths_generated)} 次运行处理结束，结果已保存 ---")
        print("--- 要进行统计评估，请运行评估函数 ---")

    else:
        print("   错误: 读取 VRP 文件失败。请检查文件路径和格式。")

    print("\n--- CVRP 处理结束 ---")


if __name__ == "__main__":
    

    solve_cvrp(f'{project_path}/dataset/Vrp-Set-Golden/Golden/Golden_1.vrp')