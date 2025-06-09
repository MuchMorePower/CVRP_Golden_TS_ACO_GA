import os
import math
import time
import json
import random
from tqdm import tqdm
from utils import *
from vrp_reader import read_vrp_file, read_sol_file



class AntColonyOptimizer:
    
    def __init__(self, vrp_data, distance_matrix, node_map, n_ants, n_iterations, alpha, beta, evaporation_rate, pheromone_constant, elitist_weight=1.0, local_search_probability=0.1):
        self.vrp_data = vrp_data
        self.distance_matrix = distance_matrix
        self.node_map = node_map
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_constant = pheromone_constant
        self.elitist_weight = elitist_weight
        self.local_search_probability = local_search_probability 
        self.n_nodes = len(distance_matrix)
        self.pheromone = [[1.0 / (self.n_nodes * self.n_nodes)] * self.n_nodes for _ in range(self.n_nodes)]
        self.depot_index = self.node_map[self.vrp_data['depot']]
        self.node_map_from_zero = {i: node_id for node_id, i in node_map.items()}

    def _local_search_2opt(self, route):
        """对单条路径应用 2-opt 局部搜索。"""
        if not route or len(route) < 2:
            return route

        best_route = route[:]
        improved = True
        while improved:
            improved = False
            full_route = [self.depot_index] + best_route + [self.depot_index]
            n = len(full_route)

            for i in range(n - 3):
                for j in range(i + 2, n - 1):
                    node_i, node_i1 = full_route[i], full_route[i+1]
                    node_j, node_j1 = full_route[j], full_route[j+1]

                    cost_change = (self.distance_matrix[node_i][node_j] +
                                   self.distance_matrix[node_i1][node_j1] -
                                   self.distance_matrix[node_i][node_i1] -
                                   self.distance_matrix[node_j][node_j1])

                    if cost_change < -1e-9:
                        segment_to_reverse = full_route[i + 1:j + 1]
                        full_route[i + 1:j + 1] = segment_to_reverse[::-1]
                        best_route = full_route[1:-1]
                        improved = True
                        break
                if improved:
                    break
        return best_route

    def _apply_local_search_to_solution(self, solution):
        """对整个解决方案应用 2-opt。"""
        improved_solution = []
        for route in solution:
            improved_route = self._local_search_2opt(route)
            improved_solution.append(improved_route)
        return improved_solution

    def solve(self):
        global_best_solution = None
        global_best_cost = float('inf')

        pbar = tqdm(range(self.n_iterations), desc="ACO-EAS+ProbLS Solving")

        for iteration in pbar:
            solutions_with_costs = []

            for _ in range(self.n_ants):
                solution = self._construct_solution()
                
                # 概率性应用局部搜索
                if random.random() < self.local_search_probability:
                    solution = self._apply_local_search_to_solution(solution)

                cost = calculate_solution_cost(solution, self.distance_matrix, self.depot_index)
                solutions_with_costs.append((solution, cost))

                if cost < global_best_cost:
                    global_best_solution = solution
                    global_best_cost = cost

            self._update_pheromone(solutions_with_costs, global_best_solution, global_best_cost)
            pbar.set_postfix(best_cost=f"{global_best_cost:.2f}")

        pbar.close()
        return global_best_solution, global_best_cost

    def _construct_solution(self):
        
        solution = []
        unvisited = list(range(self.n_nodes))
        unvisited.remove(self.depot_index)
        random.shuffle(unvisited)
        while unvisited:
            route = []
            current_capacity = self.vrp_data['capacity']
            current_node = self.depot_index
            while True: 
                next_node = self._select_next_node(current_node, unvisited, current_capacity)
                if next_node is None: break 
                route.append(next_node)
                unvisited.remove(next_node)
                node_id_1_based = self.node_map_from_zero[next_node]
                current_capacity -= self.vrp_data['demands'][node_id_1_based]
                current_node = next_node
                if not unvisited: break
            if route: solution.append(route)
        return solution

    def _select_next_node(self, current_node, unvisited, current_capacity):
         
        probabilities = []
        total_prob = 0.0
        possible_nodes = []
        for node in unvisited:
            node_id_1_based = self.node_map_from_zero[node]
            demand = self.vrp_data['demands'][node_id_1_based]
            if demand <= current_capacity:
                possible_nodes.append(node)
        if not possible_nodes: return None
        for node in possible_nodes:
            distance = self.distance_matrix[current_node][node]
            if distance == 0: distance = 1e-9
            pheromone = self.pheromone[current_node][node] ** self.alpha
            visibility = (1.0 / distance) ** self.beta
            prob_value = pheromone * visibility
            probabilities.append((node, prob_value))
            total_prob += prob_value
        if total_prob == 0: return random.choice(possible_nodes)
        rand = random.uniform(0, total_prob)
        cumulative_prob = 0.0
        for node, prob in probabilities:
            cumulative_prob += prob
            if cumulative_prob >= rand: return node
        return possible_nodes[-1]

    def _update_pheromone(self, solutions_with_costs, global_best_solution, global_best_cost):
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                self.pheromone[i][j] *= (1.0 - self.evaporation_rate)
        for solution, cost in solutions_with_costs:
            if cost == 0: continue
            pheromone_to_add = self.pheromone_constant / cost
            for route in solution:
                current_node = self.depot_index
                for node in route:
                    self.pheromone[current_node][node] += pheromone_to_add
                    current_node = node
                self.pheromone[current_node][self.depot_index] += pheromone_to_add
        if global_best_solution and global_best_cost != float('inf'):
            elitist_pheromone_to_add = self.elitist_weight * (self.pheromone_constant / global_best_cost)
            for route in global_best_solution:
                current_node = self.depot_index
                for node in route:
                    self.pheromone[current_node][node] += elitist_pheromone_to_add
                    current_node = node
                self.pheromone[current_node][self.depot_index] += elitist_pheromone_to_add

def aco_solve_cvrp(vrp_filepath):
    
    current_file = os.path.abspath(__file__)
    project_path = os.path.abspath(os.path.join(current_file, "..", ))
    try:
        potential_file_path = vrp_filepath
        if os.path.exists(potential_file_path):
            file_to_read = potential_file_path
        
    except NameError:
         print("无法自动获取路径，请确保 'file_to_read' 设置正确。")


    if not os.path.exists(file_to_read):
        print(f"错误: 文件 '{file_to_read}' 不存在。请检查路径。")
    else:
        print(f"正在尝试读取文件: {file_to_read}")
        vrp_data = read_vrp_file(file_to_read)

        if vrp_data:
            print(f"\n成功读取文件: {vrp_data.get('name', 'N/A')}")

            distance_matrix, node_map = calculate_distance_matrix(vrp_data['coords'])

            # 创建 AntColonyOptimizer 实例
            aco = AntColonyOptimizer(
                vrp_data=vrp_data,
                distance_matrix=distance_matrix,
                node_map=node_map,
                n_ants=70,           # 蚂蚁数量
                n_iterations=500,    # 迭代次数
                alpha=1.0,           # 信息素影响因子
                beta=4.0,            # 启发信息影响因子
                evaporation_rate=0.18, # 蒸发率
                pheromone_constant=250, # 信息素增加量
                elitist_weight=6.0,   # 精英权重
                local_search_probability=0.4 # 局部搜索概率
            )

            start_time = time.time()
            best_solution, best_cost = aco.solve() 
            end_time = time.time()
            execution_time = end_time - start_time

            print("\n蚁群算法求解结果:")
            print(f"  总成本: {best_cost:.2f}")
            print(f"  执行时间: {execution_time:.2f} 秒")
            print("  路径:")
            
            node_map_from_zero = {i: node_id for node_id, i in node_map.items()}
            depot_1_based = vrp_data['depot']

            if best_solution:
                for i, route in enumerate(best_solution, start=1):
                    route_1_based = [depot_1_based] + [node_map_from_zero[node] for node in route] + [depot_1_based]
                    print(f"    路径 {i}: {' -> '.join(map(str, route_1_based))}")

                

                # 准备 0-based 的需求列表
                demands_0_based = [vrp_data['demands'][node_id] for node_id in sorted(vrp_data['coords'].keys())]

                # 构造文件名 (例如: Golden_1_aco_run1_v1.json)
                problem_name = vrp_data.get('name', 'unknown').replace(' ', '_')
                algorithm_tag = 'aco' # 使用 'aco' 标签
                run_number = 1 # 可以修改或作为参数传入
                output_filename = f"{project_path}/results/{problem_name}_{algorithm_tag}_run{run_number}.json"

                
                save_solution(
                    solution=best_solution, # 0-based solution
                    cost=best_cost,
                    demands=demands_0_based, # 传入 0-based 需求列表
                    distance_matrix=distance_matrix,
                    depot_index=aco.depot_index, # 0-based depot index
                    vrp_data=vrp_data,
                    execution_time=execution_time,
                    filepath=output_filename
                )
                

            else:
                print("    未找到有效路径。")

        else:
            print("\n读取 VRP 文件失败。")


if __name__ == "__main__":
    
    file_to_read = 'path/to/your/Golden_1.vrp'

    
    try:
        current_file = os.path.abspath(__file__)
        project_path = os.path.abspath(os.path.join(current_file, "..", ))
        # 数据集在 'dataset/Vrp-Set-Golden/Golden/' 目录下
        potential_file_path = os.path.join(project_path, 'dataset', 'Vrp-Set-Golden', 'Golden', 'Golden_1.vrp')
        if os.path.exists(potential_file_path):
            file_to_read = potential_file_path
        else:
             
             potential_file_path = os.path.join(project_path, 'Golden_1.vrp')
             if os.path.exists(potential_file_path):
                 file_to_read = potential_file_path

    except NameError:
         print("无法自动获取路径，请确保 'file_to_read' 设置正确。")


    if not os.path.exists(file_to_read):
        print(f"错误: 文件 '{file_to_read}' 不存在。请检查路径。")
    else:
        print(f"正在尝试读取文件: {file_to_read}")
        vrp_data = read_vrp_file(file_to_read)

        if vrp_data:
            print(f"\n成功读取文件: {vrp_data.get('name', 'N/A')}")

            distance_matrix, node_map = calculate_distance_matrix(vrp_data['coords'])

            # 创建 AntColonyOptimizer 实例
            aco = AntColonyOptimizer(
                vrp_data=vrp_data,
                distance_matrix=distance_matrix,
                node_map=node_map,
                n_ants=70,           # 蚂蚁数量
                n_iterations=500,    # 迭代次数
                alpha=1.0,           # 信息素影响因子
                beta=4.0,            # 启发信息影响因子
                evaporation_rate=0.18, # 蒸发率
                pheromone_constant=250, # 信息素增加量
                elitist_weight=6.0,   # 精英权重
                local_search_probability=0.4 # 局部搜索概率
            )

            start_time = time.time()
            best_solution, best_cost = aco.solve() 
            end_time = time.time()
            execution_time = end_time - start_time

            print("\n蚁群算法求解结果:")
            print(f"  总成本: {best_cost:.2f}")
            print(f"  执行时间: {execution_time:.2f} 秒")
            print("  路径:")
            
            node_map_from_zero = {i: node_id for node_id, i in node_map.items()}
            depot_1_based = vrp_data['depot']

            if best_solution:
                for i, route in enumerate(best_solution, start=1):
                    route_1_based = [depot_1_based] + [node_map_from_zero[node] for node in route] + [depot_1_based]
                    print(f"    路径 {i}: {' -> '.join(map(str, route_1_based))}")

               

                # 1. 准备 0-based 的需求列表
                demands_0_based = [vrp_data['demands'][node_id] for node_id in sorted(vrp_data['coords'].keys())]

                # 2. 构造文件名 (例如: Golden_1_aco_run1_v1.json)
                problem_name = vrp_data.get('name', 'unknown').replace(' ', '_')
                algorithm_tag = 'aco' # 使用 'aco' 标签
                run_number = 1 
                output_filename = f"{project_path}/results/{problem_name}_{algorithm_tag}_run{run_number}.json"

                
                save_solution(
                    solution=best_solution, # 0-based solution
                    cost=best_cost,
                    demands=demands_0_based, # 传入 0-based 需求列表
                    distance_matrix=distance_matrix,
                    depot_index=aco.depot_index, # 0-based depot index
                    vrp_data=vrp_data,
                    execution_time=execution_time,
                    filepath=output_filename
                )
                

            else:
                print("    未找到有效路径。")

        else:
            print("\n读取 VRP 文件失败。")