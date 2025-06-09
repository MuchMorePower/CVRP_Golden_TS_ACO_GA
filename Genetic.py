import os
import re
import math
import random
import copy
import json
import time

import numpy as np
from tqdm import tqdm
from vrp_reader import read_vrp_file, read_sol_file
from utils import calculate_distance_matrix, calculate_route_cost, calculate_solution_cost, check_solution_feasibility, save_solution, calculate_insertion_cost

SEED = 42 
random.seed(SEED)



def generate_greedy_solution(demands, capacity, distance_matrix, depot_index, customer_nodes_list):
    """
    生成一个简单的贪心插入解。
    Args:
        customer_nodes_list: 不包含仓库的客户节点列表。
    """
    unrouted_customers = set(customer_nodes_list)
    solution = []

    while unrouted_customers:
        current_route = []
        current_load = 0
        
        if not unrouted_customers: break
        
        path_started = False
        temp_unrouted_list = list(unrouted_customers)
        random.shuffle(temp_unrouted_list)

        for seed_candidate in temp_unrouted_list:
            if demands[seed_candidate] <= capacity:
                current_route.append(seed_candidate)
                current_load += demands[seed_candidate]
                unrouted_customers.remove(seed_candidate)
                path_started = True
                break
        
        if not path_started: break 

        while True:
            best_insertion = None 
            min_cost_increase = float('inf')
            
            temp_unrouted_list_for_insertion = list(unrouted_customers)
            random.shuffle(temp_unrouted_list_for_insertion)

            for customer in temp_unrouted_list_for_insertion:
                if current_load + demands[customer] <= capacity:
                    for pos in range(len(current_route) + 1):
                        cost_inc = calculate_insertion_cost(current_route, customer, pos, distance_matrix, depot_index)
                        if cost_inc < min_cost_increase:
                            min_cost_increase = cost_inc
                            best_insertion = (customer, pos)
            
            if best_insertion is None: break
            
            cust_to_add, insert_pos = best_insertion
            current_route.insert(insert_pos, cust_to_add)
            current_load += demands[cust_to_add]
            unrouted_customers.remove(cust_to_add)

        if current_route:
            solution.append(current_route)
            
    return solution

def generate_nn_solution(demands, capacity, distance_matrix, depot_index, customer_nodes_list):
    """
    生成一个基于最近邻法的 VRP 解决方案。
    """
    unrouted = set(customer_nodes_list)
    solution = []

    while unrouted:
        current_route = []
        current_load = 0
        last_node = depot_index # 每条新路径都从仓库开始

        while True:
            best_next_cust = None
            min_dist = float('inf')

            # 寻找离 last_node 最近的、满足条件的未服务客户
            for cust in unrouted:
                if current_load + demands[cust] <= capacity:
                    dist = distance_matrix[last_node][cust]
                    if dist < min_dist:
                        min_dist = dist
                        best_next_cust = cust
            
            # 如果找到了合适的客户
            if best_next_cust:
                current_route.append(best_next_cust)
                current_load += demands[best_next_cust]
                last_node = best_next_cust
                unrouted.remove(best_next_cust)
            else:
                # 如果找不到可以加入当前路径的客户，则结束这条路径
                break

        if current_route:
            solution.append(current_route)
            
    return solution

def generate_farthest_insertion_solution(demands, capacity, distance_matrix, depot_index, customer_nodes_list):
    """
    生成一个基于最远客户种子 + 最佳插入法的 VRP 解决方案。
    """
    unrouted = set(customer_nodes_list)
    solution = []

    while unrouted:
        current_route = []
        current_load = 0

        
        farthest_cust = -1
        max_dist = -1.0
        
        possible_seeds = [c for c in unrouted if demands[c] <= capacity]
        if not possible_seeds: break # 没有客户能启动新路径

        for cust in possible_seeds:
            dist = distance_matrix[depot_index][cust]
            if dist > max_dist:
                max_dist = dist
                farthest_cust = cust
        
        if farthest_cust == -1: break # 如果找不到合适的种子客户

        current_route.append(farthest_cust)
        current_load += demands[farthest_cust]
        unrouted.remove(farthest_cust)
        

        
        while True:
            best_insertion_candidate = None # (customer, position, cost_increase)
            min_cost_increase = float('inf')

            for customer in unrouted:
                if current_load + demands[customer] <= capacity:
                    for pos in range(len(current_route) + 1):
                        cost_inc = calculate_insertion_cost(current_route, customer, pos, distance_matrix, depot_index)
                        if cost_inc < min_cost_increase:
                            min_cost_increase = cost_inc
                            best_insertion_candidate = (customer, pos)

            # 如果找到了最佳插入点
            if best_insertion_candidate:
                cust_to_add, insert_pos = best_insertion_candidate
                current_route.insert(insert_pos, cust_to_add)
                current_load += demands[cust_to_add]
                unrouted.remove(cust_to_add)
            else:
                # 如果找不到可以插入的客户，则结束当前路径
                break
       

        if current_route:
            solution.append(current_route)
            
    return solution

def solution_to_chromosome(solution, all_customer_nodes):
    """
    将 VRP 解 (路径列表) 转换为染色体 (客户列表的排列)。
    确保染色体包含所有客户节点且无重复，长度正确。
    """
    chromosome = []
    routed_customers_in_solution = set()

    if solution is not None:
        for route in solution:
            for customer_node in route:
                if customer_node in all_customer_nodes and customer_node not in routed_customers_in_solution:
                    chromosome.append(customer_node)
                    routed_customers_in_solution.add(customer_node)
    
    missing_customers = set(all_customer_nodes) - routed_customers_in_solution
    chromosome.extend(list(missing_customers)) 

    if len(chromosome) != len(all_customer_nodes) or len(set(chromosome)) != len(all_customer_nodes):
        # print(f"警告: 染色体转换后异常。长度: {len(chromosome)}, 唯一: {len(set(chromosome))}, 期望: {len(all_customer_nodes)}")
        return random.sample(all_customer_nodes, len(all_customer_nodes))
        
    return chromosome



def apply_2opt_on_route(route, distance_matrix, depot_index):
    """对单条路径应用 2-Opt 优化（迭代直到无改进）。"""
    if not route or len(route) < 2:
        return route

    best_route = route[:]
    improved = True
    while improved:
        improved = False
        current_cost = calculate_route_cost(best_route, distance_matrix, depot_index)

        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
               
                
                node_im1 = best_route[i-1] if i > 0 else depot_index
                node_i = best_route[i]
                node_j = best_route[j]
                node_j1 = best_route[j+1] if j + 1 < len(best_route) else depot_index

                # 原始成本 = dist(i-1, i) + dist(j, j+1)
                original_edge_cost = distance_matrix[node_im1][node_i] + distance_matrix[node_j][node_j1]
                # 新成本 = dist(i-1, j) + dist(i, j+1)
                new_edge_cost = distance_matrix[node_im1][node_j] + distance_matrix[node_i][node_j1]

                if new_edge_cost < original_edge_cost:
                    # 构造新路径 
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    
                    best_route = new_route
                    current_cost = current_cost - original_edge_cost + new_edge_cost 
                    improved = True
                    break # 找到改进后，从头开始检查新路径
            if improved:
                break
                
    return best_route

def apply_relocate(solution, distance_matrix, demands, capacity, depot_index):
    """
    尝试将单个客户移动到不同路径或位置 (首次改进策略)。
    返回 (新解决方案, 是否改进)。
    """
    current_solution = [route[:] for route in solution] 
    num_routes = len(current_solution)

    for r1_idx in range(num_routes):
        route1 = current_solution[r1_idx]
        for c1_idx in range(len(route1)):
            customer_to_move = route1[c1_idx]
            demand_to_move = demands[customer_to_move]

            
            node_prev = route1[c1_idx-1] if c1_idx > 0 else depot_index
            node_curr = route1[c1_idx]
            node_next = route1[c1_idx+1] if c1_idx + 1 < len(route1) else depot_index
            cost_removed = distance_matrix[node_prev][node_next] - \
                           distance_matrix[node_prev][node_curr] - \
                           distance_matrix[node_curr][node_next]

            
            for r2_idx in range(num_routes):
                route2 = current_solution[r2_idx]

                # 检查容量
                if r1_idx != r2_idx:
                    current_load_r2 = sum(demands[c] for c in route2)
                    if current_load_r2 + demand_to_move > capacity:
                        continue

                # 尝试插入到每个可能的位置
                for c2_idx in range(len(route2) + 1):
                    # 如果是同一条路径，避免无效移动
                    if r1_idx == r2_idx and (c2_idx == c1_idx or c2_idx == c1_idx + 1):
                         continue

                    # 计算插入客户后的成本变化 (增加的成本)
                    node_before_insert = route2[c2_idx-1] if c2_idx > 0 else depot_index
                    node_after_insert = route2[c2_idx] if c2_idx < len(route2) else depot_index
                    cost_inserted = distance_matrix[node_before_insert][customer_to_move] + \
                                    distance_matrix[customer_to_move][node_after_insert] - \
                                    distance_matrix[node_before_insert][node_after_insert]

                    cost_change = cost_removed + cost_inserted

                    # 如果成本降低 (找到改进)，则执行移动并返回
                    if cost_change < -1e-6: # 使用一个小阈值避免浮点误差
                        new_solution = [r[:] for r in current_solution]
                        
                        moved_customer_val = new_solution[r1_idx].pop(c1_idx)
                       
                        if r1_idx == r2_idx and c1_idx < c2_idx:
                            new_solution[r2_idx].insert(c2_idx - 1, moved_customer_val)
                        else:
                             new_solution[r2_idx].insert(c2_idx, moved_customer_val)

                        # 移除可能产生的空路径
                        final_solution = [r for r in new_solution if r]
                        return final_solution, True 

    return current_solution, False # 未找到改进

def apply_swap(solution, distance_matrix, demands, capacity, depot_index):
    """
    尝试交换两个客户 (路径间或路径内) (首次改进策略)。
    返回 (新解决方案, 是否改进)。
    """
    current_solution = [route[:] for route in solution]
    num_routes = len(current_solution)

    for r1_idx in range(num_routes):
        for c1_idx in range(len(current_solution[r1_idx])):
            c1 = current_solution[r1_idx][c1_idx]
            r1 = current_solution[r1_idx]
            p1 = r1[c1_idx-1] if c1_idx > 0 else depot_index
            n1 = r1[c1_idx+1] if c1_idx + 1 < len(r1) else depot_index

            
            for r2_idx in range(r1_idx, num_routes):
                r2 = current_solution[r2_idx]
                
                start_c2 = c1_idx + 1 if r1_idx == r2_idx else 0
                for c2_idx in range(start_c2, len(r2)):
                    c2 = current_solution[r2_idx][c2_idx]
                    p2 = r2[c2_idx-1] if c2_idx > 0 else depot_index
                    n2 = r2[c2_idx+1] if c2_idx + 1 < len(r2) else depot_index

                    
                    if r1_idx != r2_idx:
                        load1 = sum(demands[c] for c in r1)
                        load2 = sum(demands[c] for c in r2)
                        if load1 - demands[c1] + demands[c2] > capacity or \
                           load2 - demands[c2] + demands[c1] > capacity:
                            continue

                    
                    cost_removed = distance_matrix[p1][c1] + distance_matrix[c1][n1] + \
                                   distance_matrix[p2][c2] + distance_matrix[c2][n2]
                    
                    if r1_idx == r2_idx:
                        # 路径内交换 
                        if c2_idx == c1_idx + 1: # 邻近交换
                            cost_added = distance_matrix[p1][c2] + distance_matrix[c2][c1] + distance_matrix[c1][n2]
                        else: # 非邻近交换
                            cost_added = distance_matrix[p1][c2] + distance_matrix[c2][n1] + \
                                         distance_matrix[p2][c1] + distance_matrix[c1][n2]
                    else:
                        # 路径间交换
                        cost_added = distance_matrix[p1][c2] + distance_matrix[c2][n1] + \
                                     distance_matrix[p2][c1] + distance_matrix[c1][n2]

                    cost_change = cost_added - cost_removed

                   
                    if cost_change < -1e-6:
                        new_solution = [r[:] for r in current_solution]
                        new_solution[r1_idx][c1_idx], new_solution[r2_idx][c2_idx] = \
                            new_solution[r2_idx][c2_idx], new_solution[r1_idx][c1_idx]
                        return new_solution, True # 找到改进，立即返回

    return current_solution, False # 未找到改进

def apply_local_search(solution, distance_matrix, depot_index):
    """对整个解决方案应用局部搜索 (这里只用 2-Opt)。"""
    improved_solution = []
    for route in solution:
        optimized_route = apply_2opt_on_route(route, distance_matrix, depot_index)
        if optimized_route: # 确保不添加空路径
             improved_solution.append(optimized_route)
    return improved_solution

def apply_local_search_vns(solution, distance_matrix, demands, capacity, depot_index):
    """
    应用基于 VNS 的局部搜索 (2-Opt -> Relocate -> Swap)，直到无法改进。
    """
    current_solution = [route[:] for route in solution]
    
    while True: # 持续循环直到一整轮没有任何改进
        # print(1)
        cost_before_cycle = calculate_solution_cost(current_solution, distance_matrix, depot_index)

        
        solution_after_2opt = []
        for route in current_solution:
            optimized_route = apply_2opt_on_route(route, distance_matrix, depot_index)
            if optimized_route:
                solution_after_2opt.append(optimized_route)
        
        cost_after_2opt = calculate_solution_cost(solution_after_2opt, distance_matrix, depot_index)
        if cost_after_2opt < cost_before_cycle:
            current_solution = solution_after_2opt
            continue 

        
        solution_after_relocate, improved_rel = apply_relocate(current_solution, distance_matrix, demands, capacity, depot_index)
        if improved_rel:
            current_solution = solution_after_relocate
            continue 

        
        solution_after_swap, improved_swap = apply_swap(current_solution, distance_matrix, demands, capacity, depot_index)
        if improved_swap:
            current_solution = solution_after_swap
            continue 

        
        break

    return current_solution


def decode_chromosome(chromosome, demands, capacity, depot_index):
    solution = []
    current_route = []
    current_load = 0
    for customer_node in chromosome:
        demand = demands[customer_node]
        if current_load + demand <= capacity:
            current_route.append(customer_node)
            current_load += demand
        else:
            if current_route: solution.append(current_route)
            current_route = [customer_node]
            current_load = demand
    if current_route: solution.append(current_route)
    return solution

def tournament_selection(population_with_fitness, k=5):
    k = min(k, len(population_with_fitness))
    selected_indices = random.sample(range(len(population_with_fitness)), k)
    best_index = -1
    max_fitness = -float('inf')
    for index in selected_indices:
        if population_with_fitness[index]['fitness'] > max_fitness:
            max_fitness = population_with_fitness[index]['fitness']
            best_index = index
    return population_with_fitness[best_index]['chromosome']

def order_crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [-1] * size, [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child1[start:end+1] = parent1[start:end+1]
    child2[start:end+1] = parent2[start:end+1]
    seg1_set = set(child1[start:end+1])
    seg2_set = set(child2[start:end+1])
    
    p2_idx, p1_idx = 0, 0
    c1_idx, c2_idx = 0, 0
    
    while c1_idx < size or c2_idx < size:
        # Fill child 1
        if c1_idx == start: c1_idx = end + 1
        if c1_idx < size:
            while parent2[p2_idx] in seg1_set: p2_idx += 1
            child1[c1_idx] = parent2[p2_idx]
            c1_idx += 1
            p2_idx += 1
            
        # Fill child 2
        if c2_idx == start: c2_idx = end + 1
        if c2_idx < size:
            while parent1[p1_idx] in seg2_set: p1_idx += 1
            child2[c2_idx] = parent1[p1_idx]
            c2_idx += 1
            p1_idx += 1
            
    return child1, child2

def swap_mutation(chromosome, mutation_rate=0.1):
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(mutated_chromosome) - 1)
            mutated_chromosome[i], mutated_chromosome[j] = mutated_chromosome[j], mutated_chromosome[i]
    return mutated_chromosome

# --- 模因算法主函数 ---
def memetic_algorithm_solver(demands, capacity, distance_matrix, depot_index,
                             population_size=100, generations=100, 
                             crossover_rate=0.9, mutation_rate=0.1,
                             elitism_size=20, tournament_k=6,
                             local_search_freq=1): # LS 频率 
    """
    实现不具有混合初始化的模因算法 (GA + Local Search)。
    """
    local_search_prob = 0.5 # 局部搜索概率 
    # --- 初始化 ---
    num_customers = len(demands) - 1
    customer_nodes = [i for i in range(len(demands)) if i != depot_index]

    # 初始化种群 
    population_chromosomes = [random.sample(customer_nodes, num_customers) for _ in range(population_size)]

    best_solution_overall = None
    best_cost_overall = float('inf')

    print(f"开始模因算法: 种群大小={population_size}, 代数={generations}")

    for generation in tqdm(range(generations), desc="模因算法进度"):
        population_with_fitness = []

        
        for chrom in population_chromosomes:
            solution = decode_chromosome(chrom, demands, capacity, depot_index)

            
            # 决定是否应用 LS 
            if ((generation + 1) % local_search_freq == 0 and random.random() < local_search_prob) or best_solution_overall is None:
                improved_solution = apply_local_search(solution, distance_matrix, depot_index)
                # improved_solution = apply_local_search_vns(solution, distance_matrix, demands, capacity, depot_index)
            else:
                improved_solution = solution # 如果不应用 LS，则使用原解

            cost = calculate_solution_cost(improved_solution, distance_matrix, depot_index)
            fitness = 1.0 / (cost + 1e-6)
            population_with_fitness.append({
                'chromosome': chrom,
                'solution': improved_solution, # 存储优化后的解
                'cost': cost,
                'fitness': fitness
            })

        
        population_with_fitness.sort(key=lambda x: x['cost']) # 按成本排序
        current_best = population_with_fitness[0]

        if current_best['cost'] < best_cost_overall:
            best_cost_overall = current_best['cost']
            best_solution_overall = current_best['solution']
            tqdm.write(f"\n代 {generation+1}: 找到新最优解! 成本: {best_cost_overall:.2f}")

       
        new_population_chromosomes = []

       
        for i in range(elitism_size):
            new_population_chromosomes.append(population_with_fitness[i]['chromosome'])

       
        while len(new_population_chromosomes) < population_size:
            parent1 = tournament_selection(population_with_fitness, k=tournament_k)
            parent2 = tournament_selection(population_with_fitness, k=tournament_k)

            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)

            new_population_chromosomes.append(child1)
            if len(new_population_chromosomes) < population_size:
                new_population_chromosomes.append(child2)

        population_chromosomes = new_population_chromosomes

    print(f"\n模因算法完成。最优成本: {best_cost_overall:.2f}, 路径数: {len(best_solution_overall)}")
    return best_solution_overall, best_cost_overall

def memetic_algorithm_solver(demands, capacity, distance_matrix, depot_index,
                             population_size=100, generations=100, 
                             crossover_rate=0.9, mutation_rate=0.1,
                             elitism_size=20, tournament_k=6,
                             local_search_freq=1, 
                             num_greedy_seeds=30):  
    """
    实现具有混合初始化模因算法 (GA + Local Search)。
    """

    local_search_prob = 0.5 # 局部搜索概率 
    # local_search_prob = 0 # 局部搜索概率
    num_nn_seeds = 10 
    num_insert_seeds = 10 

    
    customer_nodes = [i for i in range(len(demands)) if i != depot_index and demands[i] > 0] 
    
    actual_num_customers_to_route = len(customer_nodes)

    if actual_num_customers_to_route == 0:
        # print("没有客户需要服务。")
        return [], 0.0

    
    population_chromosomes = []
    # print(f"生成 {num_greedy_seeds} 个贪心初始解...")
    actual_num_greedy_seeds = min(num_greedy_seeds, population_size) 
    for _ in range(actual_num_greedy_seeds):
        greedy_sol = generate_greedy_solution(demands, capacity, distance_matrix, depot_index, customer_nodes)
        greedy_chrom = solution_to_chromosome(greedy_sol, customer_nodes)
        population_chromosomes.append(greedy_chrom)

    # 生成最近邻解
    for _ in range(num_nn_seeds):
        if len(population_chromosomes) >= population_size: break
        sol = generate_nn_solution(demands, capacity, distance_matrix, depot_index, customer_nodes)
        chrom = solution_to_chromosome(sol, customer_nodes)
        population_chromosomes.append(chrom)

    # 生成最远插入解
    for _ in range(num_insert_seeds):
        if len(population_chromosomes) >= population_size: break
        sol = generate_farthest_insertion_solution(demands, capacity, distance_matrix, depot_index, customer_nodes)
        chrom = solution_to_chromosome(sol, customer_nodes)
        population_chromosomes.append(chrom)
    
    print(f"生成 {population_size - len(population_chromosomes)} 个随机初始解...")
    while len(population_chromosomes) < population_size:
        population_chromosomes.append(random.sample(customer_nodes, actual_num_customers_to_route))
    # --- 混合初始化结束 ---

    best_solution_overall = None
    best_cost_overall = float('inf')

    print(f"开始模因算法: 种群大小={population_size}, 代数={generations}") 

    for generation in tqdm(range(generations), desc="模因算法进度"):
        population_with_fitness = []

        for chrom_idx, chrom in enumerate(population_chromosomes):
            
            if not chrom or len(chrom) != actual_num_customers_to_route or len(set(chrom)) != actual_num_customers_to_route:
                # print(f"警告: 代 {generation+1}, 染色体 {chrom_idx} 无效 ({chrom})。重新生成随机染色体。")
                chrom = random.sample(customer_nodes, actual_num_customers_to_route)
                population_chromosomes[chrom_idx] = chrom

            solution = decode_chromosome(chrom, demands, capacity, depot_index)

            
            # 决定是否应用 LS 
            if ((generation + 1) % local_search_freq == 0 and random.random() < local_search_prob) or best_solution_overall is None:
                
                improved_solution = apply_local_search(solution, distance_matrix, depot_index)
            else:
                improved_solution = solution # 如果不应用 LS，则使用原解

            cost = calculate_solution_cost(improved_solution, distance_matrix, depot_index)
            fitness = 1.0 / (cost + 1e-6) # 避免除以零
            population_with_fitness.append({
                'chromosome': chrom,
                'solution': improved_solution, # 存储优化后的解
                'cost': cost,
                'fitness': fitness
            })
        
        if not population_with_fitness: # 如果所有染色体都无效导致列表为空
            
            population_chromosomes = []
            for _ in range(actual_num_greedy_seeds):
                greedy_sol = generate_greedy_solution(demands, capacity, distance_matrix, depot_index, customer_nodes)
                greedy_chrom = solution_to_chromosome(greedy_sol, customer_nodes)
                population_chromosomes.append(greedy_chrom)
            while len(population_chromosomes) < population_size:
                population_chromosomes.append(random.sample(customer_nodes, actual_num_customers_to_route))
            continue # 跳到下一代


        # 寻找当前代最佳
        population_with_fitness.sort(key=lambda x: x['cost']) # 按成本排序
        current_best_individual = population_with_fitness[0]

        if current_best_individual['cost'] < best_cost_overall:
            best_cost_overall = current_best_individual['cost']
            best_solution_overall = copy.deepcopy(current_best_individual['solution']) # 存储深拷贝
            tqdm.write(f"\n代 {generation+1}: 找到新最优解! 成本: {best_cost_overall:.2f}") 

        
        new_population_chromosomes = []


        actual_elitism_size = min(elitism_size, len(population_with_fitness))
        for i in range(actual_elitism_size):
            new_population_chromosomes.append(population_with_fitness[i]['chromosome'])

           
        while len(new_population_chromosomes) < population_size:
            parent1 = tournament_selection(population_with_fitness, k=tournament_k)
            parent2 = tournament_selection(population_with_fitness, k=tournament_k)

            
            if parent1 is None or parent2 is None:
                
                parent1 = random.sample(customer_nodes, actual_num_customers_to_route)
                parent2 = random.sample(customer_nodes, actual_num_customers_to_route)

            child1, child2 = parent1[:], parent2[:] 
            if random.random() < crossover_rate:
                
                if len(parent1) == actual_num_customers_to_route and len(parent2) == actual_num_customers_to_route:
                     child1_co, child2_co = order_crossover(parent1, parent2)
                     
                     if child1_co and len(child1_co) == actual_num_customers_to_route: child1 = child1_co
                     if child2_co and len(child2_co) == actual_num_customers_to_route: child2 = child2_co
                # else:
                    # print(f"警告: 交叉操作中父代长度不匹配。P1 len: {len(parent1)}, P2 len: {len(parent2)}")


           
            if len(child1) != actual_num_customers_to_route: child1 = random.sample(customer_nodes, actual_num_customers_to_route)
            if len(child2) != actual_num_customers_to_route: child2 = random.sample(customer_nodes, actual_num_customers_to_route)

            new_population_chromosomes.append(swap_mutation(child1, mutation_rate))
            if len(new_population_chromosomes) < population_size:
                new_population_chromosomes.append(swap_mutation(child2, mutation_rate))
        
        population_chromosomes = new_population_chromosomes
        if not population_chromosomes and generation < generations -1 : # 如果种群意外变空
            
            for _ in range(actual_num_greedy_seeds):
                greedy_sol = generate_greedy_solution(demands, capacity, distance_matrix, depot_index, customer_nodes)
                greedy_chrom = solution_to_chromosome(greedy_sol, customer_nodes)
                population_chromosomes.append(greedy_chrom)
            while len(population_chromosomes) < population_size:
                population_chromosomes.append(random.sample(customer_nodes, actual_num_customers_to_route))


    print(f"\n模因算法完成。最优成本: {best_cost_overall:.2f}, 路径数: {len(best_solution_overall) if best_solution_overall else 0}") 
    return best_solution_overall, best_cost_overall

def memetic_solve_cvrp(vrp_file_path, sol_file_path):

    project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), ".."))
    results_dir = os.path.join(project_path, 'results')
    os.makedirs(results_dir, exist_ok=True) 
    # 读取 BKS
    sol_data = read_sol_file(sol_file_path)
    bks_cost = sol_data['cost'] if sol_data else None
    print(f"BKS 成本: {bks_cost}")

    # 读取 VRP 数据
    print("\n1. 正在读取 VRP 文件...")
    vrp_raw_data = read_vrp_file(vrp_file_path)

    if vrp_raw_data:
        print(f"   文件读取成功: {vrp_raw_data.get('name', 'N/A')}")
        capacity = vrp_raw_data['capacity']
        print("\n2. 正在准备数据...")
        distance_matrix, node_map = calculate_distance_matrix(vrp_raw_data['coords'])
        demands = [vrp_raw_data['demands'][node_id] for node_id in sorted(vrp_raw_data['coords'].keys())]
        depot_index = node_map[vrp_raw_data['depot']]
        print(f"   数据准备完成. 节点数: {len(demands)}")

        # 运行模因算法
        print("\n4. 正在运行模因算法...")
        start_ma_time = time.time()
        best_sol_ma, best_c_ma = memetic_algorithm_solver(
            demands, capacity, distance_matrix, depot_index,
            population_size=250,  
            generations=500,  
            mutation_rate=0.007,
            local_search_freq=1 
        )
        end_ma_time = time.time()
        execution_time_ma = end_ma_time - start_ma_time

        
        solution_filename = f"{vrp_raw_data.get('name', 'unknown_solution')}_ma_run1_v2.json"
        solution_filepath = os.path.join(results_dir, solution_filename)
        save_solution(best_sol_ma, best_c_ma, demands, distance_matrix, depot_index, vrp_raw_data, execution_time_ma, solution_filepath)

        
        print("\n--- 评估 MA 结果 ---")
        # try:
        #      evaluate(solution_filepath, vrp_file_path, bks_cost=bks_cost)
        # except NameError:
        #      print("错误: 'evaluate' 函数未找到。请确保它已定义或导入。")

    else:
        print("读取 VRP 文件失败。")

if __name__ == '__main__':
    version = '1' # 选择 Golden 数据集版本

    
    project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), ".."))
    vrp_file_path = f'{project_path}/dataset/Vrp-Set-Golden/Golden/Golden_{version}.vrp'
    sol_file_path = f'{project_path}/dataset/Vrp-Set-Golden/Golden/Golden_{version}.sol'
    results_dir = os.path.join(project_path, 'results')
    os.makedirs(results_dir, exist_ok=True) 

    # 读取 BKS
    sol_data = read_sol_file(sol_file_path)
    bks_cost = sol_data['cost'] if sol_data else None
    print(f"Golden_{version} 的 BKS 成本: {bks_cost}")

    # 读取 VRP 数据
    print("\n1. 正在读取 VRP 文件...")
    vrp_raw_data = read_vrp_file(vrp_file_path)

    if vrp_raw_data:
        print(f"   文件读取成功: {vrp_raw_data.get('name', 'N/A')}")
        capacity = vrp_raw_data['capacity']
        print("\n2. 正在准备数据...")
        distance_matrix, node_map = calculate_distance_matrix(vrp_raw_data['coords'])
        demands = [vrp_raw_data['demands'][node_id] for node_id in sorted(vrp_raw_data['coords'].keys())]
        depot_index = node_map[vrp_raw_data['depot']]
        print(f"   数据准备完成. 节点数: {len(demands)}")

        # 运行模因算法
        print("\n4. 正在运行模因算法...")
        start_ma_time = time.time()
        best_sol_ma, best_c_ma = memetic_algorithm_solver(
            demands, capacity, distance_matrix, depot_index,
            population_size=300,  
            generations=500,   
            mutation_rate=0.007,
            local_search_freq=1 
        )
        end_ma_time = time.time()
        execution_time_ma = end_ma_time - start_ma_time

        
        solution_filename = f"{vrp_raw_data.get('name', 'unknown_solution')}_ma_run1_v2.json"
        solution_filepath = os.path.join(results_dir, solution_filename)
        save_solution(best_sol_ma, best_c_ma, demands, distance_matrix, depot_index, vrp_raw_data, execution_time_ma, solution_filepath)

        
        print("\n--- 评估 MA 结果 ---")
        # try:
        #      evaluate(solution_filepath, vrp_file_path, bks_cost=bks_cost)
        # except NameError:
        #      print("错误: 'evaluate' 函数未找到。请确保它已定义或导入。")

    else:
        print("读取 VRP 文件失败。")