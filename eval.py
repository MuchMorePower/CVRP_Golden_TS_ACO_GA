import os
import math
import numpy as np
from vrp_reader import read_sol_file, read_vrp_file
import json

current_file = os.path.abspath(__file__)
project_path = os.path.abspath(os.path.join(current_file, "..", ))

algorithms = {
        'ts': 'TS',
        'aco': 'ACO',
        'ma_v1': 'HMA+LS',
        'ma_v2': 'HMA',
        'ma_v3': 'MA'
    }

def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def process_data_list(data_list, bks_cost=None):
    algorithms = ['TS', 'ACO', 'PMA', 'HMA+LS', 'HMA']
    lines = []  
    stats = {alg: {'gaps': [], 'times': [], 'best_count': 0} for alg in algorithms}
    
    
    if bks_cost is not None and bks_cost > 0:
        global_best_cost = min(data['best_cost'] for data in data_list)
    else:
        global_best_cost = None

    for data, algorithm in zip(data_list, algorithms):
        best_cost = round(data['best_cost'], 1)
        execution_time = round(data['execution_time_seconds'], 1)
        
        
        if bks_cost is not None and bks_cost > 0:
            gap = (best_cost - bks_cost) / bks_cost * 100
            gap_str = f"{gap:.2f}"
            stats[algorithm]['gaps'].append(gap)
        else:
            gap_str = "N/A"

       
        stats[algorithm]['times'].append(execution_time)
        
        
        if global_best_cost is not None and abs(best_cost - global_best_cost) < 1e-6:
            stats[algorithm]['best_count'] += 1

        
        record_string = f"  & {algorithm} & {best_cost} & {gap_str} & {execution_time:.2f} \\\\"
        lines.append(record_string)
    
    
    summary = {}
    for alg in algorithms:
        summary[alg] = {
            'avg_gap': f"{np.mean(stats[alg]['gaps']):.2f}" if stats[alg]['gaps'] else "N/A",
            'best_count': stats[alg]['best_count'],
            'avg_time': f"{np.mean(stats[alg]['times']):.2f}"
        }
    
    
    stats_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Algorithm Performance Summary}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Algorithm & Avg Gap (\%) & Best Count & Avg Time (s) \\\\",
        "\\midrule"
    ]
    
    for alg in algorithms:
        stats_lines.append(
            f"{alg} & {summary[alg]['avg_gap']} & {summary[alg]['best_count']} & {summary[alg]['avg_time']} \\\\"
        )
    
    stats_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    stats_table = "\n".join(stats_lines)
    full_table_rows = "\n".join(lines)
    
    return full_table_rows, stats_table


def main():
    dataset_path = os.path.join(project_path, 'dataset', 'Vrp-Set-Golden', 'Golden')
    all_latex_blocks = []
    for i in range(1, 21):
        sol_file_path = os.path.join(dataset_path, f'Golden_{i}.sol')
        sol_data = read_sol_file(sol_file_path)
        bks_cost = sol_data['cost'] if sol_data else None

        ts_file = os.path.join(project_path, 'results', f'Golden_{i}_ts_run1.json')
        aco_file = os.path.join(project_path, 'results', f'Golden_{i}_aco_run1.json') 
        ma_v1_file = os.path.join(project_path, 'results', f'Golden_{i}_ma_run1_v1.json')
        ma_v2_file = os.path.join(project_path, 'results', f'Golden_{i}_ma_run1_v2.json')
        ma_v3_file = os.path.join(project_path, 'results', f'Golden_{i}_ma_run1_v3.json')

        ts_data = load_json_data(ts_file)
        aco_data = load_json_data(aco_file)
        ma_v1_data = load_json_data(ma_v1_file)
        ma_v2_data = load_json_data(ma_v2_file)
        ma_v3_data = load_json_data(ma_v3_file)

        data_list = [ts_data, aco_data, ma_v1_data, ma_v2_data, ma_v3_data]
        processed_block,  stats_table= process_data_list(data_list, bks_cost)

        block = f"\\multirow{{5}}{{*}}{{Golden{i}}}\n{processed_block}\n\\addlinespace[0.2cm]"
        all_latex_blocks.append(block)

    latex_code = '\n\n'.join(all_latex_blocks)
    print(latex_code)
    print("\n\n--- 统计表格 ---")
    print(stats_table)



if __name__ == '__main__':
    # data = load_json_data(r"E:\学习资料\大三下\智能算法\期末大作业\CVRP_Project\results\Golden_1_aco_run1.json")
    # print(data['best_cost'])
    main()