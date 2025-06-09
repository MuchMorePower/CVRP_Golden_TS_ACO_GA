import re
import os
current_file = os.path.abspath(__file__)
project_path = os.path.abspath(os.path.join(current_file, "..", ))
print(project_path)
# --- CVRPLIB 格式的 VRP 文件读取器 ---
# 该脚本用于读取 CVRPLIB 格式的 VRP 文件，解析并提取相关数据。
def read_vrp_file(filepath):
    """
    读取并解析 CVRPLIB 格式的 .vrp 文件。

    参数:
        filepath (str): .vrp 文件的路径。

    返回:
        dict: 包含 VRP 问题数据的字典，如果读取失败则返回 None。
              字典包含键: 'name', 'type', 'dimension', 'capacity',
                           'edge_weight_type', 'coords', 'demands', 'depot'。
    """
    data = {
        'coords': {},
        'demands': {},
        'depot': None
    }
    reading_section = None  # 当前正在读取的部分

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()

                # 跳过空行
                if not line:
                    continue

               
                if line.startswith("NAME"):
                    data['name'] = line.split(':')[-1].strip()
                elif line.startswith("TYPE"):
                    data['type'] = line.split(':')[-1].strip()
                elif line.startswith("DIMENSION"):
                    data['dimension'] = int(line.split(':')[-1].strip())
                elif line.startswith("CAPACITY"):
                    data['capacity'] = int(line.split(':')[-1].strip())
                elif line.startswith("EDGE_WEIGHT_TYPE"):
                    data['edge_weight_type'] = line.split(':')[-1].strip()

               
                elif line.startswith("NODE_COORD_SECTION"):
                    reading_section = "COORDS"
                    continue
                elif line.startswith("DEMAND_SECTION"):
                    reading_section = "DEMANDS"
                    continue
                elif line.startswith("DEPOT_SECTION"):
                    reading_section = "DEPOT"
                    continue
                elif line.startswith("EOF"):
                    reading_section = None
                    break

                
                parts = re.split(r'\s+', line) 
                if reading_section == "COORDS":
                    
                    if len(parts) == 3:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        data['coords'][node_id] = (x, y)

                elif reading_section == "DEMANDS":
                    
                    if len(parts) == 2:
                        node_id = int(parts[0])
                        demand = int(parts[1])
                        data['demands'][node_id] = demand

                elif reading_section == "DEPOT":
                    
                    depot_id = int(parts[0])
                    if depot_id != -1:
                        data['depot'] = depot_id
                    else:
                        reading_section = None 

       
        if 'dimension' not in data or len(data['coords']) != data['dimension']:
            print(f"警告: 坐标数量 ({len(data['coords'])}) 与 DIMENSION ({data.get('dimension')}) 不匹配。")
        if 'dimension' not in data or len(data['demands']) != data['dimension']:
             print(f"警告: 需求数量 ({len(data['demands'])}) 与 DIMENSION ({data.get('dimension')}) 不匹配。")
        if data['depot'] is None and 1 in data['coords']:
             print("警告: 未找到 DEPOT_SECTION，默认使用节点 1 作为仓库。")
             data['depot'] = 1 

        return data

    except FileNotFoundError:
        print(f"错误: 文件未找到 - {filepath}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None

def read_sol_file(filepath):
    """
    读取并解析 .sol 文件。

    参数:
        filepath (str): .sol 文件的路径。

    返回:
        dict: 包含 'routes' 和 'cost' 的字典，如果读取失败则返回 None。
              'routes' 中的节点 ID 是 1-based 的。
    """
    solution = {'routes': [], 'cost': None}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                
                route_match = re.match(r"Route #(\d+):\s*(.*)", line)
                if route_match:
                    
                    nodes_str = route_match.group(2).split()
                    route = [int(node) for node in nodes_str]
                    solution['routes'].append(route)
                    continue

                
                cost_match = re.match(r"Cost\s+([\d\.]+)", line, re.IGNORECASE)
                if cost_match:
                    solution['cost'] = float(cost_match.group(1))
                    continue
        
        
        if not solution['routes'] or solution['cost'] is None:
            print(f"警告: .sol 文件 {filepath} 可能未完全解析。")
            
        return solution

    except FileNotFoundError:
        print(f"错误: .sol 文件未找到 - {filepath}")
        return None
    except Exception as e:
        print(f"读取 .sol 文件时发生错误: {e}")
        return None

# --- 示例用法 ---
if __name__ == "__main__":
    
    file_to_read = f'{project_path}/dataset/Vrp-Set-Golden/Golden/Golden_1.vrp'

    print(f"正在尝试读取文件: {file_to_read}")
    print("=" * 30)
    print("注意：如果看到'文件未找到'的错误，请确保你已经下载了 VRP 文件，")
    print("      并且 'file_to_read' 变量指向了正确的文件路径。")
    print("=" * 30)

    vrp_data = read_vrp_file(file_to_read)

    if vrp_data:
        print(f"\n成功读取文件: {vrp_data.get('name', 'N/A')}")
        print(f"问题类型: {vrp_data.get('type', 'N/A')}")
        print(f"维度 (节点数): {vrp_data.get('dimension', 'N/A')}")
        print(f"车辆容量: {vrp_data.get('capacity', 'N/A')}")
        print(f"仓库节点: {vrp_data.get('depot', 'N/A')}")

        # 打印前 5 个节点的坐标和需求 
        print("\n前 5 个节点信息:")
        for i in range(1, min(6, vrp_data.get('dimension', 0) + 1)):
            coords = vrp_data['coords'].get(i, '(无坐标)')
            demand = vrp_data['demands'].get(i, '(无需求)')
            print(f"  节点 {i}: 坐标={coords}, 需求={demand}")

        print(f"\n总共读取了 {len(vrp_data['coords'])} 个坐标。")
        print(f"总共读取了 {len(vrp_data['demands'])} 个需求。")
    else:
        print("\n读取 VRP 文件失败。")

    sol_file_path = f'{project_path}/dataset/Vrp-Set-Golden/Golden/Golden_1.sol'
    print(f"\n正在尝试读取解决方案文件: {sol_file_path}")
    sol_data = read_sol_file(sol_file_path)
    print("\n解决方案数据:")
    if sol_data:
        print(f"  路径数量: {len(sol_data['routes'])}")
        print(f"  总成本: {sol_data['cost']}")
        for i, route in enumerate(sol_data['routes'], start=1):
            print(f"  路径 {i}: {route}")
    else:
        print("读取解决方案文件失败。")