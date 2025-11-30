import json
import subprocess
import argparse
import os
import sys
import re
from typing import List, Dict, Any
from datetime import datetime
import requests

def find_max_map50(filename):

    max_map50 = -1.0
    max_line = ""
    max_line_number = -1
    
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if line and 'MAP50:' in line:
                try:

                    map50_part = line.split('MAP50:')[-1].strip()
                    map50_value = float(map50_part.split()[0])  
                    

                    if map50_value > max_map50:
                        max_map50 = map50_value
                        max_line = line
                        max_line_number = line_number
                        
                except (ValueError, IndexError) as e:
                    print(f"警告: 第 {line_number} 行解析错误: {e}")
                    continue
    
    return max_map50, max_line_number, max_line

def load_hyperparameters(filename: str) -> List[Dict[str, Any]]:

    combinations = []
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    if 'MAP50:' in line:
                        parts = line.split('MAP50:')
                        param_part = parts[0].strip()
                        map50_part = parts[1].strip()

                        try:
                            param_set = json.loads(param_part)
                            param_set['map50'] = float(map50_part)
                            combinations.append(param_set)
                        except (json.JSONDecodeError, ValueError) as e:
                            print(f"警告: 第 {line_num} 行解析错误: {e}")
                            continue
                    else:
                        try:
                            param_set = json.loads(line)
                            combinations.append(param_set)
                        except json.JSONDecodeError as e:
                            print(f"警告: 第 {line_num} 行JSON解析错误: {e}")
                            continue
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 不存在")
        sys.exit(1)

    return combinations


def extract_map50_from_output(output: str) -> float:

    patterns = [
        r"all\s+\d+\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)",  
        r"mAP50.*?(\d+\.\d+)",  
        r"MAP50.*?(\d+\.\d+)",  
    ]

    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return -1.0


def update_hyperparam_file(filename: str, index: int, map50: float):


    with open(filename, 'r') as f:
        lines = f.readlines()

    if index >= len(lines):
        print(f"错误: 索引 {index} 超出文件范围")
        return

    line = lines[index].strip()
    if 'MAP50:' in line:

        parts = line.split('MAP50:')
        new_line = f"{parts[0].strip()} MAP50: {map50:.4f}\n"
    else:

        new_line = f"{line} MAP50: {map50:.4f}\n"


    lines[index] = new_line

    with open(filename, 'w') as f:
        f.writelines(lines)

    print(f"已更新超参数文件: 第 {index + 1} 行 MAP50: {map50:.4f}")


def create_experiment_name(param_set: Dict[str, Any], index: int) -> str:


    name = f"exp_{index:03d}"


    if 'lr0' in param_set:
        name += f"_lr{param_set['lr0']:.0e}"
    if 'batch' in param_set:
        name += f"_bs{param_set['batch']}"
    if 'optimizer' in param_set:
        name += f"_{param_set['optimizer']}"
    if 'epochs' in param_set:
        name += f"_epo{param_set['epochs']}"
    if 'lrf' in param_set:
        name += f"_lrf{param_set['lrf']:.0e}"
    if 'dropout' in param_set:
        name += f"_do{param_set['dropout']}"
    if 'workers' in param_set:
        name += f"_wk{param_set['workers']}"
    if 'iou' in param_set:
        name += f"_iou{param_set['iou']}"
    if 'momentum' in param_set:
        name += f"_mom{param_set['momentum']}"
    if 'weight_decay' in param_set:
        name += f"_wd{param_set['weight_decay']:.0e}"
    if 'warmup_epochs' in param_set:
        name += f"_wep{param_set['warmup_epochs']}"
    if 'warmup_momentum' in param_set:
        name += f"_wmom{param_set['warmup_momentum']}"
    if 'warmup_bias_lr' in param_set:
        name += f"_wbl{param_set['warmup_bias_lr']:.0e}"
    if 'box' in param_set:
        name += f"_box{param_set['box']}"
    if 'cls' in param_set:
        name += f"_cls{param_set['cls']}"
    if 'dfl' in param_set:
        name += f"_dfl{param_set['dfl']}"
    if 'pose' in param_set:
        name += f"_pose{param_set['pose']}"
    if 'kobj' in param_set:
        name += f"_kobj{param_set['kobj']}"
    if 'nbs' in param_set:
        name += f"_nbs{param_set['nbs']}"
    if 'hsv_h' in param_set:
        name += f"_hsvh{param_set['hsv_h']}"
    if 'hsv_s' in param_set:
        name += f"_hsvs{param_set['hsv_s']}"
    if 'hsv_v' in param_set:
        name += f"_hsvv{param_set['hsv_v']}"
    if 'degrees' in param_set:
        name += f"_deg{param_set['degrees']}"
    if 'translate' in param_set:
        name += f"_trans{param_set['translate']}"

    return name


def build_train_command(param_set: Dict[str, Any], index: int) -> List[str]:

    command = [
        'python', 'train.py'  
    ]


    exp_name = create_experiment_name(param_set, index)



    command.extend(['--name', exp_name])

    filtered_params = {k: v for k, v in param_set.items() if k != 'map50'}
    for key, value in filtered_params.items():
        command.extend(['--' + key, str(value)])

    return command, exp_name



def run_training(param_sets: List[Dict[str, Any]], start_idx: int, end_idx: int, hyperparam_file: str):

    successful_runs = 0
    total_runs = end_idx - start_idx
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i in range(start_idx, end_idx):
        param_set = param_sets[i]
        has_map50 = 'map50' in param_set and param_set['map50'] >= 0

        print(f"\n{'=' * 60}")
        print(f"运行参数组合 {i + 1}/{len(param_sets)}")
        print(f"参数: {json.dumps(param_set, indent=2)}")
        if has_map50:
            print(f"已有MAP50: {param_set['map50']:.4f} (将跳过训练)")
        print(f"{'=' * 60}")


        if has_map50:
            successful_runs += 1
            continue

        command, exp_name = build_train_command(param_set, i)
        project_name = f"project : {timestamp}"
        command.extend(['--project', project_name])
        print(f"执行命令: {' '.join(command)}")

        try:

            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("训练完成!")


            map50 = extract_map50_from_output(result.stdout)

            if map50 >= 0:
                print(f"MAP50: {map50:.4f}")

                update_hyperparam_file(hyperparam_file, i, map50)
                successful_runs += 1
            else:
                print("警告: 未能从输出中提取MAP50值")
                print(f"输出: {result.stdout[-500:]}")  

        except subprocess.CalledProcessError as e:
            print(f"训练失败! 错误代码: {e.returncode}")
            print(f"标准输出: {e.stdout}")
            print(f"错误输出: {e.stderr}")
        except Exception as e:
            print(f"运行命令时发生错误: {e}")

    print(f"\n{'=' * 60}")
    print(f"所有训练完成! 成功: {successful_runs}/{total_runs}")
    print(f"{'=' * 60}")


def backup_hyperparam_file(filename: str):

    if os.path.exists(filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filename}.backup_{timestamp}"
        with open(filename, 'r') as src, open(backup_name, 'w') as dst:
            dst.write(src.read())
        print(f"已创建备份: {backup_name}")


def main():
    """
    Flask端点：接收超参数网格并生成所有组合

    期望的JSON格式:
    {
        "param_grid": {
            "lr0": {
                "type": "log", //实例中为对数搜索，如果要进行网格搜索则为linear；随机搜索则为random
                "min": 0.0001,
                "max": 0.1,
                "num_samples": 5//取样粒度，即在取值范围内取几个点
            },//连续型搜索变量的格式 
            "batch": [32, 64, 128],//离散型搜索变量的格式
            "optimizer": ["Adam", "SGD"]//离散型搜索变量的格式
        },
        "output_file": "hyperparameter.txt"      //这一行的文件名尽量不要修改
    }
    """

    url = "http://localhost:5000/grid_search"


    data = {
        "param_grid": {

            "batch": [32,64,128],  
            "optimizer": ["Adam","SGD"],  
            "epochs": [15,30]
        },
        "output_file": "hyperparams.txt"
    }
    response = requests.post(url, json=data)
    print(json.dumps(response.json(), indent=2))

    parser = argparse.ArgumentParser(description='YOLO超参数训练启动器')
    parser.add_argument('--hyperparam-file', type=str, default='hyperparams.txt',
                        help='超参数文件路径')
    parser.add_argument('--start-index', type=int, default=0,
                        help='从第几个参数组合开始训练 (默认: 0)')
    parser.add_argument('--end-index', type=int, default=None,
                        help='训练到第几个参数组合结束 (默认: 所有)')
    parser.add_argument('--no-backup', action='store_true',
                        help='不创建备份文件')

    args = parser.parse_args()


    if not args.no_backup:
        backup_hyperparam_file(args.hyperparam_file)


    print(f"加载超参数文件: {args.hyperparam_file}")
    param_sets = load_hyperparameters(args.hyperparam_file)

    if not param_sets:
        print("没有找到有效的超参数组合!")
        return

    print(f"找到 {len(param_sets)} 个超参数组合")


    start_idx = max(0, args.start_index)
    end_idx = args.end_index if args.end_index is not None else len(param_sets)
    end_idx = min(end_idx, len(param_sets))

    if start_idx >= end_idx:
        print("错误: 开始索引必须小于结束索引")
        return

    print(f"训练范围: 第 {start_idx} 到 {end_idx - 1} 个组合")


    skip_count = sum(1 for i in range(start_idx, end_idx) if 'map50' in param_sets[i])
    if skip_count > 0:
        print(f"将跳过 {skip_count} 个已有MAP50值的组合")


    response = input("是否开始训练? (y/n): ")
    if response.lower() != 'y':
        print("训练取消")
        return


    run_training(param_sets, start_idx, end_idx, args.hyperparam_file)

    filename = data['output_file']
    
    try:
        max_map50, line_number, line_content = find_max_map50(filename)
        
        if max_map50 >= 0:
            print(f"\n最高MAP50值: {max_map50:.4f}")
            print(f"实验组数: 第 {line_number-1} 组")
            print(f"对应参数: {line_content}")
        else:
            print("未找到有效的MAP50值")
            
    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 不存在")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == '__main__':
    main()