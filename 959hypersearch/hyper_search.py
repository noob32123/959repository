from flask import Flask, request, jsonify
import itertools
import json
import numpy as np
from typing import Dict, List, Any, Union, Tuple

app = Flask(__name__)


def sample_continuous_parameter(param_spec: Union[List[Any], Dict[str, Any]]) -> List[Any]:

    if isinstance(param_spec, list):

        return param_spec

    elif isinstance(param_spec, dict):
        param_type = param_spec.get('type', 'linear')
        min_val = param_spec['min']
        max_val = param_spec['max']
        num_samples = param_spec.get('num_samples', 10)

        if param_type == 'linear':

            return np.linspace(min_val, max_val, num_samples).tolist()

        elif param_type == 'log':

            return np.logspace(np.log10(min_val), np.log10(max_val), num_samples).tolist()

        elif param_type == 'random':

            return np.random.uniform(min_val, max_val, num_samples).tolist()

        else:
            raise ValueError(f"不支持的采样类型: {param_type}")

    else:
        raise ValueError("参数规格必须是列表或字典")


def generate_grid_combinations(param_grid: Dict[str, Any]) -> List[Dict[str, Any]]:

    discrete_param_grid = {}

    for param_name, param_values in param_grid.items():
        discrete_param_grid[param_name] = sample_continuous_parameter(param_values)

    keys = discrete_param_grid.keys()
    values = discrete_param_grid.values()
    combinations = list(itertools.product(*values))

    return [dict(zip(keys, combo)) for combo in combinations]


def save_combinations_to_file(combinations: List[Dict[str, Any]], filename: str) -> None:

    with open(filename, 'w') as f:
        for i, combo in enumerate(combinations):
            f.write(json.dumps(combo))
            if i < len(combinations) - 1:
                f.write('\n')


@app.route('/grid_search', methods=['POST'])
def grid_search():
    """
    Flask端点：接收超参数网格并生成所有组合

    期望的JSON格式:
    {
        "param_grid": {
            "lr0": {
                "type": "log",
                "min": 0.0001,
                "max": 0.1,
                "num_samples": 5
            },
            "batch": [32, 64, 128],
            "optimizer": ["Adam", "SGD"]
        },
        "output_file": "hyperparameter.txt"
    }
    """
    try:
        data = request.get_json()

        if not data or 'param_grid' not in data:
            return jsonify({"error": "缺少必需的参数: param_grid"}), 400

        param_grid = data['param_grid']
        output_file = data.get('output_file', 'hyperparameter_combinations.txt')

        combinations = generate_grid_combinations(param_grid)

        save_combinations_to_file(combinations, output_file)

        response = {
            "message": "参数组合生成成功",
            "total_combinations": len(combinations),
            "output_file": output_file,
            "sample_combinations": combinations[:3]
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"服务器错误: {str(e)}"}), 500




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)