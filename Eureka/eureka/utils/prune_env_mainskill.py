import os
import sys
import ast
import astunparse
import yaml

def modify_python_file(reward_name, input_file, output_file):
    with open(input_file, 'r') as file:
        tree = ast.parse(file.read())

    class RewardVisitor(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'compute_reward':
                return None
            return node

    new_tree = RewardVisitor().visit(tree)
    
    with open(output_file, 'w') as file:
        file.write(astunparse.unparse(new_tree))

def prune_python_class(input_file, output_file, methods_to_keep):
    with open(input_file, 'r') as file:
        tree = ast.parse(file.read())

    class Pruner(ast.NodeTransformer):
        def visit_ClassDef(self, node):
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods_to_keep]
            return node

    new_tree = Pruner().visit(tree)
    
    with open(output_file, 'w') as file:
        file.write(astunparse.unparse(new_tree))

def modify_yaml_file(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    if 'env' in data and 'env_args' in data['env']:
        data['env']['env_args']['obs_mode'] = 'state'
    
    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

if __name__ == "__main__":
    EUREKA_ROOT_DIR = os.getcwd()
    MANISKILL_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../../ManiSkill/mani_skill"

    tasks = [
        "push_cube"#,
        # "shadow_hand_pen",
        # "shadow_hand_door_close_outward",
        # "shadow_hand_door_open_inward",
        # "shadow_hand_push_block",
        # "shadow_hand_catch_underarm",
        # "shadow_hand_catch_abreast",
        # "shadow_hand_lift_underarm",
        # "shadow_hand_kettle"
    ]

    for task in tasks:
        # Create base environment file to write reward function for
        modify_python_file(task, f"{MANISKILL_ROOT_DIR}/envs/tasks/{task}.py", f"../envs/maniskill/{task}.py")
        
        # Create a condensed version to serve as input to Eureka
        prune_python_class(f"../envs/maniskill/{task}.py", f"../envs/maniskill/{task}_obs.py", 
                        ["compute_observations", "_update_states", "compute_reward"])
        
        # Modify the corresponding YAML file
        modify_yaml_file(f"{EUREKA_ROOT_DIR}/cfg/env/{task}.yaml")

    print("ManiSkill environment pruning completed.")