import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os, sys
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time 
from utils.misc import * 
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import envs
EUREKA_ROOT_DIR = os.getcwd()

def filter_traceback(stdout_str):
    """
    Filters the traceback message from the standard output string.

    Args:
        stdout_str (str): The standard output string.

    Returns:
        str: The filtered traceback message.
    """
    traceback_msg = ""
    traceback_start = False
    lines = stdout_str.split("\n")
    for line in lines:
        if "Traceback" in line:
            traceback_start = True
        if traceback_start:
            traceback_msg += line + "\n"
    return traceback_msg


def load_tensorboard_logs(logdir):
    """
    Loads the TensorBoard logs from the specified directory.

    Args:
        logdir (str): The directory containing the TensorBoard logs.

    Returns:
        dict: A dictionary containing the TensorBoard logs.
    """
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    tensorboard_logs = {}
    for tag in event_acc.Tags()["scalars"]:
        tensorboard_logs[tag] = [
            event.value for event in event_acc.Scalars(tag)
        ]
    return tensorboard_logs


def block_until_training(filepath, log_status=False, iter_num=None, response_id=None):
    """
    Blocks until the training is finished.

    Args:
        filepath (str): The path to the output file.
        log_status (bool, optional): Whether to log the status. Defaults to False.
        iter_num (int, optional): The iteration number. Defaults to None.
        response_id (int, optional): The response ID. Defaults to None.
    """
    while True:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                content = f.read()
                print(content)
                if "Finished training." in content:
                    if log_status:
                        logging.info(f"Iteration {iter_num}, Response {response_id}: Training finished.")
                    break
        else:
            if log_status:
                logging.info(f"Iteration {iter_num}, Response {response_id}: Waiting for training to start...")


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")    #your openAI API Key

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.task.lower()
    env_parent = "maniskill"
    print(env_name)
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
        
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string  = file_to_string(task_file)
    task_obs_code_string  = file_to_string(task_obs_file)
    output_file = f"{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    task_code_string = task_code_string.replace(task, task+suffix)
    # Create Task YAML files
    # create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in model else 4

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=cfg.temperature,
                        n=chunk_size
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur["choices"])
            prompt_tokens = response_cur["usage"]["prompt_tokens"]
            total_completion_token += response_cur["usage"]["completion_tokens"]
            total_token += response_cur["usage"]["total_tokens"]

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        code_runs = [] 
        rl_runs = []
        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    
            # Add the Eureka Reward Signature to the environment code
            try:
                gpt_reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                continue

            code_runs.append(code_string)
            print(gpt_reward_signature)
            print(input_lst)
            reward_signature = [
                f"self.update_states()",
                f"reward, self.rew_dict = {gpt_reward_signature}",
                # f"self.extras['gpt_reward'] = reward",
                # f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
            ]
            indent = " " * 8
            reward_signature = "\n".join([indent + line for line in reward_signature])
            # print(task_code_string)
            if "def compute_dense_reward(self, obs: Any, action: Array, info: Dict):" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_dense_reward(self, obs: Any, action: Array, info: Dict):", "def compute_dense_reward(self, obs: Any, action: Array, info: Dict):\n" + reward_signature)
            else:
                raise NotImplementedError

            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                file.writelines(task_code_string_iter + '\n')
                file.writelines("from typing import Tuple, Dict" + '\n')
                file.writelines("import math" + '\n')
                file.writelines("import torch" + '\n')
                file.writelines("from torch import Tensor" + '\n')
                if "@torch.jit.script" not in code_string:
                    code_string = "@torch.jit.script\n" + code_string
                file.writelines(code_string + '\n')

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            # Find the freest GPU to run GPU-accelerated RL
            set_freest_gpu()
            
            # Execute the python file with flags
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            print("Training loop")
            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen(['python3', '-u', f'{EUREKA_ROOT_DIR}/maniskill_train.py',  
                                            # 'hydra/output=subprocess',
                                            # f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                            # f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                            # f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                            # f'max_iterations={cfg.max_iterations}'
                                            ],
                                            stdout=f, stderr=f)   
            block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            print("out training")
            rl_runs.append(process)



if __name__ == "__main__":
    main()