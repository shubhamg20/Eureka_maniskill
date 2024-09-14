import logging
import os
import datetime
import tyro
from omegaconf import DictConfig, OmegaConf
import sys, random
import shutil
from pathlib import Path
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo.ppo import PPO 
from ppo.agent import Args, Agent
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import mani_skill.envs   #imp
import gymnasium as gym  #imp
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# def preprocess_train_config(config_dict, args):
#     """
#     Adding common configuration parameters to the rl_games train config.
#     An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
#     variable interpolations in each config.
#     """

#     train_cfg = config_dict['params']['config']
#     train_cfg['full_experiment_name'] = args.get('full_experiment_name')

#     try:
#         model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
#         if model_size_multiplier != 1:
#             units = config_dict['params']['network']['mlp']['units']
#             for i, u in enumerate(units):
#                 units[i] = u * model_size_multiplier
#             print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
#     except KeyError:
#         pass

#     return config_dict


def launch_rlg(args):
    print("hii")
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper


    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # run_name = f"{args.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if args.checkpoint:
        args.checkpoint = to_absolute_path(args.checkpoint)

    # print_dict(cfg_dict)

    # sets seed. if seed is -1 will pick a random one
    # rank = int(os.getenv("LOCAL_RANK", "0"))
    # args.seed += rank
    # args.seed = set_seed(args.seed, torch_deterministic=args.torch_deterministic)
    # args.train.params.config.multi_gpu = args.multi_gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    env_kwargs = dict(obs_mode="rgb", control_mode="pd_joint_delta_pos", render_mode=args.render_mode, sim_backend="gpu")
    envs = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    # if args.capture_video:
    #     envs.is_vector_env = True
    #     if args.test:
    #         envs = gym.wrappers.RecordVideo(
    #             envs,
    #             f"videos/{run_name}",
    #             step_trigger=lambda step: (step % args.capture_video_freq == 0),
    #             video_length=args.capture_video_len,
    #         )
    #     else:
    #         envs = gym.wrappers.RecordVideo(
    #             envs,
    #             f"videos/{run_name}",
    #             step_trigger=lambda step: (step % args.capture_video_freq == 0) and (step > 0),
    #             video_length=args.capture_video_len,
    #         )
    

    # envs = create_maniskill_env()
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': envs,
    })
    
    # Save the environment code!
    try:
        # output_file = f"{ROOT_DIR}/tasks/{args.task.env.env_name.lower()}.py"
        output_file = f"{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}{suffix.lower()}.py"
        shutil.copy(output_file, f"env.py")
    except:
        import re
        print("Error copying in env.py")
        def camel_to_snake(name):
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        # output_file = f"{ROOT_DIR}/tasks/{camel_to_snake(args.task.name)}.py"
        # output_file = f"{ROOT_DIR}/tasks/{camel_to_snake(args.task.name)}.py"

        # shutil.copy(output_file, f"env.py")

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    # register new AMP network builder and agent

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # Create the agent
    next_obs, _ = envs.reset(seed=args.seed)
    agent = Agent(envs, sample_obs=next_obs).to(device)
    
    # Create the optimizer
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Create the PPO algorithm
    ppo = PPO(
        agent=agent,
        optimizer=optimizer,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        epochs=args.update_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        norm_adv=args.norm_adv,
        clip_vloss=args.clip_vloss,
        reward_scale=args.reward_scale,
        device=device
    )
    
     # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    eps_returns = torch.zeros(args.num_envs, dtype=torch.float, device=device)
    eps_lens = np.zeros(args.num_envs)
    place_rew = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")

    print("Done")
        
if __name__ == "__main__":
    args = tyro.cli(Args)
    print("enter maniskill_train")
    launch_rlg(args)
