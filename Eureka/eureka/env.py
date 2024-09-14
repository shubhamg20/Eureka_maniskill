from mani_skill.envs.tasks.tabletop.push_cube import PushCube as ManiSkillPushCube
import numpy as np

class PushCubeEnv(ManiSkillPushCube):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_obs(self):
        obs = super()._get_obs()
        return {'observation': obs['agent']['qpos']}

    def reset(self, seed=None, reconfigure=False):
        super().reset(seed=seed, reconfigure=reconfigure)
        return self.get_obs()

    def step(self, action):
        _, _, _, info = super().step(action)
        obs = self.get_obs()
        reward, done = self.get_reward(action)
        return obs, reward, done, info
    
    def compute_reward(self, actions):
        self.rew_buf[:], self.rew_dict = compute_reward(self.object_pos, self.goal_pos, self.hand_pos)
        self.extras['gpt_reward'] = self.rew_buf.mean()
        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
        pass
from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(object_pos: torch.Tensor, goal_pos: torch.Tensor, hand_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute distances
    distance_to_goal = torch.norm(object_pos - goal_pos, dim=-1)
    distance_hand_to_object = torch.norm(hand_pos - object_pos, dim=-1)

    # Reward for getting the object closer to the goal
    goal_distance_reward_temp = 1.0
    goal_distance_reward = torch.exp(-goal_distance_reward_temp * distance_to_goal)

    # Reward for hand being close to the object
    hand_distance_reward_temp = 0.5
    hand_distance_reward = torch.exp(-hand_distance_reward_temp * distance_hand_to_object)

    # Total reward
    total_reward = goal_distance_reward + hand_distance_reward

    # Reward components dictionary
    reward_components = {
        'goal_distance_reward': goal_distance_reward,
        'hand_distance_reward': hand_distance_reward
    }

    return total_reward, reward_components
