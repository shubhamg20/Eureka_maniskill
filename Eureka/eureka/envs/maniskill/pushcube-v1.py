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
        pass