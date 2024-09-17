import gymnasium as gym
import mani_skill.envs
from envs.maniskill.pushcubegpt import PushCubeGPTEnv
env = gym.make("PushCubeGPT")