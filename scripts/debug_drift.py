from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import trossen_ai_isaac.tasks

env_cfg = parse_env_cfg('Isaac-Reach-MobileAI-IK-Rel-Play-v0', device='cuda:0', num_envs=1)
env = gym.make('Isaac-Reach-MobileAI-IK-Rel-Play-v0', cfg=env_cfg).unwrapped
env.reset()

robot = env.scene['robot']
left_idx = robot.find_bodies('follower_left_link_6')[0][0]

for i in range(30):
    action = torch.zeros(1, 12, device='cuda:0')
    env.step(action)
    pos = robot.data.body_pos_w[0, left_idx]
    if i % 5 == 0:
        print(f'Step {i:02d}: {pos.tolist()}')

env.close()
simulation_app.close()
