# Copyright 2026 Trossen Robotics & Collaborators
"""
Custom Teleoperation and Episode Recording Script using DualShock 4 Gamepad.
Loads 'Virtual_Env.usd', randomizes object colors, and saves training data.
"""

import argparse
import os
import random
import numpy as np
import torch
import h5py

from isaaclab.app import AppLauncher

# 1. Setup CLI Arguments
parser = argparse.ArgumentParser(description="Teleoperate your custom Trossen scene and record episodes.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (Keep 1 for manual teleop).")
parser.add_argument("--output_dir", type=str, default="./recorded_episodes", help="Directory where recorded HDF5 episodes are saved.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the Omniverse Application
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# 2. Post-Launcher Imports (Must happen after AppLauncher)
import gymnasium as gym
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.materials import PreviewSurfaceMaterial
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.sim import SimulationCfg, SceneCfg
from isaaclab.devices import Se3Gamepad, Se3GamepadCfg
from isaaclab.utils import configclass

# 3. Define the Custom Scene Configuration using your USD
@configclass
class CustomSceneCfg(SceneCfg):
    """Configuration for your specific Virtual Environment USD layout."""
    def __init__(self):
        super().__init__()
        # Points Isaac Lab directly to your saved stage asset
        self.usd_path = "home/trossen_ai_isaac/assets/robots/mobile_ai/Virtual_Env.usd"

@configclass
class CustomEnvCfg(ManagerBasedEnvCfg):
    """Basic environment configuration wrapping your custom USD scene."""
    def __init__(self):
        super().__init__()
        self.env_name = "Trossen-Custom-Pick-Place-v0"
        self.sim = SimulationCfg(dt=0.01)
        self.scene = CustomSceneCfg()
        
        # Disable default timeouts so you can take your time recording
        self.terminations.time_out = None

# 4. Helper Functions for Color Manipulation and Data Storage
def randomize_cube_color():
    """Finds the cube primitive within your USD stage structure and shifts its color."""
    colors = {
        "RED": [1.0, 0.0, 0.0],
        "GREEN": [0.0, 1.0, 0.0],
        "BLUE": [0.0, 0.0, 1.0]
    }
    chosen_name, chosen_rgb = random.choice(list(colors.items()))
    
    # Common Isaac Lab object target path template. 
    # NOTE: If your cube inside Virtual_Env.usd has a specific name/path (e.g., /World/Cube), update this string!
    cube_prim_path = "/World/Cube" 
    
    if prim_utils.is_prim_path_valid(cube_prim_path):
        mat_path = f"{cube_prim_path}/looks/cube_material"
        visual_mat = PreviewSurfaceMaterial(prim_path=mat_path, diffuse_color=np.array(chosen_rgb))
        prim_utils.apply_material(cube_prim_path, visual_mat.prim)
        print(f"[RESET] Environment reloaded. Target cube color set to: {chosen_name}")
    else:
        print(f"[Warning] Could not find a cube prim at '{cube_prim_path}'. Check your USD tree hierarchy.")

def save_episode(output_dir, episode_idx, states, actions):
    """Saves telemetry arrays into a structured HDF5 file format for AI training."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"episode_{episode_idx:04d}.h5")
    
    with h5py.File(file_path, "w") as f:
        f.create_dataset("states", data=np.array(states), compression="gzip")
        f.create_dataset("actions", data=np.array(actions), compression="gzip")
    print(f"\n[DATA SAVED] Saved Episode {episode_idx} ({len(actions)} steps) -> {file_path}")

# 5. Main Control Loop
def main():
    env_cfg = CustomEnvCfg()
    
    # Create the environment using our custom USD definition
    env = ManagerBasedEnv(cfg=env_cfg)
    
    # Initialize Isaac Lab's built-in Gamepad interface (Supports standard DualShock 4 mappings)
    teleop_interface = Se3Gamepad(Se3GamepadCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    
    # Data buffers
    episode_idx = 0
    state_buffer = []
    action_buffer = []
    
    print("\n============================================================")
    print("CONTROL LAYOUT (DualShock 4 via Se3Gamepad):")
    print(" - Left Joystick: Move arm X / Y (Forward/Backward/Sides)")
    print(" - L2 / R2 Triggers: Move arm Z (Up / Down)")
    print(" - Right Joystick: Control gripper wrist orientation")
    print(" - L1 / R1 Buttons: Open / Close Gripper")
    print(" - Keyboard [R] key: Save current episode data & reset scene")
    print("============================================================\n")

    env.reset()
    randomize_cube_color()
    teleop_interface.reset()

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Read incoming actions from your connected DS4 Gamepad
                action_delta = teleop_interface.advance()
                
                # Format to match environment expectations
                actions = action_delta.repeat(env.num_envs, 1)
                
                # Step physics engine forward
                obs, reward, terminated, truncated, info = env.step(actions)
                
                # Log telemetry to memory buffers
                # You can customize exactly what tensor values from 'obs' you want to save here
                state_buffer.append(obs["policy"].cpu().numpy().squeeze())
                action_buffer.append(action_delta.cpu().numpy().squeeze())
                
                # Check if a manual or automatic reset command was issued (e.g., pressing 'R')
                if terminated[0] or truncated[0]:
                    if len(action_buffer) > 10:  # Avoid saving empty or accidental taps
                        save_episode(args_cli.output_dir, episode_idx, state_buffer, action_buffer)
                        episode_idx += 1
                    
                    # Flush data buffers for the next demonstration run
                    state_buffer, action_buffer = [], []
                    
                    env.reset()
                    randomize_cube_color()
                    teleop_interface.reset()
                    
    except KeyboardInterrupt:
        print("\nExiting and cleaning up simulation environment safely...")
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
