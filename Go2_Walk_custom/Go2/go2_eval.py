import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import math

from go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, nargs="+", default=[100])
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # Create visualization offsets to place robots side-by-side
    num_envs = len(args.ckpt)
    env_cfg["visualization_offsets"] = [(i * 2.0, 0, 0) for i in range(num_envs)]

    # Rotate robots by -90 degrees (yaw) to face the +Y axis for forward movement
    env_cfg["base_init_quat"] = [0.7071, 0.0, 0.0, -0.7071]  # (w, x, y, z) for -90 deg yaw

    # Adjust camera position to view all robots
    if num_envs > 0:
        center_x = (num_envs - 1) * 2.0 / 2.0  # Center X-coordinate of the robot span

        # Default camera settings for a single robot or when span_x is small
        camera_pos_y_base = -6.0  # A good default distance for 1 robot
        camera_pos_z_base = 3.5   # A good default height for 1 robot
        
        # Default FOV (can be overridden by env_cfg if it exists)
        camera_fov = env_cfg.get("viewer_options", {}).get("camera_fov", 40)

        if num_envs > 1:
            span_x_width = (num_envs - 1) * 2.0
            fov_half_rad = math.radians(camera_fov / 2.0)
            
            # Calculate required distance along y-axis to see the full span_x
            # Add a buffer to ensure elements at the very edge are fully visible.
            # Cover 20% more than the actual span for comfortable view
            view_coverage_width = span_x_width * 1.2 
            
            if fov_half_rad != 0:
                required_distance_y = (view_coverage_width / 2.0) / math.tan(fov_half_rad)
                camera_pos_y = -required_distance_y
            else: # Fallback for 0 FOV, though unlikely
                camera_pos_y = -100.0 # Just place it very far back
            
            # Scale Z with num_envs for larger spans
            camera_pos_z = camera_pos_z_base + (num_envs * 0.7) 
        else: # num_envs == 1
            camera_pos_y = camera_pos_y_base
            camera_pos_z = camera_pos_z_base
            
        env_cfg["viewer_options"] = {
            "camera_pos": (center_x, camera_pos_y, camera_pos_z),
            "camera_lookat": (center_x, 2.5, 0.5), # Look at a point 2.5m ahead of the robots' path
            "camera_fov": camera_fov,
            "max_FPS": 60,
        }

    env = Go2Env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Load policies
    policies = []
    for ckpt in args.ckpt:
        # Create a new ActorCritic instance for each policy
        policy_cfg = train_cfg["policy"]
        policy = ActorCritic(
            num_actor_obs=env.num_obs,
            num_critic_obs=env.num_obs,
            num_actions=env.num_actions,
            actor_hidden_dims=policy_cfg["actor_hidden_dims"],
            critic_hidden_dims=policy_cfg["critic_hidden_dims"],
            activation=policy_cfg["activation"],
            init_noise_std=policy_cfg["init_noise_std"],
        )
        policy.to(gs.device)
        policy.eval()

        # Load the state dict from the checkpoint file
        resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
        print(f"Loading policy from: {resume_path}")
        loaded_dict = torch.load(resume_path, map_location=gs.device)
        policy.load_state_dict(loaded_dict["model_state_dict"])
        policies.append(policy)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = torch.empty((num_envs, env.num_actions), device=gs.device)
            for i in range(num_envs):
                actions[i, :] = policies[i].act_inference(obs[i].unsqueeze(0))
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
