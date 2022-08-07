import cv2
import numpy as np
from scipy.special import rel_entr

from simulation.utils.utils import make_pdf, chw_to_hwc, project_to_target_direction


class Reward:
    """Simplified reward function reinforcing moving objects along a target direction."""

    def __init__(self, robot, config):
        self.physics = robot
        self.config = config

    def __call__(self, obs, new_obs, init_obj_pos, final_obj_pos, target_dir, gripper_open, controls):
        return self.agent_reward(init_obj_pos, final_obj_pos, target_dir, gripper_open, controls)

    def agent_reward(self, init_obj_pos, final_obj_pos, target_dir, gripper_open, controls):
        """Reward function for the agent."""
        reward = 0.
        # project the object position onto the target direction before and after the action
        # compute projection scalars
        init_obj_proj = project_to_target_direction(init_obj_pos[:2], target_dir)
        final_obj_proj = project_to_target_direction(final_obj_pos[:2], target_dir)

        dist_to_target_vector = np.linalg.norm(final_obj_proj * target_dir - final_obj_pos[:2])

        # dist_travel_obj = np.linalg.norm(final_obj_proj * target_dir - init_obj_proj * target_dir)
        dist_travel_obj = final_obj_proj - init_obj_proj

        if (dist_travel_obj > 0.) and (dist_travel_obj < 0.1) and (dist_to_target_vector < 0.1):
            reward = dist_travel_obj
            if not gripper_open and np.all(controls) != 0:
                reward = reward * 2
                if final_obj_pos[2] > 0:
                    reward = reward * 1.5

        #  time penalty
        # reward -= 0.01

        return reward * 100  # values are in meters and, therefore, very small


class IntrinsicReward(Reward):
    """Add intrinsic reward to encourage exploration."""
    def __init__(self, robot, config):
        super().__init__(robot, config)
        self.physics = robot
        self.config = config

    def __call__(self, obs, new_obs, init_obj_pos, final_obj_pos, target_dir, gripper_open, controls):
        agent_reward = self.agent_reward(init_obj_pos, final_obj_pos, target_dir, gripper_open, controls)
        intrinsic_reward = self.intrinsic_reward(obs, new_obs)

        return agent_reward + intrinsic_reward

    def intrinsic_reward(self, obs, new_obs):
        """Intrinsic reward function."""

        obs = chw_to_hwc(obs)
        new_obs = chw_to_hwc(new_obs)

        rgb_obs_pdf = make_pdf(cv2.cvtColor(obs[..., :3], cv2.COLOR_BGR2GRAY))
        rgb_new_obs_pdf = make_pdf(cv2.cvtColor(new_obs[..., :3], cv2.COLOR_BGR2GRAY))

        kl_div = rel_entr(rgb_obs_pdf, rgb_new_obs_pdf)
        kl_div[np.isinf(kl_div)] = 0.
        reward = sum(kl_div)

        if self.config.full_observation:
            depth_obs_pdf = make_pdf(obs[..., 3])
            depth_new_obs_pdf = make_pdf(new_obs[..., 3])
            kl_div_depth = rel_entr(depth_obs_pdf, depth_new_obs_pdf)
            kl_div_depth[np.isinf(kl_div_depth)] = 0.
            reward = (reward + sum(kl_div_depth))/2

        return float(reward)
