import numpy as np
from simulation.environment.robot_env import RobotEnv


class Reward:
    """Simplified reward function reinforcing moving objects along a target direction."""

    def __init__(self, robot, config):
        self.physics = robot
        self.config = config

    def __call__(self, obs, new_obs, init_obj_pos, final_obj_pos):
        return self.agent_reward(init_obj_pos, final_obj_pos), RobotEnv.Status.RUNNING

    def agent_reward(self, init_obj_pos, final_obj_pos):
        """Reward function for the agent."""
        reward = 0.

        target_dir = self.physics.target_direction

        init_obj_proj = (np.dot(init_obj_pos, target_dir) / np.linalg.norm(target_dir) ** 2) * target_dir
        final_obj_proj = (np.dot(final_obj_pos, target_dir) / np.linalg.norm(target_dir) ** 2) * target_dir
        dist_to_target_vector = np.linalg.norm(init_obj_proj - final_obj_pos)

        if (final_obj_proj - init_obj_proj > 0.) and (dist_to_target_vector < 0.02):
            reward = np.linalg.norm(final_obj_pos * target_dir - init_obj_pos * target_dir)

        return reward * 100  # values are in meters and, therefore, very little


class IntrinsicReward(Reward):
    """Add intrinsic reward to encourage exploration."""
    def __init__(self, robot, config):
        super().__init__(robot, config)
        self.physics = robot
        self.config = config

    def __call__(self, obs, new_obs, init_obj_pos, final_obj_pos):
        agent_reward = self.agent_reward(init_obj_pos, final_obj_pos)
        intrinsic_reward = self.intrinsic_reward(obs, new_obs)

        return agent_reward + intrinsic_reward, RobotEnv.Status.RUNNING

    def intrinsic_reward(self, obs, new_obs):
        """Intrinsic reward function."""
        reward = 0.
        rgb_obs = obs[..., :3]
        rgb_new_obs = new_obs[..., :3]

        if self.config.full_observation:
            depth_obs = obs[..., 3]
            depth_new_obs = new_obs[..., 3]


        return reward, RobotEnv.Status.RUNNING

# if the object is very far away from the target direction or gripper status is fail
# no more boundaries on the workspace
