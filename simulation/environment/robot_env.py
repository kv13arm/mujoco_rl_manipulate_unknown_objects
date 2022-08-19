import cv2
import gym
import time
import math
import numpy as np
from enum import Enum
from pathlib import Path
from gym.utils import seeding
from dm_control import mujoco
from simulation.controller.sensor import RGBDSensor
from simulation.controller.actuator import Actuator
from simulation.environment.reward import Reward, IntrinsicReward
from simulation.utils.utils import hwc_to_chw, project_to_target_direction


class RobotEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array', 'depth_array']}

    class Status(Enum):
        RUNNING = 0
        FAIL = 1
        TIME_LIMIT = 2

    def __init__(self, config):
        self.config = config
        self.physics = mujoco.Physics.from_xml_path(Path(__file__).resolve().parent.parent.parent.as_posix() + config.sim_env)

        self.np_random = self.seed()
        # self.target_direction = self._get_direction()
        if config.direction == 0:
            self.target_direction = np.array([1, 0])
        if config.direction == 45:
            self.target_direction = np.array([1, 1])

        self.obs = dict()

        self._sensor = RGBDSensor(robot=self.physics, config=config)
        self._actuator = Actuator(robot=self.physics, config=config)
        if not self.config.im_reward:
            self._reward_fn = Reward(robot=self.physics, config=config)
        else:
            self._reward_fn = IntrinsicReward(robot=self.physics, config=config)

        self.setup_spaces()

    def _get_direction(self):
        """
        Return the target direction vector.
        :return: [np.array] the target direction vector
        """
        # generate a random angle for the target direction vector
        theta = np.round(self.np_random.uniform(0, 2 * np.pi), 2)

        return np.array([np.cos(theta), np.sin(theta)])

    def reset(self):
        """
        Reset the environment.
        :return: the initial observation
        """
        # constant actuator signal
        self.physics.reset()
        # gravity compensation
        mg = -(0.438 * self.physics.model.opt.gravity[2])
        self.physics.named.data.xfrc_applied["ee", 2] = mg

        self.episode_step = 0
        self.episode_rewards = np.zeros(self.config.time_horizon)
        self.status = RobotEnv.Status.RUNNING
        self.obs["observation"] = self.get_observation()
        self.obs["achieved_goal"] = self.physics.named.data.xpos["object"][:2].copy().astype(np.float32)
        self.obs["desired_goal"] = self.target_direction.copy().astype(np.float32)
        self.gripper_open = True

        return self.obs

    def step(self, action):
        """
        Perform a step in the environment.
        :param action:
        :return:
        """
        pos_reached = {"target": False,
                       "initial": False,
                       "fail": False}

        init_obj_pos = self.physics.named.data.xpos["object"].copy()
        total_distance = None

        init_qpos = self.physics.data.qpos[:5].copy()
        open_close = action[-1]

        target_qpos = self._actuator.get_target_pose(action)

        step_limit = self.config.max_steps

        for i in range(self.config.max_steps):
            current_qpos = self.physics.data.qpos[:5]
            self.physics.data.ctrl[0:5] = self._actuator.scale_control(target_qpos - current_qpos, open_close)
            self.physics.step()
            if self.config.show_obs:
                self.render()
            # print("Iteration: ", i)
            step_limit -= 1
            deltas = abs(current_qpos - target_qpos)
            # print("qpos difference:", deltas)
            if max(deltas) < self.config.pos_tolerance:
                pos_reached["target"] = True
                self.physics.data.ctrl[0:5] = 0
                break

        if step_limit == 0:
            # print("Target not reached. Returning to initial position.")
            target_qpos = init_qpos[:5]

            for i in range(self.config.max_steps):
                current_qpos = self.physics.data.qpos[:5]
                self.physics.data.ctrl[0:5] = self._actuator.scale_control(target_qpos - current_qpos, open_close)
                self.physics.step()
                if self.config.show_obs:
                    self.render()

                deltas = abs(current_qpos - target_qpos)
                # print("qpos difference:", deltas)
                if max(deltas) < self.config.pos_tolerance:
                    pos_reached["initial"] = True
                    self.physics.data.ctrl[0:5] = 0
                    break

        if not pos_reached["target"] and not pos_reached["initial"]:
            pos_reached["fail"] = True
            self.status = RobotEnv.Status.FAIL

        object_grasped = 0

        if pos_reached["target"]:
            if open_close > 0. and not self.gripper_open:
                target_qpos = self._actuator.open_gripper()
                for i in range(self.config.max_steps):
                    deltas = abs(target_qpos - self.physics.data.qpos[5:7])
                    # print(deltas)
                    self.physics.step()
                    if self.config.show_obs:
                        self.render()
                    if (max(deltas) < self.config.grasp_tolerance) or np.all(self.physics.data.qpos[5:7] > target_qpos):
                        self.physics.data.ctrl[5:7] = 0
                        self.gripper_open = True
                        break
                self.physics.data.ctrl[5:7] = 0
            elif open_close < 0. and self.gripper_open:
                target_qpos = self._actuator.close_gripper()
                for i in range(self.config.max_steps):
                    deltas = abs(target_qpos - self.physics.data.qpos[5:7])
                    # check if the object is securely grasped
                    object_grasped = self._actuator.check_grasp("object")
                    # print("Object grasped: ", object_grasped == 3)
                    self.physics.step()
                    if self.config.show_obs:
                        self.render()
                    if max(deltas) < self.config.grasp_tolerance:
                        self.physics.data.ctrl[5:7] = 0
                        self.gripper_open = False
                        break

                    if object_grasped == 3:
                        self.gripper_open = False
                        break
                self.physics.data.ctrl[5:7] = 0

        final_obj_pos = self.physics.named.data.xpos["object"]
        final_gripper_pos = self.physics.named.data.xpos["ee"]
        if np.linalg.norm(final_obj_pos[:2] - final_gripper_pos[:2]) > 1.:
            self.status = RobotEnv.Status.FAIL

        target_dir = self.target_direction
        self.obs["desired_goal"] = (project_to_target_direction(final_obj_pos[:2], target_dir)
                                    * target_dir).astype(np.float32)
        self.obs["achieved_goal"] = final_obj_pos[:2].astype(np.float32)

        # print("Final position: ", self.physics.named.data.xpos["ee"])
        # # print("Target position: ", target_pos)
        # print("Final position: ", self.physics.named.data.xquat["ee"])
        # # print("Target position: ", target_ori)
        # print("Position reached: ", pos_reached)

        new_obs = self.get_observation()
        old_obs = self.obs["observation"].copy()
        reward = self.compute_reward(self.obs["achieved_goal"],
                                     self.obs["desired_goal"],
                                     {"old_obs": old_obs,
                                      "new_obs": new_obs,
                                      "init_obj_pos": init_obj_pos,
                                      "final_obj_pos": final_obj_pos,
                                      "target_dir": target_dir,
                                      "gripper_open": self.gripper_open,
                                      "controls": self.physics.data.ctrl[5:7],
                                      "object_grasped": object_grasped})

        self.episode_rewards[self.episode_step] = reward

        if self.status != RobotEnv.Status.RUNNING:
            done = True
        elif self.episode_step == self.config.time_horizon - 1:
            done, self.status = True, RobotEnv.Status.TIME_LIMIT
        else:
            done = False

        # if done:
        total_distance = np.linalg.norm(final_obj_pos[:2] - init_obj_pos[:2])

        # project the object position onto the target direction before and after the action
        # compute projection scalars
        init_obj_proj = project_to_target_direction(init_obj_pos[:2], target_dir)
        final_obj_proj = project_to_target_direction(final_obj_pos[:2], target_dir)
        dist_to_target_vector = np.linalg.norm(final_obj_proj * target_dir - final_obj_pos[:2])
        dist_travel_obj = final_obj_proj - init_obj_proj
        if (dist_travel_obj > 0.) and (dist_travel_obj < 0.1) and (dist_to_target_vector < 0.1):
            line_distance = dist_travel_obj
        else:
            line_distance = 0.

        self.episode_step += 1
        self.obs["observation"] = new_obs
        # print(f"S: {self.episode_step}, O: {self.gripper_open}, O/C:{open_close > 0.}")

        return self.obs, reward, done, {"old_obs": old_obs,
                                        "new_obs": new_obs,
                                        "init_obj_pos": init_obj_pos,
                                        "final_obj_pos": final_obj_pos,
                                        "target_dir": target_dir,
                                        "gripper_open": self.gripper_open,
                                        "controls": self.physics.data.ctrl[5:7],
                                        "object_grasped": object_grasped,
                                        "episode_step": self.episode_step,
                                        "episode_rewards": self.episode_rewards,
                                        "status": self.status,
                                        "gripper_position": final_gripper_pos,
                                        "object_position": final_obj_pos,
                                        "position_reached": pos_reached,
                                        "total_distance": total_distance,
                                        "line_distance": line_distance}

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the reward for the given achieved goal and desired goal.
        """
        args = info
        # check for HER implementation as it saves info as a np.array
        if isinstance(args, np.ndarray):
            env_reward = self._reward_fn(args[0]["old_obs"],
                                         args[0]["new_obs"],
                                         args[0]["init_obj_pos"],
                                         args[0]["final_obj_pos"],
                                         args[0]["target_dir"],
                                         args[0]["gripper_open"],
                                         args[0]["controls"],
                                         args[0]["object_grasped"])
        else:
            env_reward = self._reward_fn(args["old_obs"],
                                         args["new_obs"],
                                         args["init_obj_pos"],
                                         args["final_obj_pos"],
                                         args["target_dir"],
                                         args["gripper_open"],
                                         args["controls"],
                                         args["object_grasped"])

        if self.config.her_buffer:
            dist = np.linalg.norm(desired_goal - achieved_goal)
            goal_reward = 1/math.exp(dist)
            return env_reward + goal_reward
        else:
            return env_reward

    def get_observation(self):
        """
        Return the observation from the gripper camera.
        :return: [np.array] the observation
        """
        rgb, depth = self._sensor.render_images(camera_id="gripper_camera")
        sensor_pad = np.zeros(self._sensor.state_space["observation"].shape[1:3])
        sensor_pad[0][0] = self._actuator.check_grasp("object")
        sensor_pad[0][1] = self._actuator.pheromone_level(self.target_direction)

        if not self.config.full_observation:
            obs = np.dstack((rgb, sensor_pad)).astype(np.uint8)
        else:
            obs = np.dstack((rgb, depth, sensor_pad)).astype(np.uint8)

        if self.config.show_obs:
            self.render()

        return hwc_to_chw(obs)

    def setup_spaces(self):
        """
        Set up the observation and action spaces.
        """
        self.action_space = self._actuator.setup_action_space()
        self.observation_space = self._sensor.setup_observation_space()

    def render(self, mode='human'):
        """
        Render the environment.
        :param mode: the rendering mode
        """
        single_camera = [0, 1, 2]

        def rgb_or_depth(mode, camera_id):
            """
            Return the rgb or depth image from given camera.
            :param mode: rgb or depth
            :param camera_id: selected camera output
            :return: [np.array] the rgb or depth image
            """
            if mode == 'depth':
                _, depth = self._sensor.render_images(camera_id=camera_id,
                                                      w_zoom=self.config.rendering_zoom_width,
                                                      h_zoom=self.config.rendering_zoom_height)
                return depth
            else:
                rgb, _ = self._sensor.render_images(camera_id=camera_id,
                                                    w_zoom=self.config.rendering_zoom_width,
                                                    h_zoom=self.config.rendering_zoom_height)
                return rgb

        if self.config.camera_id in single_camera:
            result = rgb_or_depth(mode, self.config.camera_id)
        else:
            img = []
            for camera in range(self.config.camera_id):
                img.append(rgb_or_depth(mode, camera))
            result = np.hstack(img)

        if mode == 'rgb_array' or mode == 'depth_array':
            return result
        elif mode == 'human':
            cv2.imshow("Camera", cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            time.sleep(0.01)

    def seed(self, seed=None):
        """
        Set the random seed for the environment.
        :param seed: the random seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return self.np_random

    def close(self):
        """
        Close rendering window.
        """
        cv2.destroyAllWindows()
