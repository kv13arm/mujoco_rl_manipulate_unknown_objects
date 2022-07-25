import gym
import time
import functools
import numpy as np
from enum import Enum
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib
from simulation.controller.sensor import RGBDSensor
from simulation.controller.actuator import Actuator
from utils import transformations
import cv2


def _reset(robot, actuator):
    """
    Reset the robot to its initial state.
    :param robot: the robot to reset
    :param actuator: the actuator to reset
    """
    # constant actuator signal
    robot.reset()
    # gravity compensation
    mg = -(0.438 * robot.model.opt.gravity[2])
    robot.named.data.xfrc_applied["ee", 2] = mg


class RobotEnv(gym.Env):
    class Events(Enum):
        START_EPISODE = 0
        END_EPISODE = 1
        CLOSE = 2
        CHECKPOINT = 3

    class Status(Enum):
        RUNNING = 0
        SUCCESS = 1
        FAIL = 2
        TIME_LIMIT = 3

    def __init__(self, config):
        self.config = config
        self.physics = mujoco.Physics.from_xml_path(config.xml_path)
        self._sensor = RGBDSensor(robot=self.physics, config=config)
        self._actuator = Actuator(robot=self.physics, config=config)
        self._callbacks = {RobotEnv.Events.START_EPISODE: [],
                           RobotEnv.Events.END_EPISODE: [],
                           RobotEnv.Events.CLOSE: [],
                           RobotEnv.Events.CHECKPOINT: []}
        self.register_events()
        self.setup_spaces()

    def set_random_seed(self):
        """
        Set the random seed for the environment.
        """
        np.random.seed(self.config.seed)

    def register_events(self):
        """
        Register the events to be triggered.
        """
        # set up the reset function
        reset = functools.partial(_reset, self.physics, self._actuator)

        # Register callbacks
        self.register_callback(RobotEnv.Events.START_EPISODE, reset)
        # self.register_callback(RobotEnv.Events.START_EPISODE, self._reward_fn.reset)  # TODO: change and link to reward fn
        # self.register_callback(RobotEnv.Events.END_EPISODE, self.curriculum.update)  # TODO: change and link to history
        self.register_callback(RobotEnv.Events.CLOSE, self.close)

    def _trigger_event(self, event, *event_args):
        """
        Trigger the given event.
        :param event: the event to trigger
        :param event_args: the arguments to pass to the event
        """
        for fn, args, kwargs in self._callbacks[event]:
            fn(*(event_args + args), **kwargs)

    def register_callback(self, event, fn, *args, **kwargs):
        """
        Register a callback associated with the given event.
        :param event: the event to register the callback for
        :param fn: the callback function
        :param args: the arguments to pass to the callback function
        :param kwargs: the keyword arguments to pass to the callback function
        """
        self._callbacks[event].append((fn, args, kwargs))

    def reset(self):
        """
        Reset the environment.
        :return: the initial observation
        """
        self._trigger_event(RobotEnv.Events.START_EPISODE)
        self.episode_step = 0
        self.episode_rewards = np.zeros(self.config.time_horizon)
        self.status = RobotEnv.Status.RUNNING
        self.obs = self.get_observation()
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
        init_qpos = self.physics.data.qpos[:5].copy()
        open_close = action[-1]

        target_qpos = self._actuator.get_target_pose(action)

        step_limit = self.config.max_steps

        while not pos_reached["target"]:
            current_qpos = self.physics.data.qpos[:5]
            self.physics.data.ctrl[0:5] = self._actuator.scale_control(target_qpos - current_qpos, open_close)
            self.physics.step()
            if self.config.show_obs:
                self.render()
            step_limit -= 1
            print("Iteration: ", 1000 - step_limit)

            deltas = abs(current_qpos - target_qpos)
            print("qpos difference:", deltas)
            if max(deltas) < self.config.pos_tolerance:
                pos_reached["target"] = True
                self.physics.data.ctrl[0:5] = 0
            if step_limit == 0:
                print("Target not reached. Returning to initial position.")
                target_qpos = init_qpos[:5]

                for i in range(self.config.max_steps):
                    current_qpos = self.physics.data.qpos[:5]
                    self.physics.data.ctrl[0:5] = self._actuator.scale_control(target_qpos - current_qpos, open_close)
                    self.physics.step()
                    if self.config.show_obs:
                        self.render()
                    deltas = abs(current_qpos - target_qpos)
                    print("qpos difference:", deltas)
                    if max(deltas) < self.config.pos_tolerance:
                        pos_reached["initial"] = True
                        self.physics.data.ctrl[0:5] = 0
                        break
                if not pos_reached["target"] and not pos_reached["initial"]:
                    pos_reached["fail"] = True
                    self.status = RobotEnv.Status.FAIL

        if pos_reached["target"]:
            if open_close > 0. and not self.gripper_open:
                target_qpos = self._actuator.open_gripper()
                while not self.gripper_open:
                    deltas = abs(target_qpos - self.physics.data.qpos[5:7])
                    print(deltas)
                    self.physics.step()
                    if self.config.show_obs:
                        self.render()
                    if max(deltas) < self.config.grasp_tolerance:
                        self.physics.data.ctrl[5:7] = 0
                        self.gripper_open = True
            elif open_close < 0. and self.gripper_open:
                target_qpos = self._actuator.close_gripper()
                while self.gripper_open:
                    deltas = abs(target_qpos - self.physics.data.qpos[5:7])
                    # check if the object is securely grasped
                    # object_securely_graspped = self.physics.data.sensordata[0] > self.config.grasp_tolerance
                    object_graspped = self._actuator.check_grasp("box_1")
                    print("Object graspped: ", object_graspped)
                    self.physics.step()
                    if self.config.show_obs:
                        self.render()
                    if max(deltas) < self.config.grasp_tolerance:
                        self.physics.data.ctrl[5:7] = 0
                        self.gripper_open = False

                    if object_graspped:
                        self.gripper_open = False

        # self.status = RobotEnv.Status.SUCCESS
        # print("Final position: ", self.physics.named.data.xpos["ee"])
        # # print("Target position: ", target_pos)
        # print("Final position: ", self.physics.named.data.xquat["ee"])
        # # print("Target position: ", target_ori)
        # print("Position reached: ", pos_reached)

        new_obs = self.get_observation()

        # reward, self.status = self._reward_fn(self.obs, action, new_obs)
        # self.episode_rewards[self.episode_step] = reward

        if self.status != RobotEnv.Status.RUNNING:
            done = True
        elif self.episode_step == self.time_horizon - 1:
            done, self.status = True, RobotEnv.Status.TIME_LIMIT
        else:
            done = False

        if done:
            self._trigger_event(RobotEnv.Events.END_OF_EPISODE, self)

        self.episode_step += 1
        self.obs = new_obs

        # return self.obs, reward, done, {"is_success": self.status==RobotEnv.Status.SUCCESS,
        #                                 "episode_step": self.episode_step,
        #                                 "episode_rewards": self.episode_rewards,
        #                                 "status": self.status,
        #                                 "position_reached": pos_reached,
        #                                 "object_grasped": object_graspped}

    def get_observation(self):
        """
        Return the observation from the gripper camera.
        :return: [np.array] the observation
        """
        rgb, depth = self._sensor.render_images(camera_id="gripper_camera")
        if not self.config.full_observation:
            obs = rgb
        else:
            sensor_pad = np.zeros(self._sensor.state_space.shape[:2])
            sensor_pad[0][0] = self._actuator.get_gripper_width()
            obs = np.dstack((rgb, depth, sensor_pad)).astype(np.float32)

        if self.config.show_obs:
            self.render()
        return obs

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

    def close(self):
        """
        Close rendering window.
        """
        cv2.destroyAllWindows()
