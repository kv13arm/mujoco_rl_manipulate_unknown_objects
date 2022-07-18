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
    # reset the actuator
    actuator.reset()
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
        self._workspace = {'lower': np.array([-0.5, -0.5, 0.1]),
                           'upper': np.array([0.5, 0.5, 0.5])}
        self._roll_rotation = {'lower': -np.pi/4,
                               'upper': np.pi/4}
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

        return self.obs

    # def step(self):
    #     self.physics.step()

    def close_gripper(self):
        # self._gripper_open = False
        # closed_pos = [1.12810781, -0.59798289, -0.53003607]
        # for i in range(2):
        #     for j in range(3):
        #         self.physics.data.qpos[i * 4 + j] = closed_pos[j]
        # self.physics.step()
        self._actuator.close_gripper()

    # def open_gripper(self):
    #     self._actuator.open_gripper(half_open=False)

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

    def _enforce_constraints(self, position, orientation):
        """
        Enforce constraints on the next robot movement. The robot
        is allowed to move only within the workspace and within a
        certain angle range.
        :param position: [np.array] position of the robot
        :param orientation: [np.array] orientation of the robot
        :return: [np.array] constrained position and orientation
        """
        # if the roll is allowed, the roll angle is constrained
        # to +=-pi/4, else it is set to 0
        # the pitch angle is fixed to 0
        # the yaw angle is not constrained
        if not self.config.include_roll:
            orientation[0] = 0.
        else:
            if orientation[0] > self._roll_rotation['upper']:
                orientation[0] = self._roll_rotation['upper']
            if orientation[0] < self._roll_rotation['lower']:
                orientation[0] = self._roll_rotation['lower']
        orientation[1] = 0.

        position = np.clip(position,
                           self._workspace['lower'],
                           self._workspace['upper'])

        return position, orientation

    def move_to_pose(self, translation, rotation, steps=100):
        """
        Compute joint angles and move the robot to the given pose.
        :param translation: [np.array] translation of the robot in the local frame
        :param rotation: [np.array] rotation of the robot in the local frame
        :param steps: [int] number of maximum steps to move the robot
        """

        if not self.config.include_roll:
            rotation = np.array([0., 0., rotation[0]])
        else:
            rotation = np.array([rotation[0], 0., rotation[1]])
        current_qpos = self.physics.data.qpos[0:5]

        for i in range(steps):
            self.physics.step()
            current_pos = self.physics.named.data.xpos["ee"]
            current_ori_quat = self.physics.named.data.xquat["ee"]
            yaw, pitch, roll = transformations.euler_from_quaternion(current_ori_quat,
                                                                     axes=(0, 0, 0, 1))
            current_ori = np.array([roll, pitch, yaw])

            T_world_old = transformations.compose_matrix(
                angles=current_ori, translate=current_pos)

            T_old_to_new = transformations.compose_matrix(
                angles=rotation, translate=translation)

            T_world_new = np.dot(T_world_old, T_old_to_new)

            position = T_world_new[:3, 3]
            # orientation = current_ori + rotation
            orientation = np.array(transformations.euler_from_matrix(T_world_new))
            target_pos, target_ori = self._enforce_constraints(position, orientation)

            err_pos = target_pos - current_pos
            err_ori = target_ori - current_ori

            # compute the Jacobian
            jac_pos = np.zeros((3, self.physics.model.nv))
            jac_rot = np.zeros((3, self.physics.model.nv))

            mjlib.mj_jacBody(self.physics.model.ptr,
                         self.physics.data.ptr,
                         jac_pos,
                         jac_rot,
                         self.physics.model.name2id('ee', 'body'))

            J_full = np.vstack([jac_pos[:, :5], jac_rot[:, :5]])
            J_inv = np.linalg.pinv(J_full)

            # compute the joint angles
            qpos_next = np.dot(J_inv, np.hstack([[err_pos], [err_ori]]).T).T.squeeze()
            self.physics.data.qpos[0:5] = self.physics.data.qpos[0:5] + qpos_next
            # ori_test = current_ori + rotation
            # qpos_test = np.r_[target_pos, ori_test[0], ori_test[2]]
            # self.physics.data.qpos[0:5] = qpos_test

            self.physics.data.ctrl[0:5] = self.physics.data.qpos[0:5]

            steps -= 1
            print(f"Step {i}")
            if self.config.show_obs:
                self.render()
            # img = []
            # for camera in range(camera_id):
            #     rgb, _ = self._sensor.render_images(camera_id=camera,
            #                                         w_zoom=self.config.rendering_zoom_width,
            #                                         h_zoom=self.config.rendering_zoom_height)
            #     img.append(rgb)
            #
            # result = np.hstack(img)
            # cv2.imwrite(f'images6/{i}.png', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            # self.physics.step()
            if np.abs(target_pos - self.physics.named.data.xpos["ee"]).sum() < 0.1:
                self.physics.data.ctrl[0:5] = 0
                break
        if steps == 0:
            print("Could not move to pose")
            self.physics.data.ctrl[0:5] = current_qpos
            self.physics.step()
            self._actuator.open_gripper()

    def close(self):
        """
        Close the connection to the robot.
        """
        # self.physics.close()
        # mujoco_env.MujocoEnv.close(self)
        cv2.destroyAllWindows()
