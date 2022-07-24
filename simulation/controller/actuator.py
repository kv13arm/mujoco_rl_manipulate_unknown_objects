import gym
import cv2
import math
import numpy as np
from utils import transformations
from dm_control.mujoco.wrapper.mjbindings import mjlib
from sklearn.preprocessing import MinMaxScaler
from simulation.controller.sensor import RGBDSensor


class Actuator:
    def __init__(self, robot, config):
        self.physics = robot
        self.config = config

        self._sensor = RGBDSensor(robot=self.physics, config=config)

        # Last gripper action
        self.gripper_open = True

        self._workspace = {'lower': np.array([-0.5, -0.5, 0.1]),
                           'upper': np.array([0.5, 0.5, 0.5])}
        self._roll_rotation = {'lower': -np.pi / 4,
                               'upper': np.pi / 4}

    def reset(self):
        """
        Reset the gripper to its open position.
        """
        # self.open_gripper(half_open=True)
        self.gripper_open = True

    def _normalise_action(self, action):
        """
        Normalise the action to actual values.
        :param action: [np.array] provided action
        :return: [np.array] normalised translation, rotation and gripper action (open/close)
        """
        # denormalize action vector from values [-1, 1] to
        # the bounds of the +-max_translation and +-max_rotation
        action = self._action_scaler.inverse_transform(np.array([action]))
        action = action.squeeze()

        if self.config.include_roll:
            # parse the action vector
            # the rotation matrix includes the roll and yaw angles
            translation, rotation = self._clip_translation_vector(action[:3], action[3:5])
        else:
            # the rotation matrix includes only the yaw angle
            translation, rotation = self._clip_translation_vector(action[:3], action[3:4])
        return translation, rotation, action[-1]

    def scale_control(self, qpos):
        _low = np.r_[[-self.config.max_translation] * 3, [-self.config.max_rotation] * 2]
        _high = -_low
        # max_translation and max_rotation will be scaled to 1
        _scaler = MinMaxScaler((-1, 1))
        _scaler.fit(np.vstack((_low, _high)))
        return _scaler.transform(qpos)
        # return self._action_scaler.transform(qpos)

    def _get_current_pose(self):
        current_pos = self.physics.named.data.xpos["ee"]
        current_ori_quat = self.physics.named.data.xquat["ee"]
        yaw, pitch, roll = transformations.euler_from_quaternion(current_ori_quat,
                                                                 axes=(0, 0, 0, 1))
        current_ori = np.array([roll, pitch, yaw])
        return current_pos, current_ori

    def get_target_pose(self, action):

        translation, rotation, open_close = self._normalise_action(action)

        if not self.config.include_roll:
            rotation = np.array([0., 0., rotation[0]])
        else:
            rotation = np.array([rotation[0], 0., rotation[1]])

        current_pos, current_ori = self._get_current_pose()

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
        err_qpos = np.dot(J_inv, np.hstack([[err_pos], [err_ori]]).T).T.squeeze()

        # err_qpos = self._actuator.move_to_pose(current_pos, current_ori,
        #                                        target_pos, target_ori)
        target_qpos = self.physics.data.qpos[:5] + err_qpos
        return target_qpos

    def close_gripper(self):
        """
        Close the gripper. If an object is detected, the gripper grasps the object,
        otherwise, the gripper closes completely.
        :return: [bool] True if the gripper grasped an object, False otherwise
        """
        # debug
        # for _ in range(5):
        #     self.physics.step()

        self.gripper_open = True
        # set the ctrl (motors) controlling the gripper left and right knuckle_link to -1
        self.physics.data.ctrl[6:8] = -1
        # set the position of the gripper links when fully closed
        target_pos = [-0.4, -0.4]

        step = 0
        camera_id = 3

        while self.gripper_open:
            self.physics.step()
            step += 1

            if self.config.show_obs:
                self.physics.render()

            img = []
            for camera in range(camera_id):
                rgb, _ = self._sensor.render_images(camera_id=camera,
                                                       w_zoom=self.config.rendering_zoom_width,
                                                       h_zoom=self.config.rendering_zoom_height)
                img.append(rgb)

            result = np.hstack(img)
            cv2.imwrite(f'images6/0_{step}_close.png',  cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            print(self.physics.data.qpos)
            print('sensordata', self.physics.data.sensordata)
            print('step', step)

            # check if the object is securely grasped
            object_securely_graspped = self.physics.data.sensordata[0] > self.config.grasp_tolerance
            # check the gripper's links positions
            gripper_fully_closed = np.all(np.abs(self.physics.data.qpos[5:7] -
                                                 [target_pos]) < self.config.gripper_tolerance)
            # if object_securely_graspped or gripper_fully_closed:
            if gripper_fully_closed:
                self.physics.data.ctrl[6:8] = 0

                self.gripper_open = False
                break

        return object_securely_graspped

    def open_gripper(self, half_open=False):
        """
        Open the gripper.
        :param half_open: [bool] if True, the gripper opens half way,
               if False, the gripper opens completely
        """

        # set the position of the gripper links when open
        if half_open:
            target_pos = [0, 0]
        else:
            target_pos = [0.4, 0.4]
        # set the ctrl (motors) controlling the gripper left and right knuckle_link to 1
        self.physics.data.ctrl[6:8] = 1

        # step = 0
        # camera_id = 3

        while not self.gripper_open:
            self.physics.step()
            # step += 1
            # print(self.physics.data.qpos)
            # print('step', step)
            # print("diff", np.abs(self.physics.data.qpos[5:7] -
            #                              [target_pos]))
            # img = []
            # for camera in range(camera_id):
            #     rgb, _ = self._sensor.render_images(camera_id=camera,
            #                                            w_zoom=self.config.rendering_zoom_width,
            #                                            h_zoom=self.config.rendering_zoom_height)
            #     img.append(rgb)
            #
            # result = np.hstack(img)
            # cv2.imwrite(f'images/{step}.png', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            # check the gripper's links positions
            gripper_open = np.all(np.abs(self.physics.data.qpos[5:7] -
                                         [target_pos]) < self.config.gripper_tolerance)
            if gripper_open:
                self.physics.data.ctrl[6:8] = 0
                self.gripper_open = True
                break

    def get_gripper_width(self):
        """
        Get the current opening width of the gripper scaled to a range of [0, 1].
        :return: [float] the current opening width of the gripper
        """
        # the fully closed qpos of the left finger is -0.4
        left_finger_pos = 0.4 + self.physics.data.qpos[0]
        # the fully closed qpos of the right finger is -0.4
        right_finger_pos = 0.4 + self.physics.data.qpos[1]

        return (left_finger_pos + right_finger_pos) / 1.6

    def setup_action_space(self):
        """
        Set up the continuous action space.
        :return: [gym.spaces.Box] action space
        """
        def _set_bounds(rotations_num):
            """
            Set the shape of the action space and normalise to values between [-1, 1].
            :param rotations_num: [int] number of rotations; if 1, only yaw is allowed,
                   if 2, yaw and roll are allowed
            :return: action space scaler and action space shape
            """
            _low = np.r_[[-self.config.max_translation] * 3, [-self.config.max_rotation] * rotations_num, -1]
            _high = -_low
            # max_translation and max_rotation will be scaled to 1
            _scaler = MinMaxScaler((-1, 1))
            _scaler.fit(np.vstack((_low, _high)))
            return _scaler, _low.shape

        if self.config.include_roll:
            # action space shape is (6,): x, y, z, roll, yaw, gripper open/close
            self._action_scaler, shape = _set_bounds(2)
        else:
            # action space shape is (5,): x, y, z, yaw, gripper open/close
            self._action_scaler, shape = _set_bounds(1)

        self.action_space = gym.spaces.Box(-1., 1., shape=shape, dtype=np.float32)

        return self.action_space

    def _clip_translation_vector(self, translation, rotation):
        """
        Clip the gripper's translation and rotation at each step.
        :param translation: [np.array] the X, Y, Z position of the gripper
        :param rotation: [np.array] the roll and yaw angle of the gripper
        :return: [np.array] the clipped translation and rotation
        """
        # clip the position coordinates if the gripper translates more than 0.05 m
        length = np.linalg.norm(translation)
        if length > self.config.max_translation:
            translation *= self.config.max_translation / length

        # clip roll and yaw angle to a rotation of +- 0.15 rad
        rotation = np.clip(rotation, -self.config.max_rotation, self.config.max_rotation)

        return translation, rotation

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
