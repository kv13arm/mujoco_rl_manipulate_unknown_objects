import numpy as np
import gym
import cv2
from sklearn.preprocessing import MinMaxScaler
from simulation.controller.sensor import RGBDSensor


class Actuator:
    def __init__(self, robot, config):
        self.physics = robot
        self.config = config

        self._sensor = RGBDSensor(robot=self.physics, config=config)

        # Last gripper action
        self._gripper_open = True

    def reset(self):
        """
        Reset the gripper to its open position.
        """
        self.open_gripper(half_open=True)
        self._gripper_open = True

    def step(self, action):
        """
        Perform a step with the actuator.
        :param action: [np.array] the action to perform
        :return: moves the gripper
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
        # open/close the gripper
        open_close = action[-1]
        # move the robot
        self.robot.relative_pose(translation, rotation)
        if open_close > 0. and not self._gripper_open:
            self.open_gripper()
        elif open_close < 0. and self._gripper_open:
            self.close_gripper()

    def close_gripper(self):
        """
        Close the gripper. If an object is detected, the gripper grasps the object,
        otherwise, the gripper closes completely.
        :return: [bool] True if the gripper grasped an object, False otherwise
        """
        # debug
        # for _ in range(5):
        #     self.physics.step()

        self._gripper_open = True
        # set the ctrl (motors) controlling the gripper left and right knuckle_link to -1
        self.physics.data.ctrl[6:8] = -1
        # set the position of the gripper links when fully closed
        target_pos = [-0.4, -0.4]

        step = 0
        camera_id = 3

        while self._gripper_open:
            self.physics.step()
            step += 1

            if self.config.show_obs:
                self.physics.render()

            # img = []
            # for camera in range(camera_id):
            #     rgb, _ = self._sensor.render_images(camera_id=camera,
            #                                            w_zoom=self.config.rendering_zoom_width,
            #                                            h_zoom=self.config.rendering_zoom_height)
            #     img.append(rgb)
            #
            # result = np.hstack(img)
            # cv2.imwrite(f'images6/{step}_g.png',  cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            # print(self.physics.data.qpos)
            # print('sensordata', self.physics.data.sensordata)
            # print('step', step)

            # check if the object is securely grasped
            object_securely_graspped = self.physics.data.sensordata[0] > self.config.grasp_tolerance
            # check the gripper's links positions
            gripper_fully_closed = np.all(np.abs(self.physics.data.qpos[5:7] -
                                                 [target_pos]) < self.config.gripper_tolerance)
            if object_securely_graspped or gripper_fully_closed:
            # if gripper_fully_closed:
                self.physics.data.ctrl[6:8] = 0
                self._gripper_open = False
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

        while not self._gripper_open:
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
                self._gripper_open = True
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

    def push(self):

        # push only when the gripper is closed
        if not self._gripper_open:
            # push
            pass
        else:
            self.close_gripper()

        pass

    def pull(self):
        pass

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
        if length > self._max_translation:  # TODO: change the norm for when x, y, z are max_translation
            translation *= self._max_translation / length

        # clip roll and yaw angle to a rotation of +- 0.15 rad
        rotation = np.clip(rotation, -self.config.max_rotation, self.config.max_rotation)

        return translation, rotation
