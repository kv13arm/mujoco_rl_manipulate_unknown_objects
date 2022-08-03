import gym
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

        self._workspace = {'lower': np.array([-0.6, -0.6, 0.1]),
                           'upper': np.array([0.6, 0.6, 0.5])}
        self._roll_rotation = {'lower': -np.pi / 4,
                               'upper': np.pi / 4}

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

    def scale_control(self, qpos, open_close):
        controls = np.r_[qpos, open_close].reshape(1, -1)
        return self._action_scaler.transform(controls).squeeze()[:5]

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
        # set the position of the gripper links when fully closed
        target_pos = [-0.4, -0.4]
        # set the ctrl (motors) controlling the gripper left and right knuckle_link to -1
        self.physics.data.ctrl[5:7] = -1

        return target_pos

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
        self.physics.data.ctrl[5:7] = 0.5

        return target_pos

    def check_grasp(self, object):
        """
        Check if the gripper is grasping an object.
        :return: [bool] True if the gripper is grasping an object, False otherwise
        """
        finger_1_geom_names = ['left_inner_knuckle', 'left_inner_finger']
        finger_2_geom_names = ['right_inner_knuckle', 'right_inner_finger']
        object_geom_name = object

        # get the geom ids from the names
        finger_1_geom_ids = [
            self.physics.named.model.geom_bodyid[x] for x in finger_1_geom_names
        ]
        finger_2_geom_ids = [
            self.physics.named.model.geom_bodyid[x] for x in finger_2_geom_names
        ]
        object_geom_id = self.physics.named.model.geom_bodyid[object_geom_name]

        # touch/contact detection flag
        touch_finger_1 = False
        touch_finger_2 = False

        # int ncon: number of detected contacts
        for i in range(self.physics.data.ncon):
            # list of all detected contacts
            contact = self.physics.data.contact[i]
            # if the object is in the detected contact geom1
            if self.physics.model.geom_bodyid[contact.geom1] == object_geom_id:
                # whether the finger 1 in the detected contacts
                if self.physics.model.geom_bodyid[contact.geom2] in finger_1_geom_ids:
                    # result: finger 1 touched the object
                    touch_finger_1 = True
                if self.physics.model.geom_bodyid[contact.geom2] in finger_2_geom_ids:
                    # finger 2 touched the object
                    touch_finger_2 = True
                    # if the object is in the detected contact geom2
            elif self.physics.model.geom_bodyid[contact.geom2] == object_geom_id:
                if self.physics.model.geom_bodyid[contact.geom1] in finger_1_geom_ids:
                    # finger 1 touched the object
                    touch_finger_1 = True
                if self.physics.model.geom_bodyid[contact.geom1] in finger_2_geom_ids:
                    # finger 2 touched the object
                    touch_finger_2 = True
        if not touch_finger_1 and not touch_finger_2:
            return 0
        elif touch_finger_1 and not touch_finger_2:
            return 1
        elif not touch_finger_1 and touch_finger_2:
            return 2
        else:
            return 3

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

    def pheromone_level(self):
        """
        Get the current pheromone level of the environment.
        :return: [float] the current pheromone level of the environment
        """
        target_dir = self.physics.target_direction
        ee_pos = self.physics.named.data.xpos["ee"]
        project_ee = (np.dot(ee_pos, target_dir) / np.linalg.norm(target_dir) ** 2) * target_dir
        return -np.linalg.norm(project_ee - ee_pos)

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
