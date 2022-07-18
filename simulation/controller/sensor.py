import copy
import gym
import numpy as np


class RGBDSensor:
    def __init__(self, robot, config):
        self.physics = robot
        self.config = config

    def setup_observation_space(self):
        """
        Set up the observation space.
        :return: [gym.spaces.Box] observation space
        """
        if not self.config.full_observation:
            # RGB output
            self.state_space = gym.spaces.Box(low=0, high=255,
                                              shape=(self.config.height_capture, self.config.width_capture, 3),
                                              dtype=np.uint8)
        else:
            # RGB + Depth + gripper width
            self.state_space = gym.spaces.Box(low=0, high=255,
                                              shape=(self.config.height_capture, self.config.width_capture, 5),
                                              dtype=np.float32)
        return self.state_space

    def render_images(self, camera_id, w_zoom=1, h_zoom=1):
        """
        Render images from the camera.
        :param camera_id: selected camera
        :param w_zoom: width zoom, default 1, no zoom size is 64x64
        :param h_zoom: height zoom, default 1
        :return: [np.array] rgb and depth images
        """
        rgb = copy.deepcopy(self.physics.render(camera_id=camera_id,
                                                width=int(self.config.width_capture * w_zoom),
                                                height=int(self.config.height_capture * h_zoom)))

        # Depth is a float array, in meters
        depth = copy.deepcopy(self.physics.render(camera_id=camera_id,
                                                  width=int(self.config.width_capture * w_zoom),
                                                  height=int(self.config.height_capture * h_zoom),
                                                  depth=True))

        # For debugging
        # depth -= depth.min()
        # # Scale by 2 mean distances of near rays.
        # depth /= 2 * depth[depth <= 1].mean()
        # # Scale to [0, 255]
        # pixels = 255 * np.clip(depth, 0, 1)
        # cv2.imwrite(f"depth_{camera}.png", pixels)

        return rgb.astype(np.uint8), depth.astype(np.float32)
