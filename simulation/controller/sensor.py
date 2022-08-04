import copy
import gym
import numpy as np
from utils.utils import transform_depth


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
            # RGB + contact info and pheromone info
            self.state_space = gym.spaces.Box(low=0, high=255,
                                              shape=(self.config.height_capture, self.config.width_capture, 4),
                                              dtype=np.float32)
        else:
            # RGB + Depth + contact info and pheromone info
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

        # transform depth to uint8
        depth = transform_depth(depth)

        return rgb, depth
