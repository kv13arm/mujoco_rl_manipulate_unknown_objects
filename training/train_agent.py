from simulation.config import Config
from simulation.env.robot_env import RobotEnv
from utils import transformations
import numpy as np
import time


def run():
    config = Config()
    env = RobotEnv(config)
    env.reset()
    # env.move_to_pose(np.array([0.3, 0.3, 0.3]), np.array([0.8726646, 3.1415927]))
    # env.move_to_pose(np.array([0.05, 0., 0.]), np.array([0., 0.]))
    env.close_gripper()
    env.move_to_pose(np.array([0.0, 0., 0.05]), np.array([0., 0.]))
    env.move_to_pose(np.array([0.03, 0., 0.]), np.array([0., 0.]))
    env.move_to_pose(np.array([0.03, 0., 0.]), np.array([0., 0.]))
    env.move_to_pose(np.array([0.03, 0., 0.]), np.array([0., 0.]))
    env.move_to_pose(np.array([0.03, 0., 0.]), np.array([0., 0.]))
    # env.open_gripper()
    # env.close_gripper()
    print(env.physics.named.data.qpos)
    print(env.physics.named.data.xpos["world"])
    print(env.physics.named.data.xquat["world"])
    print(env.physics.named.data.xpos["ee"])
    print(env.physics.named.data.xquat["ee"])
    print(transformations.euler_from_quaternion(env.physics.named.data.xquat["ee"]))

    env.render()
    time.sleep(5)
    env.close()


if __name__ == '__main__':
    run()

# import gym
#
# print('begin')
# env = gym.make('gripper-env-v0', config=Config())
# print('here1')
# # env.reset()
#
# for i in range(1000):
#     print('here')
#     env.render()
#     action = [0, 0, 0, 0]
#     state, reward, done, info = env.step(action)
#     print(state)