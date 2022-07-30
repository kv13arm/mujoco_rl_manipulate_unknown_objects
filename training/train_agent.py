from simulation.config import Config
from simulation.environment.robot_env import RobotEnv
from utils import transformations
import numpy as np
import time


def run():
    config = Config()
    env = RobotEnv(config)
    env.reset()
    # print(env.physics.named.data.xpos["box_1"])
    # print(env.physics.named.data.xquat["box_1"])
    # env.step(np.array([0, 0, 0, 0, 0, -0.3]))
    # env.step(np.array([-1, 0, 0, 0, 0, -0.3]))
    # env.step(np.array([0, 0, 1, 0, 0, -0.3]))
    # env.step(np.array([0, 0, 0, 0, 0, 0.3]))
    # env.step(np.array([0.1, 0, 0, 0, 0, 0.3]))
    # env.step(np.array([0.5, 0.5, 0.5, 1, 1, 0.3]))
    # env.step(np.array([0.5, 0, 0, 1, 0, 0.3]))
    # env.step(np.array([0.5, 0, 0, -1, 0, -0.3]))
    # env.step(np.array([0.5, 0, 0, 1, 0, 0.3]))
    # env.step(np.array([0.5, 0.5, 0.5, 1, 1, 0.3]))
    # env.step(np.array([1, 0., 0, 0, 0, 0.3]))
    # env.step(np.array([-0.5, -0.3, -1, -1, -1, 0.3]))
    # env.move_to_pose(np.array([0.3, 0.3, 0.3]), np.array([0.8726646, 3.1415927]))
    # env.move_to_pose(np.array([0.05, 0., 0.]), np.array([0., 0.]))
    # env.close_gripper()

    print(env.physics.named.data.qpos)

    print(env.physics.named.data.xpos["object"])
    print(env.physics.named.data.xquat["object"])
    print(env.physics.named.data.xpos["ee"])
    print(env.physics.named.data.xquat["ee"])
    print(transformations.euler_from_quaternion(env.physics.named.data.xquat["ee"]))

    env.render()
    # time.sleep(5)
    # env.close()


if __name__ == '__main__':
    run()
    # make_video("C:/Users/an-de/Documents/MSc AI_UoE/disseration/training/images/*.jpg", "timestep_2e-3_codim_3")

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