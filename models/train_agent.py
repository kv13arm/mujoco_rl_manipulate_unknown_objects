from simulation.config import Config
from simulation.environment.robot_env import RobotEnv
from models.feature_extractor import AugmentedNatureCNN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import time


def run():
    config = Config()
    env = RobotEnv(config)
    policy_kwargs = dict(features_extractor_class=AugmentedNatureCNN,
                         share_features_extractor=True)
    env = make_vec_env(lambda: env, n_envs=1)

    model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1).learn(10000)

    # obs = env.reset()
    # env.render()
    # # print(env.observation_space)
    # # print(env.action_space)
    # # print(env.action_space.sample())
    #
    # for step in range(40):
    #     print("Step {}".format(step + 1))
    #     _, reward, done, info = env.step(env.action_space.sample())
    #     print('reward=', reward, 'done=', done)
    #     env.render()
    #     if done:
    #         print("Goal reached!", "reward=", reward)
    #         break


    # check_env(env, warn=True)
    # env.reset()
    # print(env.physics.named.data.xpos["box_1"])
    # print(env.physics.named.data.xquat["box_1"])
    # obs, _, _, info = env.step(np.array([0, 0, 0, 0, 1, -0.3]))
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
    # fext = AugmentedNatureCNN(env.observation_space)
    # fext.forward(obs["observation"])
    # print("test")
    # print(env.physics.named.data.qpos)
    #
    # print(env.physics.named.data.xpos["object"])
    # print(env.physics.named.data.xquat["object"])
    # print(env.physics.named.data.xpos["ee"])
    # print(env.physics.named.data.xquat["ee"])
    # print(transformations.euler_from_quaternion(env.physics.named.data.xquat["ee"]))

    # env.render()
    # time.sleep(5)
    # env.close()


if __name__ == '__main__':
    run()
    # make_video("C:/Users/an-de/Documents/MSc AI_UoE/disseration/models/images/*.jpg", "timestep_2e-3_codim_3")

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