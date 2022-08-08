import numpy as np
from simulation.environment.robot_env import RobotEnv
from config.eval_config import EvalConfig
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate():
    env = RobotEnv(config)
    model = SAC.load("dqn_lunar", env=env)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    total_dist = []
    line_dist = []
    robot_pos = []
    obj_pos = []

    for episode in range(config.eval_episodes):
        total_step = []
        line_step = []
        robot_step = []
        obj_step = []
        obs = env.reset()
        for i in range(config.max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            total_step.append(info["total_distance"])
            line_step.append(info["line_distance"])
            robot_step.append(info["gripper_position"])
            obj_step.append(info["object_position"])
            # env.render()
            if dones:
                break
        total_dist.append(sum(total_step))
        line_dist.append(sum(line_step))
        robot_pos.append(robot_step)
        obj_pos.append(obj_step)

    total_dist_mean = np.mean(total_dist)
    total_dist_std = np.std(total_dist)

    line_dist_mean = np.mean(line_dist)
    line_dist_std = np.std(line_dist)

    print(f"Total distance travelled by the object:{total_dist_mean:.2f} +/- {total_dist_std:.2f}")
    print(f"Distance travelled on a straight line:{line_dist_mean:.2f} +/- {line_dist_std:.2f}")


if __name__ == "__main__":
    config = EvalConfig.parse()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='./models/best_model/best_model.pkl')
    # parser.add_argument('--stochastic', action='store_true')
    # args = parser.parse_args()
    evaluate()
