import gym
import argparse
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate():
    eval_env = gym.make('Ant-Gripper-v0')

    mean_reward, std_reward = evaluate_policy('./models/best_model/best_model.pkl', eval_env, n_eval_episodes=100)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./models/best_model/best_model.pkl')
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()
    evaluate()
