from gym.envs.registration import register


register(
    id='Ant-Gripper-v0',
    entry_point='simulation.environment.robot_env:RobotEnv',
    max_episode_steps=200,
)
