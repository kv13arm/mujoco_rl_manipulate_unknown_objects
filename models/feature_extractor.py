import gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AugmentedNatureCNN(BaseFeaturesExtractor):
    """
    Copied from stable_baselines3 torch_layers.py and modified.
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 514):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space["observation"].shape[0] - 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space["observation"].sample()[:-1, ...][None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU())

    def forward(self, observations: th.Tensor, num_direct_features: int = 2) -> th.Tensor:

        # take known amount of direct features, rest are padding zeros
        other_features = observations["observation"][:, -1, :, :][:, 0, :num_direct_features]

        # obs_cnn = th.as_tensor(observations[:-1, ...][None]).float()
        # img_output = self.linear(self.cnn(obs_cnn))
        img_output = self.linear(self.cnn(observations["observation"][:, :-1, :, :]))
        concat = th.cat((img_output, other_features), dim=1)

        return concat
