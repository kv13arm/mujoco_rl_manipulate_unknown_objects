import gym
import torch as th
from torch import nn
from stable_baselines3.common.preprocessing import is_image_space


class BaseFeaturesExtractor(nn.Module):
    """
    Copied from stable_baselines3 torch_layers.py.
    Base class that represents a features extractor.
    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


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

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # assert is_image_space(observation_space, check_channels=False), (
        #     "You should use NatureCNN "
        #     f"only with images not with {observation_space}\n"
        #     "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
        #     "If you are using a custom environment,\n"
        #     "please check it using our env checker:\n"
        #     "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        # )
        n_input_channels = observation_space.shape[0]
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
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[..., :-1][None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor, num_direct_features: int = 2) -> th.Tensor:

        # take last channel as direct features
        other_features = nn.Flatten(observations[..., -1])
        # take known amount of direct features, rest are padding zeros
        other_features = other_features[:, :num_direct_features]

        img_output = self.linear(self.cnn(observations[..., :-1]))
        concat = th.cat((img_output, other_features), dim=1)

        return concat
