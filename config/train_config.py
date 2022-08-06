from .base_config import BaseConfig


class TrainConfig(BaseConfig):
    def initialize(self, parser):
        parser = BaseConfig.initialize(self, parser)

        return parser
