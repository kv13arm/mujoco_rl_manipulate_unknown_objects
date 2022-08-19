from .base_config import BaseConfig


class EvalConfig(BaseConfig):
    def initialize(self, parser):
        parser = BaseConfig.initialize(self, parser)

        parser.add_argument('--train_env', type=str, default='bread_crumb', help='trained model to evaluate')
        parser.add_argument('--plot_trajectory', type=bool, default=False, help='plot object and robot trajectories')
        parser.add_argument('--save_gif', type=bool, default=False, help='save gif of evaluation rendering')
        parser.add_argument('--reward_type', type=str, default='progress',
                            help='intrinsic motivation (im) or progress (progress) reward')
        parser.add_argument('--render', type=bool, default=False, help='render environment')

        return parser
