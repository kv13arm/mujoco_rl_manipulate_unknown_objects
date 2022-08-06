import os
import argparse


class BaseConfig():
    """
    Configuration for the simulation.
    """
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--sim_env', type=str, default='/xmls/acorn_env.xml',
                            help='path to simulation environment xml file')
        parser.add_argument('--verbose', type=bool, default=False, help='whether to show config information')

        # camera parameters
        parser.add_argument('--width_capture', type=int, default=64, help='width of the image to be captured')
        parser.add_argument('--height_capture', type=int, default=64, help='height of the image to be captured')
        parser.add_argument('--rendering_zoom_width', type=int, default=5 * 2, help='width of the rendering zoom')
        parser.add_argument('--rendering_zoom_height', type=int, default=3.75 * 2, help='height of the rendering zoom')
        parser.add_argument('--full_observation', type=bool, default=True,
                            help='True for RGBD observation, False for RGB observation')
        parser.add_argument('--camera_id', type=int, default=3,
                            help='workbench_camera: 0, upper_camera: 1, gripper_camera: 2, all: 3')
        parser.add_argument('--show_obs', type=bool, default=False, help='True for live rendering observations')

        # actuator parameters
        parser.add_argument('--max_rotation', type=float, default=0.15, help='maximum rotation of the gripper')
        parser.add_argument('--max_translation', type=float, default=0.05, help='maximum translation of the gripper')
        parser.add_argument('--grasp_tolerance', type=float, default=0.03, help='joint grasp tolerance')
        parser.add_argument('--pos_tolerance', type=float, default=0.002, help='joint position tolerance')
        parser.add_argument('--include_roll', type=bool, default=True, help='True for including roll in the action')

        # simulation parameters
        parser.add_argument('--max_steps', type=int, default=400,
                            help='maximum number of timesteps to execute an action')
        parser.add_argument('--im_reward', type=bool, default=False, help='True for adding intrinsic reward')
        parser.add_argument('--her_buffer', type=bool, default=False, help='True for adding HER buffer')

        # Markov decision process parameters
        parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor')
        parser.add_argument('--time_horizon', type=int, default=200, help='maximum number of steps per episode')

        # agent parameters
        parser.add_argument('--trained_models', type=str, default='./trained_models',
                            help='path to trained models')
        parser.add_argument('--name', type=str, default='SAC', help='name of the experiment')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: config.name = config.name + suffix: e.g., {model}_{IM_reward}')

        self.initialized = True

        return parser

    def gather_config(self):
        # initialize parser with basic configuration
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser
        config = parser.parse_args()

        return config

    def print_config(self, config):
        message = ''
        message += '---------------- Config -----------------\n'
        for k, v in sorted(vars(config).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save config to the folder
        expr_dir = os.path.join(config.trained_models, config.name)
        if isinstance(expr_dir, list) and not isinstance(expr_dir, str):
            for path in expr_dir:
                if not os.path.exists(path):
                    os.makedirs(path)
        else:
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)

        file_name = os.path.join(expr_dir, 'config.txt')
        with open(file_name, 'wt') as config_file:
            config_file.write(message)
            config_file.write('\n')

    def parse(self):

        config = self.gather_config()

        # process config.suffix
        if config.suffix:
            suffix = ('_' + config.suffix.format(**vars(config))) if config.suffix != '' else ''
            config.name = config.name + suffix

        if config.verbose:
            self.print_config(config)

        self.config = config
        return self.config
