import argparse
import os
from utils.misc import mkdir

class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.parser = None
        self.is_train = True

    
    def initialize(self, parser):
        # Basic Parameters
        parser.add_argument('--dataroot', type=str, required=True, help='path to images')
        parser.add_argument('--name', type=str, default='experiment name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models saved here')

        # Model Parameters
        parser.add_argument('--model', type=str, default='cae', help='type of compression model')
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
        parser.add_argument('--epoch', type=str, default=1, help='epoch to continue training from. Load best model if set to \'best\'')

        # Dataset Parameters
        parser.add_argument('--max_dataset_size', type=int, default=float('inf'), help='max number of samples')
        parser.add_argument('--crop_size', type=int, default=300, help='crop to this size')
        parser.add_argument('--resize_size', type=int, default=128, help='then scale images to this size')

        # Misc Parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging info')
        return parser

    
    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        self.parser = parser
        return parser.parse_args()


    def print_options(self, opt):
        message = ''
        message += '-------------------- Options --------------------'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------- End --------------------'
        print(message)

        # save to disk
        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        mkdir(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    
    def parse(self):
        opt = self.gather_options()
        opt.is_train = self.is_train

        self.print_options(opt)
        self.opt = opt
        return self.opt