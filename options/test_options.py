from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--num_test', type=int, default=10, help='how many test images to run')
        parser.set_defaults(phase='test')
        parser.set_defaults(override=True)  # we don't need optimizers and schedulers for testing
        self.is_train = False
        return parser