from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # Visualization/Print Parameters
        parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

        # Saving and Loading Parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training from a checkpoint')
        parser.add_argument('--override', action='store_true', help='override values for optimizer and scheduler with command line arguments when loading checkpoint')
        parser.add_argument('--phase', type=str, default='train', help='[train | test]')

        # Training Parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        parser.add_argument('--lr_policy', type=str, default='plateau', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--incremental', action='store_true', help='if set, perform incremental training')

        # Parameters for lr_policy == 'plateau' only
        parser.add_argument('--min_lr', type=float, default=1e-9, help='min learning rate')
        parser.add_argument('--max_val_imgs', type=int, default=200, help='number of validation images used for calculating validation loss (for lr_policy = plateau)')


        # parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        # parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        # parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        # parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser
