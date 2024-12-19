from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ckpt_dir', type=str)
        parser.add_argument('--vanilla', action='store_const', const=1, default=0)
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        return parser