from easydict import EasyDict


config = EasyDict()

config.dataset_dir = './datasets/'
config.tensorboard_dir = './tensorboard/'
config.checkpoint_dir = './checkpoints/'
config.network = 'fcn32s'  # ['fcn32s', 'fcn16s']

config.batch_size = 1
config.num_workers = 2

config.lr = 1e-4
config.momentum = 0.9
config.weight_decay = 5e-4

config.print_step = 10
config.tensorboard_step = 50
config.checkpoint_step = 5000
config.load_iter = 0
config.train_iters = 100000
config.is_train = True
config.use_cuda = True
