from easydict import EasyDict


config = EasyDict()

config.dataset_dir = './datasets/'
config.tensorboard_dir = './tensorboard/'
config.checkpoint_dir = './checkpoints/'

config.lr = 1e-4
config.momentum = 0.9
config.weight_decay = 1.6e-3

config.load_iter = 0
config.train_iters = 0
config.is_train = True
config.use_cuda = True
