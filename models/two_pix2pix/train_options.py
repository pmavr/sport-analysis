import numpy as np

class opt:

    def __init__(self):
        self.isTrain = True
        self.input_shape = (256, 256, 3)
        self.lambda_a = 100.
        self.input_nc = 3
        self.output_nc = 1
        self.ngf = 64
        self.ndf = 64
        self.norm = 'batch'
        self.no_dropout = False
        self.init_type = 'normal'
        self.use_sigmoid = True
        self.n_layers_D = 3
        self.pool_size = 0
        self.no_lsgan = True
        self.lr = .0002
        self.beta1 = .5
        self.epoch_count = 1
        self.niter = 100
        self.niter_decay = 100
        self.model = 'two_pix2pix'
        # self.print_freq = 100
        self.batchSize = 1
        self.display_id = 1
        # self.save_latest_freq = 5000
        self.save_epoch_freq = 50
        self.dataset_mode = 'two_aligned'
        # self.dataset_mode = 'aligned'
        self.phase1 = 'grass_mask_estimator_train_dataset'
        self.phase2 = 'edge_map_generator_train_dataset'
        self.phase = 'grass_mask_estimator_train_dataset'
        self.dataroot = '/home/panagiotis/dev/projects/Python_Projects/Sport_Analysis/datasets/world_cup_2014_scc/'
        self.resize_or_crop = 'resize_and_crop'
        self.serial_batches = False
        self.max_dataset_size = np.inf
        self.gpu_ids = [0]
        # self.checkpoints_dir = '/modules/scc_model/checkpoints_dir'
        # self.name = 'soccer_seg_detection_pix2pix'
        self.joint_train = 0
        self.which_direction = 'AtoB'
        self.which_epoch = 'latest'
        self.which_model_netD = 'basic'
        self.which_model_netG = 'unet_256'
        self.lr_policy = 'lambda'
        self.loadSize = 286
        self.fineSize = 256
        self.no_flip = False
        self.lambda_A = 100.
        self.continue_train = False
