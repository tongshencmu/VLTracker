class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/OSTrack/work_dirs'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/OSTrack/work_dirs/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/OSTrack/work_dirs/pretrained_networks'
        self.lasot_dir = '/ocean/projects/ele220002p/tongshen/dataset/lasot_new/LaSOTBenchmark'
        self.got10k_dir = '/ocean/projects/ele220002p/tongshen/dataset/got10k/train'
        self.lasot_lmdb_dir = '/ocean/projects/ele220002p/tongshen/dataset/lasot_lmdb'
        self.got10k_lmdb_dir = '/ocean/projects/ele220002p/tongshen/dataset/got10k_lmdb'
        self.trackingnet_dir = '/ocean/projects/ele220002p/tongshen/dataset/trackingnet'
        self.trackingnet_lmdb_dir = '/ocean/projects/ele220002p/tongshen/dataset/trackingnet_lmdb'
        self.coco_dir = '/ocean/projects/ele220002p/tongshen/dataset/coco'
        self.coco_lmdb_dir = '/ocean/projects/ele220002p/tongshen/dataset/coco_lmdb'
        self.tnl2k_dir = '/ocean/projects/ele220002p/tongshen/dataset/tnl2k'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/ocean/projects/ele220002p/tongshen/dataset/vid'
        self.imagenet_lmdb_dir = '/ocean/projects/ele220002p/tongshen/dataset/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
