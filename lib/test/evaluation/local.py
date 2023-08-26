from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/ocean/projects/ele220002p/tongshen/dataset/got10k_lmdb'
    settings.got10k_path = '/ocean/projects/ele220002p/tongshen/dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/ocean/projects/ele220002p/tongshen/dataset/lasot_lmdb'
    settings.lasot_path = '/ocean/projects/ele220002p/tongshen/dataset/lasot_new/LaSOTBenchmark'
    settings.network_path = '/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/OSTrack/work_dirs/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/ocean/projects/ele220002p/tongshen/dataset/nfs'
    settings.otb_path = '/ocean/projects/ele220002p/tongshen/dataset/OTB2015'
    settings.prj_dir = '/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/OSTrack/work_dirs'
    settings.result_plot_path = '/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/OSTrack/work_dirs/test/result_plots'
    settings.results_path = '/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/OSTrack/work_dirs/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/OSTrack/work_dirs'
    settings.segmentation_path = '/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/OSTrack/work_dirs/test/segmentation_results'
    settings.tc128_path = '/ocean/projects/ele220002p/tongshen/dataset/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/ocean/projects/ele220002p/tongshen/dataset/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/ocean/projects/ele220002p/tongshen/dataset/trackingNet'
    settings.uav_path = '/ocean/projects/ele220002p/tongshen/dataset/UAV123'
    settings.vot_path = '/ocean/projects/ele220002p/tongshen/dataset/VOT2019'
    settings.youtubevos_dir = ''

    return settings

