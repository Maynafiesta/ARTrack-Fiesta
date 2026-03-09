from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/app/data/got10k_lmdb'
    settings.got10k_path = '/app/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/app/data/itb'
    settings.lasot_extension_subset_path_path = '/app/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/app/data/lasot_lmdb'
    settings.lasot_path = '/app/data/lasot'
    settings.network_path = '/app/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/app/data/nfs'
    settings.otb_path = '/app/data/otb'
    settings.prj_dir = '/app'
    settings.result_plot_path = '/app/output/test/result_plots'
    settings.results_path = '/app/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/app/output'
    settings.segmentation_path = '/app/output/test/segmentation_results'
    settings.tc128_path = '/app/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/app/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/app/data/trackingnet'
    settings.uav_path = '/app/data/uav'
    settings.vot18_path = '/app/data/vot2018'
    settings.vot22_path = '/app/data/vot2022'
    settings.vot_path = '/app/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

