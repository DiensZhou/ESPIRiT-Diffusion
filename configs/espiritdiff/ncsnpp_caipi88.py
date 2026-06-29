from configs.espiritdiff.ncsnpp_continuous import get_config as _base_get_config


def get_config():
    config = _base_get_config()
    config.sampling.mask_file = "mask_caipi/302x256caipi_acc8.8_center48.mat"
    return config
