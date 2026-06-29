import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf
from datetime import datetime
import torch
print("🚀 Starting script...")
FLAGS = flags.FLAGS

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

# config_flags.DEFINE_config_file(
#     "config", 'configs/espiritdiff/ncsnpp_continuous.py', "Training configuration.", lock_config=True
# )
config_flags.DEFINE_config_file(
    "config", 'configs/espiritdiff/ncsnpp_continuous.py', "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", 'results', "Work directory.")
flags.DEFINE_enum("mode", 'sample', ["train", "sample"], "Running mode: train or sample") 
flags.DEFINE_string("sample_file", "", "Only sample one example .mat file, e.g. example1.mat. Empty means all examples.")
flags.DEFINE_float("sampling_snr", -1.0, "Override config.sampling.snr when >= 0.")
flags.DEFINE_float("sampling_mse", -1.0, "Override config.sampling.mse when >= 0.")
flags.DEFINE_float("sampling_corrector_mse", -1.0, "Override config.sampling.corrector_mse when >= 0.")
flags.DEFINE_integer("sample_slice_start", -1, "Override config.data.sample_slice_start when >= 0.")
flags.DEFINE_integer("sample_slice_end", -1, "Override config.data.sample_slice_end when >= 0.")
flags.DEFINE_enum("normalize_scope", "", ["", "subject", "slice"], "Override config.data.normalize_scope.")
# flags.DEFINE_enum("mode", 'sample', ["train", "sample"], "Running mode: train or sample")
flags.mark_flags_as_required(["config", "workdir", "mode"])


def main(argv):
    assert FLAGS.workdir == "results"
    if FLAGS.sample_file:
        FLAGS.config.unlock()
        FLAGS.config.data.sample_files = [FLAGS.sample_file]
        FLAGS.config.lock()
    if (
        FLAGS.sampling_snr >= 0
        or FLAGS.sampling_mse >= 0
        or FLAGS.sampling_corrector_mse >= 0
        or FLAGS.sample_slice_start >= 0
        or FLAGS.sample_slice_end >= 0
        or FLAGS.normalize_scope
    ):
        FLAGS.config.unlock()
        if FLAGS.sampling_snr >= 0:
            FLAGS.config.sampling.snr = FLAGS.sampling_snr
        if FLAGS.sampling_mse >= 0:
            FLAGS.config.sampling.mse = FLAGS.sampling_mse
        if FLAGS.sampling_corrector_mse >= 0:
            FLAGS.config.sampling.corrector_mse = FLAGS.sampling_corrector_mse
        if FLAGS.sample_slice_start >= 0:
            FLAGS.config.data.sample_slice_start = FLAGS.sample_slice_start
        if FLAGS.sample_slice_end >= 0:
            FLAGS.config.data.sample_slice_end = FLAGS.sample_slice_end
        if FLAGS.normalize_scope:
            FLAGS.config.data.normalize_scope = FLAGS.normalize_scope
        FLAGS.config.lock()
    print(FLAGS.config)
    if FLAGS.mode == "train":
        TIMESTAMP = "{0:%Y_%m_%dT%H_%M_%S}".format(datetime.now())

        MODEL_ID = "_".join(
            [
                TIMESTAMP,
                FLAGS.config.model.name,
                FLAGS.config.training.sde,
                FLAGS.config.training.estimate_csm,
                str(FLAGS.config.training.csm),
                "alpha",
                str(FLAGS.config.model.sigma_max),
                FLAGS.config.data.normalize_type,
                "N",
                str(FLAGS.config.model.num_scales),
            ]
        )
    
        FLAGS.workdir = os.path.join(FLAGS.workdir, MODEL_ID)
        if FLAGS.config.data.padding:
            FLAGS.workdir = FLAGS.workdir + 'zeropadding'
        else:
            FLAGS.workdir = FLAGS.workdir + 'net_bilinear interpolation'

        tf.io.gfile.makedirs(FLAGS.workdir)
        gfile_stream = open(os.path.join(FLAGS.workdir, "stdout.txt"), "w")
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel("INFO")
        # Run the training pipeline
        print("Start training...")
        run_lib.train(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "sample":
        FLAGS.workdir = os.path.join(FLAGS.workdir, FLAGS.config.sampling.folder)
        # Run the sampling pipeline
        if FLAGS.config.sampling.auto_tuning:
            for auto_index in range(1, 100):
                snr = auto_index / 100.0
                FLAGS.config.sampling.snr = snr
                run_lib.sample(FLAGS.config, FLAGS.workdir)
        else: 
            run_lib.sample(FLAGS.config, FLAGS.workdir)
            # test_correct_mse.sample(FLAGS.config, FLAGS.workdir)
            # test_mse.sample(FLAGS.config, FLAGS.workdir)
            # test_mask_acs.sample(FLAGS.config, FLAGS.workdir)
            # test_mask_pattern.sample(FLAGS.config, FLAGS.workdir)
            # test_mask_acclimit.sample(FLAGS.config, FLAGS.workdir)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    print("🚀 Starting main...")
    app.run(main)
