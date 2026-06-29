import argparse
import importlib.util
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

import losses
import sampling
import sde_lib
import utils.datasets as datasets
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
from run_lib import format_recon_for_subject_save
from utils.utils import restore_checkpoint, r2c


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("test_recon_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config()
    config.unlock()
    return config


def build_sde(config):
    sde_name = config.training.sde.lower()
    if sde_name == "vpsde":
        return sde_lib.VPSDE(config), 1e-3
    if sde_name == "subvpsde":
        return sde_lib.subVPSDE(config), 1e-3
    if sde_name == "vesde":
        return sde_lib.VESDE(config), 1e-5
    if sde_name == "spiritsde":
        return sde_lib.SpiritSDE(config), 1e-5
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")


def load_score_model(config, workdir):
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    ckpt_path = Path(workdir) / config.sampling.folder / "checkpoints" / f"checkpoint_{config.sampling.ckpt}.pth"
    state = restore_checkpoint(str(ckpt_path), state, device=config.device)
    print(f"load weights: {ckpt_path}")
    return state["model"]


def load_mask(mask_path, device):
    mat = sio.loadmat(mask_path)
    if "mask" not in mat:
        raise KeyError(f"{mask_path} does not contain variable 'mask'")
    mask = torch.as_tensor(mat["mask"], device=device)
    return mask.unsqueeze(0).unsqueeze(0)


def sampling_shape(config):
    if config.training.sde == "vesde" and config.training.csm:
        return (1, config.data.num_channels, config.data.image_size_nx, config.data.image_size_ny)
    if config.sampling.mode == "fastMRI":
        return (15, config.data.num_channels, config.data.image_size, config.data.image_size)
    return (18, config.data.num_channels, config.data.image_size_nx, config.data.image_size_ny)


def prepare_batch(point, config, mask):
    if len(point) == 4:
        k0, csm, kernel, subject_batch = point
        subject = subject_batch[0] if isinstance(subject_batch, (list, tuple)) else str(subject_batch)
    else:
        k0, csm, kernel = point
        subject = ""

    k0 = k0.to(config.device)
    csm = csm.to(config.device)
    kernel = kernel.to(config.device)

    k0 = torch.permute(k0, (0, 3, 1, 2))
    if config.training.sde == "vesde":
        csm = torch.permute(torch.squeeze(csm), (3, 2, 0, 1))
    elif csm.ndim == 4:
        csm = torch.permute(csm, (0, 3, 1, 2))
    else:
        csm = torch.permute(csm[:, :, :, :, 0], (0, 3, 1, 2))

    if config.training.estimate_csm != "sos":
        k0 = k0.repeat(csm.shape[0], 1, 1, 1)

    if config.sampling.mode != "prospective":
        atb = k0 * mask
    else:
        atb = k0
        mask = atb[0:1, 0:1, :, :] != 0

    if config.training.sde == "vesde" and config.training.csm and config.sampling.csm == "sense":
        atb = atb[0:1, :, :, :]
        csm = csm[0:1, :, :, :]

    return atb, csm, kernel, mask, subject


def run(args):
    config = load_config(args.config)
    config.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config.sampling.mode = "retrospective"
    if args.sample_file:
        config.data.sample_files = [args.sample_file]
    config.data.sample_slice_start = args.sample_slice_start
    config.data.sample_slice_end = args.sample_slice_end
    if args.ckpt is not None:
        config.sampling.ckpt = args.ckpt
    if args.mask_file:
        config.sampling.mask_file = args.mask_file
    if args.snr is not None:
        config.sampling.snr = args.snr
    if args.mse is not None:
        config.sampling.mse = args.mse
    if args.corrector_mse is not None:
        config.sampling.corrector_mse = args.corrector_mse

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_model = load_score_model(config, args.workdir)
    score_model.eval()
    sde, sampling_eps = build_sde(config)
    sampling_fn = sampling.get_sampling_fn(
        config,
        sde,
        sampling_shape(config),
        lambda x: x,
        sampling_eps,
    )
    mask = load_mask(config.sampling.mask_file, config.device)
    test_dl = datasets.get_dataset(config, "test")

    slice_count = len(test_dl.dataset)
    recon_chunks = []
    examples = []
    slices = []

    for index, point in enumerate(test_dl):
        print(f"reconstruct slice index {index + 1}/{slice_count}")
        atb, csm, kernel, atb_mask, example_name = prepare_batch(point, config, mask)
        examples.append(example_name)
        slices.append(args.sample_slice_start + index)
        recon, _ = sampling_fn(score_model, atb, kernel, atb_mask, csm)
        recon_complex = r2c(recon)
        recon_chunks.append(format_recon_for_subject_save(recon_complex).numpy())

    recon = np.concatenate(recon_chunks, axis=0)
    save_path = output_dir / "recon.mat"
    sio.savemat(
        save_path,
        {
            "recon": recon,
            "slices": np.asarray(slices, dtype=np.int32),
            "examples": np.asarray(examples, dtype=object),
            "source": "datasets.get_dataset(config, 'test')",
        },
    )
    print(f"saved {save_path} with recon shape {recon.shape}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reconstruction with datasets.get_dataset."
    )
    parser.add_argument("--config", default="configs/espiritdiff/ncsnpp_continuous.py")
    parser.add_argument("--workdir", default="results")
    parser.add_argument("--output_dir", default="test_results")
    parser.add_argument("--sample_file", default="")
    parser.add_argument("--sample_slice_start", type=int, default=500)
    parser.add_argument("--sample_slice_end", type=int, default=501)
    parser.add_argument("--mask_file", default="")
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--mse", type=float, default=None)
    parser.add_argument("--corrector_mse", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
