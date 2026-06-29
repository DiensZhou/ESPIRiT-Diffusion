import argparse
import importlib.util
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

import losses
import sampling
import sde_lib
from models import ddpm, ncsnpp  # noqa: F401
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
from utils.utils import Emat_xyt_complex, fft2c_2d, restore_checkpoint, r2c


MASK_DIRS = ("mask_caipi", "mask_cartesian", "mask_random")
DEFAULT_CONFIGS = {
    "spirit-diffusion": "configs/SPIRiT/ncsnpp_continuous.py",
    "espiritdiff": "configs/espiritdiff/ncsnpp_continuous.py",
}
class PaperMatDataset(Dataset):
    def __init__(self, mat_path):
        self.mat_path = mat_path
        data = sio.loadmat(mat_path)
        if "ksp" in data:
            self.ksp = data["ksp"]
        elif "kspace" in data:
            self.ksp = data["kspace"]
        else:
            raise KeyError(f"{mat_path} must contain 'ksp' or 'kspace'")
        for key in ("csm", "kernel"):
            if key not in data:
                raise KeyError(f"{mat_path} must contain '{key}'")
        self.csm = data["csm"]
        self.kernel = data["kernel"]
        self.slices = data["slices"].ravel() if "slices" in data else np.arange(self.ksp.shape[0])
        self.subject = str(data["subject"].ravel()[0]) if "subject" in data else ""

        if not (self.ksp.shape[0] == self.csm.shape[0] == self.kernel.shape[0]):
            raise ValueError(
                f"first dim mismatch: ksp={self.ksp.shape}, csm={self.csm.shape}, kernel={self.kernel.shape}"
            )

    def __len__(self):
        return self.ksp.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.ksp[idx]),
            torch.from_numpy(self.csm[idx]),
            torch.from_numpy(self.kernel[idx]),
            int(self.slices[idx]),
        )


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("spirit_paper_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config()
    config.unlock()
    return config


def load_score_model(config, workdir):
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    ckpt_path = Path(workdir) / config.sampling.folder / "checkpoints" / f"checkpoint_{config.sampling.ckpt}.pth"
    state = restore_checkpoint(str(ckpt_path), state, device=config.device)
    print(f"load weights: {ckpt_path}")
    return state["model"]


def build_sampling_fn(config):
    if config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(config)
        sampling_shape = (
            1,
            config.data.num_channels,
            config.data.image_size_nx,
            config.data.image_size_ny,
        )
    elif config.training.sde.lower() == "spiritsde":
        sde = sde_lib.SpiritSDE(config)
        sampling_shape = (
            18,
            config.data.num_channels,
            config.data.image_size_nx,
            config.data.image_size_ny,
        )
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    return sampling.get_sampling_fn(config, sde, sampling_shape, lambda x: x, 1e-5)


def load_mask(mask_path, device):
    data = sio.loadmat(mask_path)
    if "mask" not in data:
        raise KeyError(f"{mask_path} does not contain 'mask'")
    mask = torch.as_tensor(data["mask"], device=device)
    return mask.unsqueeze(0).unsqueeze(0)


def find_302x256_masks(root):
    root = Path(root)
    masks = []
    for dirname in MASK_DIRS:
        for path in sorted((root / dirname).glob("*.mat")):
            data = sio.loadmat(path)
            if "mask" not in data:
                continue
            if tuple(data["mask"].shape) == (302, 256):
                masks.append(path)
    if len(masks) != 6:
        raise ValueError(f"Expected 6 masks with shape 302x256, found {len(masks)}: {masks}")
    return masks


def prepare_spirit_inputs(batch, mask, config):
    k0, csm, kernel, slice_idx = batch
    k0 = k0.to(config.device)
    csm = csm.to(config.device)
    kernel = kernel.to(config.device)

    k0 = torch.permute(k0, (0, 3, 1, 2))
    if csm.ndim == 4:
        csm = torch.permute(csm, (0, 3, 1, 2))
    else:
        csm = torch.permute(csm[:, :, :, :, 0], (0, 3, 1, 2))

    k0 = k0.repeat(csm.shape[0], 1, 1, 1)
    atb = k0 * mask
    atb_for_spirit = torch.permute(atb, (1, 0, 2, 3))
    kernel = torch.permute(kernel, (0, 4, 3, 1, 2))
    return atb, atb_for_spirit, csm, kernel, slice_idx


def prepare_espiritdiff_inputs(batch, mask, config):
    k0, csm, kernel, slice_idx = batch
    k0 = k0.to(config.device)
    csm = csm.to(config.device)
    kernel = kernel.to(config.device)

    k0 = torch.permute(k0, (0, 3, 1, 2))
    csm = torch.permute(torch.squeeze(csm), (3, 2, 0, 1))
    if config.sampling.csm == "sense":
        csm = csm[0:1, :, :, :]
    elif config.sampling.csm != "espirit":
        raise ValueError(f"Unknown config.sampling.csm: {config.sampling.csm}")
    k0 = k0.repeat(csm.shape[0], 1, 1, 1)
    atb = k0 * mask
    return atb, csm, kernel, slice_idx


def reconstruct_mask(dataset, sampling_fn, score_model, mask, config, recon_method):
    recon_chunks = []
    slice_indices = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    for index, batch in enumerate(loader):
        print(f"  slice {index + 1}/{len(dataset)}")
        if recon_method == "spirit-diffusion":
            atb, atb_for_spirit, csm, kernel, slice_idx = prepare_spirit_inputs(batch, mask, config)
            recon, _ = sampling_fn(score_model, atb_for_spirit, kernel, mask, csm)
            recon = r2c(recon)
            recon = fft2c_2d(recon)
            recon = Emat_xyt_complex(recon.permute(1, 0, 2, 3), True, csm, 1.0)
        elif recon_method == "espiritdiff":
            atb, csm, kernel, slice_idx = prepare_espiritdiff_inputs(batch, mask, config)
            recon, _ = sampling_fn(score_model, atb, kernel, mask, csm)
            recon = r2c(recon)
        else:
            raise ValueError(f"Unknown recon_method: {recon_method}")
        recon_chunks.append(torch.permute(recon[:, 0], (1, 2, 0)).unsqueeze(0).detach().cpu().numpy())
        slice_indices.append(int(slice_idx.item()))

    return np.concatenate(recon_chunks, axis=0), np.asarray(slice_indices, dtype=np.int32)


def dataset_id(subject, mat_path):
    if subject:
        return subject
    return Path(mat_path).stem


def output_prefix(recon_method, config):
    if recon_method == "spirit-diffusion":
        return "spiritdiff"
    if recon_method == "espiritdiff":
        if config.sampling.csm == "sense":
            return "sensediff"
        if config.sampling.csm == "espirit":
            return "espiritdiff"
        raise ValueError(f"Unknown config.sampling.csm: {config.sampling.csm}")
    raise ValueError(f"Unknown recon_method: {recon_method}")


def run(args):
    config_path = args.config or DEFAULT_CONFIGS[args.recon_method]

    config = load_config(config_path)
    expected_sde = "spiritsde" if args.recon_method == "spirit-diffusion" else "vesde"
    if config.training.sde.lower() != expected_sde:
        raise ValueError(
            f"{args.recon_method} requires config.training.sde={expected_sde}, "
            f"but {config_path} has {config.training.sde}"
        )
    config.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config.sampling.mode = "retrospective"
    if args.ckpt is not None:
        config.sampling.ckpt = args.ckpt
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    prefix = output_prefix(args.recon_method, config)
    output_dir = Path(args.output_dir or f"{prefix}_paper")
    dataset = PaperMatDataset(args.paper_mat)
    masks = find_302x256_masks(args.mask_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {config.device}")
    print(f"recon_method: {args.recon_method}")
    print(f"config: {config_path}")
    print(f"workdir: {args.workdir}")
    print(f"mask_root: {args.mask_root}")
    print(f"output_dir: {output_dir}")
    print(f"dataset: {args.paper_mat}, slices={dataset.slices.tolist()}, name={dataset_id(dataset.subject, args.paper_mat)}")
    print("masks:")
    for path in masks:
        print(f"  {path}")

    score_model = load_score_model(config, args.workdir)
    score_model.eval()
    sampling_fn = build_sampling_fn(config)

    for mask_path in masks:
        mask = load_mask(mask_path, config.device)
        save_path = output_dir / f"{prefix}{mask_path.stem}{dataset_id(dataset.subject, args.paper_mat)}.mat"
        if save_path.exists() and not args.overwrite:
            print(f"skip existing {save_path}")
            continue

        print(f"reconstruct with {mask_path}")
        recon, slice_indices = reconstruct_mask(
            dataset, sampling_fn, score_model, mask, config, args.recon_method
        )
        data_dict = {
            "recon": recon,
            "mask": mask.detach().cpu().numpy().squeeze(),
            "mask_file": str(mask_path),
            "paper_mat": args.paper_mat,
            "recon_method": args.recon_method,
            "dataset_name": dataset_id(dataset.subject, args.paper_mat),
            "slices": slice_indices,
        }
        if args.save_inputs:
            data_dict.update(
                {
                    "kspace": dataset.ksp,
                    "ksp": dataset.ksp,
                    "csm": dataset.csm,
                    "kernel": dataset.kernel,
                }
            )
        sio.savemat(save_path, data_dict)
        print(f"saved {save_path} with recon shape {recon.shape}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SPIRiT-Diffusion or ESPIRiTDiff on an example .mat file for all 302x256 paper masks."
    )
    parser.add_argument(
        "--recon_method",
        default="spirit-diffusion",
        choices=("spirit-diffusion", "espiritdiff"),
    )
    parser.add_argument("--paper_mat", default="example1.mat")
    parser.add_argument("--config", default="")
    parser.add_argument("--workdir", default="results")
    parser.add_argument("--mask_root", default=".")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_inputs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
