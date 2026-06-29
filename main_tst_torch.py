import argparse
import math
from pathlib import Path

import numpy as np
import scipy.io as sio


DEFAULT_EXAMPLES = ("example1.mat", "example2.mat")


def ifft2c(kspace):
    _, _, nx, ny = kspace.shape
    image = np.fft.ifftshift(kspace, axes=2)
    image = np.transpose(image, [0, 1, 3, 2])
    image = np.fft.ifft(image, axis=-1)
    image = np.transpose(image, [0, 1, 3, 2])
    image = np.fft.fftshift(image, axes=2) * math.sqrt(nx)
    image = np.fft.ifftshift(image, axes=3)
    image = np.fft.ifft(image, axis=-1)
    image = np.fft.fftshift(image, axes=3) * math.sqrt(ny)
    return image


def load_mask(mask_path, shape):
    if not mask_path:
        return np.ones(shape, dtype=np.complex64)

    mat = sio.loadmat(mask_path)
    if "mask" not in mat:
        raise KeyError(f"{mask_path} does not contain variable 'mask'")
    mask = mat["mask"].astype(np.complex64, copy=False)
    if mask.shape != shape:
        raise ValueError(f"{mask_path} has mask shape {mask.shape}, expected {shape}")
    return mask


def reconstruct_example(mat_path, mask_path="", map_index=0):
    mat = sio.loadmat(mat_path)
    for key in ("ksp", "csm"):
        if key not in mat:
            raise KeyError(f"{mat_path} does not contain variable '{key}'")

    ksp = mat["ksp"].astype(np.complex64, copy=False)
    csm = mat["csm"].astype(np.complex64, copy=False)
    if ksp.ndim != 4:
        raise ValueError(f"{mat_path} ksp should be 4D, got {ksp.shape}")
    if csm.ndim != 5:
        raise ValueError(f"{mat_path} csm should be 5D, got {csm.shape}")
    if ksp.shape[0] != 1:
        raise ValueError(f"{mat_path} should contain one example, got ksp shape {ksp.shape}")

    k0 = np.transpose(ksp[0], (2, 0, 1))[None, ...]
    maps = np.transpose(csm[0], (3, 2, 0, 1))
    if map_index < 0 or map_index >= maps.shape[0]:
        raise ValueError(f"--map_index must be in [0, {maps.shape[0] - 1}], got {map_index}")

    mask = load_mask(mask_path, ksp.shape[1:3])
    coil_images = ifft2c(k0 * mask[None, None, :, :])
    recon_maps = np.sum(coil_images[0][None, ...] * np.conj(maps), axis=1)
    recon = recon_maps[map_index]
    return recon.astype(np.complex64, copy=False), recon_maps.astype(np.complex64, copy=False), mask


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for example in args.examples:
        mat_path = Path(example)
        recon, recon_maps, mask = reconstruct_example(mat_path, args.mask_file, args.map_index)
        save_path = output_dir / f"{mat_path.stem}_recon.mat"
        data = {
            "recon": recon,
            "recon_maps": recon_maps,
            "source_file": str(mat_path),
            "map_index": np.asarray(args.map_index, dtype=np.int32),
            "method": "zero_filled_espirit_sense",
        }
        if args.mask_file:
            data["mask"] = mask
            data["mask_file"] = str(args.mask_file)
        sio.savemat(save_path, data)
        print(f"Saved {save_path} with recon shape {recon.shape}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct example .mat files with zero-filled ESPIRiT/SENSE coil combination."
    )
    parser.add_argument("examples", nargs="*", default=list(DEFAULT_EXAMPLES))
    parser.add_argument("--mask_file", default="", help="Optional undersampling mask .mat file.")
    parser.add_argument("--output_dir", default="recon_results")
    parser.add_argument("--map_index", type=int, default=0, help="ESPIRiT map index to save as recon.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
