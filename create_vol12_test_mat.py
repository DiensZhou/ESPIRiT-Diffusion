import argparse
import math
import os

import numpy as np
import scipy.io as sio


def read_cfl_dims(name):
    with open(name + ".hdr", "rt") as f:
        f.readline()
        dims = [int(i) for i in f.readline().split()]
    n = int(np.prod(dims))
    dims_prod = np.cumprod(dims)
    return dims[: np.searchsorted(dims_prod, n) + 1]


def cfl_memmap(name):
    dims = read_cfl_dims(name)
    data = np.memmap(name + ".cfl", dtype=np.complex64, mode="r")
    return np.ndarray(shape=tuple(dims), dtype=np.complex64, buffer=data, order="F")


def selected_processed_ksp(ksp, slices, y_block=4):
    """Match CFLDataset._process_ksp for selected zero-based output slices."""
    nx, ny, nz, nc = ksp.shape
    shifted_x = (np.arange(nx) + nx // 2) % nx
    out = np.empty((len(slices), ny, nz, nc), dtype=np.complex64)

    for out_idx, slice_idx in enumerate(slices):
        if slice_idx < 0 or slice_idx >= nx:
            raise ValueError(f"slice {slice_idx} is out of range [0, {nx - 1}]")

        ifft_idx = (slice_idx + nx // 2) % nx
        weight = np.exp(2j * np.pi * np.arange(nx) * ifft_idx / nx).astype(np.complex64)
        weight /= math.sqrt(nx)

        for y0 in range(0, ny, y_block):
            y1 = min(y0 + y_block, ny)
            block = np.asarray(ksp[shifted_x, y0:y1, :, :])
            out[out_idx, y0:y1, :, :] = np.tensordot(weight, block, axes=(0, 0))

    return out


def processed_ksp_std(ksp, y_block=16):
    """Compute np.std(CFLDataset._process_ksp(ksp)) without materializing it.

    _process_ksp applies shifts plus a unitary IFFT along dim 0. The transform
    preserves the global sum of |x|^2; only the global mean needs special care.
    """
    nx, ny, nz, nc = ksp.shape
    total = nx * ny * nz * nc
    sum_abs2 = 0.0
    for y0 in range(0, ny, y_block):
        y1 = min(y0 + y_block, ny)
        block = np.asarray(ksp[:, y0:y1, :, :])
        sum_abs2 += float(np.sum(np.abs(block) ** 2, dtype=np.float64))

    center_line = np.asarray(ksp[nx // 2, :, :, :])
    processed_mean = math.sqrt(nx) * np.sum(center_line, dtype=np.complex128) / total
    variance = sum_abs2 / total - float(np.abs(processed_mean) ** 2)
    return math.sqrt(max(variance, 0.0))


def main():
    parser = argparse.ArgumentParser(
        description="Create example .mat files from selected vol12 slices."
    )
    parser.add_argument(
        "--root",
        default="/data1/jiasen/VWI_ksp_test/retrospective302*256",
        help="Directory containing vol12_ksp/csm/kernel CFL files.",
    )
    parser.add_argument("--subject", default="vol12")
    parser.add_argument("--slices", type=int, nargs="+", default=[301, 351])
    parser.add_argument("--output", default="test.mat")
    parser.add_argument("--normalize-coeff", type=float, default=1.5)
    parser.add_argument("--y-block", type=int, default=4)
    parser.add_argument(
        "--selected-std",
        action="store_true",
        help="Normalize by only the selected slices instead of the full vol12 processed k-space.",
    )
    args = parser.parse_args()

    subject_base = os.path.join(args.root, args.subject)
    ksp_path = subject_base + "_ksp"
    csm_path = subject_base + "_csm"
    kernel_path = subject_base + "_kernel"

    ksp_raw = cfl_memmap(ksp_path)
    csm_raw = cfl_memmap(csm_path)
    kernel_raw = cfl_memmap(kernel_path) if os.path.exists(kernel_path + ".cfl") else csm_raw

    ksp = selected_processed_ksp(ksp_raw, args.slices, args.y_block)
    ksp_std = np.std(ksp) if args.selected_std else processed_ksp_std(ksp_raw)
    if ksp_std == 0:
        raise ValueError("selected k-space std is zero; cannot normalize")
    ksp = ksp / (args.normalize_coeff * ksp_std)

    csm = np.asarray(csm_raw[args.slices, ...])
    kernel = np.asarray(kernel_raw[args.slices, ...])

    sio.savemat(
        args.output,
        {
            "ksp": ksp,
            "kspace": ksp,
            "csm": csm,
            "kernel": kernel,
            "subject": args.subject,
            "slices": np.asarray(args.slices, dtype=np.int32),
            "normalize_coeff": np.asarray(args.normalize_coeff, dtype=np.float32),
            "ksp_std": np.asarray(ksp_std, dtype=np.float32),
            "normalization_scope": "selected_slices" if args.selected_std else "full_subject",
        },
    )
    print(f"saved {args.output}")
    print(f"ksp shape: {ksp.shape}, csm shape: {csm.shape}, kernel shape: {kernel.shape}")


if __name__ == "__main__":
    main()
