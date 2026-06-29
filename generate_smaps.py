import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time

bart_python_path = os.environ.get("BART_PYTHON_PATH")
if bart_python_path:
    sys.path.append(bart_python_path)

import h5py
import numpy as np
import scipy.io as scio
from tqdm import tqdm
import cfl


def get_bart():
    try:
        from bart import bart
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import BART python module. Please check sys.path and TOOLBOX_PATH "
            "for your BART installation. You can set BART_PYTHON_PATH to the BART python folder."
        ) from exc
    return bart


def check_gpu():
    """Print available NVIDIA GPU information via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print("GPU check failed: nvidia-smi is unavailable or cannot communicate with the NVIDIA driver.")
        print(str(exc))
        return False

    print("Detected NVIDIA GPUs:")
    print(result.stdout.strip())
    return "A40" in result.stdout


def ecalib_supports_gpu():
    """Return whether the installed BART ecalib command advertises a GPU flag."""
    try:
        result = subprocess.run(["bart", "ecalib", "-h"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return False
    help_text = result.stdout + result.stderr
    return "-g" in help_text or "GPU" in help_text


def _expand_values(values, name, max_dims=3):
    if len(values) == 1:
        return [values[0]] * max_dims
    if len(values) in (2, 3):
        return list(values)
    raise ValueError(f"{name} expects 1, 2, or 3 values, got {values}")


def build_ecalib_cmd(
    num_maps,
    calib,
    crop,
    kernel=None,
    soft_sense=False,
    use_weights=False,
    use_gpu=False,
):
    calib = _expand_values(calib, "calib")
    flags = ["ecalib", "-d", "0"]
    if kernel is not None:
        kernel = _expand_values(kernel, "kernel")
        flags.extend(["-k", *[str(v) for v in kernel]])
    flags.extend(["-r", *[str(v) for v in calib]])
    if soft_sense:
        flags.append("-S")
    flags.extend(["-m", str(num_maps), "-c", str(crop)])
    if use_weights:
        flags.append("-W")
    if use_gpu:
        if ecalib_supports_gpu():
            flags.append("-g")
        else:
            print("Warning: --use_gpu was requested, but this BART ecalib does not advertise a -g/GPU option.")
            print("         Falling back to CPU ecalib.")
    return flags


def call_bart_ecalib(
    kspace,
    num_maps,
    calib,
    crop,
    kernel=None,
    soft_sense=False,
    use_weights=False,
    use_gpu=False,
):
    """Call BART ecalib via python API when available, otherwise via CLI CFL files."""
    cmd = build_ecalib_cmd(
        num_maps,
        calib,
        crop,
        kernel=kernel,
        soft_sense=soft_sense,
        use_weights=use_weights,
        use_gpu=use_gpu,
    )
    print("Running BART:", " ".join(cmd), flush=True)
    try:
        bart = get_bart()
        start = time.time()
        smaps = bart(1, " ".join(shlex.quote(x) for x in cmd), np.ascontiguousarray(kspace))
        print(f"BART finished in {time.time() - start:.1f}s", flush=True)
        return smaps
    except ModuleNotFoundError:
        pass

    with tempfile.TemporaryDirectory(prefix="bart_ecalib_") as tmpdir:
        input_name = os.path.join(tmpdir, "kspace")
        output_name = os.path.join(tmpdir, "sens")
        cfl.writecfl(input_name, np.ascontiguousarray(kspace))
        start = time.time()
        subprocess.run(["bart", *cmd, input_name, output_name], check=True)
        print(f"BART finished in {time.time() - start:.1f}s", flush=True)
        return cfl.readcfl(output_name)


def as_complex_array(x):
    """Convert common HDF5/MAT complex layouts to a complex64 ndarray."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return np.ascontiguousarray(x.astype(np.complex64))

    # Common real/imag layout: [..., 2].
    if x.ndim >= 1 and x.shape[-1] == 2 and x.dtype.kind in ("f", "i", "u"):
        x = x[..., 0] + 1j * x[..., 1]
        return np.ascontiguousarray(x.astype(np.complex64))

    raise ValueError(
        "Input k-space is not complex. Expected complex dtype or real/imag last dimension [..., 2]."
    )


def load_kspace(path, key):
    if path.endswith(".cfl"):
        return as_complex_array(cfl.readcfl(path[:-4]))

    if path.endswith((".h5", ".hdf5")):
        with h5py.File(path, "r") as hf:
            if key not in hf:
                raise KeyError(f"{path} does not contain dataset key '{key}'. Available keys: {list(hf.keys())}")
            return as_complex_array(hf[key][:])

    if path.endswith(".mat"):
        mat = scio.loadmat(path)
        if key not in mat:
            raise KeyError(f"{path} does not contain variable '{key}'. Available keys: {list(mat.keys())}")
        return as_complex_array(mat[key])

    raise ValueError(f"Unsupported file type: {path}")


def normalize_kspace_layout(kspace):
    """Ensure k-space layout is [Nx, Ny, Nz, Nc]."""
    if kspace.ndim != 4:
        raise ValueError(f"Expected 3D multi-coil k-space [Nx, Ny, Nz, Nc], got {kspace.shape}")

    # If the coil dimension is accidentally first, e.g. [Nc, Nx, Ny, Nz], move it to last.
    if kspace.shape[0] <= 64 and all(dim > 64 for dim in kspace.shape[1:3]):
        kspace = np.transpose(kspace, (1, 2, 3, 0))

    if kspace.shape[-1] > 128:
        raise ValueError(
            f"The last dimension should be coil count Nc, but got shape {kspace.shape}. "
            "Please transpose the data to [Nx, Ny, Nz, Nc]."
        )

    return np.ascontiguousarray(kspace.astype(np.complex64))


def squeeze_bart_maps(smaps, kspace_shape, num_maps):
    """Normalize BART ecalib output to [Nx, Ny, Nz, Nc, Nm]."""
    smaps = np.asarray(smaps)

    # Remove only trailing singleton dimensions beyond [x, y, z, coil, maps].
    while smaps.ndim > 5 and smaps.shape[-1] == 1:
        smaps = np.squeeze(smaps, axis=-1)

    if smaps.ndim == 4:
        smaps = smaps[..., np.newaxis]

    if smaps.ndim != 5:
        raise ValueError(f"Expected BART maps to have 5 dims [Nx, Ny, Nz, Nc, Nm], got {smaps.shape}")

    if smaps.shape[:4] != kspace_shape:
        raise ValueError(
            f"BART map shape {smaps.shape} does not match k-space shape {kspace_shape} "
            "in the first four dimensions."
        )

    if smaps.shape[-1] != num_maps:
        print(f"Warning: requested {num_maps} map sets, BART returned {smaps.shape[-1]} map sets.")

    return np.ascontiguousarray(smaps.astype(np.complex64))


def squeeze_bart_maps_2d(smaps, nx, ny, nc, num_maps):
    """Normalize 2D BART ecalib output to [Nx, Ny, 1, Nc, Nm]."""
    smaps = np.asarray(smaps)
    while smaps.ndim > 5 and smaps.shape[-1] == 1:
        smaps = np.squeeze(smaps, axis=-1)
    if smaps.ndim == 4:
        smaps = smaps[..., np.newaxis]

    # Expected for input [1, Nx, Ny, Nc]: [1, Nx, Ny, Nc, Nm].
    if smaps.ndim == 5 and smaps.shape[0] == 1 and smaps.shape[1] == nx and smaps.shape[2] == ny:
        smaps = np.transpose(smaps, (1, 2, 0, 3, 4))

    # Some BART builds may return [Nx, Ny, 1, Nc, Nm].
    if smaps.ndim == 5 and smaps.shape[:4] == (nx, ny, 1, nc):
        return np.ascontiguousarray(smaps.astype(np.complex64))

    raise ValueError(
        f"Expected 2D BART maps compatible with [Nx, Ny, 1, Nc, Nm] = "
        f"{(nx, ny, 1, nc, num_maps)}, got {smaps.shape}"
    )


def estimate_3d_smaps(
    kspace,
    num_maps=4,
    calib=(24,),
    crop=0.95,
    kernel=None,
    soft_sense=False,
    use_weights=False,
    use_gpu=False,
):
    """Estimate multiple 3D ESPIRiT sensitivity map sets.

    Args:
        kspace: complex ndarray with shape [Nx, Ny, Nz, Nc].
        num_maps: number of ESPIRiT map sets, i.e. BART ecalib -m.
        calib: ACS/calibration size, i.e. BART ecalib -r.
        crop: eigenvalue crop threshold, i.e. BART ecalib -c.
        use_weights: include BART -W.

    Returns:
        smaps: complex ndarray with shape [Nx, Ny, Nz, Nc, Nm].
    """
    smaps = call_bart_ecalib(
        kspace,
        num_maps=num_maps,
        calib=calib,
        crop=crop,
        kernel=kernel,
        soft_sense=soft_sense,
        use_weights=use_weights,
        use_gpu=use_gpu,
    )
    return squeeze_bart_maps(smaps, kspace.shape, num_maps)


def estimate_slice_wise_smaps(
    kspace,
    num_maps=4,
    calib=(24,),
    crop=0.95,
    kernel=None,
    soft_sense=False,
    use_weights=False,
    use_gpu=False,
):
    """Estimate multiple 2D ESPIRiT map sets for every z slice.

    Input is still [Nx, Ny, Nz, Nc], output is [Nx, Ny, Nz, Nc, Nm].
    """
    nx, ny, nz, nc = kspace.shape
    smaps = np.zeros((nx, ny, nz, nc, num_maps), dtype=np.complex64)
    calib_2d = [calib[0], calib[0]] if len(calib) == 1 else calib[:2]
    if kernel is None:
        kernel_2d = None
    else:
        kernel_2d = [kernel[0], kernel[0]] if len(kernel) == 1 else kernel[:2]

    for z in tqdm(range(nz), desc="Estimating slice-wise CSM"):
        # BART 2D ecalib convention keeps the coil dimension at dim 3:
        # [1, Nx, Ny, Nc], matching common fastMRI-style calls.
        ksp_z = np.ascontiguousarray(kspace[:, :, z, :][np.newaxis, ...])
        smap_z = call_bart_ecalib(
            ksp_z,
            num_maps=num_maps,
            calib=calib_2d,
            crop=crop,
            kernel=kernel_2d,
            soft_sense=soft_sense,
            use_weights=use_weights,
            use_gpu=use_gpu,
        )
        smap_z = squeeze_bart_maps_2d(smap_z, nx, ny, nc, num_maps)
        smaps[:, :, z : z + 1, :, :] = smap_z
    return smaps


def coil_combine(kspace, smaps):
    """Create map-wise combined images [Nx, Ny, Nz, Nm] for quick checking."""
    image_coil = np.fft.ifftshift(
        np.fft.ifftn(np.fft.fftshift(kspace, axes=(0, 1, 2)), axes=(0, 1, 2), norm="ortho"),
        axes=(0, 1, 2),
    )
    return np.sum(image_coil[..., np.newaxis] * np.conj(smaps), axis=3)


def save_output(path, kspace, smaps, combined):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".cfl"):
        cfl.writecfl(path[:-4], smaps)
        return

    with h5py.File(path, "w") as hf:
        hf.create_dataset("kspace", data=kspace)
        hf.create_dataset("s_maps", data=smaps)
        hf.create_dataset("combined_img", data=combined)


def find_input_files(source_path):
    if os.path.isfile(source_path):
        return [source_path]

    files = []
    for root, _, names in os.walk(source_path):
        for name in names:
            if not name.endswith(".cfl"):
                continue
            if name.endswith("_csm.cfl"):
                continue
            files.append(os.path.join(root, name))
    return sorted(files)


def make_output_path(input_path, source_path, save_path):
    if input_path.endswith(".cfl"):
        rel = os.path.relpath(input_path, source_path) if os.path.isdir(source_path) else os.path.basename(input_path)
        if rel.endswith("_ksp.cfl"):
            rel = rel[:-8] + "_csm.cfl"
        elif "_ksp" in rel:
            rel = rel.replace("_ksp", "_csm", 1)
        else:
            rel = os.path.splitext(rel)[0] + "_csm.cfl"
        return os.path.join(save_path, rel)

    if os.path.isfile(source_path):
        base = os.path.basename(input_path)
    else:
        base = os.path.relpath(input_path, source_path)
    stem, _ = os.path.splitext(base)
    return os.path.join(save_path, stem + "_csm.h5")


def main(args):
    if args.check_gpu:
        has_a40 = check_gpu()
        if has_a40:
            print("A40 GPU detected.")
        else:
            print("A40 GPU was not detected or the driver is currently unavailable.")
        print("BART ecalib GPU flag supported:", ecalib_supports_gpu())
        if args.check_only:
            return

    input_files = find_input_files(args.source_path)
    if not input_files:
        raise FileNotFoundError(f"No input .cfl files found under {args.source_path}")

    if args.limit is not None:
        input_files = input_files[: args.limit]

    for file_index, input_file in enumerate(tqdm(input_files, desc="Processing files")):
        output_file = make_output_path(input_file, args.source_path, args.save_path)
        if args.skip_existing and os.path.exists(output_file):
            print(f"[{file_index + 1}/{len(input_files)}] Skip existing {output_file}", flush=True)
            continue

        print(f"[{file_index + 1}/{len(input_files)}] Loading {input_file}", flush=True)
        kspace = load_kspace(input_file, args.kspace_key)
        kspace = normalize_kspace_layout(kspace)
        print(f"  kspace shape: {kspace.shape}, dtype: {kspace.dtype}", flush=True)

        if args.slice_wise:
            smaps = estimate_slice_wise_smaps(
                kspace,
                num_maps=args.num_maps,
                calib=args.calib,
                crop=args.crop,
                kernel=args.kernel,
                soft_sense=args.soft_sense,
                use_weights=args.use_weights,
                use_gpu=args.use_gpu,
            )
        else:
            smaps = estimate_3d_smaps(
                kspace,
                num_maps=args.num_maps,
                calib=args.calib,
                crop=args.crop,
                kernel=args.kernel,
                soft_sense=args.soft_sense,
                use_weights=args.use_weights,
                use_gpu=args.use_gpu,
            )

        combined = None if output_file.endswith(".cfl") else coil_combine(kspace, smaps)
        save_output(output_file, kspace, smaps, combined)
        print(f"Saved {output_file}")
        print(f"  kspace shape: {kspace.shape}")
        print(f"  s_maps shape: {smaps.shape}")
        if combined is not None:
            print(f"  combined_img shape: {combined.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate multiple 3D ESPIRiT CSM sets with BART.")
    parser.add_argument("--source_path", default=".", help="Input file or folder.")
    parser.add_argument("--save_path", default=".", help="Output folder.")
    parser.add_argument("--kspace_key", default="label", help="HDF5/MAT key for k-space data.")
    parser.add_argument("--num_maps", type=int, default=4, help="Number of ESPIRiT map sets, BART ecalib -m.")
    parser.add_argument(
        "--calib",
        type=int,
        nargs="+",
        default=[24],
        help="Calibration region size, BART ecalib -r. Use one value or three values, e.g. --calib 32 24 24.",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        nargs="+",
        default=None,
        help="Kernel size, BART ecalib -k. Use one value or three values, e.g. --kernel 6 6 6.",
    )
    parser.add_argument("--crop", type=float, default=0.95, help="Eigenvalue crop threshold, BART ecalib -c.")
    parser.add_argument("--soft_sense", action="store_true", help="Add BART ecalib -S for Soft-SENSE maps.")
    parser.add_argument("--use_weights", action="store_true", help="Add BART ecalib -W.")
    parser.add_argument("--use_gpu", action="store_true", help="Use BART GPU ecalib when supported by this BART build.")
    parser.add_argument("--check_gpu", action="store_true", help="Print nvidia-smi and BART GPU support information before running.")
    parser.add_argument("--check_only", action="store_true", help="Only run --check_gpu diagnostics and exit.")
    parser.add_argument("--slice_wise", action="store_true", help="Estimate 2D maps slice by slice along Nz.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N input files.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files whose output *_csm.cfl already exists.")
    main(parser.parse_args())
