# ESPIRiT Diffusion

This repository contains code for VWI MRI reconstruction with ESPIRiT/SENSE coil
sensitivity maps and score-based diffusion sampling.

The GitHub package is intentionally source-focused. Large experiment folders,
logs, reconstruction outputs, raw CFL data, and model checkpoints are not
included.

## Contents

- `models/`, `configs/`, `utils/`, `src/`, `op/`: model, config, data, and CUDA
  extension code.
- `main.py`: train/sample entry point using `ml_collections` configs.
- `test_recon.py`: focused reconstruction test entry point.
- `create_vol12_test_mat.py`: utility to build small `.mat` examples from CFL
  k-space, CSM, and kernel files.
- `generate_smaps.py`: BART ESPIRiT CSM generation helper.
- `mask_random/`, `mask_cartesian/`, `mask_caipi/`: sampling masks.
- `example1.mat`, `example2.mat`: small examples generated from `vol12` slices
  301 and 351.

## Example Data

The included examples were generated with:

```bash
python create_vol12_test_mat.py \
  --root "/data1/jiasen/VWI_ksp_test/retrospective302*256" \
  --subject vol12 \
  --slices 301 \
  --output example1.mat

python create_vol12_test_mat.py \
  --root "/data1/jiasen/VWI_ksp_test/retrospective302*256" \
  --subject vol12 \
  --slices 351 \
  --output example2.mat
```

Each file contains `ksp`, `kspace`, `csm`, `kernel`, `subject`, `slices`,
`normalize_coeff`, `ksp_std`, and `normalization_scope`.

## Setup

Install PyTorch for your CUDA version first, then install the remaining Python
dependencies:

```bash
pip install -r requirements.txt
```

If you need to estimate ESPIRiT maps with `generate_smaps.py`, install BART and
ensure both the `bart` command and the BART Python module are available.

## Notes

- Checkpoints are expected under `results/<run_name>/checkpoints/` when running
  sampling.
- Raw CFL files are expected to follow the project convention:
  `<subject>_ksp.cfl`, `<subject>_csm.cfl`, and optionally
  `<subject>_kernel.cfl`.
- The default config in `configs/ve/ncsnpp_continuous.py` uses the VWI
  retrospective data path and can be overridden from the command line.
